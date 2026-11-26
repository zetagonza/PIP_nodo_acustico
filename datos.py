"""
datos.py - Gestión del dataset: carpeta datos/ + datos/caracteristicas_captura.csv

  python datos.py resumen                    # registros por estado/sesión, calidad y espacio
  python datos.py descartar --estado normal --sesion s1
  python datos.py descartar --estado normal --sesion s1 --archivo normal_s1_007_....wav
  python datos.py verificar [--reparar]      # WAV sin fila en el CSV, filas sin WAV, restos .tmp
  python datos.py limpiar                    # BORRA lo descartado (pide confirmación)
  python datos.py limpiar --dias 7 --si      # BORRA lo descartado hace más de 7 días, sin preguntar
  python datos.py vivo                       # resultado de las pruebas en vivo (diagnostico.py --real)

"Descartar" NO borra: mueve los WAV a datos/_descartados/<fecha-hora>/ y saca las
filas del CSV (las guarda en filas.csv dentro de esa carpeta). Para que el disco
no se llene, cada vez que descartás se borran solos los descartados con más de
DIAS_RETENCION días, y captura.py se detiene si quedan menos de MIN_LIBRE_MB libres.
"""
import argparse
import csv
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

# Estados que reconoce todo el proyecto (el orden es el del menú de ensayo.py).
# Para agregar uno nuevo, sumalo acá. Sin guiones bajos: el nombre del archivo se
# separa por "_".
#   apagado     motor detenido (ruido de fondo del lugar)
#   normal      sin fallas
#   combustible falla de combustible (llave de paso parcialmente cerrada)
#   aire        falla de aire (admisión restringida con el cebador)
# "desbalanceo" quedó afuera porque no se puede inducir en este generador. Si algún día
# se graba EN ESTE MISMO generador, alcanza con agregarlo a la lista (ensayo.py ya trae sus
# instrucciones). No mezclar acá audios de otra máquina: ver la nota en entrenar.py.
ESTADOS = ["apagado", "normal", "combustible", "aire"]
CARPETA = "datos"
NOMBRE_CSV = "caracteristicas_captura.csv"
DIAS_RETENCION = 7        # los descartados más viejos que esto se borran solos
MIN_LIBRE_MB = 500        # captura.py se detiene si queda menos libre que esto
MAX_DESCARTADOS_MB = 200  # por encima, se avisa
MIN_N = {"apagado": 15}   # mínimo recomendado por estado (el resto: 30)


def ruta_csv(carpeta=CARPETA):
    return os.path.join(carpeta, NOMBRE_CSV)


def leer_filas(carpeta=CARPETA):
    p = ruta_csv(carpeta)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def agregar_fila(fila, carpeta=CARPETA):
    """Agrega una fila al CSV. Si el CSV ya existe con otras columnas (por
    ejemplo, cambiaste caracteristicas.py o captura.py), avisa en vez de mezclar."""
    p = ruta_csv(carpeta)
    campos = list(fila.keys())
    nuevo = not os.path.exists(p) or os.path.getsize(p) == 0
    if not nuevo:
        with open(p, newline="") as fh:
            existentes = next(csv.reader(fh))
        if existentes != campos:
            raise ValueError(
                "El CSV existente tiene columnas distintas a las actuales "
                "(¿cambiaste caracteristicas.py o captura.py?). Renombralo (p. ej. a "
                "caracteristicas_captura_v1.csv) y volvé a empezar, o "
                "regenerá todo desde los WAV con procesar_dataset.py.")
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def contar(estado, sesion, carpeta=CARPETA):
    return sum(1 for f in leer_filas(carpeta)
               if f.get("estado") == estado and f.get("sesion") == sesion)


def siguiente_indice(estado, sesion, carpeta=CARPETA):
    """Próximo número de registro (para --agregar: sigue donde quedó)."""
    mx = 0
    for f in leer_filas(carpeta):
        if f.get("estado") == estado and f.get("sesion") == sesion:
            try:
                mx = max(mx, int(f.get("archivo", "").split("_")[2]))
            except (IndexError, ValueError):
                pass
    return mx + 1


# ----------------------------------------------------------------- espacio
def _tam_dir(ruta):
    total = 0
    for base, _, archivos in os.walk(ruta):
        for a in archivos:
            try:
                total += os.path.getsize(os.path.join(base, a))
            except OSError:
                pass
    return total


def espacio(carpeta=CARPETA):
    """(MB del dataset, MB en _descartados, MB libres en el disco)."""
    desc = os.path.join(carpeta, "_descartados")
    total = _tam_dir(carpeta) if os.path.isdir(carpeta) else 0
    d = _tam_dir(desc) if os.path.isdir(desc) else 0
    return (total - d) / 1e6, d / 1e6, espacio_libre_mb(carpeta)


def espacio_libre_mb(carpeta=CARPETA):
    return shutil.disk_usage(carpeta if os.path.isdir(carpeta) else ".").free / 1e6


def avisos_espacio(carpeta=CARPETA):
    """Lista de textos de aviso (vacía si todo está bien)."""
    av = []
    libre = espacio_libre_mb(carpeta)
    if libre < 2 * MIN_LIBRE_MB:
        av.append(f"quedan {libre:.0f} MB libres en el disco (la captura se detiene por debajo "
                  f"de {MIN_LIBRE_MB} MB). Liberá espacio:  python datos.py limpiar")
    desc = espacio(carpeta)[1]
    if desc > MAX_DESCARTADOS_MB:
        av.append(f"_descartados ocupa {desc:.0f} MB.  Para borrarlo:  python datos.py limpiar")
    huerfanos, _, tmps = verificar(carpeta)
    if huerfanos or tmps:
        av.append(f"{len(huerfanos)} WAV sin fila en el CSV y {len(tmps)} restos .tmp.  "
                  "Para arreglarlo:  python datos.py verificar --reparar")
    return av


def _edad_dias(nombre, ruta):
    try:
        t = datetime.strptime(nombre, "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        t = os.path.getmtime(ruta)
    return (time.time() - t) / 86400


def limpiar_descartados(carpeta=CARPETA, dias=None):
    """Borra DEFINITIVAMENTE lo que hay en carpeta/_descartados: todo si dias es
    None, o solo lo anterior a `dias` días. Devuelve los MB liberados."""
    base = os.path.join(carpeta, "_descartados")
    if not os.path.isdir(base):
        return 0.0
    liberado = 0
    for nombre in sorted(os.listdir(base)):
        ruta = os.path.join(base, nombre)
        if dias is not None and _edad_dias(nombre, ruta) <= dias:
            continue
        if os.path.isdir(ruta):
            liberado += _tam_dir(ruta)
            shutil.rmtree(ruta, ignore_errors=True)
        else:
            liberado += os.path.getsize(ruta)
            os.remove(ruta)
    return liberado / 1e6


# ------------------------------------------------------ descartar / verificar
def descartar(estado, sesion, archivo=None, carpeta=CARPETA):
    """Mueve (no borra) los registros de esa tanda, o uno solo si se da
    `archivo`. Después borra los descartados con más de DIAS_RETENCION días.
    Devuelve (n_filas, n_wavs, carpeta_destino)."""
    filas = leer_filas(carpeta)
    quitar = [f for f in filas if f.get("estado") == estado and f.get("sesion") == sesion
              and (archivo is None or f.get("archivo") == archivo)]
    destino = os.path.join(carpeta, "_descartados",
                           datetime.now().strftime("%Y%m%d-%H%M%S"))
    prefijo = f"{estado}_{sesion}_"          # el "_" final evita confundir s1 con s10
    wavs = 0
    if os.path.isdir(carpeta):
        for nombre in sorted(os.listdir(carpeta)):
            if nombre.startswith(prefijo) and nombre.endswith(".wav") \
                    and (archivo is None or nombre == archivo):
                os.makedirs(destino, exist_ok=True)
                shutil.move(os.path.join(carpeta, nombre), os.path.join(destino, nombre))
                wavs += 1
    if quitar:
        os.makedirs(destino, exist_ok=True)
        campos = list(filas[0].keys())
        ids = {id(f) for f in quitar}
        with open(os.path.join(destino, "filas.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(quitar)
        tmp = ruta_csv(carpeta) + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(f for f in filas if id(f) not in ids)
        os.replace(tmp, ruta_csv(carpeta))
    limpiar_descartados(carpeta, DIAS_RETENCION)
    return len(quitar), wavs, destino


def verificar(carpeta=CARPETA, reparar=False):
    """Compara los WAV del disco con las filas del CSV.
    Devuelve (wav_sin_fila, filas_sin_wav, restos_tmp). Con reparar=True mueve los
    WAV sin fila a _descartados/ y borra los restos .tmp."""
    if not os.path.isdir(carpeta):
        return [], [], []
    en_csv = {f.get("archivo") for f in leer_filas(carpeta)}
    nombres = os.listdir(carpeta)
    wavs = {n for n in nombres if n.endswith(".wav")}
    huerfanos = sorted(wavs - en_csv)
    sin_wav = sorted(a for a in en_csv - wavs if a)
    tmps = sorted(n for n in nombres if n.endswith(".wav.tmp"))
    if reparar:
        destino = os.path.join(carpeta, "_descartados",
                               datetime.now().strftime("%Y%m%d-%H%M%S"))
        for n in huerfanos:
            os.makedirs(destino, exist_ok=True)
            shutil.move(os.path.join(carpeta, n), os.path.join(destino, n))
        for n in tmps:
            os.remove(os.path.join(carpeta, n))
    return huerfanos, sin_wav, tmps


# ------------------------------------------------------------------ resumen
def resumen(carpeta=CARPETA, min_peine=4.0, max_clip=0.01):
    filas = leer_filas(carpeta)
    if not filas:
        print(f"No hay registros en {ruta_csv(carpeta)}")
        return
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["estado"], f["sesion"])].append(f)
    orden = lambda k: (ESTADOS.index(k[0]) if k[0] in ESTADOS else 99, k[1])
    print(f"{'estado':<13}{'sesión':<8}{'n':>4}{'carga%':>8}{'peine dB':>10}"
          f"{'clip':>8}{'perdidos':>9}  avisos")
    for k in sorted(grupos, key=orden):
        g = grupos[k]
        peine = median(float(x["prominencia_peine_db"]) for x in g)
        clip = max(float(x["clip_frac"]) for x in g)
        perd = sum(int(float(x["perdidos"])) for x in g)
        carga = median(float(x["carga_pct"]) for x in g)
        av = []
        if k[0] != "apagado" and peine < min_peine:
            av.append("sin armónicos (¿motor apagado/viento?)")
        if clip > max_clip:
            av.append("saturación")
        if perd:
            av.append("hubo pérdida de datos")
        print(f"{k[0]:<13}{k[1]:<8}{len(g):>4}{carga:>8.0f}{peine:>10.1f}"
              f"{clip:>8.4f}{perd:>9}  {'; '.join(av)}")
    print("\nTotal por estado (mínimo recomendado, en >= 3 sesiones):")
    for e in ESTADOS:
        n = sum(len(g) for k, g in grupos.items() if k[0] == e)
        ses = len({k[1] for k in grupos if k[0] == e})
        marca = "" if n >= MIN_N.get(e, 30) and ses >= 3 else "   <-- falta"
        print(f"  {e:<13}{n:>4} registros en {ses} sesión(es){marca}")
    usado, desc, libre = espacio(carpeta)
    print(f"\nEspacio: dataset {usado:.0f} MB | descartados {desc:.0f} MB | "
          f"libre en disco {libre / 1000:.1f} GB")
    for a in avisos_espacio(carpeta):
        print("AVISO:", a)


# --------------------------------------------------------------------- vivo
def vivo(ruta_log="diagnosticos.csv"):
    """Matriz de confusión de las pruebas en vivo (diagnostico.py --real ...)."""
    if not os.path.exists(ruta_log):
        print(f"No existe {ruta_log}. Corré antes:  python diagnostico.py --real normal --n 10")
        return
    with open(ruta_log, newline="") as fh:
        filas = [f for f in csv.DictReader(fh) if f.get("real")]
    if not filas:
        print(f"{ruta_log} no tiene pruebas con --real.")
        return
    reales = [e for e in ESTADOS if any(f["real"] == e for f in filas)]
    reales += sorted({f["real"] for f in filas} - set(reales))
    pred = [e for e in ESTADOS if any(f["estado"] == e for f in filas)]
    pred += sorted({f["estado"] for f in filas} - set(pred) - {"DESCONOCIDO", "INVÁLIDO"})
    pred += [x for x in ("DESCONOCIDO", "INVÁLIDO") if any(f["estado"] == x for f in filas)]
    conf = defaultdict(Counter)
    for f in filas:
        conf[f["real"]][f["estado"]] += 1
    ancho = max(len(r) for r in reales) + 2
    print("Filas = estado REAL, columnas = lo que respondió el nodo\n")
    print(" " * ancho + "".join(f"{p[:12]:>13}" for p in pred) + f"{'aciertos':>10}")
    total = ok = 0
    for r in reales:
        n = sum(conf[r].values())
        a = conf[r][r]
        total, ok = total + n, ok + a
        print(f"{r:<{ancho}}" + "".join(f"{conf[r][p]:>13}" for p in pred)
              + f"{a / n:>10.0%}")
    print(f"\nExactitud global: {ok / total:.1%}  ({ok} de {total} diagnósticos)")
    print("\n| Estado real | " + " | ".join(pred) + " | Aciertos |")
    print("|" + "---|" * (len(pred) + 2))
    for r in reales:
        n = sum(conf[r].values())
        print(f"| {r} | " + " | ".join(str(conf[r][p]) for p in pred)
              + f" | {conf[r][r] / n:.0%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("resumen")
    s.add_argument("--carpeta", default=CARPETA)
    d = sub.add_parser("descartar")
    d.add_argument("--estado", required=True, choices=ESTADOS)
    d.add_argument("--sesion", required=True)
    d.add_argument("--archivo", default=None, help="descartar solo este registro")
    d.add_argument("--carpeta", default=CARPETA)
    v = sub.add_parser("verificar")
    v.add_argument("--reparar", action="store_true",
                   help="mueve los WAV sin fila a _descartados/ y borra restos .tmp")
    v.add_argument("--carpeta", default=CARPETA)
    c = sub.add_parser("limpiar")
    c.add_argument("--dias", type=float, default=None,
                   help="solo lo descartado hace más de N días (por defecto: todo)")
    c.add_argument("--si", action="store_true", help="no pedir confirmación")
    c.add_argument("--carpeta", default=CARPETA)
    w = sub.add_parser("vivo")
    w.add_argument("--log", default="diagnosticos.csv")
    a = ap.parse_args()

    if a.cmd == "resumen":
        resumen(a.carpeta)
    elif a.cmd == "descartar":
        nf, nw, dest = descartar(a.estado, a.sesion, a.archivo, a.carpeta)
        if nf == 0 and nw == 0:
            print("No había nada para descartar.")
        else:
            print(f"Movidos {nw} WAV y {nf} filas a {dest}")
    elif a.cmd == "verificar":
        huerfanos, sin_wav, tmps = verificar(a.carpeta, a.reparar)
        print(f"WAV sin fila en el CSV : {len(huerfanos)}")
        for n in huerfanos[:20]:
            print("   ", n)
        print(f"Filas sin WAV          : {len(sin_wav)}  (normal si grabaste con --no-wav)")
        print(f"Restos .tmp            : {len(tmps)}")
        if (huerfanos or tmps) and not a.reparar:
            print("\nPara repararlo:  python datos.py verificar --reparar")
        elif a.reparar:
            print("\nReparado: los WAV sin fila se movieron a _descartados/.")
    elif a.cmd == "limpiar":
        desc = espacio(a.carpeta)[1]
        if desc == 0:
            print("No hay nada descartado para borrar.")
            return
        alcance = "todo lo descartado" if a.dias is None else f"lo descartado hace más de {a.dias:g} días"
        if not a.si and input(f"Se BORRA definitivamente {alcance} (hoy {desc:.0f} MB en "
                              f"{a.carpeta}/_descartados). ¿Seguir? [s/N] ").strip().lower() != "s":
            print("Cancelado.")
            return
        print(f"Liberados {limpiar_descartados(a.carpeta, a.dias):.0f} MB")
    else:
        vivo(a.log)


if __name__ == "__main__":
    main()