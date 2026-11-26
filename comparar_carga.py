"""
comparar_carga.py - Prueba con carga: compara el estado NORMAL sin carga con el estado
NORMAL con carga (~50 %), y diagnostica los registros con carga con el modelo entrenado.

Grabación (en la Pi, el mismo día, sin mover el micrófono entre una y otra):
  python captura.py --estado normal --sesion c0  --n-registros 40 --carga 1
  (conectar la carga de ~2 kW y esperar 2 minutos a que se estabilice)
  python captura.py --estado normal --sesion c50 --n-registros 40 --carga 50

Análisis (en la Pi o en la PC, con la carpeta datos/ y modelo.npz):
  python comparar_carga.py datos/caracteristicas_captura.csv --carga c50 --ref c0
  python comparar_carga.py datos/caracteristicas_captura.csv --carga c50          # ref = s1..s5

--ref c0     compara contra el normal sin carga grabado ese mismo día (lo correcto: así
             la diferencia es solo la carga y no el día ni la posición del micrófono).
sin --ref    compara contra el normal de las sesiones de entrenamiento (s1, s2, ...).
--con-fallas agrega a la figura de bandas las fallas de las sesiones de entrenamiento.

IMPORTANTE: el modelo tiene que estar entrenado SIN las sesiones de la prueba con carga:
  python entrenar.py datos/caracteristicas_captura.csv --modo unico --excluir c0 c50

Genera en informe/ (sin textos explicativos dentro de los gráficos):
  fig_carga_bandas.png      energía relativa por banda, normal sin carga y con carga
  fig_carga_espectro.png    densidad espectral media (Welch) de los dos grupos
  fig_carga_bandas_abs.png  nivel ABSOLUTO de cada banda (dBFS), sin carga y con carga
  carga_tabla.csv           media, desvío y diferencia de las características básicas
  carga_diagnostico.csv     diagnóstico del modelo para cada registro de los dos grupos
"""
import argparse
import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import modelo
from caracteristicas import BANDAS_HZ, F_MAX, F_MIN, FS, _SOS_HP
from procesar_dataset import leer_bloques
from reporte import COLORES, MARCAS, SUPERFICIE, TINTA2, estilo, separar_etiquetas

COLOR_CARGA = "#6250d6"
BASICAS = [("f_fire_hz", "Frecuencia de encendido [Hz]"),
           ("rpm", "Velocidad equivalente [rpm]"),
           ("vel_inestabilidad_pct", "Inestabilidad de velocidad [%]"),
           ("rms_dbfs", "Nivel RMS [dBFS]"),
           ("prominencia_peine_db", "Prominencia del peine [dB]"),
           ("centroide_hz", "Centroide espectral [Hz]"),
           ("rolloff85_hz", "Roll-off al 85 % [Hz]"),
           ("h01_db", "Primer armónico h01 [dB]"),
           ("ratio_impar_par_db", "Relación impares/pares [dB]")]


def valor(f, c):
    if c == "rpm":
        return 120.0 * float(f.get("f_fire_hz", "nan") or "nan")
    try:
        return float(f.get(c, "nan") or "nan")
    except ValueError:
        return float("nan")


def psd_media(carpeta, filas, max_wav):
    """Densidad espectral media (potencia lineal promediada) de hasta max_wav registros,
    con el mismo pasa-altos y la misma estimación de Welch que el nodo."""
    acum, n = None, 0
    rutas = [os.path.join(carpeta, f["archivo"]) for f in filas]
    rutas = [r for r in rutas if os.path.exists(r)]
    if len(rutas) > max_wav:
        idx = np.linspace(0, len(rutas) - 1, max_wav).round().astype(int)
        rutas = [rutas[i] for i in idx]
    for r in rutas:
        x = np.concatenate(list(leer_bloques(r)))
        y = signal.sosfilt(_SOS_HP, x, zi=signal.sosfilt_zi(_SOS_HP) * x[0])[0]
        f, p = signal.welch(y, FS, window="hann", nperseg=FS, noverlap=0, detrend=False)
        acum = p if acum is None else acum + p
        n += 1
    return (f, acum / n, n) if n else (None, None, 0)


def diagnosticar(m, filas, max_clip):
    cols = [str(c) for c in m["columnas"]]
    salida = []
    for f in filas:
        if valor(f, "clip_frac") > max_clip:
            salida.append("INVÁLIDO")
            continue
        x = np.array([float(f[c]) for c in cols])
        r = modelo.predecir(m, x)
        salida.append("DESCONOCIDO" if r["rechazo"] else r["clase"])
    return salida


def fig_bandas(grupos, cols, ruta):
    bcols = sorted([c for c in cols if c.startswith("banda_")], key=lambda c: int(c.split("_")[1]))
    etiquetas = [c.split("_")[1] + "–" + c.split("_")[2] for c in bcols]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    finales = []
    for k, (nombre, filas, col, estilo_linea) in enumerate(grupos):
        med = np.array([np.nanmean([valor(f, c) for f in filas]) for c in bcols])
        ax.plot(range(len(bcols)), med, color=col, linewidth=2, linestyle=estilo_linea,
                marker=MARCAS[k % len(MARCAS)], markersize=7, markeredgecolor=SUPERFICIE,
                markeredgewidth=1.5, label=nombre)
        finales.append((nombre, med[-1]))
    todos = [v for _, v in finales]
    rango = max(1e-9, max(todos) - min(todos)) if len(todos) > 1 else 1.0
    for (nombre, _), yy in zip(finales, separar_etiquetas(todos, max(0.6, 0.05 * rango))):
        ax.annotate(nombre, (len(bcols) - 1, yy), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TINTA2)
    ax.set_xticks(range(len(bcols)), etiquetas)
    ax.set_xlim(-0.3, len(bcols) - 1 + 2.2)
    ax.set_xlabel("Banda de frecuencia (Hz)")
    ax.set_ylabel("Energía relativa a la total (dB)")
    ax.set_title("Distribución de energía por banda, normal sin carga y con carga")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(grupos), fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_bandas_abs(espectros, ruta):
    """Nivel ABSOLUTO de cada banda (dBFS), integrando la densidad espectral media."""
    etiquetas = [f"{lo}–{hi}" for lo, hi in BANDAS_HZ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ancho = 0.8 / len(espectros)
    niveles = []
    for nombre, f, p, col in espectros:
        df = f[1] - f[0]
        niveles.append([10 * np.log10(p[(f >= lo) & (f < hi)].sum() * df + 1e-20)
                        for lo, hi in BANDAS_HZ])
    piso = np.floor((min(min(n) for n in niveles) - 6) / 5) * 5
    for k, ((nombre, f, p, col), nv) in enumerate(zip(espectros, niveles)):
        x = np.arange(len(BANDAS_HZ)) + (k - (len(espectros) - 1) / 2) * ancho
        ax.bar(x, np.array(nv) - piso, width=ancho, bottom=piso, color=col, label=nombre)
    ax.set_xticks(range(len(BANDAS_HZ)), etiquetas)
    ax.set_ylim(piso, max(max(n) for n in niveles) + 3)
    ax.set_xlabel("Banda de frecuencia (Hz)")
    ax.set_ylabel("Nivel de la banda (dBFS)")
    ax.set_title("Nivel absoluto por banda, normal sin carga y con carga")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(espectros), fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return niveles


def fig_espectro(espectros, ruta):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for nombre, f, p, col in espectros:
        db = 10 * np.log10(p + 1e-20)
        m = (f >= F_MIN) & (f <= F_MAX)
        ax[0].semilogx(f[m], db[m], color=col, linewidth=0.9, label=nombre)
        z = (f >= F_MIN) & (f <= 160)
        ax[1].plot(f[z], db[z], color=col, linewidth=1.1, label=nombre)
    ax[0].set(title="Densidad espectral media, 20 Hz a 6 kHz", xlabel="Frecuencia (Hz)",
              ylabel="Densidad (dB re FS²/Hz)", xlim=(F_MIN, F_MAX))
    ax[1].set(title="Detalle de 20 a 160 Hz", xlabel="Frecuencia (Hz)",
              ylabel="Densidad (dB re FS²/Hz)", xlim=(F_MIN, 160))
    ax[0].grid(True, which="both", alpha=0.6)
    ax[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--carga", nargs="+", required=True, help="sesiones con carga (ej.: c50)")
    ap.add_argument("--ref", nargs="*", default=None,
                    help="sesiones de referencia sin carga (ej.: c0). Por defecto: s1, s2, ...")
    ap.add_argument("--modelo", default="modelo.npz")
    ap.add_argument("--con-fallas", action="store_true")
    ap.add_argument("--max-wav", type=int, default=40, help="registros por grupo para el espectro")
    ap.add_argument("--max-clip", type=float, default=0.01)
    ap.add_argument("--salida", default="informe")
    ap.add_argument("--nombre-ref", default=None, help='leyenda de la referencia, ej.: "Normal, carga de 40 W"')
    ap.add_argument("--nombre-carga", default=None, help='leyenda con carga, ej.: "Normal, carga del 50 %%"')
    a = ap.parse_args()
    estilo()

    filas = modelo.leer_csv(a.csv)
    carpeta = os.path.dirname(os.path.abspath(a.csv))
    sesiones = sorted({f.get("sesion", "") for f in filas})
    ref = a.ref if a.ref else [s for s in sesiones if s.startswith("s")]
    normal = [f for f in filas if f.get("estado") == "normal"]
    g_ref = [f for f in normal if f.get("sesion") in ref]
    g_car = [f for f in normal if f.get("sesion") in a.carga]
    if not g_car:
        raise SystemExit(f"No hay registros 'normal' en las sesiones {a.carga}")
    if not g_ref:
        raise SystemExit(f"No hay registros 'normal' en las sesiones de referencia {ref}")

    def pct(g):
        v = [valor(f, "carga_pct") for f in g]
        v = [x for x in v if np.isfinite(x)]
        return f"{np.mean(v):.0f} %" if v else "?"
    n_ref = a.nombre_ref or f"Normal sin carga ({pct(g_ref)})"
    n_car = a.nombre_carga or f"Normal con carga ({pct(g_car)})"
    print(f"Referencia: {len(g_ref)} registros de {', '.join(ref)}  ->  {n_ref}")
    print(f"Con carga:  {len(g_car)} registros de {', '.join(a.carga)}  ->  {n_car}")
    os.makedirs(a.salida, exist_ok=True)

    # ------------------------------------------------------------------ tabla
    print(f"\n{'característica':<34}{'sin carga':>18}{'con carga':>18}{'diferencia':>12}{'d':>7}")
    tabla = []
    cols_csv = list(filas[0].keys())
    extra = [(c, f"Banda {c.split('_')[1]}–{c.split('_')[2]} Hz [dB]") for c in cols_csv
             if c.startswith("banda_")]
    for c, nombre in BASICAS + extra:
        r = np.array([valor(f, c) for f in g_ref])
        k = np.array([valor(f, c) for f in g_car])
        r, k = r[np.isfinite(r)], k[np.isfinite(k)]
        if not len(r) or not len(k):
            continue
        dif = k.mean() - r.mean()
        d = dif / (np.sqrt((r.var() + k.var()) / 2) + 1e-12)
        tabla.append({"caracteristica": nombre, "sin_carga_media": r.mean(), "sin_carga_desvio": r.std(),
                      "con_carga_media": k.mean(), "con_carga_desvio": k.std(),
                      "diferencia": dif, "d": d})
        print(f"{nombre:<34}{r.mean():>10.2f} ±{r.std():>6.2f}{k.mean():>10.2f} ±{k.std():>6.2f}"
              f"{dif:>+12.2f}{d:>+7.2f}")
    with open(os.path.join(a.salida, "carga_tabla.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tabla[0].keys()))
        w.writeheader()
        for t in tabla:
            w.writerow({kk: (f"{v:.3f}" if isinstance(v, float) else v) for kk, v in t.items()})

    # ------------------------------------------------------------------ diagnóstico
    if os.path.exists(a.modelo):
        m = modelo.cargar(a.modelo)
        if modelo.es_ponderado(m):
            print("\nAviso: el modelo es ponderado; se usa igual.")
        print(f"\nDiagnóstico con {a.modelo}:")
        filas_diag = []
        for nombre, g in ((n_ref, g_ref), (n_car, g_car)):
            diag = diagnosticar(m, g, a.max_clip)
            cuenta = Counter(diag)
            ok = cuenta.get("normal", 0)
            print(f"  {nombre:<28} {len(g):>4} registros | normal {ok} ({ok / len(g):.0%}) | "
                  + ", ".join(f"{e} {n}" for e, n in cuenta.most_common() if e != "normal"))
            filas_diag += [{"grupo": nombre, "archivo": f.get("archivo", ""), "sesion": f.get("sesion", ""),
                            "diagnostico": dd} for f, dd in zip(g, diag)]
        if not a.ref:
            print("  (la referencia son las sesiones de entrenamiento: su acierto es optimista)")
        with open(os.path.join(a.salida, "carga_diagnostico.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas_diag[0].keys()))
            w.writeheader()
            w.writerows(filas_diag)
    else:
        print(f"\nNo encontré {a.modelo}: se omite el diagnóstico.")

    # ------------------------------------------------------------------ figuras
    grupos = [(n_ref, g_ref, COLORES["normal"], "-"), (n_car, g_car, COLOR_CARGA, "--")]
    if a.con_fallas:
        for e in ("combustible", "aire"):
            g = [f for f in filas if f.get("estado") == e and f.get("sesion", "").startswith("s")]
            if g:
                grupos.append((e, g, COLORES[e], ":"))
    fig_bandas(grupos, cols_csv, os.path.join(a.salida, "fig_carga_bandas.png"))

    espectros = []
    for nombre, g, col in ((n_ref, g_ref, COLORES["normal"]), (n_car, g_car, COLOR_CARGA)):
        f, p, n = psd_media(carpeta, g, a.max_wav)
        if n:
            espectros.append((nombre, f, p, col))
            print(f"Espectro medio de '{nombre}': {n} registros")
    if espectros:
        fig_espectro(espectros, os.path.join(a.salida, "fig_carga_espectro.png"))
        niveles = fig_bandas_abs(espectros, os.path.join(a.salida, "fig_carga_bandas_abs.png"))
        if len(niveles) == 2:
            print("\nNivel absoluto por banda [dBFS] (de la densidad espectral media):")
            print(f"{'banda':<14}{'sin carga':>11}{'con carga':>11}{'diferencia':>12}")
            for (lo, hi), r0, r1 in zip(BANDAS_HZ, niveles[0], niveles[1]):
                print(f"{f'{lo}-{hi} Hz':<14}{r0:>11.1f}{r1:>11.1f}{r1 - r0:>+12.1f}")
    else:
        print("No encontré los WAV: se omite la figura del espectro.")
    print(f"\nArchivos en {a.salida}/: fig_carga_bandas.png, fig_carga_espectro.png, "
          "fig_carga_bandas_abs.png, carga_tabla.csv, carga_diagnostico.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())