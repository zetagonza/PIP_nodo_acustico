"""
source venv/bin/activate

ensayo.py - Captura GUIADA de una sesión completa, un estado a la vez.

  python ensayo.py --sesion s1 --carga 50

Muestra un menú con los estados y cuántos registros tiene cada uno en esta
sesión. Al elegir uno te da las instrucciones para inducir ese estado, espera a
que lo confirmes, deja pasar un tiempo de estabilización y graba. Si ese estado
ya tenía registros, te pregunta si querés reemplazarlos o agregar más.

obs: Es el mismo motor de captura que captura.py (mismos archivos y mismo CSV).
"""
import argparse
import sys
import time

import datos
from captura import capturar_serie, dispositivo_arg, DISPOSITIVO


N_POR_ESTADO = {"apagado": 20}      # cantidad por defecto; el resto usa 40

INSTRUCCIONES = {
    "apagado": "Motor DETENIDO, micrófono en su lugar de siempre, sin tocar nada. Es el "
               "ruido de fondo del lugar: sirve para medir la relación señal/ruido de las "
               "demás grabaciones y para que el nodo reconozca 'apagado'. NO se resta de los "
               "otros audios.",
    "normal": "Motor caliente, en régimen estable, con carga resistiva constante. "
              "Sin ninguna falla inducida.",
    "combustible": "FALLA DE COMBUSTIBLE: cerrar PARCIALMENTE la llave de combustible hasta que el motor "
                   "funcione irregular pero sin detenerse (marcá la posición de la llave). "
                   "Esperar a que se estabilice. "
                   "Mantener carga resistiva (nada sensible conectado).",
    "desbalanceo": "Con el motor DETENIDO, montar la masa al eje, bien sujeta "
                   "y protegida con guarda. Recién ahí arrancar y esperar el régimen.",
    "aire": "FALLA DE AIRE: restringir la admisión de aire con el cebador (restricción "
            "parcial y constante, ~40 %; marcá la posición de la perilla para repetirla en "
            "cada sesión). Esperar a que se estabilice.",
}


def preguntar(texto, validas=None):
    while True:
        r = input(texto).strip().lower()
        if validas is None or r in validas:
            return r


def estabilizar(segundos):
    for resto in range(int(segundos), 0, -1):
        print(f"\r  estabilizando... {resto:3d} s ", end="", flush=True)
        time.sleep(1)
    if segundos:
        print("\r" + " " * 32 + "\r", end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesion", required=True, help="s1, s2, ... (sin '_')")
    ap.add_argument("--n-registros", type=int, default=None,
                    help="registros por estado (por defecto 40; 20 para 'apagado')")
    ap.add_argument("--duracion", type=int, default=10)
    ap.add_argument("--espera", type=int, default=10,
                    help="segundos de estabilización antes de grabar cada estado")
    ap.add_argument("--rpm", type=float, default=3000.0)
    ap.add_argument("--carga", type=float, default=0.0)
    ap.add_argument("--ruido", default="ninguno")
    ap.add_argument("--distancia", type=float, default=1.0)
    ap.add_argument("--canal", type=int, default=0, choices=[0, 1])
    ap.add_argument("--dispositivo", default=DISPOSITIVO)
    ap.add_argument("--salida", default=datos.CARPETA)
    a = ap.parse_args()
    if "_" in a.sesion:
        print("ERROR: el nombre de sesión no puede tener '_'")
        return 2

    try:
        while True:
            print(f"\n=== Sesión {a.sesion} ===")
            for aviso in datos.avisos_espacio(a.salida):
                print("AVISO:", aviso)
            for i, e in enumerate(datos.ESTADOS, 1):
                print(f"  [{i}] {e:<12} {datos.contar(e, a.sesion, a.salida):>3} registros")
            op = preguntar("Estado (1-%d), [r]esumen, [q] salir: " % len(datos.ESTADOS),
                           [str(i) for i in range(1, len(datos.ESTADOS) + 1)] + ["r", "q"])
            if op == "q":
                return 0
            if op == "r":
                datos.resumen(a.salida)
                continue
            estado = datos.ESTADOS[int(op) - 1]

            existentes = datos.contar(estado, a.sesion, a.salida)
            if existentes:
                m = preguntar(f"'{estado}' ya tiene {existentes} registros en {a.sesion}. "
                              "[r]eemplazar / [a]gregar / [c]ancelar: ", ["r", "a", "c"])
                if m == "c":
                    continue
                if m == "r":
                    nf, nw, dest = datos.descartar(estado, a.sesion, carpeta=a.salida)
                    print(f"  Anteriores movidos a {dest}")

            print(f"\n>>> {estado.upper()}\n{INSTRUCCIONES[estado]}")
            if preguntar("Enter cuando esté listo ([c] cancelar): ", None) == "c":
                continue
            estabilizar(a.espera)
            inicio = datos.siguiente_indice(estado, a.sesion, a.salida)
            n = capturar_serie(estado, a.sesion,
                               a.n_registros or N_POR_ESTADO.get(estado, 40), inicio=inicio,
                               duracion=a.duracion, rpm=a.rpm, canal=a.canal,
                               dispositivo=dispositivo_arg(a.dispositivo), carga=a.carga,
                               ruido=a.ruido, distancia=a.distancia, carpeta=a.salida)
            print(f"\nGrabados {n} registros de '{estado}'.")
            datos.resumen(a.salida)
    except (KeyboardInterrupt, EOFError):
        print("\nFin.")
    except ValueError as e:
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())