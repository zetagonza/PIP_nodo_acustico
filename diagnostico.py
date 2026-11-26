"""
diagnostico.py - Toma una muestra del micrófono (NO guarda audio), calcula las
características y dice si el generador está normal o en alguna falla.

  python diagnostico.py                       # una muestra de 10 s
  python diagnostico.py --continuo --pausa 300  # una muestra cada 5 min
  python diagnostico.py --modelo modelo.npz --rpm 3000 --canal 0

Resultados posibles: uno de los estados entrenados, DESCONOCIDO (no se parece a
ninguno: sensor movido, otra falla, etc.) o INVÁLIDO (saturado o sin señal del
motor: apagado/viento). Cada diagnóstico se agrega a diagnosticos.csv (unas
pocas decenas de bytes por línea).
"""
import argparse
import csv
import os
import queue
import time
from datetime import datetime

import numpy as np

import datos
from captura import grabar_registro, dispositivo_arg, DISPOSITIVO
from modelo import cargar, es_ponderado, predecir


def diagnosticar(m, feat, min_peine, max_clip):
    if feat["clip_frac"] > max_clip:
        return "INVÁLIDO", 0.0, "micrófono saturado: alejalo o bajá la ganancia", None
    if feat["prominencia_peine_db"] < min_peine and "apagado" not in m["clases"]:
        return "INVÁLIDO", 0.0, "sin armónicos del motor (¿apagado, viento fuerte?)", None
    x = np.array([feat[c] for c in m["columnas"]])
    r = predecir(m, x)
    k = int(np.argmax(r["prob"]))
    if r["rechazo"]:
        return "DESCONOCIDO", float(r["prob"][k]), \
            f"más parecido a '{r['clase']}' pero fuera de rango (d²={r['d2'][k]:.0f})", r
    return r["clase"], float(r["prob"][k]), "", r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="modelo.npz")
    ap.add_argument("--duracion", type=int, default=10)
    ap.add_argument("--rpm", type=float, default=3000.0)
    ap.add_argument("--canal", type=int, default=0, choices=[0, 1])
    ap.add_argument("--dispositivo", default=DISPOSITIVO)
    ap.add_argument("--continuo", action="store_true")
    ap.add_argument("--n", type=int, default=None,
                    help="tomar N muestras seguidas y terminar (útil con --real)")
    ap.add_argument("--pausa", type=float, default=60.0, help="s entre muestras (modo continuo)")
    ap.add_argument("--min-peine", type=float, default=4.0,
                    help="prominencia mínima (dB) del peine armónico para considerar válido")
    ap.add_argument("--max-clip", type=float, default=0.01)
    ap.add_argument("--log", default="diagnosticos.csv")
    ap.add_argument("--real", default=None, choices=datos.ESTADOS,
                    help="estado REAL del motor en esta prueba (para validar el nodo; "
                         "después:  python datos.py vivo)")
    args = ap.parse_args()

    m = cargar(args.modelo)
    f_fire_nom = args.rpm / 120.0
    print(f"Modelo: {', '.join(m['clases'])}  |  Ctrl+C para salir")
    if es_ponderado(m):
        print("Ponderado por sesión: " + ", ".join(
            f"{ss} {w:.2f}" for ss, w in zip(m["sesiones"], m["pesos"])))

    hechas = 0
    try:
        while True:
            try:
                feat, perdidos = grabar_registro(args.duracion, args.canal,
                                                 dispositivo_arg(args.dispositivo), f_fire_nom, None)
            except queue.Empty:
                print("ERROR: el micrófono no entrega datos")
                return
            faltan = [c for c in m["columnas"] if c not in feat]
            if faltan:
                print("ERROR: el modelo usa columnas que el código actual no calcula "
                      f"({', '.join(faltan[:3])}...). Reentrená:  python entrenar.py ...")
                return
            estado, prob, nota, r = diagnosticar(m, feat, args.min_peine, args.max_clip)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] >>> {estado}" + (f"  (confianza {prob:.0%})" if prob else "")
                  + (f"  - {nota}" if nota else ""))
            if r is not None:
                orden = np.argsort(-r["prob"])
                print("    " + " | ".join(f"{m['clases'][i]} {r['prob'][i]:.0%}" for i in orden))
            columnas = ["timestamp", "estado", "confianza", "f_fire_hz",
                        "rms_dbfs", "peine_db", "perdidos", "real"]
            if os.path.exists(args.log):
                with open(args.log, newline="") as fh:
                    if next(csv.reader(fh), None) != columnas:   # formato viejo
                        os.replace(args.log, args.log + ".bak")
            nuevo = not os.path.exists(args.log)
            with open(args.log, "a", newline="") as fh:
                w = csv.writer(fh)
                if nuevo:
                    w.writerow(columnas)
                w.writerow([ts, estado, f"{prob:.3f}", f"{feat['f_fire_hz']:.2f}",
                            f"{feat['rms_dbfs']:.1f}", f"{feat['prominencia_peine_db']:.1f}",
                            perdidos, args.real or ""])
            hechas += 1
            if args.n is not None:
                if hechas >= args.n:
                    break
            elif not args.continuo:
                break
            time.sleep(args.pausa)
    except KeyboardInterrupt:
        print("\nFin.")


if __name__ == "__main__":
    main()