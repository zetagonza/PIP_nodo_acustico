"""
procesar_dataset.py - Recorre una carpeta de WAV (16 o 32 bits, mono/estéreo,
16 kHz) y arma el CSV de características. Lee de a 1 s: la RAM no crece con el
largo del archivo. Sirve para re-procesar todo el dataset cuando cambies un
algoritmo (el WAV crudo es la "fuente de verdad").

Convención de nombre:  <estado>_<sesion>_<n>_<timestamp>.wav
  ej: normal_s1_003_20261101-171500.wav

uso:  python procesar_dataset.py datos --rpm 3000 --salida caracteristicas.csv
"""
import argparse
import csv
import glob
import os
import wave

import numpy as np

from caracteristicas import AcumuladorRegistro, FS_RAW, BLOQUE


def leer_bloques(ruta):
    """Genera bloques float64 (+-1) de 1 s (16000 muestras). Descarta el resto."""
    with wave.open(ruta, "rb") as wf:
        if wf.getframerate() != FS_RAW:
            pista = (" (es del dataset anterior, a 48 kHz: movelo a otra carpeta; el dataset se "
                     "regraba a 16 kHz)" if wf.getframerate() == 48000 else "")
            raise ValueError(f"{ruta}: se esperaba {FS_RAW} Hz, tiene {wf.getframerate()}{pista}")
        ancho, canales = wf.getsampwidth(), wf.getnchannels()
        if ancho not in (2, 4):
            raise ValueError(f"{ruta}: solo WAV de 16 o 32 bits")
        dtype = "<i2" if ancho == 2 else "<i4"
        escala = 2.0 ** (8 * ancho - 1)
        while True:
            datos = wf.readframes(BLOQUE)
            if len(datos) // (ancho * canales) < BLOQUE:
                break
            x = np.frombuffer(datos, dtype=dtype).reshape(-1, canales)[:, 0]
            yield x.astype(np.float64) / escala


def procesar_archivo(ruta, f_fire_nom):
    acc = AcumuladorRegistro(f_fire_nom)
    for bloque in leer_bloques(ruta):
        acc.agregar_bloque(bloque)
    return acc.resultado()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta", nargs="?", default="datos")
    ap.add_argument("--rpm", type=float, default=3000.0)
    ap.add_argument("--salida", default="caracteristicas.csv")
    args = ap.parse_args()

    f_fire_nom = args.rpm / 120.0
    rutas = sorted(r for r in glob.glob(os.path.join(args.carpeta, "**", "*.wav"), recursive=True)
                   if "_descartados" not in os.path.relpath(r, args.carpeta).split(os.sep))
    if not rutas:
        print("No se encontraron WAV en", args.carpeta)
        return
    meta_prev = {}                     # carga, ruido... del CSV de captura (si existe)
    csv_prev = os.path.join(args.carpeta, "caracteristicas_captura.csv")
    if os.path.exists(csv_prev):
        with open(csv_prev, newline="") as fh:
            meta_prev = {f.get("archivo"): f for f in csv.DictReader(fh)}
    filas = []
    for ruta in rutas:
        nombre = os.path.basename(ruta)[:-4]
        partes = nombre.split("_")
        previo = meta_prev.get(os.path.basename(ruta), {})
        meta = {"archivo": os.path.basename(ruta), "fs_hz": FS_RAW, "estado": partes[0],
                "sesion": partes[1] if len(partes) > 1 else "",
                "carga_pct": previo.get("carga_pct", ""),
                "ruido": previo.get("ruido", "ninguno"),
                "distancia_m": previo.get("distancia_m", ""),
                "rpm_nominal": previo.get("rpm_nominal", args.rpm)}
        try:
            filas.append({**meta, **procesar_archivo(ruta, f_fire_nom)})
            print("OK  ", nombre)
        except Exception as e:                      # un WAV malo no frena el lote
            print("FALLO", nombre, "->", e)
    if filas:
        with open(args.salida, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\n{len(filas)} registros -> {args.salida}")


if __name__ == "__main__":
    main()