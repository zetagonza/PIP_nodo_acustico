"""
grabar_mic.py  -  (se corre en la Raspberry)  SOLO graba. No filtra, no calcula espectros.

Guarda el audio crudo del ICS-43434 (48 kHz por defecto, o --fs 16000; mono, 32 bit) en un WAV, escribiendo bloque a
bloque (la RAM no crece con la duración). Todo el análisis se hace después, en la PC.

USO
  python grabar_mic.py --medidor                       # ajustar el volumen (no guarda nada)
  python grabar_mic.py --duracion 160                  # graba 160 s
  python grabar_mic.py --duracion 160 --etiqueta pos1  # agrega 'pos1' al nombre del archivo

Después copiar el WAV a la PC (VS Code: botón derecho > Download, o scp) y analizarlo:
  python analizar_mic.py prueba_mic/grabacion_pos1_....wav --senal senal_prueba/senal_prueba.json

Dispositivos de audio:  python -m sounddevice    (o:  arecord -l)
"""
import argparse
import os
import queue
import shutil
import sys
import time
import wave
from datetime import datetime

import numpy as np

FS = 48000
BLOQUE_CB = 4800                 # 100 ms por callback


def dbfs(v):
    return 20 * np.log10(max(float(v), 1e-9))


def dispositivo_arg(txt):
    return int(txt) if txt is not None and txt.isdigit() else txt


def medidor(a):
    import sounddevice as sd
    cola = queue.Queue(maxsize=100)

    def callback(indata, frames, tiempo, status):
        try:
            cola.put_nowait(indata[:, a.canal].astype(np.float64) / 2.0 ** 31)
        except queue.Full:
            pass

    print("Medidor en vivo (Ctrl+C para salir). Objetivo con el tono de 1 kHz: PICO entre "
          "-35 y -15 dBFS.\n")
    try:
        with sd.InputStream(samplerate=a.fs, channels=2, dtype="int32", blocksize=BLOQUE_CB,
                            device=dispositivo_arg(a.dispositivo), callback=callback):
            while True:
                x = cola.get(timeout=3)
                pico = dbfs(np.max(np.abs(x)))
                rms = dbfs(np.sqrt(np.mean(x ** 2)))
                n = int(max(0, min(40, (pico + 60) / 60 * 40)))
                aviso = "  <-- SATURA" if pico > -1 else ("  (muy bajo)" if pico < -50 else "")
                print(f"\r pico {pico:6.1f} dBFS  rms {rms:6.1f} dBFS  |{'#' * n:<40}|{aviso}      ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print()
    return 0


def grabar(a):
    import sounddevice as sd
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    necesario = a.duracion * a.fs * 4
    libre = shutil.disk_usage(os.path.dirname(os.path.abspath(a.salida))).free
    if libre < necesario + 200e6:
        print(f"ERROR: no alcanza el disco ({libre / 1e6:.0f} MB libres, hacen falta "
              f"{(necesario + 200e6) / 1e6:.0f} MB).")
        return 1

    cola = queue.Queue(maxsize=200)
    perdidos = [0]
    canal = a.canal

    def callback(indata, frames, tiempo, status):
        if status:
            perdidos[0] += 1
        try:
            cola.put_nowait(indata[:, canal].copy())
        except queue.Full:
            perdidos[0] += 1

    for r in range(a.espera, 0, -1):
        print(f"\r  empieza en {r} s ", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 24 + "\r", end="")

    total = int(a.duracion * a.fs)
    hechas = 0
    pico = 0
    saturadas = 0
    ok = False
    tmp = a.salida + ".tmp"
    wf = wave.open(tmp, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(4)
    wf.setframerate(a.fs)
    t_ini = time.time()
    proximo_aviso = 10
    print(f"GRABANDO {a.duracion:g} s  ->  {a.salida}")
    try:
        with sd.InputStream(samplerate=a.fs, channels=2, dtype="int32", blocksize=BLOQUE_CB,
                            device=dispositivo_arg(a.dispositivo), callback=callback):
            while hechas < total:
                x = cola.get(timeout=3)[:total - hechas]
                wf.writeframes(x.astype("<i4").tobytes())
                hechas += len(x)
                a_abs = np.abs(x)
                pico = max(pico, int(a_abs.max()))
                saturadas += int(np.count_nonzero(a_abs > 0.99 * 2 ** 31))
                if hechas / a.fs >= proximo_aviso:
                    print(f"  {proximo_aviso:4d} s  pico {dbfs(pico / 2.0 ** 31):6.1f} dBFS")
                    proximo_aviso += 10
        ok = True
    except KeyboardInterrupt:
        print("\nInterrumpido: la grabación incompleta se descarta.")
    except queue.Empty:
        print("ERROR: el micrófono no entrega datos (¿overlay I2S? ¿dispositivo?).")
    finally:
        wf.close()
        if ok:
            os.replace(tmp, a.salida)
        elif os.path.exists(tmp):
            os.remove(tmp)
    if not ok:
        return 1

    p = dbfs(pico / 2.0 ** 31)
    print(f"\nListo: {a.salida}  ({os.path.getsize(a.salida) / 1e6:.1f} MB, "
          f"{time.time() - t_ini:.0f} s)")
    print(f"  pico máximo: {p:.1f} dBFS   muestras saturadas: {saturadas}   bloques perdidos: {perdidos[0]}")
    if saturadas or p > -3:
        print("  AVISO: el micrófono saturó o estuvo muy cerca. Bajá el volumen de la PC y repetí.")
    elif p < -45:
        print("  AVISO: nivel muy bajo. Subí el volumen de la PC o acercá el parlante.")
    if perdidos[0]:
        print("  AVISO: se perdieron bloques de audio: repetí la grabación.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duracion", type=float, default=None, help="segundos a grabar")
    ap.add_argument("--medidor", action="store_true", help="medidor en vivo, sin guardar")
    ap.add_argument("--etiqueta", default="", help="texto que se agrega al nombre (pos1, ref...)")
    ap.add_argument("--salida", default=None, help="archivo de salida (por defecto prueba_mic/...)")
    ap.add_argument("--espera", type=int, default=3, help="segundos antes de empezar")
    ap.add_argument("--canal", type=int, default=0, choices=[0, 1],
                    help="0 = izquierdo (pin L/R a GND), 1 = derecho (pin L/R a VDD)")
    ap.add_argument("--dispositivo", default="hw:0,0",
                    help="por defecto el hardware I2S directo (sin remuestreo de ALSA)")
    ap.add_argument("--fs", type=int, default=FS, help="48000 o 16000")
    a = ap.parse_args()
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        print("Falta sounddevice (pip install sounddevice) o no hay audio en este equipo.")
        return 1
    try:
        if a.medidor:
            return medidor(a)
        if a.duracion is None:
            ap.error("indicá --duracion (o usá --medidor)")
        if a.salida is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            et = f"_{a.etiqueta}" if a.etiqueta else ""
            a.salida = os.path.join("prueba_mic", f"grabacion{et}_{ts}.wav")
        return grabar(a)
    except Exception as e:
        print(f"ERROR de audio: {e}\nListá los dispositivos con:  python -m sounddevice")
        return 1


if __name__ == "__main__":
    sys.exit(main())