"""
probar_fs.py  -  (se corre en la Raspberry)  ¿La Raspberry entrega el audio del ICS-43434
directamente a otra frecuencia de muestreo (por ejemplo 16 kHz)?

Para cada frecuencia pedida hace, sin necesitar la PC ni ningún sonido de prueba:
  1. Pregunta a ALSA si acepta abrir el micrófono a esa frecuencia.
  2. Graba unos segundos y mide la frecuencia REAL con el reloj de pared
     (muestras recibidas / segundos transcurridos).
  3. Lee la frecuencia que está usando el HARDWARE (/proc/asound/.../hw_params). Si ALSA
     remuestrea por software, acá se ve: el hardware seguiría a otra frecuencia.
  4. Mide ruido de fondo en la banda útil, DC, pico y bloques perdidos.
  5. Guarda el WAV para poder analizarlo después.

USO
  python probar_fs.py                         # prueba 48000 y 16000 Hz, 10 s cada una
  python probar_fs.py --fs 48000 16000 8000 --duracion 15
  python probar_fs.py --dispositivo 1         # índice o nombre (python -m sounddevice)

Hacelo en silencio (parlante apagado): así el ruido medido es el del micrófono + ambiente.
"""
import argparse
import glob
import os
import queue
import re
import sys
import time
import wave
from datetime import datetime

import numpy as np

try:
    from scipy import signal
except ImportError:
    signal = None

FS_PROPIA = {  # rangos del datasheet DS-000069 v1.2
    "bajo consumo": (6250, 18750),
    "alto rendimiento": (23000, 51600),
}


def modo_datasheet(fs):
    for nombre, (a, b) in FS_PROPIA.items():
        if a <= fs <= b:
            return nombre
    return None


def dbfs(v):
    return 20 * np.log10(max(float(v), 1e-12))


def dispositivo_arg(txt):
    return int(txt) if txt is not None and str(txt).isdigit() else txt


def hw_params_activos():
    """Devuelve las frecuencias que ALSA reporta en uso por el hardware (captura)."""
    out = []
    for ruta in glob.glob("/proc/asound/card*/pcm*c/sub*/hw_params"):
        try:
            txt = open(ruta).read()
        except OSError:
            continue
        if txt.strip() == "closed":
            continue
        m = re.search(r"rate:\s*(\d+)", txt)
        if m:
            out.append((ruta.split("/proc/asound/")[1], int(m.group(1))))
    return out


def ruido_banda(x, fs, f_alto):
    """RMS en dBFS entre 20 Hz y f_alto (sin DC)."""
    x = x - np.mean(x)
    if signal is None:
        return dbfs(np.sqrt(np.mean(x ** 2)))
    nper = min(len(x), int(fs))
    f, P = signal.welch(x, fs, window="hann", nperseg=nper)
    m = (f >= 20) & (f <= f_alto)
    return 10 * np.log10(np.sum(P[m]) * (f[1] - f[0]) + 1e-30)


def probar(fs, a, sd):
    print(f"\n=== {fs} Hz ===")
    modo = modo_datasheet(fs)
    if modo:
        print(f"  datasheet: {fs} Hz cae en el modo '{modo}' del micrófono")
    else:
        print("  datasheet: esa frecuencia NO está en ningún rango de operación normal "
              "(6,25-18,75 o 23-51,6 kHz): el micrófono no la garantiza")
    dev = dispositivo_arg(a.dispositivo)
    try:
        sd.check_input_settings(device=dev, channels=2, dtype="int32", samplerate=fs)
    except Exception as e:                              # noqa: BLE001
        print(f"  ALSA RECHAZA abrir a {fs} Hz: {e}")
        return {"fs": fs, "ok": False, "motivo": "rechazada"}
    print("  ALSA acepta abrir a esa frecuencia")

    bloque = max(int(fs * 0.1), 256)
    cola = queue.Queue(maxsize=400)
    perdidos = [0]
    t_cb = []

    def callback(indata, frames, tiempo, status):
        if status:
            perdidos[0] += 1
        t_cb.append(time.monotonic())
        try:
            cola.put_nowait(indata[:, a.canal].copy())
        except queue.Full:
            perdidos[0] += 1

    total = int(a.duracion * fs)
    bloques = []
    hechas = 0
    hw = []
    try:
        with sd.InputStream(samplerate=fs, channels=2, dtype="int32", blocksize=bloque,
                            device=dev, callback=callback):
            t_ini = time.monotonic()
            while hechas < total:
                x = cola.get(timeout=5)
                bloques.append(x)
                hechas += len(x)
                if not hw and hechas > total * 0.3:
                    hw = hw_params_activos()
            t_fin = time.monotonic()
    except Exception as e:                              # noqa: BLE001
        print(f"  ERROR al grabar: {e}")
        return {"fs": fs, "ok": False, "motivo": "error"}
    x = np.concatenate(bloques)[:total].astype(np.float64) / 2.0 ** 31

    # frecuencia real con el reloj de pared (se descarta el primer bloque)
    n_util = (len(bloques) - 1) * bloque
    dt = t_cb[len(bloques) - 1] - t_cb[0] if len(t_cb) >= len(bloques) else t_fin - t_ini
    fs_real = n_util / dt if dt > 0 else float("nan")
    err = (fs_real / fs - 1) * 100
    print(f"  frecuencia real medida con el reloj: {fs_real:.1f} Hz  (error {err:+.2f} %)")
    if hw:
        for ruta, r in hw:
            marca = "OK" if r == fs else "DISTINTA: ALSA está remuestreando por software"
            print(f"  hardware ({ruta}): {r} Hz -> {marca}")
    else:
        print("  hardware: no pude leer /proc/asound (¿no es la Raspberry o no hay hw_params?)")

    pico = dbfs(np.max(np.abs(x)))
    dc = dbfs(abs(np.mean(x)))
    f_alto = a.f_comun
    rb = ruido_banda(x, fs, f_alto)
    print(f"  pico {pico:.1f} dBFS   DC {dc:.1f} dBFS   bloques perdidos: {perdidos[0]}")
    print(f"  ruido 20 Hz-{f_alto / 1000:.2f} kHz: {rb:.1f} dBFS RMS "
          f"(misma banda para todas las frecuencias, así se pueden comparar)")

    os.makedirs(a.salida, exist_ok=True)
    ruta = os.path.join(a.salida, f"fs{fs}_{datetime.now():%Y%m%d-%H%M%S}.wav")
    with wave.open(ruta, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(fs)
        wf.writeframes((x * 2.0 ** 31).clip(-2 ** 31, 2 ** 31 - 1).astype("<i4").tobytes())
    print(f"  guardado: {ruta}")

    problemas = []
    if abs(err) > 0.5:
        problemas.append("la frecuencia real no coincide con la pedida")
    if hw and any(r != fs for _, r in hw):
        problemas.append("el hardware no corre a esa frecuencia (remuestreo por software)")
    if perdidos[0]:
        problemas.append("se perdieron bloques")
    if pico > -3:
        problemas.append("pico muy alto: hay saturación o ruido fuerte")
    if pico < -90:
        problemas.append("no hay señal: ¿el micrófono entrega datos?")
    return {"fs": fs, "ok": not problemas, "motivo": "; ".join(problemas), "ruido": rb,
            "err": err, "hw": [r for _, r in hw]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=int, nargs="+", default=[48000, 16000])
    ap.add_argument("--duracion", type=float, default=10.0)
    ap.add_argument("--canal", type=int, default=0, choices=[0, 1])
    ap.add_argument("--dispositivo", default="hw:0,0")
    ap.add_argument("--salida", default="prueba_fs")
    a = ap.parse_args()
    try:
        import sounddevice as sd
    except ImportError:
        print("Falta sounddevice (pip install sounddevice) o no hay audio en este equipo.")
        return 1
    try:
        info = sd.query_devices(dispositivo_arg(a.dispositivo), "input")
        print(f"Dispositivo de entrada: {info['name']}")
    except Exception:                                   # noqa: BLE001
        print("(no pude consultar el dispositivo; sigo igual)")
    a.f_comun = 0.417 * min(a.fs)          # banda plana común a todas las frecuencias probadas
    res = [probar(fs, a, sd) for fs in a.fs]

    print("\n================ RESUMEN ================")
    for r in res:
        estado = "FUNCIONA" if r["ok"] else f"PROBLEMA ({r['motivo']})"
        print(f"  {r['fs']:>6} Hz: {estado}")
    con_ruido = [r for r in res if "ruido" in r]
    if len(con_ruido) > 1:
        base = con_ruido[0]
        for r in con_ruido[1:]:
            print(f"  ruido en 20 Hz-{a.f_comun / 1000:.2f} kHz: {r['fs']} Hz vs {base['fs']} Hz = "
                  f"{r['ruido'] - base['ruido']:+.1f} dB (el datasheet da unos 1 dB más de ruido en "
                  "bajo consumo; el ambiente puede pesar más)")
    print("\nSi 16000 dice FUNCIONA: seguí con la prueba de tonos (ver instrucciones).")
    return 0


if __name__ == "__main__":
    sys.exit(main())