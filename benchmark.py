"""
benchmark.py - Mide cuánta RAM y CPU consume el procesamiento. Correlo EN LA
Raspberry Pi: los números sirven como evidencia en el informe (Pruebas y
validaciones) de que el nodo cumple con los 512 MB.

  python benchmark.py                    # 60 bloques de audio sintético (60 s)
  python benchmark.py --wav datos/normal_s1_001_....wav
  python benchmark.py --bloques 300

Mide: tiempo por bloque de 1 s (si es < 1 s, el nodo procesa más rápido que
tiempo real), RAM base y pico del proceso, y temperatura de CPU. Al final imprime
una tabla lista para pegar en el informe.
"""
import argparse
import subprocess
import time

import numpy as np

from caracteristicas import AcumuladorRegistro, BLOQUE, FS_RAW


def memoria_mb():
    """(RSS actual, pico) del proceso en MB, leído de /proc (Linux)."""
    rss = pico = None
    try:
        with open("/proc/self/status") as fh:
            for linea in fh:
                if linea.startswith("VmRSS"):
                    rss = int(linea.split()[1]) / 1024
                elif linea.startswith("VmHWM"):
                    pico = int(linea.split()[1]) / 1024
    except OSError:
        pass
    return rss, pico


def temperatura_cpu():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read()) / 1000.0
    except (OSError, ValueError):
        return None


def throttled():
    try:
        r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                           text=True, timeout=2)
        return r.stdout.strip() or None
    except Exception:
        return None


def bloques(args):
    if args.wav:
        from procesar_dataset import leer_bloques
        for i, b in enumerate(leer_bloques(args.wav)):
            if i >= args.bloques:
                return
            yield b
    else:
        rng = np.random.default_rng(0)
        t = np.arange(BLOQUE) / FS_RAW
        for _ in range(args.bloques):
            yield (0.05 * np.sin(2 * np.pi * 25 * t) + 0.01 * rng.standard_normal(BLOQUE))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bloques", type=int, default=60)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--rpm", type=float, default=3000.0)
    args = ap.parse_args()

    acc = AcumuladorRegistro(args.rpm / 120.0)
    rss0, _ = memoria_mb()
    t0 = temperatura_cpu()
    wall, cpu = [], []
    for b in bloques(args):
        w, c = time.perf_counter(), time.process_time()
        acc.agregar_bloque(b)
        cpu.append(time.process_time() - c)
        wall.append(time.perf_counter() - w)
    w, c = time.perf_counter(), time.process_time()
    acc.resultado()
    t_res = time.perf_counter() - w
    if not wall:
        print("No se procesó ningún bloque.")
        return 1
    rss1, pico = memoria_mb()
    t1 = temperatura_cpu()
    wall_ms, cpu_ms = np.array(wall) * 1e3, np.array(cpu) * 1e3

    print(f"Bloques de 1 s procesados : {len(wall)}")
    print(f"Tiempo por bloque (pared) : media {wall_ms.mean():.0f} ms | "
          f"p95 {np.percentile(wall_ms, 95):.0f} ms | máx {wall_ms.max():.0f} ms")
    print(f"CPU por bloque            : media {cpu_ms.mean():.0f} ms")
    print(f"Factor de tiempo real     : {wall_ms.mean() / 1000:.3f}  (< 1 = procesa más rápido que el audio)")
    print(f"Cierre del registro       : {t_res * 1e3:.0f} ms")
    if rss0 is not None:
        print(f"RAM del proceso           : base {rss0:.0f} MB | final {rss1:.0f} MB | "
              f"pico {pico:.0f} MB (de 512 MB totales)")
    if t0 is not None:
        print(f"Temperatura CPU           : {t0:.1f} -> {t1:.1f} °C")
    th = throttled()
    if th:
        print(f"Estado de alimentación    : {th}   (0x0 = sin subtensión ni limitación)")

    print("\n| Métrica | Valor |\n|---|---|")
    print(f"| Bloques de 1 s procesados | {len(wall)} |")
    print(f"| Tiempo de procesamiento por bloque (media / p95) | "
          f"{wall_ms.mean():.0f} ms / {np.percentile(wall_ms, 95):.0f} ms |")
    print(f"| Factor de tiempo real | {wall_ms.mean() / 1000:.3f} |")
    if pico is not None:
        print(f"| RAM pico del proceso | {pico:.0f} MB |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())