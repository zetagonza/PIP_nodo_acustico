"""
analisis_fft.py - Gráficos de diagnóstico de uno o varios WAV superpuestos
(ideal para comparar normal vs. falla). Reemplaza al FFT de registro completo.

  1) PSD tipo Welch 0-500 Hz  (líneas = armónicos de la f_fire estimada)
  2) PSD tipo Welch 20-6000 Hz (eje de frecuencia logarítmico)
  3) Espectro de envolvente 0-150 Hz (profundidad de modulación)

uso:  python analisis_fft.py normal_s1_001_*.wav combustible_s1_001_*.wav --rpm 3000
Genera espectro_fft.png (no necesita pantalla: sirve por SSH).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caracteristicas import AcumuladorRegistro
from procesar_dataset import leer_bloques


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--rpm", type=float, default=3000.0)
    ap.add_argument("--salida", default="espectro_fft.png")
    args = ap.parse_args()
    f_nom = args.rpm / 120.0

    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    for i, ruta in enumerate(args.wavs):
        acc = AcumuladorRegistro(f_nom)
        for bloque in leer_bloques(ruta):
            acc.agregar_bloque(bloque)
        sp = acc.espectros()
        f, psd_db = sp["f"], 10 * np.log10(sp["psd"] + 1e-20)
        etiqueta = f"{os.path.basename(ruta)[:28]} (f_fire {sp['f_fire']:.1f} Hz)"
        ax[0].plot(f[:501], psd_db[:501], lw=0.9, label=etiqueta)
        ax[1].semilogx(f[20:6001], psd_db[20:6001], lw=0.8, label=etiqueta)
        ax[2].plot(f[:151], sp["env"][:151], lw=0.9, label=etiqueta)
        if i == 0:
            for k in range(1, 13):
                ax[0].axvline(k * sp["f_fire"], color="gray", ls=":", lw=0.6)
            for m in range(1, 9):
                ax[2].axvline(m * sp["f_fire"] / 2, color="gray", ls=":", lw=0.6)

    ax[0].set(title="PSD (Welch) 0-500 Hz - líneas: k·f_fire", xlabel="Frecuencia (Hz)",
              ylabel="dB re FS²/Hz")
    ax[1].set(title="PSD (Welch) 20-6000 Hz", xlabel="Frecuencia (Hz)", ylabel="dB re FS²/Hz")
    ax[2].set(title="Espectro de envolvente (banda 1-6 kHz) - líneas: m·f_fire/2",
              xlabel="Frecuencia (Hz)", ylabel="Profundidad de modulación")
    for a in ax:
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.salida, dpi=110)
    print("Gráfico guardado en", args.salida)


if __name__ == "__main__":
    main()