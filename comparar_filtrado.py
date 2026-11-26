"""
comparar_filtrado.py - Compara una grabación SIN filtrar con la misma grabación después del
filtrado que aplica el nodo (caracteristicas.py), y mide cuánto ruido se quita.

Los WAV de datos/ se guardan SIN filtrar, así que se pueden usar directamente.

Filtros del nodo (los mismos coeficientes, importados de caracteristicas.py):
  - Pasa-altos Butterworth de orden 4, 20 Hz: se aplica a toda la señal.
  - Pasa-banda Butterworth de 1 a 6 kHz: solo para la rama de envolvente.

  python comparar_filtrado.py datos/normal_s2_001_*.wav
  python comparar_filtrado.py datos/normal_s2_001_*.wav --ruido datos/apagado_s2_001_*.wav

Con --ruido (una grabación con el motor apagado, en el mismo lugar) además calcula la
relación señal/ruido por banda, antes y después del filtrado.

Genera en comparacion_filtrado/:
  <nombre>_filtrado.png   las tres gráficas juntas
  <nombre>_1_forma_onda.png          forma de onda, sin filtrar y filtrada
  <nombre>_2_envolvente.png          pasa-banda 1-6 kHz y su envolvente
  <nombre>_3_densidad_motor.png      densidad espectral del motor, sin filtrar y filtrada
  <nombre>_4_densidad_ruido.png      densidad espectral del motor apagado, sin filtrar y filtrada (con --ruido)
  <nombre>_5_densidad_motor_y_ruido.png   motor sin filtrar y filtrado, y apagado filtrado
  (las figuras 3 y 4 tienen la misma escala vertical, para poder compararlas)
  <nombre>_filtrado.csv   potencias, nivel de continua y relación señal/ruido por banda
"""
import argparse
import csv
import os

import numpy as np
from scipy import signal

from caracteristicas import BANDAS_HZ, F_MAX, F_MIN, FS, _SOS_ENV, _SOS_HP
from procesar_dataset import leer_bloques

AZUL, NARANJA, VERDE, GRIS = "#3b6fb6", "#e09f3e", "#2a9d8f", "#777777"


def leer(ruta):
    bloques = list(leer_bloques(ruta))
    if not bloques:
        raise SystemExit(f"{ruta}: no tiene ni 1 s de audio")
    return np.concatenate(bloques)


def filtrar(x):
    """Mismo filtrado que el nodo: pasa-altos con estado inicial en la 1.ª muestra."""
    y, _ = signal.sosfilt(_SOS_HP, x, zi=signal.sosfilt_zi(_SOS_HP) * x[0])
    xb = signal.sosfilt(_SOS_ENV, y)
    return y, xb


def psd(x):
    """Densidad espectral por Welch, bloques de 1 s con ventana de Hann (como el nodo)."""
    return signal.welch(x, FS, window="hann", nperseg=FS, noverlap=0, detrend=False)


def potencia(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.sum(P[m]) * (f[1] - f[0]))


def db(v):
    return 10 * np.log10(max(v, 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="grabación a analizar (por ejemplo, estado normal)")
    ap.add_argument("--ruido", default=None, help="grabación con el motor apagado")
    ap.add_argument("--salida", default="comparacion_filtrado")
    ap.add_argument("--ver", action="store_true", help="abrir el gráfico en una ventana")
    ap.add_argument("--ms", type=float, default=200.0,
                    help="milisegundos que se muestran en la forma de onda y la envolvente")
    ap.add_argument("--inicio", type=float, default=1.0,
                    help="segundo desde el que se muestra (se evita el inicio del registro)")
    a = ap.parse_args()
    import matplotlib
    if not a.ver:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = leer(a.wav)
    y, xb = filtrar(x)
    f, Px = psd(x)
    _, Py = psd(y)
    bandas = [(0, F_MIN, "0-20 Hz")] + [(lo, hi, f"{lo}-{hi} Hz") for lo, hi in BANDAS_HZ]

    ruido = None
    if a.ruido:
        xr = leer(a.ruido)
        yr, _ = filtrar(xr)
        _, Prx = psd(xr)
        _, Pry = psd(yr)
        ruido = (Prx, Pry)

    # ------------------------------------------------------------ números
    dc = float(np.mean(x))
    filas = []
    print(f"\n{os.path.basename(a.wav)}  ({len(x) / FS:.0f} s)")
    print(f"Componente continua sin filtrar: {dc:+.2e} ({20 * np.log10(abs(dc) + 1e-12):.1f} dBFS)")
    print(f"RMS sin filtrar: {db(np.mean(x ** 2)):.1f} dBFS   filtrada: {db(np.mean(y ** 2)):.1f} dBFS")
    cab = f"{'banda':<14}{'sin filtrar':>13}{'filtrada':>11}{'quitado':>10}"
    if ruido:
        cab += f"{'SNR sin filtrar':>17}{'SNR filtrada':>14}"
    print("\nPotencia por banda [dBFS]" + ("  y relación señal/ruido [dB]" if ruido else ""))
    print(cab)
    for lo, hi, nombre in bandas + [(F_MIN, F_MAX, "20-6000 Hz"), (0, FS / 2, "total")]:
        px, py = potencia(f, Px, lo, hi), potencia(f, Py, lo, hi)
        fila = {"banda": nombre, "sin_filtrar_dbfs": db(px), "filtrada_dbfs": db(py),
                "quitado_db": db(px) - db(py)}
        linea = f"{nombre:<14}{db(px):>13.1f}{db(py):>11.1f}{db(px) - db(py):>10.1f}"
        if ruido:
            nx, ny = potencia(f, ruido[0], lo, hi), potencia(f, ruido[1], lo, hi)
            fila["snr_sin_filtrar_db"] = db(px) - db(nx)
            fila["snr_filtrada_db"] = db(py) - db(ny)
            linea += f"{db(px) - db(nx):>17.1f}{db(py) - db(ny):>14.1f}"
        filas.append(fila)
        print(linea)

    # ------------------------------------------------------------ archivos
    os.makedirs(a.salida, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.wav))[0]
    ruta_csv = os.path.join(a.salida, base + "_filtrado.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for fila in filas:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in fila.items()})

    t0 = a.inicio                                  # por defecto desde 1 s (sin el inicio)
    n0, n1 = int(t0 * FS), int((t0 + a.ms / 1000) * FS)
    t = np.arange(n0, n1) / FS * 1000
    env = np.abs(signal.hilbert(xb))
    m = f > 0

    def onda(ax):
        ax.plot(t, x[n0:n1], color=GRIS, lw=0.8, label="Sin filtrar")
        ax.plot(t, y[n0:n1], color=AZUL, lw=0.8, label="Pasa-altos 20 Hz")
        ax.set(title=f"Forma de onda ({a.ms:.0f} ms)", xlabel="Tiempo [ms]", ylabel="Amplitud [FS]")
        ax.legend(fontsize=9, loc="upper right")

    def envolvente(ax):
        ax.plot(t, xb[n0:n1], color=VERDE, lw=0.8, label="Pasa-banda 1-6 kHz")
        ax.plot(t, env[n0:n1], color=NARANJA, lw=1.2, label="Envolvente")
        ax.set(title=f"Rama de envolvente ({a.ms:.0f} ms)", xlabel="Tiempo [ms]", ylabel="Amplitud [FS]")
        ax.legend(fontsize=9, loc="upper right")

    def densidad(ax):
        ax.semilogx(f[m], 10 * np.log10(Px[m] + 1e-20), color=GRIS, lw=0.8, label="Sin filtrar")
        ax.semilogx(f[m], 10 * np.log10(Py[m] + 1e-20), color=AZUL, lw=0.8, label="Pasa-altos 20 Hz")
        if ruido:
            ax.semilogx(f[m], 10 * np.log10(ruido[1][m] + 1e-20), color=NARANJA, lw=0.8,
                        label="Motor apagado, pasa-altos 20 Hz")
        ax.axvline(F_MIN, color="k", ls="--", lw=0.8)
        ax.set(title="Densidad espectral de potencia (Welch, bloques de 1 s, ventana de Hann)",
               xlabel="Frecuencia [Hz]", ylabel="Densidad [dB re FS²/Hz]", xlim=(1, FS / 2))
        ax.legend(fontsize=9, loc="lower left")

    todos = [Px[m], Py[m]] + ([ruido[0][m], ruido[1][m]] if ruido else [])
    ylim = (10 * np.log10(min(v.min() for v in todos) + 1e-20) - 3,
            10 * np.log10(max(v.max() for v in todos) + 1e-20) + 5)

    def par(ax, p_sin, p_con, col, titulo, etiqueta):
        """Densidad espectral de una misma grabación sin filtrar y filtrada."""
        ax.semilogx(f[m], 10 * np.log10(p_sin[m] + 1e-20), color=GRIS, lw=0.8,
                    label=f"{etiqueta}, sin filtrar")
        ax.semilogx(f[m], 10 * np.log10(p_con[m] + 1e-20), color=col, lw=0.8,
                    label=f"{etiqueta}, pasa-altos 20 Hz")
        ax.axvline(F_MIN, color="k", ls="--", lw=0.8)
        ax.set(title=titulo, xlabel="Frecuencia [Hz]", ylabel="Densidad [dB re FS²/Hz]",
               xlim=(1, FS / 2), ylim=ylim)
        ax.legend(fontsize=9, loc="lower left")

    def motor(ax):
        par(ax, Px, Py, AZUL, "Densidad espectral del motor en funcionamiento normal", "Motor")

    def apagado(ax):
        par(ax, ruido[0], ruido[1], NARANJA, "Densidad espectral del ruido de fondo (motor apagado)",
            "Motor apagado")

    # figura combinada (como antes)
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), dpi=130,
                           gridspec_kw={"height_ratios": [1, 1, 1.4]})
    for fn, a_ in zip((onda, envolvente, densidad), ax):
        fn(a_)
        a_.grid(alpha=.3, which="both")
    fig.tight_layout()
    ruta_png = os.path.join(a.salida, base + "_filtrado.png")
    fig.savefig(ruta_png)

    # una figura por gráfico, para insertarlas por separado en el informe
    rutas = []
    separadas = [(onda, "_1_forma_onda", 3.8), (envolvente, "_2_envolvente", 3.8),
                 (motor, "_3_densidad_motor", 5.0)]
    if ruido:
        separadas += [(apagado, "_4_densidad_ruido", 5.0), (densidad, "_5_densidad_motor_y_ruido", 5.0)]
    else:
        separadas += [(densidad, "_5_densidad_motor_y_ruido", 5.0)]
    for fn, sufijo, alto in separadas:
        fg, ax1 = plt.subplots(figsize=(10, alto))
        fn(ax1)
        ax1.grid(alpha=.3, which="both")
        fg.tight_layout()
        r = os.path.join(a.salida, base + sufijo + ".png")
        fg.savefig(r, dpi=300)
        plt.close(fg)
        rutas.append(r)
    print("\nArchivos:\n  " + "\n  ".join([ruta_png] + rutas + [ruta_csv]))
    if a.ver:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())