"""
reporte.py - Arma el material del capítulo de RESULTADOS a partir del CSV del
dataset (correlo en la PC, no hace falta en la Pi). Genera en informe/:

  ranking_caracteristicas.csv   qué características separan mejor los estados
  tabla_resultados.md           media ± desvío por estado de las 8 mejores
  fig_top_caracteristicas.png   distribución de las 4 mejores, por estado
  fig_bandas.png                energía por banda de frecuencia, por estado
  fig_confusion.png             matriz de confusión (validación por sesión)
  estres_ruido.txt              entrena SIN ruido y prueba CON ruido (si hay)

  python reporte.py datos/caracteristicas_captura.csv

La validación deja una sesión afuera por vez (leave-one-session-out): con >= 3
sesiones por estado es la cifra honesta para reportar.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import datos
import modelo

# Un color fijo por estado (la identidad no cambia si filtrás estados).
COLORES = {"normal": "#2a78d6", "combustible": "#eb6834",
           "desbalanceo": "#1baf7a", "aire": "#eda100", "cebador": "#eda100", "apagado": "#e87ba4"}
EXTRA = ["#008300", "#6250d6", "#e34948", "#898781"]
SUPERFICIE, TINTA, TINTA2, MUDO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE = "#e1e0d9", "#c3c2b7"
MARCAS = ["o", "s", "^", "D", "v", "P"]


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE, "font.family": "sans-serif",
        "text.color": TINTA, "axes.labelcolor": TINTA2, "axes.edgecolor": EJE,
        "xtick.color": MUDO, "ytick.color": MUDO, "axes.titlecolor": TINTA,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def color(estado, orden):
    if estado in COLORES:
        return COLORES[estado]
    return EXTRA[(orden.index(estado)) % len(EXTRA)]


def razon_f(X, y):
    """Estadístico F de ANOVA por característica (varianza entre estados /
    varianza dentro de cada estado). Más alto = separa mejor los estados."""
    clases = sorted(set(y.tolist()))
    k, n = len(clases), len(y)
    if n <= k:
        return np.zeros(X.shape[1])
    mu = X.mean(axis=0)
    ssb = sum((y == c).sum() * (X[y == c].mean(axis=0) - mu) ** 2 for c in clases)
    ssw = sum(((X[y == c] - X[y == c].mean(axis=0)) ** 2).sum(axis=0) for c in clases)
    return (ssb / (k - 1)) / (ssw / (n - k) + 1e-12)


def fig_top(X, y, cols, idx, F, orden, ruta):
    fig, ejes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, j in zip(ejes.ravel(), idx[:4]):
        datos_e = [X[y == e, j] for e in orden]
        bp = ax.boxplot(datos_e, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops={"color": TINTA, "linewidth": 1.5},
                        whiskerprops={"color": MUDO, "linewidth": 1},
                        capprops={"color": MUDO, "linewidth": 1},
                        flierprops={"marker": "o", "markersize": 3,
                                    "markerfacecolor": MUDO, "markeredgecolor": SUPERFICIE})
        for parche, e in zip(bp["boxes"], orden):
            parche.set(facecolor=color(e, orden), edgecolor=SUPERFICIE, linewidth=1.5)
        ax.set_xticks(range(1, len(orden) + 1), orden, fontsize=8)
        ax.set_title(f"{cols[j]}  (F = {F[j]:.0f})", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Características que mejor separan los estados", x=0.07, ha="left",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def separar_etiquetas(ys, hueco):
    """Empuja etiquetas para que no se pisen (de menor a mayor y)."""
    orden = np.argsort(ys)
    pos = np.array(ys, dtype=float)
    for a, b in zip(orden[:-1], orden[1:]):
        if pos[b] - pos[a] < hueco:
            pos[b] = pos[a] + hueco
    return pos


def fig_bandas(X, y, cols, orden, ruta):
    bcols = sorted([c for c in cols if c.startswith("banda_")],
                   key=lambda c: int(c.split("_")[1]))
    if not bcols:
        return False
    ix = [cols.index(c) for c in bcols]
    etiquetas = [c.split("_")[1] + "–" + c.split("_")[2] for c in bcols]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    medias = {e: X[y == e][:, ix].mean(axis=0) for e in orden}
    for k, e in enumerate(orden):
        ax.plot(range(len(ix)), medias[e], color=color(e, orden), linewidth=2,
                marker=MARCAS[k % len(MARCAS)], markersize=7,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=e)
    ys = [medias[e][-1] for e in orden]
    rango = max(max(m) for m in medias.values()) - min(min(m) for m in medias.values())
    for e, yy in zip(orden, separar_etiquetas(ys, 0.05 * rango)):
        ax.annotate(e, (len(ix) - 1, yy), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TINTA2)
    ax.set_xticks(range(len(ix)), etiquetas)
    ax.set_xlim(-0.3, len(ix) - 1 + 1.3)
    ax.set_xlabel("Banda de frecuencia (Hz)")
    ax.set_ylabel("Energía relativa a la total (dB)")
    ax.set_title("Distribución de energía por banda, media por estado")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(orden), fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_confusion(clases, conf, acc, modo, ruta, orden):
    idx = [clases.index(e) for e in orden if e in clases]   # mismo orden que el resto
    clases = [clases[i] for i in idx]
    conf = conf[np.ix_(idx, idx)]
    norm =conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    mapa = LinearSegmentedColormap.from_list("azul", ["#f3f8fe", "#86b6ef", "#256abf", "#104281"])
    n = len(clases)
    fig, ax = plt.subplots(figsize=(1.3 * n + 2.2, 1.1 * n + 1.8))
    ax.imshow(norm, cmap=mapa, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            t = TINTA if norm[i, j] < 0.55 else "#ffffff"
            ax.text(j, i, f"{conf[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color=t)
    ax.set_xticks(range(n), clases, fontsize=9)
    ax.set_yticks(range(n), clases, fontsize=9)
    ax.set_xlabel("Estado predicho")
    ax.set_ylabel("Estado real")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Exactitud {acc:.1%} · validación {modo.split(' (')[0]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def estres_ruido(X, y, ruido, ruta):
    entrena = ruido == "ninguno"
    if entrena.all() or not entrena.any():
        return "Sin registros con ruido (--ruido) y sin ruido a la vez: prueba de estrés omitida."
    m = modelo.ajustar(X[entrena], y[entrena])
    ok = []
    for xi, yi in zip(X[~entrena], y[~entrena]):
        ok.append(modelo.predecir(m, xi)["clase"] == yi)
    ok = np.array(ok)
    lineas = [f"Entrenado con {entrena.sum()} registros SIN ruido; probado con "
              f"{(~entrena).sum()} registros CON ruido ({', '.join(sorted(set(ruido[~entrena])))}).",
              f"Exactitud global con ruido: {ok.mean():.1%}", "Por estado:"]
    for e in sorted(set(y[~entrena].tolist())):
        mk = y[~entrena] == e
        lineas.append(f"  {e:<13}{ok[mk].mean():.1%}  (n = {mk.sum()})")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=datos.ruta_csv())
    ap.add_argument("--salida", default="informe")
    ap.add_argument("--max-clip", type=float, default=0.01)
    a = ap.parse_args()

    filas = modelo.leer_csv(a.csv)
    X, cols = modelo.matriz(filas)
    y = np.array([f["estado"] for f in filas])
    ses = np.array([f.get("sesion", "") for f in filas])
    ruido = np.array([f.get("ruido", "ninguno") or "ninguno" for f in filas])
    clip = np.array([float(f.get("clip_frac", 0) or 0) for f in filas])
    ok = np.isfinite(X).all(axis=1) & (clip <= a.max_clip)
    X, y, ses, ruido = X[ok], y[ok], ses[ok], ruido[ok]
    orden = [e for e in datos.ESTADOS if e in set(y.tolist())] + \
            sorted(set(y.tolist()) - set(datos.ESTADOS))
    if len(orden) < 2:
        print("Hacen falta al menos 2 estados en el CSV.")
        return 1
    os.makedirs(a.salida, exist_ok=True)
    estilo()

    F = razon_f(X, y)
    idx = np.argsort(-F)
    with open(os.path.join(a.salida, "ranking_caracteristicas.csv"), "w") as fh:
        fh.write("posicion,caracteristica,F\n")
        for r, j in enumerate(idx, 1):
            fh.write(f"{r},{cols[j]},{F[j]:.2f}\n")
    print("Mejores 10 características (F de ANOVA):")
    for r, j in enumerate(idx[:10], 1):
        print(f"  {r:>2}. {cols[j]:<24}{F[j]:>10.1f}")

    top = idx[:8]
    enc = "| Característica | " + " | ".join(orden) + " |"
    md = [enc, "|" + "---|" * (len(orden) + 1)]
    for j in top:
        celdas = [f"{X[y == e, j].mean():.3g} ± {X[y == e, j].std():.2g}" for e in orden]
        md.append(f"| {cols[j]} | " + " | ".join(celdas) + " |")
    with open(os.path.join(a.salida, "tabla_resultados.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    fig_top(X, y, cols, idx, F, orden, os.path.join(a.salida, "fig_top_caracteristicas.png"))
    hay_bandas = fig_bandas(X, y, cols, orden, os.path.join(a.salida, "fig_bandas.png"))
    clases, conf, acc, modo, omit = modelo.validar(X, y, ses)
    fig_confusion(clases, conf, acc, modo, os.path.join(a.salida, "fig_confusion.png"), orden)
    print(f"\nValidación {modo}: exactitud {acc:.1%}"
          + (f" ({omit} pliegues omitidos)" if omit else ""))
    txt = estres_ruido(X, y, ruido, None)
    with open(os.path.join(a.salida, "estres_ruido.txt"), "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"\nArchivos en {a.salida}/" + ("" if hay_bandas else "  (sin fig_bandas: faltan columnas banda_*)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())