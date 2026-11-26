"""
evaluar_sesion.py - Clasifica con el modelo YA ENTRENADO todos los registros de una o más
sesiones que el modelo no usó (por ejemplo la prueba con carga c50) y arma la matriz de
confusión, con el mismo estilo que fig_confusion.png.

  python evaluar_sesion.py datos/caracteristicas_captura.csv --sesion c50
  python evaluar_sesion.py datos/caracteristicas_captura.csv --sesion c50 --titulo "Carga del 50 %"

Cada registro se diagnostica igual que en el nodo: si supera el umbral de distancia se
informa DESCONOCIDO, y si está saturado, INVÁLIDO. Se reportan además los indicadores de
detección de falla (falla = combustible o aire).

Genera en informe/ (sin textos explicativos dentro del gráfico):
  fig_confusion_<sesiones>.png
  confusion_<sesiones>.csv
"""
import argparse
import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import datos
import modelo
from reporte import TINTA, estilo

FALLAS = {"combustible", "aire"}


def diagnosticar(m, f, cols, max_clip):
    if float(f.get("clip_frac", 0) or 0) > max_clip:
        return "INVÁLIDO"
    r = modelo.predecir(m, np.array([float(f[c]) for c in cols]))
    return "DESCONOCIDO" if r["rechazo"] else r["clase"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--sesion", nargs="+", required=True)
    ap.add_argument("--modelo", default="modelo.npz")
    ap.add_argument("--titulo", default=None, help="texto del título del gráfico")
    ap.add_argument("--max-clip", type=float, default=0.01)
    ap.add_argument("--salida", default="informe")
    a = ap.parse_args()
    estilo()

    m = modelo.cargar(a.modelo)
    cols = [str(c) for c in m["columnas"]]
    filas = [f for f in modelo.leer_csv(a.csv) if f.get("sesion") in a.sesion]
    if not filas:
        raise SystemExit(f"No hay registros de las sesiones {a.sesion}")
    reales = [e for e in datos.ESTADOS if any(f["estado"] == e for f in filas)]
    pred = [diagnosticar(m, f, cols, a.max_clip) for f in filas]
    columnas = [str(c) for c in datos.ESTADOS if c in [str(x) for x in m["clases"]]]
    columnas += [x for x in ("DESCONOCIDO", "INVÁLIDO") if x in pred]
    conf = np.array([[sum(1 for f, p in zip(filas, pred) if f["estado"] == r and p == c)
                      for c in columnas] for r in reales])

    # ------------------------------------------------------------ texto
    ancho = max(len(r) for r in reales) + 2
    print(f"Sesiones {', '.join(a.sesion)}: {len(filas)} registros, modelo {a.modelo}\n")
    print(" " * ancho + "".join(f"{c[:11]:>12}" for c in columnas) + f"{'aciertos':>10}")
    for i, r in enumerate(reales):
        n = conf[i].sum()
        ok = conf[i, columnas.index(r)] if r in columnas else 0
        print(f"{r:<{ancho}}" + "".join(f"{v:>12}" for v in conf[i]) + f"{ok / n:>10.0%}")
    ok_tot = sum(conf[i, columnas.index(r)] for i, r in enumerate(reales) if r in columnas)
    print(f"\nExactitud: {ok_tot / len(filas):.1%}  ({ok_tot} de {len(filas)})")

    reales_f = [(f["estado"], p) for f, p in zip(filas, pred)]
    con_falla = [p for e, p in reales_f if e in FALLAS]
    normales = [p for e, p in reales_f if e == "normal"]
    if con_falla:
        det = sum(p in FALLAS for p in con_falla)
        tipo = sum(e == p for e, p in reales_f if e in FALLAS)
        print(f"Fallas detectadas como falla: {det} de {len(con_falla)} ({det / len(con_falla):.1%})")
        print(f"Tipo de falla correcto:       {tipo} de {len(con_falla)} ({tipo / len(con_falla):.1%})")
    if normales:
        fa = sum(p in FALLAS for p in normales)
        print(f"Falsas alarmas en normal:     {fa} de {len(normales)} ({fa / len(normales):.1%})")
    otros = Counter(p for p in pred if p in ("DESCONOCIDO", "INVÁLIDO"))
    if otros:
        print("Rechazados: " + ", ".join(f"{k} {v}" for k, v in otros.items()))

    # ------------------------------------------------------------ archivos
    os.makedirs(a.salida, exist_ok=True)
    nombre = "_".join(a.sesion)
    with open(os.path.join(a.salida, f"confusion_{nombre}.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["real"] + columnas)
        for r, fila in zip(reales, conf):
            w.writerow([r] + list(fila))

    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    mapa = LinearSegmentedColormap.from_list("azul", ["#f3f8fe", "#86b6ef", "#256abf", "#104281"])
    fig, ax = plt.subplots(figsize=(1.35 * len(columnas) + 2.2, 1.1 * len(reales) + 1.8))
    ax.imshow(norm, cmap=mapa, vmin=0, vmax=1, aspect="auto")
    for i in range(len(reales)):
        for j in range(len(columnas)):
            t = TINTA if norm[i, j] < 0.55 else "#ffffff"
            ax.text(j, i, f"{conf[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color=t)
    ax.set_xticks(range(len(columnas)), [c.lower() if c.isupper() else c for c in columnas],
                  fontsize=9)
    ax.set_yticks(range(len(reales)), reales, fontsize=9)
    ax.set_xlabel("Estado predicho")
    ax.set_ylabel("Estado real")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    titulo = a.titulo or f"Sesión {', '.join(a.sesion)}"
    ax.set_title(f"{titulo} · exactitud {ok_tot / len(filas):.1%}", fontsize=10)
    fig.tight_layout()
    ruta = os.path.join(a.salida, f"fig_confusion_{nombre}.png")
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"\nArchivos: {ruta}, {os.path.join(a.salida, f'confusion_{nombre}.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())