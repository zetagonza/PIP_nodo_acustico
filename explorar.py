"""
explorar.py - Diagnóstico del dataset cuando la clasificación no da bien.
Solo lee el CSV: no graba ni modifica nada. Se puede correr en la Pi o en la PC.

  python explorar.py datos/caracteristicas_captura.csv

Responde cuatro preguntas:
  1. ¿Falla igual en las dos direcciones? (entrenar con s1 y probar con s2, y al revés)
  2. ¿Cambian los valores básicos del motor entre estados y entre sesiones?
     (velocidad, inestabilidad, nivel, armónicos)
  3. ¿Qué características separan normal de aire, y lo hacen IGUAL en todas las sesiones?
  4. ¿Mejora la exactitud con algunas variantes del modelo? (sin tocar los audios)
"""
import argparse
from collections import Counter

import numpy as np

import modelo

BASICAS = ["f_fire_hz", "vel_inestabilidad_pct", "rms_dbfs", "prominencia_peine_db",
           "curtosis_1k_6k", "centroide_hz", "h01_db", "h02_db", "ratio_impar_par_db"]
N_CALIB = 20      # registros 'normal' de cada sesión usados como calibración (variante D)


def cargar(ruta, max_clip):
    filas = modelo.leer_csv(ruta)
    X, cols = modelo.matriz(filas)
    y = np.array([f["estado"] for f in filas])
    ses = np.array([f.get("sesion", "") for f in filas])
    arch = np.array([f.get("archivo", "") for f in filas])
    clip = np.array([float(f.get("clip_frac", 0) or 0) for f in filas])
    extra = {c: np.array([float(f.get(c, "nan") or "nan") for f in filas]) for c in BASICAS}
    ok = np.isfinite(X).all(axis=1) & (clip <= max_clip)
    return X[ok], cols, y[ok], ses[ok], arch[ok], {c: v[ok] for c, v in extra.items()}


def imprimir_conf(clases, conf):
    ancho = max(len(c) for c in clases) + 2
    print(" " * ancho + "".join(f"{c[:11]:>12}" for c in clases))
    for i, c in enumerate(clases):
        print(f"{c:<{ancho}}" + "".join(f"{v:>12}" for v in conf[i]))


def evaluar(X, y, ses, transformar=None, mascara_test=None):
    """Validación dejando una sesión afuera. transformar(Xtr, ytr, Xte) -> (Xtr', Xte')
    se ajusta SOLO con el entrenamiento de cada pliegue (sin fuga de información)."""
    clases = sorted(set(y.tolist()))
    conf = np.zeros((len(clases), len(clases)), dtype=int)
    for s in sorted(set(ses.tolist())):
        te = ses == s
        tr = ~te
        if mascara_test is not None:
            te = te & mascara_test
        Xtr, Xte = X[tr], X[te]
        if transformar is not None:
            Xtr, Xte = transformar(Xtr, y[tr], Xte)
        m = modelo.ajustar(Xtr, y[tr])
        for xi, yi in zip(Xte, y[te]):
            conf[clases.index(yi), clases.index(modelo.predecir(m, xi)["clase"])] += 1
    return np.trace(conf) / max(conf.sum(), 1), clases, conf


def fisher(X, y):
    """Razón F de una ANOVA de un factor, por columna (cuánto separa las clases)."""
    clases = sorted(set(y.tolist()))
    mu = X.mean(axis=0)
    entre = sum((y == c).sum() * (X[y == c].mean(axis=0) - mu) ** 2 for c in clases)
    dentro = sum(((X[y == c] - X[y == c].mean(axis=0)) ** 2).sum(axis=0) for c in clases)
    return (entre / (len(clases) - 1)) / (dentro / max(len(y) - len(clases), 1) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--max-clip", type=float, default=0.01)
    a = ap.parse_args()
    X, cols, y, ses, arch, extra = cargar(a.csv, a.max_clip)
    sesiones = sorted(set(ses.tolist()))
    estados = sorted(set(y.tolist()))
    print(f"{len(y)} registros, {len(cols)} características, sesiones {sesiones}")

    # ------------------------------------------------------------ 1
    print("\n=== 1. Cada dirección por separado ===")
    clases = estados
    for s in sesiones:
        tr, te = ses != s, ses == s
        m = modelo.ajustar(X[tr], y[tr])
        conf = np.zeros((len(clases), len(clases)), dtype=int)
        for xi, yi in zip(X[te], y[te]):
            conf[clases.index(yi), clases.index(modelo.predecir(m, xi)["clase"])] += 1
        print(f"\nEntrena con {sorted(set(ses[tr].tolist()))}, prueba con {s}: "
              f"exactitud {np.trace(conf) / conf.sum():.1%}")
        imprimir_conf(clases, conf)

    # ------------------------------------------------------------ 2
    print("\n=== 2. Valores básicos por estado y sesión (media ± desvío) ===")
    for c in BASICAS:
        v = extra[c]
        if not np.isfinite(v).any():
            continue
        print(f"\n{c}")
        for e in estados:
            linea = f"  {e:<12}"
            for s in sesiones:
                k = (y == e) & (ses == s)
                linea += f"  {s}: {np.nanmean(v[k]):9.3f} ± {np.nanstd(v[k]):7.3f}"
            print(linea)

    # ------------------------------------------------------------ 3
    if "normal" in estados and "aire" in estados:
        print("\n=== 3. Lo que separa 'aire' de 'normal', sesión por sesión ===")
        print("d = (media aire - media normal) / desvío. |d| > 1 separa; el signo tiene que")
        print("ser el MISMO en todas las sesiones, si no, es efecto de la sesión y no de la falla.")
        filas = []
        for j, c in enumerate(cols):
            ds = []
            for s in sesiones:
                a_ = X[(y == "aire") & (ses == s), j]
                n_ = X[(y == "normal") & (ses == s), j]
                sd = np.sqrt((a_.var() + n_.var()) / 2) + 1e-12
                ds.append((a_.mean() - n_.mean()) / sd)
            ds = np.array(ds)
            consistente = np.all(ds > 0) or np.all(ds < 0)
            filas.append((np.min(np.abs(ds)) if consistente else 0.0, c, ds))
        filas.sort(key=lambda t: -t[0])
        print(f"{'característica':<24}" + "".join(f"{s:>9}" for s in sesiones))
        for _, c, ds in filas[:12]:
            print(f"{c:<24}" + "".join(f"{d:>+9.2f}" for d in ds))
        n_buenas = sum(1 for f in filas if f[0] > 1)
        print(f"Características que separan aire de normal en TODAS las sesiones (|d|>1): {n_buenas}")

    # ------------------------------------------------------------ 4
    print("\n=== 4. Variantes del modelo (validación por sesión) ===")
    res = []
    acc, _, _ = evaluar(X, y, ses)
    res.append(("A. Modelo actual (63 características)", acc))

    for k in (10, 20, 30):
        def top_k(Xtr, ytr, Xte, k=k):
            idx = np.argsort(-fisher(Xtr, ytr))[:k]
            return Xtr[:, idx], Xte[:, idx]
        acc, _, _ = evaluar(X, y, ses, top_k)
        res.append((f"B. Solo las {k} mejores (elegidas con el entrenamiento)", acc))

    agregadas = [c for c in ("f_fire_hz",) if np.isfinite(extra[c]).all()]
    if agregadas:
        Xf = np.column_stack([X] + [extra[c] for c in agregadas])
        acc, _, _ = evaluar(Xf, y, ses)
        res.append(("C. + frecuencia de encendido (velocidad) como característica", acc))

    # D: calibración por sesión: se resta la media de los primeros N_CALIB registros
    # 'normal' de ESA sesión (en el nodo: unos minutos de 'normal' al instalarlo)
    if "normal" in estados:
        Xc = X.copy()
        calib = np.zeros(len(y), dtype=bool)
        for s in sesiones:
            idx = np.flatnonzero((y == "normal") & (ses == s))
            idx = idx[np.argsort(arch[idx])][:N_CALIB]
            calib[idx] = True
            Xc[ses == s] -= X[idx].mean(axis=0)
        acc, cl, conf_d = evaluar(Xc, y, ses, mascara_test=~calib)
        res.append((f"D. Calibración por sesión ({N_CALIB} registros 'normal' por sesión)", acc))

    for nombre, acc in res:
        print(f"  {acc:6.1%}  {nombre}")
    if "normal" in estados:
        print("\nMatriz de confusión de la variante D:")
        imprimir_conf(cl, conf_d)
    print("\nRegistros por estado:", dict(Counter(y.tolist())))


if __name__ == "__main__":
    main()