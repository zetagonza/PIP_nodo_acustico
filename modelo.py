"""
modelo.py - Clasificador liviano (SOLO numpy, sin scikit-learn) para la Pi.

Modelo: análisis discriminante lineal con covarianza compartida y "shrinkage"
(estable con pocas muestras y muchas características) + rechazo por distancia
de Mahalanobis: si la muestra está demasiado lejos de TODAS las clases
conocidas, se informa "desconocido" en lugar de forzar una etiqueta.

Hay dos formas de modelo, con la misma interfaz (predecir):
  - ÚNICO: un solo modelo entrenado con todas las sesiones juntas.
  - PONDERADO POR SESIÓN: un modelo por sesión de grabación. Cada uno se evalúa con las
    OTRAS sesiones y recibe un peso según qué tan bien las reconoce (exactitud por encima
    del azar). En el diagnóstico, las probabilidades de cada modelo se combinan con esos
    pesos: una sesión mal grabada (falla irregular, ruido externo) pesa poco.

El modelo entrenado ocupa unos pocos KB por sesión (modelo.npz).
"""
import csv

import numpy as np

# Columnas del CSV que NO son características del modelo (metadatos, calidad y
# variables que dependen de la ganancia/posición del micrófono o de la carga).
# Si querés que la caída de RPM cuente como síntoma, sacá "f_fire_hz" de acá.
NO_FEATURES = {"timestamp", "archivo", "fs_hz", "estado", "sesion", "carga_pct", "ruido",
               "distancia_m", "rpm_nominal", "perdidos", "clip_frac", "rms_dbfs", "rms_dbfs_std",
               "f_fire_hz", "prominencia_peine_db"}


def leer_csv(ruta):
    with open(ruta, newline="") as fh:
        return list(csv.DictReader(fh))


def matriz(filas, columnas=None):
    if columnas is None:
        columnas = [c for c in filas[0].keys() if c not in NO_FEATURES]
    X = np.array([[float(f[c]) for c in columnas] for f in filas])
    return X, list(columnas)


def ajustar(X, y, shrink=0.2, factor_rechazo=1.5):
    y = np.asarray(y)
    clases = sorted(set(y.tolist()))
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    medias = np.array([Z[y == c].mean(axis=0) for c in clases])
    idx = np.array([clases.index(c) for c in y])
    R = Z - medias[idx]
    d = Z.shape[1]
    S = R.T @ R / max(len(Z) - len(clases), 1)
    S = (1 - shrink) * S + shrink * np.eye(d) * np.trace(S) / d
    P = np.linalg.inv(S)
    d2 = np.einsum("ij,jk,ik->i", R, P, R)
    return {"clases": np.array(clases), "mu": mu, "sd": sd, "medias": medias,
            "P": P, "umbral": np.array(factor_rechazo * np.percentile(d2, 99))}


def _lote_simple(m, X):
    """Probabilidades, distancias y rechazo de un modelo único para varias muestras."""
    Z = (np.atleast_2d(np.asarray(X, dtype=float)) - m["mu"]) / m["sd"]
    dif = Z[:, None, :] - m["medias"][None, :, :]
    d2 = np.einsum("nkd,de,nke->nk", dif, m["P"], dif)
    lp = -0.5 * d2
    p = np.exp(lp - lp.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    k = np.argmax(p, axis=1)
    rech = d2[np.arange(len(k)), k] > float(m["umbral"])
    return p, d2, rech


def es_ponderado(m):
    return "pesos" in m


def submodelo(m, i):
    """Modelo único número i de un modelo ponderado."""
    return {"clases": m["clases"], "mu": m["mu"][i], "sd": m["sd"][i],
            "medias": m["medias"][i], "P": m["P"][i], "umbral": m["umbral"][i]}


def predecir_lote(m, X):
    """Devuelve (prob [n, clases], d2 [n, clases], rechazo [n]) para un modelo único o
    ponderado. En el ponderado: prob y d2 son promedios pesados de los de cada sesión, y
    la muestra se rechaza si los modelos que la rechazan suman al menos la mitad del peso."""
    if not es_ponderado(m):
        return _lote_simple(m, X)
    p = d2 = rech = 0.0
    for i, w in enumerate(m["pesos"]):
        pi, di, ri = _lote_simple(submodelo(m, i), X)
        p = p + w * pi
        d2 = d2 + w * di
        rech = rech + w * ri
    return p, d2, rech >= 0.5


def predecir(m, x):
    """x: vector de características en el orden de m['columnas']."""
    p, d2, rech = predecir_lote(m, x)
    p, d2 = p[0], d2[0]
    k = int(np.argmax(p))
    return {"clase": str(m["clases"][k]), "prob": p, "d2": d2, "rechazo": bool(rech[0])}


def exactitud(m, X, y):
    p, _, _ = predecir_lote(m, X)
    pred = np.asarray(m["clases"])[np.argmax(p, axis=1)]
    return float(np.mean(pred == np.asarray(y)))


def ajustar_ponderado(X, y, ses, **kw):
    """Un modelo por sesión y un peso por modelo.

    Peso de la sesión s = exactitud del modelo de s sobre las DEMÁS sesiones, menos la del
    azar (1 / cantidad de clases), con mínimo 0; luego se normalizan para que sumen 1.
    Solo se usan sesiones que tengan todas las clases. Devuelve (modelo, informe)."""
    y, ses = np.asarray(y), np.asarray(ses)
    clases = sorted(set(y.tolist()))
    usadas, saltadas, mods = [], [], []
    for s in sorted(set(ses.tolist())):
        k = ses == s
        if set(y[k].tolist()) != set(clases):
            saltadas.append(s)
            continue
        mods.append(ajustar(X[k], y[k], **kw))
        usadas.append(s)
    if len(usadas) < 2:
        raise ValueError("Hacen falta al menos 2 sesiones con todos los estados para ponderar")
    acc = []
    for i, s in enumerate(usadas):
        otras = (ses != s) & np.isin(ses, usadas)
        acc.append(exactitud(mods[i], X[otras], y[otras]))
    acc = np.array(acc)
    w = np.clip(acc - 1.0 / len(clases), 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(len(usadas), 1.0 / len(usadas))
    m = {"clases": np.array(clases), "sesiones": np.array(usadas), "pesos": w,
         "exactitud_otras": acc}
    for c in ("mu", "sd", "medias", "P", "umbral"):
        m[c] = np.stack([np.asarray(mi[c]) for mi in mods])
    return m, {"usadas": usadas, "saltadas": saltadas, "exactitud": acc, "pesos": w}


def validar(X, y, grupos, **kw):
    """Validación cruzada. Por sesión (leave-one-session-out) si hay >=2
    sesiones; si no, 5 pliegues al azar (optimista: avisa).
    Con ponderado=True, en cada pliegue el modelo ponderado se arma y sus pesos se
    calculan SOLO con las sesiones de entrenamiento (la sesión de prueba no influye en
    los pesos): es la medida honesta del modelo ponderado. Necesita >= 3 sesiones."""
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    clases = sorted(set(y.tolist()))
    conf = np.zeros((len(clases), len(clases)), dtype=int)
    if len(set(grupos.tolist())) >= 2:
        modo = "por sesión"
        pliegues = [grupos == g for g in sorted(set(grupos.tolist()))]
    else:
        modo = "aleatorio (OPTIMISTA: grabá más sesiones)"
        orden = np.random.default_rng(0).permutation(len(y))
        pliegues = []
        for k in range(5):
            mask = np.zeros(len(y), dtype=bool)
            mask[orden[k::5]] = True
            pliegues.append(mask)
    omitidos = 0
    ponderado = kw.pop("ponderado", False)
    for test in pliegues:
        train = ~test
        if set(y[train].tolist()) != set(clases):
            omitidos += 1
            continue
        if ponderado:
            if len(set(grupos[train].tolist())) < 2:
                omitidos += 1
                continue
            m, _ = ajustar_ponderado(X[train], y[train], grupos[train], **kw)
        else:
            m = ajustar(X[train], y[train], **kw)
        p, _, _ = predecir_lote(m, X[test])
        for yi, pi in zip(y[test], p):
            conf[clases.index(yi), int(np.argmax(pi))] += 1
    acc = np.trace(conf) / max(conf.sum(), 1)
    return clases, conf, acc, modo, omitidos


def guardar(m, ruta):
    np.savez(ruta, **m)


def cargar(ruta):
    d = np.load(ruta)
    return {k: d[k] for k in d.files}