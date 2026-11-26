"""
entrenar.py - Entrena el clasificador a partir del CSV de características.
Se puede correr en la Pi o en la PC (solo numpy).

  python entrenar.py datos/caracteristicas_captura.csv
  python entrenar.py datos/caracteristicas_captura.csv --modo unico
  python entrenar.py datos/caracteristicas_captura.csv --excluir s1

Dos modos de modelo:
  ponderado (por defecto)  un modelo por sesión; cada uno pesa según qué tan bien reconoce
                           las OTRAS sesiones. Una sesión mal grabada pesa poco.
  unico                    un solo modelo con todas las sesiones juntas.

Siempre imprime la validación POR SESIÓN de los dos modos (la medida honesta: el modelo se
prueba con una sesión que no vio al entrenar, y en el ponderado esa sesión tampoco
participa del cálculo de los pesos), para elegir con datos. Con menos de 3 sesiones el
modo ponderado no se puede validar y se usa el único.

Estados (sin guiones bajos): apagado, normal, combustible, aire  (ver datos.ESTADOS)

IMPORTANTE: todas las clases tienen que estar grabadas en el MISMO generador, con el mismo
micrófono y en el mismo lugar. Si una clase viene de otra máquina (por ejemplo un audio de
"desbalanceo" bajado de internet), el modelo la separa perfecto, pero porque aprende a
distinguir la máquina y el micrófono, no la falla: el resultado no vale.
"""
import argparse
from collections import Counter

import numpy as np

import modelo


def imprimir_conf(clases, conf):
    ancho = max(len(c) for c in clases) + 2
    print(" " * ancho + "".join(f"{c[:11]:>12}" for c in clases) + f"{'aciertos':>10}")
    for i, c in enumerate(clases):
        n = conf[i].sum()
        print(f"{c:<{ancho}}" + "".join(f"{v:>12}" for v in conf[i])
              + (f"{conf[i, i] / n:>10.0%}" if n else f"{'-':>10}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--salida", default="modelo.npz")
    ap.add_argument("--modo", choices=["ponderado", "unico"], default="ponderado",
                    help="modelo que se guarda (por defecto: ponderado por sesión)")
    ap.add_argument("--excluir", nargs="*", default=[],
                    help="sesiones que no se usan (ej.: --excluir s1)")
    ap.add_argument("--max-clip", type=float, default=0.01,
                    help="descarta registros con más de esta fracción saturada")
    args = ap.parse_args()

    filas = modelo.leer_csv(args.csv)
    X, cols = modelo.matriz(filas)
    y = np.array([f["estado"] for f in filas])
    ses = np.array([f.get("sesion", "") for f in filas])
    clip = np.array([float(f.get("clip_frac", 0) or 0) for f in filas])

    ok = np.isfinite(X).all(axis=1) & (clip <= args.max_clip) & ~np.isin(ses, args.excluir)
    if args.excluir:
        print(f"Sesiones excluidas: {', '.join(args.excluir)}")
    malos = (~np.isfinite(X).all(axis=1) | (clip > args.max_clip)).sum()
    if malos:
        print(f"Descartados {malos} registros (NaN o saturados)")
    X, y, ses = X[ok], y[ok], ses[ok]

    sesiones = sorted(set(ses.tolist()))
    clases = sorted(set(y.tolist()))
    print("\nRegistros por estado y sesión:")
    print(f"{'':<14}" + "".join(f"{s:>7}" for s in sesiones) + f"{'total':>8}")
    for c in clases:
        print(f"{c:<14}" + "".join(f"{((y == c) & (ses == s)).sum():>7}" for s in sesiones)
              + f"{(y == c).sum():>8}")
    for c, n in Counter(y.tolist()).items():
        if n < 30:
            print(f"  AVISO: '{c}' tiene solo {n} registros (recomendado >= 30-50)")
    if len(clases) < 2:
        print("Hace falta al menos 2 estados para entrenar.")
        return 1

    # ---------------------------------------------------- validación de los dos modos
    print("\n================ VALIDACIÓN POR SESIÓN ================")
    cl, conf_u, acc_u, modo_v, om_u = modelo.validar(X, y, ses)
    print(f"\nModelo ÚNICO  -  validación {modo_v}: exactitud = {acc_u:.1%}")
    if om_u:
        print(f"  ({om_u} pliegues omitidos: faltaban estados en el entrenamiento)")
    imprimir_conf(cl, conf_u)

    acc_p = None
    if len(sesiones) >= 3:
        cl, conf_p, acc_p, _, om_p = modelo.validar(X, y, ses, ponderado=True)
        print(f"\nModelo PONDERADO POR SESIÓN  -  validación por sesión: exactitud = {acc_p:.1%}")
        if om_p:
            print(f"  ({om_p} pliegues omitidos: faltaban estados o sesiones)")
        imprimir_conf(cl, conf_p)
    else:
        print(f"\nModelo PONDERADO: no se puede validar con {len(sesiones)} sesión(es); "
              "hacen falta al menos 3.")

    # ---------------------------------------------------- modelo final
    modo = args.modo
    if modo == "ponderado" and len(sesiones) < 3:
        print("  -> Se guarda el modelo ÚNICO.")
        modo = "unico"
    if acc_p is not None:
        mejor = "ponderado" if acc_p > acc_u else "unico"
        print(f"\nMejor en la validación: {mejor.upper()} "
              f"(único {acc_u:.1%}, ponderado {acc_p:.1%}).")
        if mejor != modo:
            print(f"  Se guarda el modo pedido ({modo}); para guardar el otro: --modo {mejor}")

    if modo == "ponderado":
        m, inf = modelo.ajustar_ponderado(X, y, ses)
        print("\nPesos de cada sesión en el modelo final")
        print("(exactitud del modelo de esa sesión al reconocer las DEMÁS sesiones):")
        print(f"{'sesión':<10}{'exactitud':>11}{'peso':>8}")
        for s, a, w in zip(inf["usadas"], inf["exactitud"], inf["pesos"]):
            print(f"{s:<10}{a:>11.1%}{w:>8.2f}")
        if inf["saltadas"]:
            print(f"  Sesiones sin todos los estados (no se usan): {', '.join(inf['saltadas'])}")
    else:
        m = modelo.ajustar(X, y)
    m["columnas"] = np.array(cols)
    modelo.guardar(m, args.salida)
    print(f"\nModelo {modo.upper()} guardado en {args.salida} ({len(cols)} características, "
          f"{len(m['clases'])} estados: {', '.join(m['clases'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())