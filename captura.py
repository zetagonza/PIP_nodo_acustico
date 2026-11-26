"""
captura.py - Captura por streaming (sin acumular audio en RAM) para el nodo.

Por cada "registro" (por defecto 10 s):
  - lee el micrófono ICS-43434 (I2S, 16 kHz directo, 32 bits) con un callback + cola,
  - guarda el WAV crudo en 16 bits mono, escribiendo bloque a bloque en disco,
  - calcula las características online (caracteristicas.AcumuladorRegistro),
  - agrega una fila a datos/caracteristicas_captura.csv con los metadatos.

UN solo script para todos los estados: lo único que cambia es --estado
(apagado | normal | combustible | aire). Para una captura guiada, con
instrucciones por estado, usá  python ensayo.py --sesion s1.

Si ya hay registros de ese estado y sesión, NO los pisa ni los mezcla: te pide
elegir  --reemplazar  (los viejos van a datos/_descartados/)  o  --agregar.

Ejemplos:
  python captura.py --estado normal --sesion s1 --n-registros 40 --carga 50
  python captura.py --estado normal --sesion s1 --n-registros 40 --reemplazar
  python captura.py --estado normal --sesion s1 --n-registros 10 --agregar
  python captura.py --estado normal --sesion s2 --ruido ventilador

Entrada de audio: por defecto hw:0,0 (hardware I2S directo, 16 kHz reales).
Otros dispositivos:  python -m sounddevice   y   --dispositivo <índice o nombre>
"""
import argparse
import csv
import glob
import os
import queue
import re
import sys
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd

import datos
from caracteristicas import AcumuladorRegistro, FS_RAW, BLOQUE

# Entrada de audio por defecto: el hardware I2S directo, SIN la capa de conversión de ALSA
# (con "default" ALSA puede remuestrear por software). Requiere dtoverlay=adau7002-simple en
# /boot/firmware/config.txt. Si la tarjeta cambia de número:  arecord -l  y usar --dispositivo.
DISPOSITIVO = "hw:0,0"

# Al abrir la entrada de audio, el reloj I2S arranca y el micrófono sale de reposo: las
# primeras muestras traen un transitorio de gran amplitud (medido: ~1 ms cerca del fondo de
# escala). Se descarta el primer medio segundo de cada registro antes de procesar y guardar.
DESCARTE_INICIAL_S = 0.5

BLOQUE_CB = FS_RAW // 10   # 100 ms por callback -> 10 callbacks = 1 bloque de 1 s
assert BLOQUE % BLOQUE_CB == 0

_hw_verificado = [False]


def verificar_frecuencia_hw():
    """Avisa (una sola vez) si el hardware I2S corre a otra frecuencia que FS_RAW, es decir,
    si ALSA estuviera remuestreando por software. Solo funciona en Linux/ALSA."""
    if _hw_verificado[0]:
        return
    _hw_verificado[0] = True
    for ruta in glob.glob("/proc/asound/card*/pcm*c/sub*/hw_params"):
        try:
            txt = open(ruta).read()
        except OSError:
            continue
        m = re.search(r"rate:\s*(\d+)", txt)
        if m and int(m.group(1)) != FS_RAW:
            print(f"AVISO: el hardware de audio corre a {m.group(1)} Hz y se pidieron {FS_RAW} Hz: "
                  "ALSA está remuestreando por software. Corré  python probar_fs.py")


def grabar_registro(duracion_s, canal, dispositivo, f_fire_nom, ruta_wav):
    """Graba `duracion_s` segundos. Devuelve (dict de características, perdidos).
    Si se interrumpe o falla, borra el WAV incompleto."""
    cola = queue.Queue(maxsize=100)          # colchón de 10 s
    perdidos = [0]

    def callback(indata, frames, tiempo, status):
        if status:                           # overflow/underflow de ALSA
            perdidos[0] += 1
        try:
            cola.put_nowait(indata[:, canal].copy())
        except queue.Full:
            perdidos[0] += 1

    acc = AcumuladorRegistro(f_fire_nom)
    buf = np.empty(BLOQUE, dtype=np.int32)
    lleno = 0
    hechos = 0
    a_descartar = int(round(DESCARTE_INICIAL_S * FS_RAW / BLOQUE_CB))   # callbacks iniciales
    wf = None
    ok = False
    if ruta_wav:
        wf = wave.open(ruta_wav, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)                   # 16 bits (el MEMS tiene ~65 dB SNR)
        wf.setframerate(FS_RAW)

    try:
        with sd.InputStream(samplerate=FS_RAW, channels=2, dtype="int32",
                            blocksize=BLOQUE_CB, device=dispositivo,
                            callback=callback):
            while hechos < duracion_s:
                trozo = cola.get(timeout=3)
                verificar_frecuencia_hw()
                if a_descartar > 0:
                    a_descartar -= 1
                    continue
                buf[lleno:lleno + len(trozo)] = trozo
                lleno += len(trozo)
                if lleno >= BLOQUE:
                    acc.agregar_bloque(buf.astype(np.float64) / 2.0 ** 31)
                    if wf:
                        # 24 bits justificados a la izquierda en int32 -> top 16 bits
                        wf.writeframes((buf >> 16).astype("<i2").tobytes())
                    lleno = 0
                    hechos += 1
        ok = True
    finally:
        if wf:
            wf.close()
        if ruta_wav and not ok and os.path.exists(ruta_wav):
            os.remove(ruta_wav)              # nunca dejar un WAV a medias
    return acc.resultado(), perdidos[0]


def capturar_serie(estado, sesion, n_registros, *, inicio=1, duracion=10, rpm=3000.0,
                   canal=0, dispositivo=DISPOSITIVO, carga=0.0, ruido="ninguno",
                   distancia=1.0, carpeta=datos.CARPETA, guardar_wav=True, pausa=0.0):
    """Graba `n_registros` y agrega cada uno al CSV. Devuelve cuántos quedaron
    grabados. Ctrl+C corta limpio: el registro en curso se descarta."""
    os.makedirs(carpeta, exist_ok=True)
    f_fire_nom = rpm / 120.0
    grabados = 0
    error_audio = getattr(sd, "PortAudioError", OSError)
    try:
        for k in range(n_registros):
            if guardar_wav and datos.espacio_libre_mb(carpeta) < datos.MIN_LIBRE_MB:
                print(f"DETENIDO: quedan menos de {datos.MIN_LIBRE_MB} MB libres en el disco.\n"
                      "Liberá espacio (python datos.py limpiar) y seguí con --agregar.")
                break
            i = inicio + k
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            nombre = f"{estado}_{sesion}_{i:03d}_{ts}.wav"
            ruta = os.path.join(carpeta, nombre) if guardar_wav else None
            tmp = ruta + ".tmp" if ruta else None      # el WAV es definitivo solo con su fila
            try:
                feat, perdidos = grabar_registro(duracion, canal, dispositivo, f_fire_nom, tmp)
                meta = {"timestamp": ts, "archivo": nombre, "fs_hz": FS_RAW, "estado": estado,
                        "sesion": sesion, "carga_pct": carga, "ruido": ruido,
                        "distancia_m": distancia, "rpm_nominal": rpm, "perdidos": perdidos}
                datos.agregar_fila({**meta, **feat}, carpeta)
                if tmp:
                    os.replace(tmp, ruta)
            finally:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            grabados += 1
            print(f"[{k + 1:03d}/{n_registros}] {nombre} | RMS {feat['rms_dbfs']:.1f} dBFS | "
                  f"frec {feat['f_fire_hz']:.2f} Hz | espectro {feat['prominencia_peine_db']:.1f} dB | "
                  f"clip {feat['clip_frac']:.4f} | perdidos {perdidos}")
            if pausa > 0 and k < n_registros - 1:
                time.sleep(pausa)
    except KeyboardInterrupt:
        print("\nInterrumpido: el registro en curso se descartó.")
    except queue.Empty:
        print("ERROR: el micrófono no entrega datos (¿overlay I2S? ¿dispositivo?)")
    except error_audio as e:
        print(f"ERROR de audio: {e}\nListá los dispositivos con:  python -m sounddevice")
    return grabados


def dispositivo_arg(txt):
    """'1' -> 1 (índice); 'nombre' -> 'nombre'; None -> None."""
    return int(txt) if txt is not None and txt.isdigit() else txt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estado", required=True, choices=datos.ESTADOS)
    ap.add_argument("--sesion", default="s1", help="identificador de sesión/día (sin '_')")
    ap.add_argument("--n-registros", type=int, default=30)
    ap.add_argument("--duracion", type=int, default=10, help="segundos por registro")
    ap.add_argument("--pausa", type=float, default=0.0, help="segundos entre registros")
    ap.add_argument("--rpm", type=float, default=3000.0,
                    help="RPM nominales (f_fire = rpm/120)")
    ap.add_argument("--canal", type=int, default=0, choices=[0, 1],
                    help="0 = izquierdo (pin L/R a GND), 1 = derecho (pin L/R a VDD)")
    ap.add_argument("--dispositivo", default=DISPOSITIVO,
                    help="entrada de audio (por defecto el hardware directo hw:0,0)")
    ap.add_argument("--carga", type=float, default=0.0, help="%% de carga eléctrica")
    ap.add_argument("--ruido", default="ninguno", help="ninguno | ventilador | ...")
    ap.add_argument("--distancia", type=float, default=1.0, help="metros micrófono-motor")
    ap.add_argument("--salida", default=datos.CARPETA)
    ap.add_argument("--no-wav", action="store_true", help="no guardar el audio crudo")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--reemplazar", action="store_true",
                   help="mueve los registros viejos de este estado+sesión a _descartados/")
    g.add_argument("--agregar", action="store_true",
                   help="suma registros a los existentes (sigue la numeración)")
    args = ap.parse_args()

    if "_" in args.sesion:
        print("ERROR: el nombre de sesión no puede tener '_' (usá s1, s2, dia2...)")
        return 2

    for aviso in datos.avisos_espacio(args.salida):
        print("AVISO:", aviso)
    existentes = datos.contar(args.estado, args.sesion, args.salida)
    if existentes and not (args.reemplazar or args.agregar):
        print(f"Ya hay {existentes} registros de '{args.estado}' en la sesión '{args.sesion}'.\n"
              f"  --reemplazar  los viejos se mueven a {args.salida}/_descartados/ y se regraba\n"
              f"  --agregar     se suman a los existentes\n"
              f"(Para ver qué hay:  python datos.py resumen)")
        return 1
    if args.reemplazar and existentes:
        nf, nw, dest = datos.descartar(args.estado, args.sesion, carpeta=args.salida)
        print(f"Registros anteriores movidos a {dest} ({nw} WAV, {nf} filas)")

    try:
        inicio = datos.siguiente_indice(args.estado, args.sesion, args.salida)
        capturar_serie(args.estado, args.sesion, args.n_registros, inicio=inicio,
                       duracion=args.duracion, rpm=args.rpm, canal=args.canal,
                       dispositivo=dispositivo_arg(args.dispositivo), carga=args.carga,
                       ruido=args.ruido, distancia=args.distancia, carpeta=args.salida,
                       guardar_wav=not args.no_wav, pausa=args.pausa)
    except ValueError as e:      # CSV con columnas distintas
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())