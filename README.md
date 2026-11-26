# Nodo embebido de monitoreo acústico para motogeneradores

Código y conjunto de datos del Proyecto Integrador Profesional **"Diseño e implementación de un nodo embebido basado en Raspberry Pi para monitoreo acústico y diagnóstico de fallas en motogeneradores a combustión off-grid"**, Facultad de Ingeniería, Universidad Nacional del Comahue (2026).

- Autor: Gonzalo Zambrano

El nodo escucha un grupo electrógeno con un micrófono MEMS, calcula 63 características acústicas por registro de 10 s y diagnostica su estado: **apagado**, **normal**, **falla de combustible** o **falla de aire**. Todo se ejecuta en la Raspberry Pi, sin conexión a internet.

## Resultados principales

| Indicador | Valor |
| --- | --- |
| Exactitud (validación cruzada dejando una sesión afuera, 1184 registros) | 76,7 % |
| Detección de falla (sin importar el tipo) | 87,8 % |
| Falsas alarmas en funcionamiento normal | 13,9 % |
| Tiempo de procesamiento por segundo de audio | 38 ms |
| Memoria RAM utilizada | 84 MB |
| Tamaño del modelo | 35 KB |

## Hardware

| Componente | Modelo |
| --- | --- |
| Computadora | Raspberry Pi Zero 2 W (512 MB de RAM) |
| Micrófono | TDK InvenSense ICS-43434 (MEMS, salida I²S, 24 bits) |
| Grupo electrógeno ensayado | Sincrolamp LT6500S (monocilíndrico, 4 tiempos, 5 kW) |

Conexión del micrófono:

| ICS-43434 | Raspberry Pi |
| --- | --- |
| VDD | Pin 1 (3,3 V) |
| GND | Pin 6 |
| SCK | Pin 12 (GPIO 18) |
| WS | Pin 35 (GPIO 19) |
| SD | Pin 38 (GPIO 20) |
| L/R | GND (canal izquierdo) |

## Instalación

1. Habilitar el micrófono I²S a 16 kHz. Agregar al final de `/boot/firmware/config.txt` y reiniciar:

   ```
   dtparam=i2s=on
   dtoverlay=adau7002-simple
   ```

2. Instalar dependencias:

   ```bash
   sudo apt install -y python3-venv libportaudio2
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy scipy sounddevice matplotlib
   ```

## Uso

```bash
# Grabar registros de un estado (repetir para cada estado y sesión)
python captura.py --estado normal --sesion s1 --n-registros 40

# Ver cuántos registros hay por estado y sesión
python datos.py resumen

# Entrenar el modelo (valida por sesión y guarda modelo.npz)
python entrenar.py datos/caracteristicas_captura.csv --modo unico

# Diagnóstico desde la terminal (una muestra, o continuo)
python diagnostico.py
python diagnostico.py --continuo --pausa 60

# Interfaz web de monitoreo en tiempo real (abrir http://<ip-de-la-raspberry>:8000)
python monitor.py
```

Análisis complementarios:

```bash
python comparar_filtrado.py datos/normal_s3_001_*.wav --ruido datos/apagado_s3_001_*.wav
python comparar_carga.py datos/caracteristicas_captura.csv --ref c0
python evaluar_sesion.py datos/caracteristicas_captura.csv --sesion c50 --titulo "Carga del 50 %"
```

## Estructura

| Archivo | Función |
| --- | --- |
| `captura.py` | Graba registros de 10 s, guarda el WAV y agrega sus características al CSV |
| `caracteristicas.py` | Filtrado y cálculo de las 63 características, por bloques de 1 s |
| `modelo.py` | Análisis discriminante lineal regularizado, predicción y rechazo de muestras desconocidas |
| `entrenar.py` | Entrenamiento y validación cruzada dejando una sesión afuera |
| `diagnostico.py` | Diagnóstico desde la línea de comandos |
| `monitor.py` | Interfaz web de monitoreo en tiempo real |
| `datos.py` | Gestión del conjunto de datos |
| `explorar.py`, `reporte.py` | Figuras de las firmas acústicas y de la matriz de confusión |
| `comparar_filtrado.py` | Efecto del filtrado y relación señal-ruido por banda |
| `comparar_carga.py` | Comparación del estado normal con y sin carga |
| `evaluar_sesion.py` | Diagnóstico de sesiones no usadas para entrenar |
| `datos/` | Conjunto de datos: WAV de 10 s (16 kHz, 16 bits) y `caracteristicas_captura.csv` |

## Conjunto de datos

Los archivos de `datos/` se nombran `<estado>_<sesión>_<número>_<fecha-hora>.wav`.

| Sesión | Fecha | Registros por estado | Condición |
| --- | --- | --- | --- |
| s1, s2 | 14/11/2026 | 72 | Carga de 40 W |
| s3 | 18/11/2026 | 72 | Carga de 40 W |
| s4, s5 | 18/11/2026 | 40 | Carga de 40 W |
| c0 | 20/11/2026 | 40 (solo normal) | Carga de 40 W |
| c50 | 20/11/2026 (normal) y 23/11/2026 (fallas) | 40 normal, 20 por falla | Carga de unos 2000 a 2500 W |

Las fallas se indujeron manualmente: la de combustible, cerrando parcialmente la llave de paso, y la de aire, cerrando parcialmente el cebador.

## Cómo citar

Zambrano Pincheira, G. M. (2026). *PIP_nodo_acustico: Nodo embebido para monitoreo acústico y diagnóstico de fallas en motogeneradores* [Software y conjunto de datos]. GitHub. https://github.com/zetagonza/PIP_nodo_acustico
