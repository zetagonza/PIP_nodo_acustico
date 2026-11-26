"""
caracteristicas.py - Extracción de características acústicas para el nodo
embebido (Raspberry Pi Zero 2 W, 512 MB de RAM). Solo numpy + scipy.

El audio se procesa en bloques de 1 s y se van ACUMULANDO estadísticos, de modo
que la RAM usada es constante (unos pocos MB) sin importar la duración del
registro.

El micrófono se muestrea DIRECTAMENTE a 16 kHz (modo de bajo consumo del ICS-43434, rango
6,25-18,75 kHz). No hay filtro FIR ni diezmado en la Raspberry: el filtro decimador
interno del micrófono (pasa-bajos que escala con fs: plano hasta 0,417*fs = 6,67 kHz,
media amplitud en fs/2 = 8 kHz, 58 dB de atenuación de banda) hace de anti-alias
ANTES de digitalizar.

Cadena por bloque (16 kHz):
  1. Pasa-altos Butterworth 20 Hz (SOS) (quita DC del MEMS y viento muy lento)
  2. Estadísticos temporales            (RMS, crest, curtosis, ZCR)
  3. Periodograma Hann de 1 s (bins de 1 Hz) acumulado  -> equivale a Welch
     + frecuencia de encendido de ESE segundo (para medir la inestabilidad de velocidad)
  4. Envolvente: pasa-banda 1-6 kHz + Hilbert, y su espectro; curtosis de esa banda
  5. MFCC livianos (se acumulan media y desvío, no se guardan las tramas)

Motor: 4 tiempos, monocilíndrico, 3000 rpm -> un encendido cada 2 vueltas:
  ciclo = 40 ms, frecuencia de encendido f_fire = 25 Hz, giro del eje = 50 Hz = 2*f_fire.

Banda de análisis: 20 Hz - 6 kHz, dentro de la banda plana del micrófono a 16 kHz
(hasta 6,67 kHz). Lo que cae por encima de 8 kHz se refleja (aliasing) hacia abajo, pero
el filtro del micrófono ya lo atenuó: los sonidos que se reflejarían dentro de la banda de
análisis (<= 6 kHz) vienen de más de 10 kHz, donde la atenuación es la de banda de rechazo.
Se verifica con analizar_fs.py.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.fft import rfft, dct

# ----------------------------------------------------------------- parámetros
FS = 16000                     # Hz, muestreo directo del ICS-43434 (modo de bajo consumo)
FS_RAW = FS                    # nombre histórico: los demás scripts lo importan
BLOQUE = FS                    # 1 s de audio por bloque (bins de 1 Hz en el periodograma)

# Frecuencia de encendido (firing) de un 4T monocilíndrico = RPM/2/60.
# 3000 rpm (50 Hz de red) -> 25 Hz.
F_FIRE_NOM = 25.0

BANDAS_HZ = [(20, 100), (100, 250), (250, 500), (500, 1000),
             (1000, 2000), (2000, 4000), (4000, 6000)]
F_MIN, F_MAX = 20, 6000        # banda de análisis
N_ARM_PSD = 12                 # armónicos de f_fire a medir en el PSD
N_ARM_ENV = 8                  # múltiplos de f_fire/2 a medir en la envolvente
ANCHO_ARM = 0.02               # semiancho relativo de la ventana de cada armónico (+-2 %)

# Trama de 128 ms = 3,2 ciclos del motor (ciclo de 40 ms). Con una trama más corta que un
# par de ciclos, cada trama cae en otra fase del ciclo y el desvío de los MFCC mide ese
# desfasaje en vez de la irregularidad real del motor.
N_FFT_MFCC = 2048              # 128 ms a 16 kHz (7,8 Hz de resolución)
HOP_MFCC = 1024                # solapamiento del 50 %
N_MEL = 26
N_MFCC = 13                    # se calcula c0..c12; en el CSV se usan c1..c12
PREENFASIS = 0.97

# ---------------------------------------------------- filtros (se crean 1 vez)
_SOS_HP = signal.butter(4, 20, "highpass", fs=FS, output="sos")
_SOS_ENV = signal.butter(4, [1000, 6000], "bandpass", fs=FS, output="sos")


def _banco_mel(n_filt, n_fft, fs, fmin, fmax):
    """Banco de filtros triangulares en escala Mel: m = 2595*log10(1 + f/700)."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    imel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    puntos = imel(np.linspace(mel(fmin), mel(fmax), n_filt + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / fs)
    fb = np.zeros((n_filt, len(bins)))
    for i in range(n_filt):
        izq, cen, der = puntos[i:i + 3]
        sube = (bins - izq) / (cen - izq)
        baja = (der - bins) / (der - cen)
        fb[i] = np.clip(np.minimum(sube, baja), 0.0, None)
    return fb


_FB_MEL = _banco_mel(N_MEL, N_FFT_MFCC, FS, 50.0, F_MAX)
_VENT_MFCC = np.hamming(N_FFT_MFCC)


def estimar_f_fire(psd, f_nom, tol=0.06, n_arm=N_ARM_PSD):
    """Busca la f_fire (en +-tol de la nominal) que mejor alinea un 'peine' de
    armónicos con el PSD. Devuelve (f_fire_estimada, prominencia_dB).
    La prominencia ~0 dB indica que no hay estructura armónica (motor apagado,
    mucho viento, etc.)."""
    f = np.arange(len(psd), dtype=float)
    lp = np.log10(psd + 1e-20)
    cands = np.arange(f_nom * (1 - tol), f_nom * (1 + tol) + 1e-9, 0.05)
    ks = np.arange(1, n_arm + 1)
    puntaje = np.interp(cands[:, None] * ks[None, :], f, lp).mean(axis=1)
    i = int(np.argmax(puntaje))
    base = np.median(lp[F_MIN:1001])
    return float(cands[i]), float(10.0 * (puntaje[i] - base))


def potencia_armonico(psd, f_c):
    """Potencia del armónico centrado en f_c (Hz): SUMA del PSD en una ventana de
    +-ANCHO_ARM*f_c (mínimo +-2 bins = lóbulo principal de la ventana de Hann).
    Si las rpm oscilan, el armónico se reparte en varios bins de 1 Hz: sumar la ventana
    recupera toda su potencia; tomar solo el bin máximo la subestima (más cuanto mayor k)."""
    c = int(round(f_c))
    w = max(2, int(round(ANCHO_ARM * f_c)))
    return float(psd[max(c - w, 0):c + w + 1].sum())


class AcumuladorRegistro:
    """Procesa un registro (varios bloques de 1 s) con memoria constante.

    uso:
        acc = AcumuladorRegistro(f_fire_nom=25.0)
        for bloque in bloques:          # np.ndarray float, len == 16000, +-1
            acc.agregar_bloque(bloque)
        fila = acc.resultado()          # dict de ~70 características
    """

    def __init__(self, f_fire_nom=F_FIRE_NOM):
        self.f_fire_nom = f_fire_nom
        self.n = 0
        self._zi_hp = self._zi_env = None
        self._prev = 0.0
        self._w = signal.windows.hann(FS, sym=False)
        self._psd = np.zeros(FS // 2 + 1)
        self._env2 = np.zeros(FS // 2 + 1)
        self._env_media = 0.0
        self._t = []                       # (rms_db, crest, curtosis, zcr, curtosis 1-6 kHz)
        self._f_bloque = []                # f_fire estimada en cada bloque de 1 s
        self._resto = np.zeros(0)          # cola de muestras para la próxima trama de MFCC
        self._clip = 0.0
        self._mfcc_s = np.zeros(N_MFCC)
        self._mfcc_q = np.zeros(N_MFCC)
        self._n_tramas = 0

    # ------------------------------------------------------------------
    def agregar_bloque(self, x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) != BLOQUE:
            raise ValueError(f"El bloque debe tener {BLOQUE} muestras")
        self._clip += float(np.mean(np.abs(x) > 0.99))

        # 1) pasa-altos 20 Hz (estado inicial = DC de la 1ª muestra)
        if self._zi_hp is None:
            self._zi_hp = signal.sosfilt_zi(_SOS_HP) * x[0]
            self._zi_env = np.zeros((_SOS_ENV.shape[0], 2))
        y, self._zi_hp = signal.sosfilt(_SOS_HP, x, zi=self._zi_hp)

        # 2) estadísticos temporales
        p2 = float(np.mean(y ** 2)) + 1e-24
        rms = np.sqrt(p2)
        t_bloque = [20 * np.log10(rms),
                    float(np.max(np.abs(y)) / rms),
                    float(np.mean(y ** 4) / p2 ** 2),
                    float(np.mean(y[:-1] * y[1:] < 0))]

        # 3) PSD (periodograma de 1 s, escala unilateral)
        X = rfft(y * self._w)
        p = (np.abs(X) ** 2) / (FS * np.sum(self._w ** 2))
        p[1:-1] *= 2.0
        self._psd += p
        self._f_bloque.append(estimar_f_fire(p, self.f_fire_nom)[0])

        # 4) envolvente de la banda 1-6 kHz (impactos / combustión)
        xb, self._zi_env = signal.sosfilt(_SOS_ENV, y, zi=self._zi_env)
        pb2 = float(np.mean(xb ** 2)) + 1e-24
        t_bloque.append(float(np.mean(xb ** 4) / pb2 ** 2))   # golpes: curtosis > 3
        self._t.append(tuple(t_bloque))
        env = np.abs(signal.hilbert(xb))
        mu = float(env.mean())
        A = 2.0 * np.abs(rfft((env - mu) * self._w)) / np.sum(self._w)
        self._env2 += A ** 2
        self._env_media += mu

        # 5) MFCC (tramas de 128 ms, salto 64 ms; la cola pasa al bloque siguiente)
        pe = np.empty_like(y)
        pe[0] = y[0] - PREENFASIS * self._prev
        pe[1:] = y[1:] - PREENFASIS * y[:-1]
        self._prev = y[-1]
        pe = np.concatenate((self._resto, pe))
        tramas = sliding_window_view(pe, N_FFT_MFCC)[::HOP_MFCC]
        self._resto = pe[len(tramas) * HOP_MFCC:].copy()
        P = (np.abs(rfft(tramas * _VENT_MFCC, axis=1)) ** 2) / N_FFT_MFCC
        logmel = np.log(P @ _FB_MEL.T + 1e-10)
        c = dct(logmel, type=2, norm="ortho", axis=1)[:, :N_MFCC]
        self._mfcc_s += c.sum(axis=0)
        self._mfcc_q += (c ** 2).sum(axis=0)
        self._n_tramas += c.shape[0]

        self.n += 1

    # ------------------------------------------------------------------
    def espectros(self):
        """Espectros promediados (para graficar)."""
        n = max(self.n, 1)
        psd = self._psd / n
        env = np.sqrt(self._env2 / n) / (self._env_media / n + 1e-12)
        f_fire, prom = estimar_f_fire(psd, self.f_fire_nom)
        return {"f": np.arange(len(psd), dtype=float), "psd": psd,
                "env": env, "f_fire": f_fire, "prominencia_db": prom}

    def resultado(self):
        if self.n == 0:
            raise RuntimeError("No se agregó ningún bloque")
        n = self.n
        sp = self.espectros()
        psd, env, f_fire = sp["psd"], sp["env"], sp["f_fire"]
        f = sp["f"]
        t = np.array(self._t)
        r = {}

        # calidad / nivel
        r["clip_frac"] = self._clip / n
        r["f_fire_hz"] = f_fire
        r["prominencia_peine_db"] = sp["prominencia_db"]

        # dominio del tiempo
        r["rms_dbfs"] = t[:, 0].mean()
        r["rms_dbfs_std"] = t[:, 0].std()
        r["crest"] = t[:, 1].mean()
        r["curtosis"] = t[:, 2].mean()
        r["zcr"] = t[:, 3].mean()
        r["curtosis_1k_6k"] = t[:, 4].mean()

        # inestabilidad de velocidad: cuánto cambia f_fire de un segundo a otro (en %)
        fb_ = np.array(self._f_bloque)
        r["vel_inestabilidad_pct"] = 100.0 * fb_.std() / fb_.mean()

        # energía por bandas, relativa a la total (invariante a la ganancia)
        p_tot = psd[F_MIN:F_MAX + 1].sum()
        for lo, hi in BANDAS_HZ:
            r[f"banda_{lo}_{hi}_db"] = 10 * np.log10(psd[lo:hi].sum() / p_tot + 1e-20)

        # descriptores espectrales
        fb, pb = f[F_MIN:F_MAX + 1], psd[F_MIN:F_MAX + 1]
        s = pb.sum()
        cen = (fb * pb).sum() / s
        r["centroide_hz"] = cen
        r["ancho_banda_hz"] = np.sqrt((((fb - cen) ** 2) * pb).sum() / s)
        r["planitud"] = np.exp(np.mean(np.log(pb + 1e-20))) / (pb.mean() + 1e-20)
        r["rolloff85_hz"] = fb[np.searchsorted(np.cumsum(pb) / s, 0.85)]
        pn = pb / s
        r["entropia"] = -(pn * np.log2(pn + 1e-20)).sum() / np.log2(len(pn))

        # armónicos de la frecuencia de encendido (k = 1..12) relativos a 20-2000 Hz
        # (potencia integrada en +-2 % alrededor de cada uno)
        p_ref = psd[F_MIN:2001].sum()
        h = np.empty(N_ARM_PSD)
        for k in range(1, N_ARM_PSD + 1):
            h[k - 1] = potencia_armonico(psd, k * f_fire)
            r[f"h{k:02d}_db"] = 10 * np.log10(h[k - 1] / p_ref + 1e-20)
        # k par = múltiplos del giro del eje (50 Hz); k impar = órdenes 0,5x, 1,5x... que solo
        # produce la combustión (un encendido cada 2 vueltas)
        r["ratio_impar_par_db"] = 10 * np.log10(h[0::2].sum() / (h[1::2].sum() + 1e-20) + 1e-20)

        # espectro de envolvente: profundidad de modulación en m*f_fire/2
        for m in range(1, N_ARM_ENV + 1):
            c = int(round(m * f_fire / 2))
            r[f"env{m:02d}_mod"] = env[max(c - 1, 0):c + 2].max()
        r["env_mod_total"] = float(np.sqrt(np.sum(env[3:201] ** 2) / 2))

        # MFCC c1..c12 (se descarta c0 por depender de la ganancia)
        mu = self._mfcc_s / self._n_tramas
        sd = np.sqrt(np.maximum(self._mfcc_q / self._n_tramas - mu ** 2, 0.0))
        for q in range(1, N_MFCC):
            r[f"mfcc{q:02d}_mean"] = mu[q]
        for q in range(1, N_MFCC):
            r[f"mfcc{q:02d}_std"] = sd[q]

        return {k: float(v) for k, v in r.items()}