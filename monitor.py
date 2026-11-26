"""
monitor.py - Diagnóstico en vivo con interfaz web. Corre en la Pi; la interfaz se abre
desde el navegador de la PC (o de un celular) conectado a la misma red.

  python monitor.py
  -> en la PC abrir:  http://gonzagz.local:8000

El nodo escucha continuamente (NO guarda audio), diagnostica cada 10 s y la página marca
el estado detectado: verde si es normal, rojo con aviso si es una falla, gris si está
apagado. Muestra además la hora del diagnóstico, la velocidad, el nivel y una línea de tiempo
con los últimos 24 diagnósticos. La tecla V oculta los controles de validación (para filmar).

Validación en vivo: en la página se elige el "estado real" del generador (los botones de
abajo). Cada diagnóstico se guarda en el CSV de --log junto con ese estado real, con el
mismo formato que diagnostico.py, así que después:
  python datos.py vivo --log vivo_monitor.csv
Al cambiar el estado del generador conviene tocar "Pausar", inducir el estado, esperar a que
se estabilice, elegir el estado real y "Reanudar": así ningún diagnóstico mezcla dos estados.

Opciones:
  python monitor.py --puerto 8000 --log vivo_monitor.csv --modelo modelo.npz
  python monitor.py --simular datos/caracteristicas_captura.csv   # sin micrófono (prueba)

No se puede usar al mismo tiempo que captura.py o diagnostico.py (el micrófono es uno solo).
"""
import argparse
import csv
import json
import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import datos
import modelo

COLUMNAS_LOG = ["timestamp", "estado", "confianza", "f_fire_hz", "rms_dbfs", "peine_db",
                "perdidos", "real"]
NOMBRES = {"apagado": "Apagado", "normal": "Normal", "combustible": "Combustible",
           "aire": "Aire"}
FALLAS = {"combustible": "Falla de combustible", "aire": "Falla de aire"}


def diagnosticar(m, feat, min_peine, max_clip):
    """Misma lógica que diagnostico.py (copiada para no depender del micrófono)."""
    if feat["clip_frac"] > max_clip:
        return "INVÁLIDO", 0.0, "micrófono saturado"
    if feat["prominencia_peine_db"] < min_peine and "apagado" not in m["clases"]:
        return "INVÁLIDO", 0.0, "sin armónicos del motor"
    x = np.array([float(feat[c]) for c in m["columnas"]])
    r = modelo.predecir(m, x)
    k = int(np.argmax(r["prob"]))
    if r["rechazo"]:
        return "DESCONOCIDO", float(r["prob"][k]), f"más parecido a '{r['clase']}'"
    return r["clase"], float(r["prob"][k]), ""


class Monitor:
    def __init__(self, a):
        self.a = a
        self.m = modelo.cargar(a.modelo)
        self.lock = threading.Lock()
        self.pausado = False
        self.real = ""
        self.ultimo = None
        self.historial = deque(maxlen=24)
        self.inicio_captura = None
        self.inicio_falla = None
        self.error = ""
        if a.simular:
            filas = modelo.leer_csv(a.simular)
            self.sim = [f for f in filas if all(c in f for c in self.m["columnas"])]
            if not self.sim:
                raise SystemExit("El CSV de --simular no tiene las columnas del modelo")
        else:
            from captura import DISPOSITIVO, dispositivo_arg, grabar_registro  # usa el micrófono
            self._grabar = grabar_registro
            self._disp = dispositivo_arg(a.dispositivo or DISPOSITIVO)

    # ------------------------------------------------------------------ captura
    def registro(self):
        if self.a.simular:
            time.sleep(self.a.duracion)
            preferidas = [f for f in self.sim if f.get("estado") == self.real] or self.sim
            f = random.choice(preferidas)
            feat = {k: (float(v) if k not in ("estado", "sesion", "archivo", "timestamp", "ruido")
                        and v not in ("", None) else v) for k, v in f.items()}
            feat.setdefault("clip_frac", 0.0)
            return feat, 0
        return self._grabar(self.a.duracion, self.a.canal, self._disp, self.a.rpm / 120.0, None)

    def bucle(self):
        while True:
            with self.lock:
                pausado = self.pausado
            if pausado:
                with self.lock:
                    self.inicio_captura = None
                time.sleep(0.3)
                continue
            with self.lock:
                self.inicio_captura = time.time()
                real = self.real
            try:
                feat, perdidos = self.registro()
            except Exception as e:                       # noqa: BLE001
                with self.lock:
                    self.error = f"Error de captura: {e}"
                time.sleep(2)
                continue
            with self.lock:
                # se pausó o se cambió el estado real durante la captura: se descarta,
                # porque ese registro puede mezclar dos estados
                if self.pausado or self.real != real:
                    continue
            estado, prob, nota = diagnosticar(self.m, feat, self.a.min_peine, self.a.max_clip)
            ahora = datetime.now()
            d = {"estado": estado, "confianza": prob, "nota": nota,
                 "hora": ahora.strftime("%H:%M:%S"), "fecha": ahora.strftime("%d/%m/%Y"),
                 "f_fire_hz": float(feat.get("f_fire_hz", float("nan"))),
                 "rms_dbfs": float(feat.get("rms_dbfs", float("nan"))), "real": real}
            with self.lock:
                if estado in FALLAS:
                    if not (self.ultimo and self.ultimo["estado"] == estado and self.inicio_falla):
                        self.inicio_falla = d["fecha"] + " " + d["hora"]
                else:
                    self.inicio_falla = None
                d["desde"] = self.inicio_falla
                self.ultimo = d
                self.historial.appendleft(d)
                self.error = ""
            self.guardar(ahora, estado, prob, feat, perdidos, real)

    def guardar(self, ahora, estado, prob, feat, perdidos, real):
        nuevo = not os.path.exists(self.a.log)
        with open(self.a.log, "a", newline="") as fh:
            w = csv.writer(fh)
            if nuevo:
                w.writerow(COLUMNAS_LOG)
            w.writerow([ahora.strftime("%Y-%m-%d %H:%M:%S"), estado, f"{prob:.3f}",
                        f"{float(feat.get('f_fire_hz', 0)):.2f}", f"{float(feat.get('rms_dbfs', 0)):.1f}",
                        f"{float(feat.get('prominencia_peine_db', 0)):.1f}", perdidos, real])

    def foto(self):
        with self.lock:
            prog = None
            if self.inicio_captura:
                prog = min(1.0, (time.time() - self.inicio_captura) / self.a.duracion)
            return {"ultimo": self.ultimo, "historial": list(self.historial),
                    "pausado": self.pausado, "real": self.real, "progreso": prog,
                    "error": self.error, "estados": [e for e in datos.ESTADOS],
                    "simulado": bool(self.a.simular), "log": self.a.log}


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitor acústico del grupo electrógeno</title>
<style>
:root{
  --bg:#0b0f14;--panel:#121821;--panel2:#18202b;--linea:#243041;
  --txt:#e8edf3;--txt2:#9aa7b6;--txt3:#5f6c7b;
  --normal:#22c55e;--falla:#ef4444;--apagado:#94a3b8;--otro:#f59e0b;--acento:#38bdf8;
  --c:var(--txt3);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--txt);font-family:"Inter","Segoe UI",system-ui,Roboto,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
main{max-width:1500px;margin:0 auto;padding:28px 40px 24px;display:flex;flex-direction:column;gap:22px;min-height:100%}
header{display:flex;justify-content:space-between;align-items:flex-start}
.titulo{font-size:22px;font-weight:650;letter-spacing:.01em}
.subtitulo{font-size:15px;color:var(--txt2);margin-top:4px}
.vivo{display:flex;align-items:center;gap:14px;font-variant-numeric:tabular-nums}
.chip{display:flex;align-items:center;gap:8px;padding:7px 14px;border-radius:999px;background:var(--panel2);
  border:1px solid var(--linea);font-size:13px;font-weight:650;letter-spacing:.12em}
.punto{width:9px;height:9px;border-radius:50%;background:var(--falla);box-shadow:0 0 0 0 rgba(239,68,68,.7);
  animation:latido 1.6s infinite}
.pausa .punto{background:var(--txt3);animation:none}
@keyframes latido{0%{box-shadow:0 0 0 0 rgba(239,68,68,.6)}70%{box-shadow:0 0 0 10px rgba(239,68,68,0)}100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}}
.reloj{font-size:26px;font-weight:600;color:var(--txt2)}

/* panel principal */
.heroe{position:relative;overflow:hidden;border-radius:20px;background:var(--panel);border:1px solid var(--linea);
  padding:38px 44px;display:flex;align-items:center;gap:36px;min-height:230px;transition:border-color .4s,background .4s}
.heroe::before{content:"";position:absolute;inset:0;background:radial-gradient(120% 140% at 0% 50%,var(--c),transparent 55%);
  opacity:.16;transition:opacity .4s}
.icono{position:relative;flex:none;width:136px;height:136px;border-radius:50%;display:grid;place-items:center;
  border:5px solid var(--c);color:var(--c);font-size:72px;font-weight:700;line-height:1;transition:all .4s}
.heroe.falla .icono{animation:alerta 1.2s ease-in-out infinite}
@keyframes alerta{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.45)}50%{box-shadow:0 0 0 18px rgba(239,68,68,0)}}
.textos{position:relative;flex:1;min-width:0}
.etiqueta{font-size:15px;font-weight:650;letter-spacing:.14em;color:var(--txt2);text-transform:uppercase}
.estado{font-size:64px;font-weight:750;line-height:1.05;margin-top:8px;color:var(--c);transition:color .4s}
.detalle{font-size:19px;color:var(--txt2);margin-top:12px;font-variant-numeric:tabular-nums}
.analisis{position:absolute;left:0;right:0;bottom:0;height:5px;background:rgba(255,255,255,.05)}
.analisis>div{height:100%;width:0;background:var(--acento);transition:width .9s linear}

/* estados */
.estados{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
.tile{border-radius:16px;background:var(--panel);border:2px solid var(--linea);padding:26px 22px;display:flex;
  align-items:center;gap:14px;font-size:24px;font-weight:650;color:var(--txt3);transition:all .35s}
.tile .marca{width:14px;height:14px;border-radius:50%;background:var(--linea);transition:all .35s}
.tile.on{color:var(--txt);border-color:var(--c);background:linear-gradient(180deg,rgba(255,255,255,.03),transparent),var(--panel2);
  box-shadow:0 0 0 1px var(--c),0 10px 40px -12px var(--c)}
.tile.on .marca{background:var(--c);box-shadow:0 0 14px var(--c)}

/* métricas y línea de tiempo */
.fila{display:grid;grid-template-columns:1fr 1fr 1fr 2.2fr;gap:18px}
.metrica,.tiempo{border-radius:16px;background:var(--panel);border:1px solid var(--linea);padding:18px 22px}
.metrica .n{font-size:13px;font-weight:650;letter-spacing:.12em;color:var(--txt3);text-transform:uppercase}
.metrica .v{font-size:34px;font-weight:700;margin-top:8px;font-variant-numeric:tabular-nums}
.metrica .u{font-size:16px;color:var(--txt2);font-weight:500;margin-left:6px}
.barras{display:flex;gap:5px;height:44px;margin-top:12px;align-items:stretch}
.barras div{flex:1;border-radius:5px;background:var(--linea);position:relative}
.barras div.x::after{content:"";position:absolute;left:50%;top:-9px;width:6px;height:6px;margin-left:-3px;border-radius:50%;background:#fff}
.ejes{display:flex;justify-content:space-between;font-size:12px;color:var(--txt3);margin-top:6px;font-variant-numeric:tabular-nums}

/* controles de validación */
.controles{margin-top:auto;border-radius:14px;background:var(--panel);border:1px dashed var(--linea);padding:14px 18px;
  display:flex;flex-wrap:wrap;align-items:center;gap:10px;font-size:14px;color:var(--txt2)}
.controles.oculto{display:none}
.controles b{color:var(--txt);font-weight:600;margin-right:4px}
button{font:inherit;font-size:14px;padding:8px 14px;border-radius:9px;border:1px solid var(--linea);background:var(--panel2);
  color:var(--txt);cursor:pointer}
button:hover{border-color:var(--txt3)}
button.sel{background:var(--txt);color:var(--bg);border-color:var(--txt)}
button.pausar{border-color:var(--acento);color:var(--acento)}
.sep{flex:1}
.ayuda{font-size:12px;color:var(--txt3);text-align:right}
@media (max-width:1000px){main{padding:20px}.estado{font-size:42px}.icono{width:96px;height:96px;font-size:50px}
  .estados{grid-template-columns:repeat(2,1fr)}.fila{grid-template-columns:1fr 1fr}.tiempo{grid-column:1/-1}}
</style></head><body><main>
<header>
  <div><div class="titulo">Nodo de monitoreo acústico</div>
       <div class="subtitulo">Grupo electrógeno Sincrolamp LT6500S · micrófono MEMS ICS-43434 · Raspberry Pi Zero 2 W</div></div>
  <div class="vivo"><div class="chip" id="chip"><span class="punto"></span><span id="chiptxt">EN VIVO</span></div>
       <div class="reloj" id="reloj">--:--:--</div></div>
</header>

<section class="heroe" id="heroe">
  <div class="icono" id="icono">…</div>
  <div class="textos">
    <div class="etiqueta">Estado detectado</div>
    <div class="estado" id="estado">Esperando diagnóstico</div>
    <div class="detalle" id="detalle">El primer resultado aparece a los 10 s</div>
  </div>
  <div class="analisis"><div id="prog"></div></div>
</section>

<section class="estados" id="tiles"></section>

<section class="fila">
  <div class="metrica"><div class="n">Velocidad</div><div class="v"><span id="rpm">–</span><span class="u">rpm</span></div></div>
  <div class="metrica"><div class="n">Frecuencia de encendido</div><div class="v"><span id="fenc">–</span><span class="u">Hz</span></div></div>
  <div class="metrica"><div class="n">Nivel de sonido</div><div class="v"><span id="nivel">–</span><span class="u">dBFS</span></div></div>
  <div class="tiempo"><div class="metrica" style="padding:0;border:0;background:none"><div class="n">Últimos diagnósticos</div></div>
    <div class="barras" id="barras"></div><div class="ejes"><span id="t0"></span><span id="t1"></span></div></div>
</section>

<section class="controles" id="controles">
  <b>Validación · estado real:</b><span id="reales"></span>
  <button class="pausar" id="bpausa" onclick="pausa()">Pausar</button>
  <span class="sep"></span>
  <span id="cuenta"></span>
  <div class="ayuda">Tecla <b>V</b>: ocultar o mostrar estos controles</div>
</section>
</main>
<script>
const NOM={apagado:"Apagado",normal:"Normal",combustible:"Combustible",aire:"Aire"};
const TXT={apagado:"Motor apagado",normal:"Funcionamiento normal",combustible:"Falla de combustible",
  aire:"Falla de aire",DESCONOCIDO:"Estado desconocido","INVÁLIDO":"Diagnóstico inválido"};
const COL={normal:"var(--normal)",combustible:"var(--falla)",aire:"var(--falla)",apagado:"var(--apagado)"};
const ICO={normal:"✓",combustible:"!",aire:"!",apagado:"–"};
let estados=[],N=24;
const $=id=>document.getElementById(id);
function color(e){return COL[e]||"var(--otro)"}
async function post(u){await fetch(u,{method:"POST"});actualizar()}
function pausa(){post("/pausa")}
function real(e){post("/real?e="+encodeURIComponent(e))}
document.addEventListener("keydown",ev=>{if(ev.key==="v"||ev.key==="V")$("controles").classList.toggle("oculto")});
setInterval(()=>{$("reloj").textContent=new Date().toLocaleTimeString("es-AR",{hour12:false})},250);
async function actualizar(){
  let r;try{r=await(await fetch("/estado")).json()}catch(e){$("chiptxt").textContent="SIN CONEXIÓN";return}
  if(!estados.length){estados=r.estados;
    $("tiles").innerHTML=estados.map(e=>`<div class="tile" id="t_${e}"><span class="marca"></span>${NOM[e]||e}</div>`).join("");
    $("reales").innerHTML=[...estados,""].map(e=>`<button id="r_${e}" onclick="real('${e}')">${NOM[e]||"Sin indicar"}</button>`).join(" ");}
  $("chip").className="chip"+(r.pausado?" pausa":"");$("chiptxt").textContent=r.pausado?"EN PAUSA":(r.simulado?"SIMULACIÓN":"EN VIVO");
  const u=r.ultimo,h=$("heroe");
  if(u){const c=color(u.estado),falla=(u.estado==="combustible"||u.estado==="aire");
    document.documentElement.style.setProperty("--c",c);
    h.className="heroe"+(falla?" falla":"");
    $("icono").textContent=ICO[u.estado]||"?";
    $("estado").textContent=TXT[u.estado]||u.estado;
    $("detalle").textContent=(falla?"¡Atención! · ":"")+"Detectado a las "+u.hora+" del "+u.fecha+(u.desde&&falla?" · presente desde las "+u.desde.split(" ")[1]:"")+(u.nota&&!falla?" · "+u.nota:"");
    estados.forEach(e=>$("t_"+e).className="tile"+(u.estado===e?" on":""));
    const f=u.f_fire_hz,ok=isFinite(f)&&u.estado!=="apagado";
    $("rpm").textContent=ok?(f*120).toFixed(0):"–";$("fenc").textContent=ok?f.toFixed(2):"–";
    $("nivel").textContent=isFinite(u.rms_dbfs)?u.rms_dbfs.toFixed(1).replace("-","−"):"–";}
  $("prog").style.width=(r.progreso==null?0:r.progreso*100)+"%";
  const hs=r.historial.slice(0,N).reverse(),vac=N-hs.length;
  $("barras").innerHTML="<div></div>".repeat(vac)+hs.map(x=>`<div class="${x.real&&x.real!==x.estado?"x":""}" title="${x.hora} · ${TXT[x.estado]||x.estado}" style="background:${color(x.estado)}"></div>`).join("");
  $("t0").textContent=hs.length?hs[0].hora:"";$("t1").textContent=hs.length?hs[hs.length-1].hora:"";
  [...estados,""].forEach(e=>{const b=$("r_"+e);if(b)b.className=(r.real===e?"sel":"")});
  $("bpausa").textContent=r.pausado?"Reanudar":"Pausar";
  const con=r.historial.filter(x=>x.real);const ac=con.filter(x=>x.real===x.estado).length;
  $("cuenta").textContent=con.length?`Aciertos recientes: ${ac} de ${con.length} · registro: ${r.log}`:`Registro: ${r.log}`;
}
actualizar();setInterval(actualizar,1000);
</script></body></html>"""


def servidor(mon):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _enviar(self, cuerpo, tipo):
            b = cuerpo.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            ruta = urlparse(self.path).path
            if ruta == "/estado":
                self._enviar(json.dumps(mon.foto()), "application/json")
            elif ruta == "/":
                self._enviar(PAGINA, "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            with mon.lock:
                if u.path == "/pausa":
                    mon.pausado = not mon.pausado
                elif u.path == "/real":
                    e = parse_qs(u.query).get("e", [""])[0]
                    mon.real = e if e in datos.ESTADOS else ""
            self._enviar("{}", "application/json")

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="modelo.npz")
    ap.add_argument("--puerto", type=int, default=8000)
    ap.add_argument("--log", default="vivo_monitor.csv")
    ap.add_argument("--duracion", type=int, default=10, help="segundos por diagnóstico")
    ap.add_argument("--rpm", type=float, default=3000.0)
    ap.add_argument("--canal", type=int, default=0, choices=[0, 1])
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--min-peine", type=float, default=4.0)
    ap.add_argument("--max-clip", type=float, default=0.01)
    ap.add_argument("--simular", default=None, help="CSV de características para probar sin micrófono")
    a = ap.parse_args()

    mon = Monitor(a)
    threading.Thread(target=mon.bucle, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.puerto), servidor(mon))
    print(f"Monitor en marcha. En la PC abrí:  http://gonzagz.local:{a.puerto}")
    print(f"Diagnósticos guardados en {a.log}.  Ctrl+C para salir.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nFin.")


if __name__ == "__main__":
    main()