import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pydeck as pdk
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET
from math import radians, cos, sin, asin, sqrt
import pathlib, zipfile, io, re, random

st.set_page_config(page_title="Peores Rutas — CDMX", page_icon="🚇", layout="wide")

# ===== UI =====
STYLE = """
<style>
:root{
  --bg:#0a0f1c; --panel:#0f1630; --ink:#e5ecff; --muted:#9fb2ff; --line:#22305b; --chip:#101a3a;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg); color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1228 0%, #091024 100%)}
.block-title{font-weight:900;font-size:1.9rem;margin:.1rem 0 .4rem}
.subtle{color:var(--muted);font-size:.95rem}
.card{border:1px solid var(--line); background:var(--panel); border-radius:18px; padding:16px}
.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.kpi>div{border:1px solid var(--line);background:#0e1736;border-radius:14px;padding:12px}
.badge{display:inline-flex;gap:.45rem;align-items:center;padding:.25rem .6rem;border-radius:999px;
  border:1px solid var(--line);background:var(--chip);font-size:.78rem}
hr{border:none;border-top:1px solid var(--line);margin:1rem 0}
a{color:#7dd3fc}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ===== Paths =====
DATA_DIR = pathlib.Path("data")
KML_PATH = DATA_DIR / "Movilidad Integrada ZMVM(1).kml"
CSV_METRO = DATA_DIR / "metro.csv"

# ===== Colors =====
LINE_COLORS = {
    # Metro
    "L1":"#F05A91","L2":"#0055A5","L3":"#8CC63E","L4":"#00B5E2","L5":"#FFC20E","L6":"#E10600",
    "L7":"#F39C12","L8":"#00A859","L9":"#8B5E3C","L12":"#C8B273","A":"#7F3FBF","B":"#7AC142",
    # Metrobus
    "MB1":"#C62828","MB2":"#1565C0","MB3":"#2E7D32","MB4":"#EF6C00","MB5":"#6A1B9A","MB6":"#00838F","MB7":"#795548",
    # Cablebus
    "CB1":"#00ACC1","CB2":"#26A69A",
    # Trolebus
    "TB1":"#9C27B0","TB2":"#7B1FA2","TB3":"#8E24AA","TB4":"#6A1B9A",
    # Tren Ligero / otros
    "TL":"#00A651","SUB":"#455A64","MXB":"#9E9D24","RTP":"#90A4AE",
    "DEFAULT":"#8894c7"
}
SERVICE_COLOR = {"METRO":"#ED5480","METROBUS":"#E53935","CABLEBUS":"#26C6DA","TROLEBUS":"#B388FF","TREN LIGERO":"#00A651","SUBURBANO":"#455A64","MEXIBUS":"#C0CA33","RTP":"#90A4AE","OTROS":"#8894c7"}

# ===== Utils =====
def hav_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    lon1, lat1, lon2, lat2 = map(radians, [lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1; r = 6371.0
    c = 2 * asin(sqrt(sin(dlat/2)**2 + cos(lat1)*cos(lat2)*(sin(dlon/2)**2)))
    return r * c

def _norm(s:str) -> str:
    s = s.lower()
    s = s.replace("í","i").replace("á","a").replace("é","e").replace("ó","o").replace("ú","u").replace("ü","u")
    return s

def detect_service(text: str) -> str:
    t = _norm(text)
    if "metrobus" in t or "metrobús" in t: return "METROBUS"
    if "cablebus" in t or "cablebus" in t: return "CABLEBUS"
    if "trolebus" in t: return "TROLEBUS"
    if "tren ligero" in t or "trenligero" in t or "tl" == t.strip(): return "TREN LIGERO"
    if "suburbano" in t: return "SUBURBANO"
    if "mexibus" in t: return "MEXIBUS"
    if "rtp" in t: return "RTP"
    if "metro" in t or "linea" in t or "línea" in text: return "METRO"
    return "OTROS"

def canonical_line(service: str, name: str) -> str:
    t = _norm(name)
    if service=="METRO":
        if "linea a" in t or t.strip()=="a": return "A"
        if "linea b" in t or t.strip()=="b": return "B"
        m = re.search(r'(\d+)', t)
        if m and m.group(1) in {"1","2","3","4","5","6","7","8","9","12"}: return f"L{m.group(1)}"
    if service=="METROBUS":
        m = re.search(r'(\d+)', t); return f"MB{m.group(1)}" if m else "MB1"
    if service=="CABLEBUS":
        m = re.search(r'(\d+)', t); return f"CB{m.group(1)}" if m else "CB1"
    if service=="TROLEBUS":
        m = re.search(r'(\d+)', t); return f"TB{m.group(1)}" if m else "TB1"
    if service=="TREN LIGERO": return "TL"
    if service=="SUBURBANO": return "SUB"
    if service=="MEXIBUS": return "MXB"
    if service=="RTP": return "RTP"
    return None

# ===== DEMO (fallback) =====
LINES_DEMO = {
    "L1":["Observatorio","Tacubaya","Juanacatlán","Chapultepec","Sevilla","Insurgentes","Cuauhtémoc","Balderas",
          "Salto del Agua","Isabel la Católica","Pino Suárez","Merced","Candelaria","San Lázaro","Moctezuma",
          "Balbuena","Blvd. Puerto Aéreo","Gómez Farías","Zaragoza","Pantitlán"],
    "L3":["Indios Verdes","Deportivo 18 de Marzo","Potrero","La Raza","Tlatelolco","Guerrero","Hidalgo","Juárez",
          "Balderas","Niños Héroes","Hospital General","Centro Médico","Etiopía/Plaza de la Transparencia","Eugenia",
          "División del Norte","Zapata","Coyoacán","Viveros/DH","Miguel Ángel de Quevedo","Copilco","Universidad"]
}
def build_graph_demo():
    G = nx.Graph()
    for ln, seq in LINES_DEMO.items():
        for u,v in zip(seq[:-1], seq[1:]): G.add_edge(u,v,length_m=1000.0,lines=set([ln]),service="METRO")
    for n in G.nodes(): G.nodes[n]["is_transfer"] = (G.degree[n]>=3)
    return G, {}

# ===== KML parsing (hierarchy-aware) =====
KML_NS = {"kml":"http://www.opengis.net/kml/2.2"}

@st.cache_data(show_spinner=False)
def read_kml_text(path: pathlib.Path) -> str:
    data = path.read_bytes()
    if data[:2]==b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name=next((n for n in zf.namelist() if n.lower().endswith(".kml")),None)
            if not name: return ""
            return zf.read(name).decode("utf-8","ignore")
    return data.decode("utf-8","ignore")

def _walk(el, trail, out_pm):
    name_el = el.find("kml:name", KML_NS)
    label = name_el.text.strip() if name_el is not None and name_el.text else None
    t2 = trail + ([label] if label else [])
    for pm in el.findall("kml:Placemark", KML_NS):
        out_pm.append((t2, pm))
    for sub in el.findall("kml:Folder", KML_NS):
        _walk(sub, t2, out_pm)

def parse_kml_with_services(kml_text: str):
    if not kml_text: return [], []
    try:
        root = ET.fromstring(kml_text)
    except Exception:
        return [], []
    doc = root.find(".//kml:Document", KML_NS) or root
    items = []
    _walk(doc, [], items)
    stations, lines = [], []
    for trail, pm in items:
        name_el = pm.find("kml:name", KML_NS)
        pm_name = name_el.text.strip() if name_el is not None and name_el.text else "Placemark"
        service = detect_service(" / ".join([t for t in trail if t] + [pm_name]))
        pt = pm.find("kml:Point/kml:coordinates", KML_NS)
        ls = pm.find("kml:LineString/kml:coordinates", KML_NS)
        if pt is not None:
            parts = pt.text.strip().split(",")
            if len(parts)>=2:
                lon, lat = float(parts[0]), float(parts[1])
                stations.append({"name": pm_name, "lat": lat, "lon": lon, "service": service})
        elif ls is not None:
            coords=[]
            for chunk in ls.text.replace("\n"," ").split():
                parts=chunk.split(",")
                if len(parts)>=2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coords.append((lat,lon))
            if coords:
                lines.append({"name": pm_name, "coords": coords, "service": service})
    return stations, lines

def snap_lines_to_stations(stations: List[Dict], lines: List[Dict], snap_threshold_m=150.0):
    if not stations or not lines: return []
    names=[s["name"] for s in stations]; coords=[(s["lat"],s["lon"]) for s in stations]
    def nearest(lat,lon):
        best_i,best_d=None,1e18
        for i,c in enumerate(coords):
            d=hav_km((lat,lon), c)
            if d<best_d: best_i,best_d=i,d
        return names[best_i], best_d*1000.0
    edges=[]
    for L in lines:
        seq=[]
        for (lat,lon) in L["coords"]:
            n,d=nearest(lat,lon)
            if d<=snap_threshold_m and (not seq or seq[-1]!=n): seq.append(n)
        for u,v in zip(seq[:-1], seq[1:]):
            if u!=v:
                dist_m = hav_km((stations[names.index(u)]["lat"], stations[names.index(u)]["lon"]),
                                (stations[names.index(v)]["lat"], stations[names.index(v)]["lon"])) * 1000.0
                code = canonical_line(L["service"], L["name"])
                edges.append((u, v, dist_m, code, L["service"]))
    return edges

def build_graph_from_edges(stations: List[Dict], edges):
    G = nx.Graph()
    used = set([u for u,_,_,_,_ in edges] + [v for _,v,_,_,_ in edges])
    for s in stations:
        if s["name"] in used:
            G.add_node(s["name"], lat=s["lat"], lon=s["lon"])
    for u,v,w,code,svc in edges:
        if not G.has_node(u) or not G.has_node(v): continue
        if G.has_edge(u,v):
            G[u][v]["length_m"] = min(G[u][v]["length_m"], w)
            G[u][v].setdefault("lines", set()).add(code) if code else None
            if "service" not in G[u][v]: G[u][v]["service"] = svc
        else:
            G.add_edge(u, v, length_m=float(w), lines=set([code]) if code else set(), service=svc)
    for n in G.nodes(): G.nodes[n]["is_transfer"] = (G.degree[n] >= 3)
    return G

# ===== CSV metro (distancias oficiales) =====
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols=[]; 
    for c in df.columns:
        x=str(c).strip().lower()
        x=re.sub(r'[áàä]','a',x); x=re.sub(r'[éèë]','e',x); x=re.sub(r'[íìï]','i',x)
        x=re.sub(r'[óòö]','o',x); x=re.sub(r'[úùü]','u',x); x=x.replace("ñ","n")
        x=re.sub(r'[^a-z0-9_]+','_',x).strip('_'); cols.append(x)
    df.columns=cols; return df

def merge_lengths_from_csv(G: nx.Graph, csv_path: pathlib.Path) -> int:
    if not csv_path.exists(): return 0
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = _normalize_cols(df)
    len_col = next((c for c in df.columns if "long" in c), None)
    def to_m(x): 
        s=str(x).replace(",","").replace(" m","").strip()
        try: return float(s)
        except: return np.nan
    df["__m__"] = df[len_col].apply(to_m) if len_col else np.nan
    cnt = 0
    for _,r in df.dropna(subset=["__m__"]).iterrows():
        u, v = str(r["origen"]).strip(), str(r["destino"]).strip()
        if G.has_edge(u,v):
            G[u][v]["length_m"] = float(r["__m__"]); cnt += 1
    return cnt

# ===== Build graph (with service filter) =====
@st.cache_data(show_spinner=True)
def build_graph_filtered(services_selected: List[str], snap_threshold_m: float = 150.0):
    if KML_PATH.exists():
        txt = read_kml_text(KML_PATH)
        stns_all, lines_all = parse_kml_with_services(txt)
        stns = [s for s in stns_all if s["service"] in services_selected]
        lins = [l for l in lines_all if l["service"] in services_selected]
        edges = snap_lines_to_stations(stns, lins, snap_threshold_m)
        G = build_graph_from_edges(stns, edges)
        if "METRO" in services_selected and CSV_METRO.exists():
            merge_lengths_from_csv(G, CSV_METRO)
        coords = {n:(G.nodes[n].get("lat"), G.nodes[n].get("lon")) for n in G.nodes()}
        return G, coords, len(stns), len(edges)
    # fallback demo
    G, coords = build_graph_demo()
    return G, coords, G.number_of_nodes(), G.number_of_edges()

# ===== Metrics & routing =====
def compute_metrics(G: nx.Graph, path: List[str], base_min=2.0, transfer_penalty=5.0, speed_kmh=32.0):
    e = list(zip(path[:-1], path[1:])); dist=0.0; seq=[]
    for u,v in e:
        w=G[u][v].get("length_m", np.nan); dist += 0 if np.isnan(w) else w
        lnset=G[u][v].get("lines"); seq.append(sorted(lnset)[0] if isinstance(lnset,set) and lnset else None)
    if any(x is not None for x in seq):
        t=0; prev=seq[0]
        for cur in seq[1:]:
            if cur!=prev: t+=1; prev=cur
        transfers=t
    else:
        transfers=sum(1 for n in path[1:-1] if G.nodes[n].get("is_transfer",False))
    t_dist=(dist/1000.0)/speed_kmh*60.0 if dist>0 else np.nan
    t_fb=base_min*len(e)+transfer_penalty*transfers
    t = t_dist if not np.isnan(t_dist) else t_fb
    return {"edges":len(e),"transfers":transfers,"dist_km":round(dist/1000.0,2) if dist>0 else None,"time_min":round(t,1),"lines_seq":seq}

def enumerate_paths_worst(G: nx.Graph, src: str, dst: str, mode: str, cutoff: int, limit_paths: int, base_min: float, transfer_penalty: float, speed_kmh: float):
    allp=[]
    for p in nx.all_simple_paths(G, source=src, target=dst, cutoff=cutoff):
        allp.append(p)
        if len(allp)>=limit_paths: break
    if not allp: return [], {}, [], pd.DataFrame()
    scored=[]
    for p in allp:
        m=compute_metrics(G,p,base_min,transfer_penalty,speed_kmh)
        s = m["time_min"] if mode=="Más tiempo" else m["transfers"] if mode=="Más transbordos" else m["edges"] if mode=="Más estaciones" else (m["dist_km"] or 0)
        scored.append((s,p,m))
    scored.sort(key=lambda x:x[0], reverse=True)
    top = scored[:min(10, len(scored))]
    df = pd.DataFrame([{"rank":i+1,"score":s,"tiempo_min":m["time_min"],"transbordos":m["transfers"],"estaciones":m["edges"],"dist_km":m["dist_km"],"ruta":" → ".join(p)} for i,(s,p,m) in enumerate(top)])
    return scored[0][1], scored[0][2], [p for _,p,_ in scored[:5]], df

# ===== Layout & plots (no SciPy) =====
def get_layout(G: nx.Graph, method: str):
    try:
        if method=="Kamada-Kawai":
            return nx.kamada_kawai_layout(G)
        if method=="Circular":
            return nx.circular_layout(G)
        return nx.random_layout(G, seed=42)
    except Exception:
        return nx.random_layout(G, seed=42)

def plot_topological(G: nx.Graph, highlight: List[str], hide_lines: List[str], layout_method: str):
    pos = get_layout(G, layout_method)
    groups={}
    for u,v,d in G.edges(data=True):
        ln = (sorted(d.get("lines"))[0] if d.get("lines") else None)
        svc = d.get("service","OTROS")
        key = ln or svc
        groups.setdefault(key, []).append((u,v,ln,svc))
    fig = go.Figure()
    for key, ev in groups.items():
        if hide_lines and key in hide_lines: continue
        color = LINE_COLORS.get(key, SERVICE_COLOR.get(ev[0][3], LINE_COLORS["DEFAULT"]))
        xs,ys=[],[]
        for (u,v,_,_) in ev:
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
        fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(width=3,color=color),name=key,hoverinfo="none"))
    node_x,node_y,texts=[],[],[]
    for n in G.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1]); texts.append(n)
    fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",
        marker=dict(size=10, line=dict(width=1,color="#0d1329")),
        text=[t if (not highlight or t in highlight) else "" for t in texts],
        textposition="top center", hovertext=texts, hoverinfo="text"))
    if highlight and len(highlight)>1:
        hx,hy=[],[]
        for u,v in zip(highlight[:-1], highlight[1:]):
            hx += [pos[u][0], pos[v][0], None]
            hy += [pos[u][1], pos[v][1], None]
        fig.add_trace(go.Scatter(x=hx,y=hy,mode="lines",line=dict(width=7,dash="dot",color="#ffffff"),name="Ruta"))
    fig.update_layout(showlegend=True,height=640,margin=dict(l=10,r=10,t=10,b=10),
                      xaxis=dict(visible=False),yaxis=dict(visible=False),
                      paper_bgcolor="#0a0f1c",plot_bgcolor="#0a0f1c")
    st.plotly_chart(fig, use_container_width=True, theme=None)

def plot_geo(path: List[str], coords: Dict[str,Tuple[float,float]], step:int=None):
    if not path or not coords: st.info("Carga KML para mapa geográfico."); return
    if not all(s in coords and coords[s][0] and coords[s][1] for s in path): st.info("Faltan coords para alguna estación."); return
    pts=[{"name":s,"lat":coords[s][0],"lon":coords[s][1]} for s in path[:(step or len(path))]]
    segs=[]; 
    for u,v in zip(path[:(step or len(path))][:-1], path[:(step or len(path))][1:]):
        a,b=coords[u],coords[v]; segs.append({"path":[[a[1],a[0]],[b[1],b[0]]]})
    view=pdk.ViewState(latitude=pts[0]["lat"], longitude=pts[0]["lon"], zoom=11)
    layer_pts=pdk.Layer("ScatterplotLayer", data=pts, get_position=["lon","lat"], get_radius=70, pickable=True)
    layer_path=pdk.Layer("PathLayer", data=segs, get_path="path", get_width=5, get_color=[255,255,255])
    st.pydeck_chart(pdk.Deck(layers=[layer_path,layer_pts], initial_view_state=view, tooltip={"text":"{name}"}))

# ===== UI =====
st.markdown('<div class="block-title">Peores Rutas — CDMX</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Filtra por servicio (Metro, Metrobús, Cablebús, etc.) y calcula la <span class="badge">peor</span> ruta por tiempo, transbordos, estaciones o distancia.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🚦 Servicios incluidos")
    services_all = ["METRO","METROBUS","CABLEBUS","TROLEBUS","TREN LIGERO","SUBURBANO","MEXIBUS","RTP"]
    services_sel = st.multiselect("Selecciona servicios", services_all, default=["METRO"])
    snap_m = st.slider("Umbral de 'encaje' KML→estación (m)", 60, 300, 150, 10)
    st.markdown("### ⚙️ Parámetros")
    speed_kmh = st.slider("Velocidad (km/h)", 24, 45, 32)
    base_min = st.slider("Minutos por tramo (fallback)", 1.0, 5.0, 2.0, .5)
    penalty = st.slider("Penalización por transbordo (min)", 1.0, 12.0, 5.0, .5)
    cutoff = st.slider("Máx. estaciones por ruta", 10, 60, 28, 1)
    limitp = st.slider("Máx. rutas a evaluar", 200, 3000, 1200, 100)
    layout_method = st.selectbox("Layout del grafo (sin SciPy)", ["Kamada-Kawai","Circular","Random"])
    st.markdown("### 🎨 Líneas visibles")
    # Se llena después de construir el grafo

# Build graph con filtro
G, coords, n_stations, n_edges = build_graph_filtered(services_sel, snap_threshold_m=snap_m)

if G.number_of_nodes()==0:
    st.error("No se pudo construir la red. Verifica data/Movilidad Integrada ZMVM(1).kml y data/metro.csv")
    st.stop()

all_line_codes = sorted({ln for _,_,d in G.edges(data=True) for ln in (d.get("lines") or set()) if ln})
hide_lines = st.sidebar.multiselect("Ocultar líneas", all_line_codes, [])

stations_all = sorted(G.nodes())
c1,c2,c3,c4,c5 = st.columns([2,2,2,2,1])
with c1:
    src = st.selectbox("Origen", stations_all, index=0)
with c2:
    dst = st.selectbox("Destino", stations_all, index=min(1,len(stations_all)-1))
with c3:
    mode = st.radio("Criterio", ["Más tiempo","Más transbordos","Más estaciones","Más distancia"], horizontal=True)
with c4:
    anim = st.toggle("Animación", value=True)
with c5:
    if st.button("🎲 Random"):
        src = random.choice(stations_all); dst = random.choice([s for s in stations_all if s!=src])

st.markdown(
    f"<div class='kpi card'>"
    f"<div><div class='subtle'>Servicios</div><div class='block-title' style='font-size:1.2rem'>{', '.join(services_sel) or '—'}</div></div>"
    f"<div><div class='subtle'>Estaciones</div><div class='block-title' style='font-size:1.2rem'>{G.number_of_nodes()}</div></div>"
    f"<div><div class='subtle'>Tramos</div><div class='block-title' style='font-size:1.2rem'>{G.number_of_edges()}</div></div>"
    f"<div><div class='subtle'>KML</div><div class='block-title' style='font-size:1.2rem'>{n_stations} est · {n_edges} tramos</div></div>"
    f"</div>", unsafe_allow_html=True
)

go = st.button("🔎 Calcular peor ruta", type="primary", use_container_width=True)

if go:
    if src==dst:
        st.warning("Elige estaciones distintas.")
    else:
        path, met, top5, df_top = enumerate_paths_worst(G, src, dst, mode, int(cutoff), int(limitp), float(base_min), float(penalty), float(speed_kmh))
        if not path:
            st.error("No encontré rutas con los parámetros actuales.")
        else:
            a,b = st.columns([3,2])
            with a:
                st.markdown(f"<div class='badge'>Ruta ({mode})</div>", unsafe_allow_html=True)
                st.markdown("**" + " → ".join(path) + "**")
                st.markdown(
                    f"- ⏱️ **{met['time_min']} min**  \n"
                    f"- 🔁 **{met['transfers']}** transbordos  \n"
                    f"- 🚉 **{met['edges']}** estaciones  \n" +
                    (f"- 📏 **{met['dist_km']} km**" if met.get('dist_km') else "")
                )
            with b:
                chips=[]
                for ln in met["lines_seq"]:
                    if ln: chips.append(f"<span class='badge' style='border-color:#2a3a77'>{ln}</span>")
                st.markdown("**Líneas**")
                st.markdown((" ".join(chips)) if chips else "<span class='subtle'>sin etiquetas</span>", unsafe_allow_html=True)
                st.download_button("⬇️ Descargar ruta (CSV)",
                    pd.DataFrame({"paso":range(1,len(path)+1),"estacion":path}).to_csv(index=False).encode("utf-8"),
                    "ruta.csv","text/csv")

            tabs = st.tabs(["🕸️ Topológico","🗺️ Geográfico","🏆 Top 10 peores"])
            with tabs[0]:
                plot_topological(G, highlight=path, hide_lines=hide_lines, layout_method=layout_method)
            with tabs[1]:
                step = st.slider("Paso", 1, len(path), len(path)) if anim else len(path)
                plot_geo(path, coords, step=step)
            with tabs[2]:
                st.dataframe(df_top, use_container_width=True)
else:
    st.info("Elige origen/destino, ajusta parámetros y pulsa **Calcular peor ruta**.")
