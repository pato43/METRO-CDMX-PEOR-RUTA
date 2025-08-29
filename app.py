import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pydeck as pdk
from streamlit.components.v1 import html
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET
from math import radians, cos, sin, asin, sqrt
import zipfile, io, os, re, pathlib, random

# -------------------- Config & Styles --------------------
st.set_page_config(page_title="Peores Rutas — Metro CDMX", page_icon="🚇", layout="wide")

STYLE = """
<style>
:root{
  --bg:#0a0f1c; --panel:#0f1630; --ink:#e5ecff; --muted:#9fb2ff; --line:#22305b; --chip:#101a3a; --accent:#4F46E5;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg); color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1228 0%, #091024 100%); color:var(--ink)}
h1,h2,h3{letter-spacing:.2px}
.block-title{font-weight:900;font-size:1.9rem;margin:.1rem 0 .4rem}
.subtle{color:var(--muted);font-size:.95rem}
.card{border:1px solid var(--line); background:var(--panel); border-radius:18px; padding:16px}
.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.kpi>div{border:1px solid var(--line);background:#0e1736;border-radius:14px;padding:12px}
.badge{display:inline-flex;gap:.5rem;align-items:center;padding:.25rem .6rem;border-radius:999px;
  border:1px solid var(--line);background:var(--chip);font-size:.78rem}
hr{border:none;border-top:1px solid var(--line);margin:1rem 0}
a{color:#7dd3fc}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

LINE_COLORS = {
    "L1":"#F05A91","L2":"#0055A5","L3":"#8CC63E","L4":"#00B5E2","L5":"#FFC20E","L6":"#E10600",
    "L7":"#F39C12","L8":"#00A859","L9":"#8B5E3C","L12":"#C8B273","A":"#7F3FBF","B":"#7AC142","DEFAULT":"#8894c7"
}

DATA_DIR = pathlib.Path("data")
CSV_INTER = DATA_DIR / "metro.csv"

# -------------------- Utils --------------------
def hav_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    lon1, lat1, lon2, lat2 = map(radians, [lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1; r = 6371.0
    c = 2 * asin(sqrt(sin(dlat/2)**2 + cos(lat1)*cos(lat2)*(sin(dlon/2)**2)))
    return r * c

def canonical_line_code(name: str) -> str:
    if not name: return None
    t = name.lower()
    t = t.replace("línea","linea").replace("line ","linea ").replace("linea ","l").replace(" ","")
    for code in ["l1","l2","l3","l4","l5","l6","l7","l8","l9","l12","a","b"]:
        if t.startswith(code) or re.search(rf'\b{code}\b', t): return code.upper()
    m = re.search(r'(\d+)$', t)
    if m and m.group(1) in {"1","2","3","4","5","6","7","8","9","12"}: return f"L{m.group(1)}"
    if "dorada" in t: return "L12"
    if t.strip()=="a": return "A"
    if t.strip()=="b": return "B"
    return None

# -------------------- DEMO fallback --------------------
LINES_DEMO = {
    "L1":["Observatorio","Tacubaya","Juanacatlán","Chapultepec","Sevilla","Insurgentes","Cuauhtémoc","Balderas",
          "Salto del Agua","Isabel la Católica","Pino Suárez","Merced","Candelaria","San Lázaro","Moctezuma",
          "Balbuena","Blvd. Puerto Aéreo","Gómez Farías","Zaragoza","Pantitlán"],
    "L2":["Cuatro Caminos","Panteones","Tacuba","Cuitláhuac","Popotla","Colegio Militar","Normal","San Cosme",
          "Revolución","Hidalgo","Bellas Artes","Allende","Zócalo/Tenochtitlán","Pino Suárez","San Antonio Abad",
          "Chabacano","Viaducto","Xola","Villa de Cortés","Nativitas","Portales","Ermita","General Anaya","Tasqueña"],
    "L3":["Indios Verdes","Deportivo 18 de Marzo","Potrero","La Raza","Tlatelolco","Guerrero","Hidalgo","Juárez",
          "Balderas","Niños Héroes","Hospital General","Centro Médico","Etiopía/Plaza de la Transparencia","Eugenia",
          "División del Norte","Zapata","Coyoacán","Viveros/DH","Miguel Ángel de Quevedo","Copilco","Universidad"]
}
def build_graph_demo():
    G = nx.Graph()
    for ln, seq in LINES_DEMO.items():
        for u,v in zip(seq[:-1], seq[1:]): G.add_edge(u,v,length_m=1000.0,lines=set([ln]))
    for n in G.nodes(): nx.set_node_attributes(G,{n:{"is_transfer":(G.degree[n]>=3)}})
    return G, {}

# -------------------- CSV interestaciones --------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols=[]
    for c in df.columns:
        x=str(c).strip().lower()
        x=re.sub(r'[áàä]','a',x); x=re.sub(r'[éèë]','e',x); x=re.sub(r'[íìï]','i',x)
        x=re.sub(r'[óòö]','o',x); x=re.sub(r'[úùü]','u',x); x=x.replace("ñ","n")
        x=re.sub(r'[^a-z0-9_]+','_',x).strip('_'); cols.append(x)
    df.columns=cols; return df

def build_graph_from_interstations_df(df: pd.DataFrame) -> nx.Graph:
    df=_normalize_cols(df)
    len_col = next((c for c in df.columns if "long" in c), None)
    if len_col is None: raise ValueError("CSV: no encuentro columna de longitud (m).")
    def to_m(x): 
        s=str(x).replace(",","").replace(" m","").strip()
        try: return float(s)
        except: return np.nan
    df["__m__"]=df[len_col].apply(to_m)
    if "origen" not in df.columns or "destino" not in df.columns:
        raise ValueError("CSV: faltan columnas 'Origen' y 'Destino'.")
    G=nx.Graph()
    for _,r in df.dropna(subset=["__m__"]).iterrows():
        u,v=str(r["origen"]).strip(), str(r["destino"]).strip()
        if u and v and u!=v:
            w=float(r["__m__"])
            if G.has_edge(u,v): G[u][v]["length_m"]=min(G[u][v]["length_m"], w)
            else: G.add_edge(u,v,length_m=w,lines=set())
    for n in G.nodes(): nx.set_node_attributes(G,{n:{"is_transfer":(G.degree[n]>=3)}})
    return G

def merge_lengths_from_interstations_df(G: nx.Graph, df: pd.DataFrame) -> int:
    df=_normalize_cols(df)
    cand_len=[c for c in df.columns if "long" in c]; len_col=cand_len[0] if cand_len else "longitud"
    def to_m(x):
        s=str(x).replace(",","").replace(" m","").strip()
        try: return float(s)
        except: return np.nan
    df["__m__"]=df[len_col].apply(to_m); cnt=0
    for _,r in df.dropna(subset=["__m__"]).iterrows():
        u,v=str(r["origen"]).strip(), str(r["destino"]).strip()
        if G.has_node(u) and G.has_node(v):
            if G.has_edge(u,v): G[u][v]["length_m"]=float(r["__m__"]); cnt+=1
            else: G.add_edge(u,v,length_m=float(r["__m__"]), lines=set()); cnt+=1
    return cnt

# -------------------- KML/KMZ --------------------
KML_NS={"kml":"http://www.opengis.net/kml/2.2"}

@st.cache_data(show_spinner=False)
def read_kml_text_from_path(path: pathlib.Path) -> str:
    data = path.read_bytes()
    if data[:2]==b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name=next((n for n in zf.namelist() if n.lower().endswith(".kml")),None)
            if not name: return ""
            return zf.read(name).decode("utf-8","ignore")
    return data.decode("utf-8","ignore")

def parse_kml_text(kml_text: str):
    if not kml_text: return {}, []
    try:
        root=ET.fromstring(kml_text)
    except Exception:
        return {}, []
    stations, lines = {}, []
    for pm in root.findall(".//kml:Placemark",KML_NS):
        name_el=pm.find("kml:name",KML_NS)
        name=name_el.text.strip() if name_el is not None and name_el.text else None
        pt=pm.find("kml:Point/kml:coordinates",KML_NS)
        ls=pm.find("kml:LineString/kml:coordinates",KML_NS)
        if pt is not None and name:
            parts=pt.text.strip().split(",")
            if len(parts)>=2:
                lon,lat=float(parts[0]),float(parts[1]); stations[name]=(lat,lon)
        elif ls is not None:
            coords=[]
            for chunk in ls.text.replace("\n"," ").split():
                parts=chunk.split(",")
                if len(parts)>=2:
                    lon,lat=float(parts[0]),float(parts[1]); coords.append((lat,lon))
            if coords: lines.append({"name":name or "LineString","coords":coords})
    return stations, lines

def snap_lines_to_stations(stations: Dict[str,Tuple[float,float]], lines: List[Dict], snap_threshold_m=150.0):
    if not stations or not lines: return []
    names=list(stations.keys()); coords=[stations[n] for n in names]
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
        for u,v in zip(seq[:-1],seq[1:]):
            if u!=v:
                dist_m=hav_km(stations[u],stations[v])*1000.0
                edges.append((u,v,dist_m, canonical_line_code(L.get("name") or "")))
    return edges

def build_graph_from_kml(stations: Dict[str,Tuple[float,float]], edges_kml: List[Tuple[str,str,float,str]]) -> nx.Graph:
    G = nx.Graph()
    for n,(lat,lon) in stations.items(): G.add_node(n, lat=lat, lon=lon)
    for u,v,w,ln in edges_kml:
        if u in G and v in G:
            if G.has_edge(u,v):
                G[u][v]["length_m"]=min(G[u][v]["length_m"], w)
                if ln: G[u][v].setdefault("lines", set()).add(ln)
            else:
                G.add_edge(u,v,length_m=float(w), lines=set([ln]) if ln else set())
    for n in G.nodes(): nx.set_node_attributes(G,{n:{"is_transfer":(G.degree[n]>=3)}})
    return G

# -------------------- Build network (auto) --------------------
def find_best_kml() -> pathlib.Path | None:
    cand = [
        DATA_DIR/"Movilidad Integrada ZMVM(1).kml",
        DATA_DIR/"Movilidad Integrada ZMVM.kml",
        DATA_DIR/"Movilidad Integrada ZMVM.kmz",
        DATA_DIR/"Movilidad Integrada ZMVM(1).kmz",
    ]
    for p in cand:
        if p.exists(): 
            txt = read_kml_text_from_path(p)
            stations, lines = parse_kml_text(txt)
            if stations and lines: return p
    return None

@st.cache_data(show_spinner=True)
def autoload_graph():
    kml_path = find_best_kml()
    G=None; coords={}; stations_count=0; edges_count=0
    if kml_path:
        txt = read_kml_text_from_path(kml_path)
        stations, lines = parse_kml_text(txt)
        edges = snap_lines_to_stations(stations, lines, 150.0)
        if stations and edges:
            G = build_graph_from_kml(stations, edges)
            coords = {n:(G.nodes[n]["lat"],G.nodes[n]["lon"]) for n in G.nodes()}
            stations_count, edges_count = len(stations), len(edges)
    if (not G) and CSV_INTER.exists():
        df=pd.read_csv(CSV_INTER, sep=None, engine="python")
        G = build_graph_from_interstations_df(df)
    if G is not None and CSV_INTER.exists():
        try:
            df=pd.read_csv(CSV_INTER, sep=None, engine="python")
            merge_lengths_from_interstations_df(G, df)
        except Exception:
            pass
    if (not G) or (G.number_of_nodes()==0): G, coords = build_graph_demo()
    return G, coords, stations_count, edges_count, bool(kml_path)

# -------------------- Metrics & routing --------------------
def compute_metrics(G: nx.Graph, path: List[str], base_minutes_per_edge=2.0, transfer_penalty_min=5.0, speed_kmh=32.0):
    edges = list(zip(path[:-1], path[1:]))
    dist_m=0.0; lines_seq=[]
    for u,v in edges:
        w=G[u][v].get("length_m", np.nan)
        if not np.isnan(w): dist_m+=w
        lnset=G[u][v].get("lines")
        ln = sorted(lnset)[0] if isinstance(lnset,set) and lnset else None
        lines_seq.append(ln)
    if any(l is not None for l in lines_seq):
        t=0; prev=lines_seq[0]
        for cur in lines_seq[1:]:
            if cur!=prev: t+=1
            prev=cur
        transfers=t
    else:
        transfers=sum(1 for n in path[1:-1] if G.nodes[n].get("is_transfer",False))
    time_dist=(dist_m/1000.0)/speed_kmh*60.0 if dist_m>0 else np.nan
    time_fallback=base_minutes_per_edge*len(edges)+transfer_penalty_min*transfers
    time_min=time_dist if not np.isnan(time_dist) else time_fallback
    return {"edges":len(edges),"transfers":transfers,"dist_km":round(dist_m/1000.0,2) if dist_m>0 else None,
            "time_min":round(time_min,1),"lines_seq":lines_seq}

def enumerate_paths_worst(G: nx.Graph, src: str, dst: str, mode: str,
                          cutoff: int, limit_paths: int,
                          base_minutes_per_edge: float,
                          transfer_penalty_min: float,
                          speed_kmh: float):
    all_paths=[]
    for p in nx.all_simple_paths(G, source=src, target=dst, cutoff=cutoff):
        all_paths.append(p)
        if len(all_paths)>=limit_paths: break
    if not all_paths: return [], {}, [], pd.DataFrame()
    scored=[]
    for p in all_paths:
        m=compute_metrics(G,p,base_minutes_per_edge,transfer_penalty_min,speed_kmh)
        score = m["time_min"] if mode=="Más tiempo" else \
                m["transfers"] if mode=="Más transbordos" else \
                m["edges"] if mode=="Más estaciones" else (m["dist_km"] or 0)
        scored.append((score,p,m))
    scored.sort(key=lambda x:x[0], reverse=True)
    top = scored[:min(10, len(scored))]
    df = pd.DataFrame([{
        "rank": i+1, "score": s, "tiempo_min":m["time_min"], "transbordos":m["transfers"],
        "estaciones":m["edges"], "dist_km":m["dist_km"], "ruta":" → ".join(p)
    } for i,(s,p,m) in enumerate(top)])
    return scored[0][1], scored[0][2], [p for _,p,_ in scored[:5]], df

# -------------------- Plots --------------------
def plot_topological(G: nx.Graph, highlight: List[str] = None, hide_lines: List[str]=None):
    pos = nx.spring_layout(G, seed=42, k=0.6)
    groups={}
    for u,v,data in G.edges(data=True):
        ln="DEFAULT"
        if "lines" in data and data["lines"]: ln=sorted(data["lines"])[0]
        groups.setdefault(ln,[]).append((u,v))
    fig=go.Figure()
    for ln, ev in groups.items():
        if hide_lines and ln in hide_lines: continue
        xs,ys=[],[]
        for (u,v) in ev:
            xs+=[pos[u][0],pos[v][0],None]; ys+=[pos[u][1],pos[v][1],None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(width=3, color=LINE_COLORS.get(ln, LINE_COLORS["DEFAULT"])),
            name=ln, hoverinfo="none"))
    node_x,node_y,texts=[],[],[]
    for n in G.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1]); texts.append(n)
    fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",
        marker=dict(size=10, line=dict(width=1,color="#0d1329")),
        text=[t if (not highlight or t in highlight) else "" for t in texts],
        textposition="top center", hovertext=texts, hoverinfo="text"))
    if highlight and len(highlight)>1:
        hx,hy=[],[]
        for u,v in zip(highlight[:-1],highlight[1:]):
            hx+=[pos[u][0],pos[v][0],None]; hy+=[pos[u][1],pos[v][1],None]
        fig.add_trace(go.Scatter(x=hx,y=hy,mode="lines",
            line=dict(width=7, dash="dot", color="#ffffff"), name="Ruta"))
    fig.update_layout(showlegend=True, height=640, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      paper_bgcolor="#0a0f1c", plot_bgcolor="#0a0f1c")
    st.plotly_chart(fig, use_container_width=True, theme=None)

def plot_geo(path: List[str], coords: Dict[str,Tuple[float,float]], step:int=None):
    if not path or not coords:
        st.info("Carga KML para ver el mapa geográfico."); return
    if not all(s in coords for s in path):
        st.info("Hay estaciones de la ruta sin coord. en KML."); return
    pts=[{"name":s,"lat":coords[s][0],"lon":coords[s][1]} for s in path[:(step or len(path))]]
    segs=[]
    for u,v in zip(path[:(step or len(path))][:-1], path[:(step or len(path))][1:]):
        a,b=coords[u],coords[v]
        segs.append({"path":[[a[1],a[0]],[b[1],b[0]]]})
    view=pdk.ViewState(latitude=pts[0]["lat"], longitude=pts[0]["lon"], zoom=11)
    layer_pts=pdk.Layer("ScatterplotLayer", data=pts, get_position=["lon","lat"], get_radius=70, pickable=True)
    layer_path=pdk.Layer("PathLayer", data=segs, get_path="path", get_width=5, get_color=[255,255,255])
    st.pydeck_chart(pdk.Deck(layers=[layer_path,layer_pts], initial_view_state=view, tooltip={"text":"{name}"}))

# -------------------- UI --------------------
st.markdown('<div class="block-title">Peores Rutas — Metro CDMX</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Calcula la <span class="badge">PEOR</span> ruta por tiempo, transbordos, estaciones o distancia — con colores por línea y mapa animado.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📦 Datos (auto)")
    G, coords, n_stations_kml, n_edges_kml, had_kml = autoload_graph()
    st.markdown(f"- **KML**: {'✅ '+str(n_stations_kml)+' estaciones' if had_kml and n_stations_kml>0 else '— demo/CSV'}")
    st.markdown(f"- **CSV interestaciones**: {'✅' if CSV_INTER.exists() else '—'}")

    st.markdown("### ⚙️ Parámetros")
    speed_kmh = st.slider("Velocidad (km/h)", 24, 45, 32, help="Sólo si hay distancias.")
    base_minutes = st.slider("Minutos por tramo (fallback)", 1.0, 5.0, 2.0, .5, help="Usado cuando no hay longitudes.")
    penalty_transfer = st.slider("Penalización por transbordo (min)", 1.0, 12.0, 5.0, .5)
    cutoff = st.slider("Máx. estaciones por ruta", 10, 60, 28, 1)
    limit_paths = st.slider("Máx. rutas a evaluar", 200, 3000, 1200, 100)
    st.markdown("### 🎨 Ocultar líneas")
    all_lines = sorted({ln for _,_,d in G.edges(data=True) for ln in (d.get("lines") or set())})
    hide_lines = st.multiselect("Elige líneas a ocultar", all_lines, [])

if not G or G.number_of_nodes()==0:
    st.error("No se pudo construir la red. Revisa archivos en /data."); st.stop()

stations_all = sorted(G.nodes())

c1,c2,c3,c4,c5 = st.columns([2,2,2,2,1])
with c1:
    src = st.selectbox("Origen", stations_all, index=0)
with c2:
    dst = st.selectbox("Destino", stations_all, index=min(1,len(stations_all)-1))
with c3:
    mode_worst = st.radio("Criterio", ["Más tiempo","Más transbordos","Más estaciones","Más distancia"], horizontal=True)
with c4:
    snap = st.toggle("Animación", value=True)
with c5:
    if st.button("🎲 Random"):
        src = random.choice(stations_all); dst = random.choice([s for s in stations_all if s!=src])

st.markdown(
    f"<div class='kpi card'>"
    f"<div><div class='subtle'>Estaciones</div><div class='block-title' style='font-size:1.2rem'>{G.number_of_nodes()}</div></div>"
    f"<div><div class='subtle'>Tramos</div><div class='block-title' style='font-size:1.2rem'>{G.number_of_edges()}</div></div>"
    f"<div><div class='subtle'>Estaciones KML</div><div class='block-title' style='font-size:1.2rem'>{n_stations_kml}</div></div>"
    f"<div><div class='subtle'>Tramos KML</div><div class='block-title' style='font-size:1.2rem'>{n_edges_kml}</div></div>"
    f"</div>", unsafe_allow_html=True
)

go = st.button("🔎 Calcular peor ruta", type="primary", use_container_width=True)

if go:
    if src==dst:
        st.warning("Elige estaciones distintas.")
    else:
        path, metrics, top5, df_top = enumerate_paths_worst(
            G, src, dst, mode_worst, int(cutoff), int(limit_paths),
            float(base_minutes), float(penalty_transfer), float(speed_kmh)
        )
        if not path:
            st.error("No encontré rutas con los parámetros actuales.")
        else:
            a,b = st.columns([3,2])
            with a:
                st.markdown(f"<div class='badge'>Ruta ({mode_worst})</div>", unsafe_allow_html=True)
                st.markdown("**" + " → ".join(path) + "**")
                st.markdown(
                    f"- ⏱️ **{metrics['time_min']} min**  \n"
                    f"- 🔁 **{metrics['transfers']}** transbordos  \n"
                    f"- 🚉 **{metrics['edges']}** estaciones  \n" +
                    (f"- 📏 **{metrics['dist_km']} km**" if metrics.get('dist_km') else "")
                )
            with b:
                chips=[]
                for ln in metrics["lines_seq"]:
                    if ln:
                        chips.append(f"<span class='badge' style='border-color:#2a3a77'>{ln}</span>")
                st.markdown("**Líneas en ruta**")
                st.markdown((" ".join(chips)) if chips else "<span class='subtle'>sin etiquetas</span>", unsafe_allow_html=True)
                st.download_button("⬇️ Descargar ruta (CSV)",
                    pd.DataFrame({"paso":range(1,len(path)+1),"estacion":path}).to_csv(index=False).encode("utf-8"),
                    "ruta.csv", "text/csv")

            tabs = st.tabs(["🕸️ Topológico","🗺️ Geográfico","🏆 Top 10 peores","🌐 My Maps"])
            with tabs[0]:
                plot_topological(G, highlight=path, hide_lines=hide_lines)
            with tabs[1]:
                if snap:
                    step = st.slider("Paso", 1, len(path), len(path))
                else:
                    step = len(path)
                plot_geo(path, coords, step=step)
            with tabs[2]:
                st.dataframe(df_top, use_container_width=True)
            with tabs[3]:
                url_default = "https://www.google.com/maps/d/embed?mid=1X_5plYmiuNx09w0jWBuakiTS3vW_6ts&ehbc=2E312F"
                url = st.text_input("URL de My Maps (embed)", value=url_default)
                if url:
                    html(f"""
                        <div style="position:relative;padding-bottom:66%;height:0;overflow:hidden;border-radius:16px;border:1px solid #22305b">
                          <iframe src="{url}" style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;" loading="lazy"></iframe>
                        </div>
                    """, height=520)
else:
    st.info("Elige origen/destino, ajusta parámetros y pulsa **Calcular peor ruta**.")
