import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pydeck as pdk
from streamlit.components.v1 import html
from typing import Dict, List, Tuple
from math import radians, cos, sin, asin, sqrt
from xml.etree import ElementTree as ET
import zipfile, io, requests, os, re, random, pathlib

st.set_page_config(page_title="Peores Rutas — Metro CDMX", page_icon="🚇", layout="wide")

CSS = """
<style>
:root{
  --bg:#0b1020; --text:#eef2ff; --muted:#a5b4fc; --card:#0f1530; --border:#232a48;
  --accent:#00B5E2;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text)}
.block-title{font-weight:900;font-size:1.8rem;margin:.2rem 0 .6rem}
.subtle{color:var(--muted);font-size:.95rem}
.card{border:1px solid var(--border);background:var(--card);border-radius:18px;padding:16px}
hr{border:none;border-top:1px solid var(--border);margin:1rem 0}
.badge{display:inline-flex;gap:.5rem;align-items:center;padding:.25rem .6rem;border-radius:999px;
  border:1px solid var(--border);background:#10183a;font-size:.78rem}
.kpi{display:flex;gap:14px}
.kpi>div{flex:1;border:1px solid var(--border);background:#0e1736;border-radius:14px;padding:12px}
a{color:#7dd3fc} .ok{color:#22c55e} .warn{color:#f59e0b} .err{color:#ef4444}
.fade{animation:fadein .35s ease-in-out} @keyframes fadein{from{opacity:0}to{opacity:1}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LINE_COLORS = {
    "L1":"#F05A91","L2":"#0055A5","L3":"#8CC63E","L4":"#00B5E2","L5":"#FFC20E","L6":"#E10600",
    "L7":"#F39C12","L8":"#00A859","L9":"#8B5E3C","L12":"#C8B273","A":"#7F3FBF","B":"#7AC142",
    "DEFAULT":"#7a88b8"
}

def hex_to_rgb(h): 
    h=h.lstrip("#"); return [int(h[i:i+2],16) for i in (0,2,4)]

def hav_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    lon1, lat1, lon2, lat2 = map(radians, [lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    r = 6371.0
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
    if t.strip() == "a": return "A"
    if t.strip() == "b": return "B"
    return None

# ---------- DEMO de respaldo ----------
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
def build_graph_demo() -> Tuple[nx.Graph, Dict[str,Tuple[float,float]]]:
    G = nx.Graph()
    for ln, seq in LINES_DEMO.items():
        for u,v in zip(seq[:-1], seq[1:]):
            G.add_edge(u, v, length_m=1000.0, lines=set([ln]))
    for n in G.nodes(): nx.set_node_attributes(G, {n: {"is_transfer": (G.degree[n] >= 3)}})
    return G, {}

# ---------- CSV interestaciones ----------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols=[]; 
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
    G = nx.Graph()
    for _,r in df.dropna(subset=["__m__"]).iterrows():
        u,v = str(r["origen"]).strip(), str(r["destino"]).strip()
        if u and v and u!=v:
            w=float(r["__m__"])
            if G.has_edge(u,v): G[u][v]["length_m"]=min(G[u][v]["length_m"], w)
            else: G.add_edge(u,v,length_m=w, lines=set())
    for n in G.nodes(): nx.set_node_attributes(G,{n:{"is_transfer":(G.degree[n]>=3)}})
    return G

# ---------- KML/KMZ ----------
KML_NS={"kml":"http://www.opengis.net/kml/2.2"}

def read_kml_text(file_or_bytes) -> str:
    data = file_or_bytes.read() if hasattr(file_or_bytes,"read") else file_or_bytes
    if isinstance(data,str): return data
    if data[:2]==b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name=next((n for n in zf.namelist() if n.lower().endswith(".kml")),None)
            if not name: raise ValueError("KMZ sin .kml")
            return zf.read(name).decode("utf-8","ignore")
    return data.decode("utf-8","ignore")

def resolve_networklinks(kml_text:str, allow_download=False)->str:
    root=ET.fromstring(kml_text)
    links=root.findall(".//kml:NetworkLink/kml:Link/kml:href",KML_NS)
    if not links or not allow_download: return kml_text
    url=links[0].text.strip(); r=requests.get(url,timeout=20); r.raise_for_status(); return r.text

def parse_kml(file_or_bytes, allow_download=False):
    kml_text=read_kml_text(file_or_bytes); kml_text=resolve_networklinks(kml_text, allow_download)
    root=ET.fromstring(kml_text)
    stations, lines = {}, []
    for pm in root.findall(".//kml:Placemark",KML_NS):
        name_el=pm.find("kml:name",KML_NS); name=name_el.text.strip() if name_el is not None and name_el.text else None
        pt=pm.find("kml:Point/kml:coordinates",KML_NS)
        ls=pm.find("kml:LineString/kml:coordinates",KML_NS)
        if pt is not None and name:
            lon,lat=map(float, pt.text.strip().split(",")[:2]); stations[name]=(lat,lon)
        elif ls is not None:
            coords=[]
            for chunk in ls.text.replace("\n"," ").split():
                lon,lat=map(float, chunk.split(",")[:2]); coords.append((lat,lon))
            if coords: lines.append({"name":name or "LineString","coords":coords})
    return stations, lines

def snap_lines_to_stations(stations: Dict[str,Tuple[float,float]], lines: List[Dict], snap_threshold_m=150.0):
    if not stations or not lines: return []
    names=list(stations.keys()); coords=[stations[n] for n in names]
    def nearest(lat,lon):
        best_i,best_d=None,1e18
        for i,c in enumerate(coords):
            d=hav_km((lat,lon),c)
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

def set_edge_line_if_missing(G: nx.Graph, edges_kml: List[Tuple[str,str,float,str]]) -> int:
    cnt=0
    for u,v,_,ln in edges_kml:
        if not ln: continue
        if G.has_edge(u,v):
            L = G[u][v].setdefault("lines", set())
            if ln not in L: L.add(ln); cnt+=1
    return cnt

def compute_metrics(G: nx.Graph, path: List[str], base_minutes_per_edge=2.0, transfer_penalty_min=5.0, speed_kmh=32.0):
    edges = list(zip(path[:-1], path[1:]))
    dist_m=0.0; lines_seq=[]
    for u,v in edges:
        w=G[u][v].get("length_m", np.nan)
        if not np.isnan(w): dist_m+=w
        ln=None
        lnset=G[u][v].get("lines")
        if isinstance(lnset,set) and lnset: ln=sorted(lnset)[0]
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
    if not all_paths: return [], {}, []
    scored=[]
    for p in all_paths:
        m=compute_metrics(G,p,base_minutes_per_edge,transfer_penalty_min,speed_kmh)
        score = m["time_min"] if mode=="Más tiempo" else \
                m["transfers"] if mode=="Más transbordos" else \
                m["edges"] if mode=="Más estaciones" else (m["dist_km"] or 0)
        scored.append((score,p,m))
    scored.sort(key=lambda x:x[0], reverse=True)
    top5=[p for _,p,_ in scored[:5]]
    return scored[0][1], scored[0][2], top5

def plot_topological(G: nx.Graph, highlight: List[str] = None, hide_lines: List[str]=None):
    pos = nx.spring_layout(G, seed=42, k=0.6)
    groups={}
    for u,v,data in G.edges(data=True):
        ln="DEFAULT"
        if "lines" in data and data["lines"]:
            ln=sorted(data["lines"])[0]
        groups.setdefault(ln,[]).append((u,v))
    fig=go.Figure()
    for ln, ev in groups.items():
        if hide_lines and ln in hide_lines: continue
        xs,ys=[],[]
        for (u,v) in ev:
            xs+=[pos[u][0],pos[v][0],None]; ys+=[pos[u][1],pos[v][1],None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=3, color=LINE_COLORS.get(ln, LINE_COLORS["DEFAULT"])),
            name=ln, hoverinfo="none"
        ))
    node_x,node_y,texts=[],[],[]
    for n in G.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1]); texts.append(n)
    fig.add_trace(go.Scatter(
        x=node_x,y=node_y,mode="markers+text",
        marker=dict(size=10, line=dict(width=1,color="#0d1329")),
        text=[t if (not highlight or t in highlight) else "" for t in texts],
        textposition="top center", hovertext=texts, hoverinfo="text"
    ))
    if highlight and len(highlight)>1:
        hx,hy=[],[]
        for u,v in zip(highlight[:-1], highlight[1:]):
            hx+=[pos[u][0],pos[v][0],None]; hy+=[pos[u][1],pos[v][1],None]
        fig.add_trace(go.Scatter(
            x=hx,y=hy,mode="lines",
            line=dict(width=7, dash="dot", color="#ffffff"), name="Ruta destacada"
        ))
    fig.update_layout(showlegend=True, height=640, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="#0b1020", plot_bgcolor="#0b1020")
    st.plotly_chart(fig, use_container_width=True, theme=None)

def plot_geo(path: List[str], coords: Dict[str,Tuple[float,float]], color_seq: List[str]=None, step:int=None):
    if not path or not coords:
        st.info("Faltan coordenadas. Carga KML/KMZ para el mapa geográfico."); return
    if not all(s in coords for s in path):
        st.info("Alguna estación no tiene coordenadas en KML."); return
    pts=[{"name":s,"lat":coords[s][0],"lon":coords[s][1]} for s in path[:(step or len(path))]]
    segs=[]
    for u,v in zip(path[:(step or len(path))][:-1], path[:(step or len(path))][1:]):
        a,b=coords[u],coords[v]
        segs.append({"path":[[a[1],a[0]],[b[1],b[0]]]})
    view=pdk.ViewState(latitude=pts[0]["lat"], longitude=pts[0]["lon"], zoom=11)
    layer_pts=pdk.Layer("ScatterplotLayer", data=pts, get_position=["lon","lat"], get_radius=70, pickable=True)
    layer_path=pdk.Layer("PathLayer", data=segs, get_path="path", get_width=5,
                         get_color=[255,255,255])
    st.pydeck_chart(pdk.Deck(layers=[layer_path,layer_pts], initial_view_state=view, tooltip={"text":"{name}"}))

# ---------- Auto-carga de datos locales ----------
DATA_DIR = pathlib.Path("data")
CAND_KMLS = [
    DATA_DIR/"Movilidad Integrada ZMVM.kmz",
    DATA_DIR/"Movilidad Integrada ZMVM.kml",
    DATA_DIR/"Movilidad Integrada ZMVM(1).kml",
]
CSV_INTER = DATA_DIR/"metro.csv"

def try_autoload():
    G=None; coords={}
    kml_ok=False; edges_kml=[]
    stations_kml={}; lines_kml=[]
    for p in CAND_KMLS:
        if p.exists():
            try:
                with open(p,"rb") as f:
                    stations_kml, lines_kml = parse_kml(f, allow_download=False)
                edges_kml = snap_lines_to_stations(stations_kml, lines_kml, 150.0)
                G = build_graph_from_kml(stations_kml, edges_kml)
                coords = {n:(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in G.nodes()}
                kml_ok=True
                break
            except Exception:
                continue
    if not kml_ok:
        G, coords = build_graph_demo()
    if CSV_INTER.exists() and G:
        try:
            df=pd.read_csv(CSV_INTER, sep=None, engine="python")
            merge_lengths_from_interstations_df(G, df)
            if kml_ok: set_edge_line_if_missing(G, edges_kml)
        except Exception:
            pass
    return G, coords, (len(stations_kml) if stations_kml else 0), (len(edges_kml) if edges_kml else 0)

# ---------- UI ----------
st.markdown('<div class="block-title fade">Peores Rutas — Metro CDMX</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Cálculo de la <span class="badge">PEOR</span> ruta por tiempo, transbordos, estaciones o distancia, con colores por línea.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📦 Datos (auto)")
    G, coords, n_stations_kml, n_edges_kml = try_autoload()
    ok_kml = n_stations_kml>0 and n_edges_kml>0
    st.markdown(f"- KML/KMZ: {'✅' if ok_kml else '⚠️ demo'}  \n- CSV interestaciones: {'✅' if CSV_INTER.exists() else '—'}")
    st.markdown("### ⚙️ Parámetros")
    speed_kmh = st.slider("Velocidad (km/h)", 24, 45, 32)
    base_minutes = st.slider("Minutos por tramo (fallback)", 1.0, 5.0, 2.0, .5)
    penalty_transfer = st.slider("Penalización por transbordo (min)", 1.0, 12.0, 5.0, .5)
    cutoff = st.slider("Máx. estaciones por ruta", 10, 60, 28, 1)
    limit_paths = st.slider("Máx. rutas a evaluar", 200, 3000, 1200, 100)
    st.markdown("### 🎨 Líneas visibles")
    all_lines = sorted({ln for _,_,d in G.edges(data=True) for ln in (d.get("lines") or set())} | {"DEFAULT"})
    hide_lines = st.multiselect("Ocultar líneas", [l for l in all_lines if l!="DEFAULT"], [])

if not G or G.number_of_nodes()==0:
    st.error("No se pudo construir la red. Revisa los archivos en /data.")
    st.stop()

stations_all = sorted(G.nodes())
c1,c2,c3,c4 = st.columns([2,2,2,2])
with c1:
    src = st.selectbox("Origen", stations_all, index=0)
with c2:
    dst = st.selectbox("Destino", stations_all, index=min(1,len(stations_all)-1))
with c3:
    mode_worst = st.selectbox("Criterio de 'peor'", ["Más tiempo","Más transbordos","Más estaciones","Más distancia"])
with c4:
    go = st.button("🔎 Calcular", type="primary", use_container_width=True)

st.markdown('<div class="card kpi"><div><div class="subtle">Nodos</div><div class="block-title" style="font-size:1.2rem;">'+str(G.number_of_nodes())+'</div></div><div><div class="subtle">Aristas</div><div class="block-title" style="font-size:1.2rem;">'+str(G.number_of_edges())+'</div></div><div><div class="subtle">Estaciones KML</div><div class="block-title" style="font-size:1.2rem;">'+str(n_stations_kml)+'</div></div><div><div class="subtle">Tramos KML</div><div class="block-title" style="font-size:1.2rem;">'+str(n_edges_kml)+'</div></div></div>', unsafe_allow_html=True)

if go:
    if src==dst:
        st.warning("Elige estaciones distintas.")
    else:
        path, metrics, top5 = enumerate_paths_worst(
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
                        chips.append(f"<span class='badge' style='border-color:#1f2760'>{ln}</span>")
                st.markdown("**Líneas**")
                st.markdown((" ".join(chips)) if chips else "<span class='subtle'>sin etiquetas</span>", unsafe_allow_html=True)
                df_dl = pd.DataFrame({"paso":list(range(1,len(path)+1)),"estacion":path})
                st.download_button("⬇️ Descargar ruta (CSV)", df_dl.to_csv(index=False).encode("utf-8"), "ruta.csv", "text/csv")
            tabs = st.tabs(["🕸️ Topológico","🗺️ Geográfico","🔀 Alternativas","🌐 My Maps"])
            with tabs[0]:
                plot_topological(G, highlight=path, hide_lines=hide_lines)
            with tabs[1]:
                st.markdown("**Animación de recorrido**")
                step = st.slider("Paso", 1, len(path), len(path))
                plot_geo(path, coords, step=step)
            with tabs[2]:
                if len(top5)<=1:
                    st.info("No hay suficientes alternativas.")
                else:
                    for i,pth in enumerate(top5, start=1):
                        m = compute_metrics(G, pth, base_minutes, penalty_transfer, speed_kmh)
                        with st.expander(f"Alternativa #{i} — ⏱ {m['time_min']} min · 🔁 {m['transfers']} · 🚉 {m['edges']}" + (f" · 📏 {m['dist_km']} km" if m.get('dist_km') else "")):
                            st.write(" → ".join(pth))
            with tabs[3]:
                url_default = "https://www.google.com/maps/d/embed?mid=1X_5plYmiuNx09w0jWBuakiTS3vW_6ts&ehbc=2E312F"
                url = st.text_input("URL de My Maps (embed)", value=url_default)
                if url:
                    html(f"""
                        <div style="position:relative;padding-bottom:66%;height:0;overflow:hidden;border-radius:16px;border:1px solid #232a48">
                          <iframe src="{url}" style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;" loading="lazy"></iframe>
                        </div>
                    """, height=520)
else:
    st.info("Selecciona origen/destino y pulsa **Calcular**. También puedes ocultar líneas desde la barra lateral.")
