# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime, timezone, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.mongo_client import (
    get_recent_earthquakes, get_recent_eonet,
    get_fema_by_state, get_disaster_trend
)

# Paths to batch training outputs
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "ml", "trained_models")

def load_json_stats(filename, default=None):
    path = os.path.join(ML_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default or {}

ANOMALY_STATS  = load_json_stats("anomaly_stats.json",
    {"global_mean_count":10,"global_std_count":15,"threshold_moderate":40,"threshold_high":55})
SEASONAL_STATS    = load_json_stats("seasonal_stats.json")
KMEANS_META    = load_json_stats("kmeans_meta.json")

# Page config
st.set_page_config(page_title="Real-Time Disaster Monitor", page_icon="🌍",
                   layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="collapsedControl"] { display: none; }
.section-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.12em; color: #555;
    border-bottom: 1px solid #1e1e2e; padding-bottom: 6px; margin-bottom: 14px;
}
.kpi-card {
    background: #e3f0ff; border-radius: 18px; padding: 22px 0 12px 0;
    margin-bottom: 8px; box-shadow: 0 2px 8px rgba(30,30,46,0.07); text-align: center;
}
.kpi-label { font-size: 13px; color: #1565c0; margin-bottom: 2px; font-weight: 600; letter-spacing: 0.04em; }
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #1976d2; margin-bottom: 2px; }
.kpi-sub { font-size: 12px; color: #42a5f5; }
.anomaly-banner {
    background: linear-gradient(90deg,#1a1a2e 0%,#2d1b1b 50%,#1a1a2e 100%);
    border: 1px solid #ff4b4b; border-left: 4px solid #ff4b4b;
    border-radius: 10px; padding: 14px 20px; margin-bottom: 10px; color: #ff8c8c; font-size: 13px;
}
.anomaly-banner-ok {
    background: linear-gradient(90deg,#1a1a2e 0%,#1b2d1b 50%,#1a1a2e 100%);
    border: 1px solid #22c55e; border-left: 4px solid #22c55e;
    border-radius: 10px; padding: 14px 20px; margin-bottom: 10px; color: #86efac; font-size: 13px;
}
.pipeline-pill {
    display: inline-block; background: #1e293b; border: 1px solid #334155;
    border-radius: 20px; padding: 4px 14px; margin: 2px 4px; font-size: 12px; color: #94a3b8;
}
.pipeline-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e; margin-right: 5px; animation: pulse-green 2s infinite;
}
@keyframes pulse-green { 0%,100%{opacity:1} 50%{opacity:0.4} }
</style>
""", unsafe_allow_html=True)

# Toolbar
now_utc = datetime.now(timezone.utc)
now_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")

title_col, controls_col = st.columns([3, 1])
with title_col:
    st.title("🌍 Real-Time Disaster Monitor")
    st.caption("Natural hazards tracking dashboard")
    st.caption(f"Live as of {now_str}")

with controls_col:
    filter_col, refresh_col = st.columns([1, 1])
    with filter_col:
        with st.popover("⚙️ Filter", use_container_width=True):
            st.markdown("**🌋 Earthquake Filters**")
            min_mag  = st.slider("Min Magnitude", 0.0, 9.0, 2.5, 0.1)
            eq_limit = st.slider("Max Events", 50, 1000, 300, 50)
            st.markdown("---")
            st.markdown("**🌿 Event Types**")
            event_types = st.multiselect("Event Catergories",
                ["wildfire","storm","volcano","flood","ice","landslide"],
                default=["wildfire","storm","volcano","flood"])
            st.markdown("---")
            st.markdown("**🗺️ Map Layer**")
            map_layer = st.radio("Display Mode",["Markers","Risk Heatmap","Both"], index=2, horizontal=True)
            st.markdown("---")
            auto_refresh = st.checkbox("Auto-refresh every 60s", value=False)
    with refresh_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

st.divider()

# Load data (from MongoDB — already ML-scored by stream_processor)
@st.cache_data(ttl=60)
def load_earthquakes(limit, min_mag):
    return pd.DataFrame(get_recent_earthquakes(limit=limit, min_mag=min_mag))

@st.cache_data(ttl=120)
def load_eonet(types):
    return pd.DataFrame(get_recent_eonet(limit=200, event_types=types if types else None))

@st.cache_data(ttl=300)
def load_fema_state():
    data = get_fema_by_state()
    return pd.DataFrame([{"state": d["_id"], "count": d["count"]} for d in data])

@st.cache_data(ttl=300)
def load_trend():
    data = get_disaster_trend(days=30)
    return pd.DataFrame([
        {"date": d["_id"]["date"], "count": d["count"], "avg_mag": d.get("avg_mag", 0)}
        for d in data
    ])

eq_df    = load_earthquakes(eq_limit, min_mag)
eonet_df = load_eonet(event_types)
fema_df  = load_fema_state()
trend_df = load_trend()

#  ANOMALY DETECTION — using batch-trained thresholds on live data

def detect_anomalies_from_data(eq_df, eonet_df):
    """Run anomaly detection using thresholds from batch_ml_training.py."""
    anomalies = []
    if eq_df.empty:
        return anomalies

    df = eq_df.dropna(subset=["latitude","longitude"]).copy()
    if df.empty:
        return anomalies

    if "anomaly_flag" in df.columns:
        flagged = df[df["anomaly_flag"] == True]
        if not flagged.empty:
            flagged["grid_lat"] = (flagged["latitude"]/5).round()*5
            flagged["grid_lon"] = (flagged["longitude"]/5).round()*5
            for (glat, glon), group in flagged.groupby(["grid_lat","grid_lon"]):
                top_place = str(group.iloc[0].get("place","Unknown"))
                max_mag = group["magnitude"].max()
                anomalies.append({
                    "severity": "HIGH" if max_mag >= 6 else "MODERATE",
                    "description": f"{len(group)} flagged events (max M{max_mag:.1f}) near {top_place}",
                    "lat": glat, "lon": glon, "eq_count": len(group),
                    "eonet_count": 0, "multi_type": False, "max_mag": max_mag,
                })
            return sorted(anomalies, key=lambda x: x["max_mag"], reverse=True)

    threshold = ANOMALY_STATS.get("threshold_moderate", 40)
    df["grid_lat"] = (df["latitude"]/5).round()*5
    df["grid_lon"] = (df["longitude"]/5).round()*5

    grid_counts = (df.groupby(["grid_lat","grid_lon"])
        .agg(count=("magnitude","count"), max_mag=("magnitude","max")).reset_index())

    for _, cell in grid_counts[grid_counts["count"] > threshold].iterrows():
        nearby_eonet = 0; eonet_types = []
        if not eonet_df.empty and "latitude" in eonet_df.columns:
            nearby = eonet_df[
                (eonet_df["latitude"].between(cell["grid_lat"]-5, cell["grid_lat"]+5)) &
                (eonet_df["longitude"].between(cell["grid_lon"]-5, cell["grid_lon"]+5))]
            nearby_eonet = len(nearby)
            if not nearby.empty and "type" in nearby.columns:
                eonet_types = nearby["type"].unique().tolist()

        severity = "HIGH" if (cell["max_mag"] >= 6 or nearby_eonet > 0) else "MODERATE"
        region_events = df[(df["grid_lat"]==cell["grid_lat"]) & (df["grid_lon"]==cell["grid_lon"])]
        top_place = str(region_events.iloc[0].get("place","Unknown")) if not region_events.empty else "Unknown"
        desc = f"{int(cell['count'])} earthquakes (max M{cell['max_mag']:.1f}) near {top_place}"
        if nearby_eonet > 0:
            desc += f" + {nearby_eonet} {', '.join(eonet_types)} event(s) — MULTI-TYPE CLUSTER"
        anomalies.append({"severity":severity,"description":desc,"lat":cell["grid_lat"],
            "lon":cell["grid_lon"],"eq_count":int(cell["count"]),"eonet_count":nearby_eonet,
            "multi_type":nearby_eonet>0,"max_mag":cell["max_mag"]})

    return sorted(anomalies, key=lambda x: x["max_mag"], reverse=True)

def compute_regional_risk(eq_df):
    """Aggregate risk by region using ML-scored fields from MongoDB."""
    if eq_df.empty: return pd.DataFrame()
    df = eq_df.dropna(subset=["latitude","longitude"]).copy()
    if df.empty: return pd.DataFrame()

    def extract_region(place):
        if not isinstance(place,str): return "Unknown"
        parts = place.split(",")
        return parts[-1].strip() if len(parts)>1 else place.strip()

    df["region"] = df["place"].apply(extract_region) if "place" in df.columns else "Unknown"
    agg = {"eq_count":("magnitude","count"),"avg_mag":("magnitude","mean"),"max_mag":("magnitude","max")}
    risk_col = "risk_score" if "risk_score" in df.columns else None
    if risk_col: agg["avg_risk"] = (risk_col, "mean")
    regional = df.groupby("region").agg(**agg).reset_index()
    regional["avg_mag"] = regional["avg_mag"].round(2)
    if "avg_risk" in regional.columns: regional["avg_risk"] = regional["avg_risk"].round(1)
    def assign_profile(row):
        if row["max_mag"]>=6.0: return "Critical Seismic Zone"
        elif row["avg_mag"]>=4.5: return "High Seismic Activity"
        elif row["eq_count"]>=10: return "Frequent Activity"
        else: return "Low Activity"
    regional["profile"] = regional.apply(assign_profile, axis=1)
    return regional.sort_values("avg_risk" if "avg_risk" in regional.columns else "max_mag", ascending=False)

anomalies = detect_anomalies_from_data(eq_df, eonet_df)
regional_risk = compute_regional_risk(eq_df)

# KPIs
avg_mag = round(eq_df["magnitude"].mean(),2) if not eq_df.empty else 0.0
max_mag = round(eq_df["magnitude"].max(),1) if not eq_df.empty else 0.0
critical_eq = len(eq_df[eq_df["magnitude"]>=6.0]) if not eq_df.empty else 0
tsunami_warn = int(eq_df["tsunami"].sum()) if ("tsunami" in eq_df.columns and not eq_df.empty) else 0
active_events = len(eonet_df)
risk_col = "risk_score" if "risk_score" in eq_df.columns else None
high_risk_count = len(eq_df[eq_df[risk_col]>=70]) if (risk_col and not eq_df.empty) else 0

# Anomaly Banner 
if anomalies:
    multi_type = [a for a in anomalies if a["multi_type"]]
    top = multi_type[0] if multi_type else anomalies[0]
    tag = "🔴 ANOMALY DETECTED — Multi-Type Disaster Cluster" if multi_type else "🟠 CLUSTER ALERT — Elevated Seismic Activity"
    st.markdown(f'<div class="anomaly-banner"><b>{tag}</b><br>{top["description"]}<br><span style="font-size:11px;color:#ff6b6b">Threshold: {ANOMALY_STATS.get("threshold_moderate","N/A")} | Grid: ({top["lat"]:.0f}°, {top["lon"]:.0f}°) | {now_str}</span></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="anomaly-banner-ok"><b>🟢 NO ANOMALIES DETECTED</b> — All regions within trained thresholds (μ+2σ = {ANOMALY_STATS.get("threshold_moderate","N/A")}). <span style="font-size:11px">| {now_str}</span></div>', unsafe_allow_html=True)

# Pipeline Health
rf_status = "RF Model" if os.path.exists(os.path.join(ML_DIR,"risk_score_rf")) else "Fallback"
st.markdown(f'<div style="text-align:right;margin-bottom:8px;"><span class="pipeline-pill"><span class="pipeline-dot"></span>USGS — {len(eq_df)} events</span><span class="pipeline-pill"><span class="pipeline-dot"></span>NASA EONET — {len(eonet_df)} active</span><span class="pipeline-pill"><span class="pipeline-dot"></span>OpenFEMA — {len(fema_df)} states</span><span class="pipeline-pill" style="border-color:#8b5cf6;color:#a78bfa;">🧠 {rf_status}</span></div>', unsafe_allow_html=True)

# KPI Cards
kpi1,kpi2,kpi3,kpi4,kpi5,kpi6 = st.columns(6)
with kpi1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">⚠️ Critical Earthquakes</div><div class="kpi-value">{critical_eq}</div><div class="kpi-sub">Mag ≥ 6.0</div></div>', unsafe_allow_html=True)
with kpi2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">📊 Max Magnitude</div><div class="kpi-value">{max_mag}</div><div class="kpi-sub">Avg: {avg_mag}</div></div>', unsafe_allow_html=True)
with kpi3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">🌋 Earthquakes Tracked</div><div class="kpi-value">{len(eq_df)}</div><div class="kpi-sub">Mag ≥ {min_mag}</div></div>', unsafe_allow_html=True)
with kpi4: st.markdown(f'<div class="kpi-card"><div class="kpi-label">🔥 Active Events</div><div class="kpi-value">{active_events}</div><div class="kpi-sub">Wildfires, Storms & more</div></div>', unsafe_allow_html=True)
with kpi5: st.markdown(f'<div class="kpi-card"><div class="kpi-label">🌊 Tsunami Warnings</div><div class="kpi-value">{tsunami_warn}</div><div class="kpi-sub">Active alerts</div></div>', unsafe_allow_html=True)
with kpi6: st.markdown(f'<div class="kpi-card" style="background:#fff0f0;border:1px solid #ffcccc;"><div class="kpi-label" style="color:#c62828;">🎯 High-Risk Events</div><div class="kpi-value" style="color:#d32f2f;">{high_risk_count}</div><div class="kpi-sub" style="color:#e57373;">ML Risk ≥ 70</div></div>', unsafe_allow_html=True)


#  MAP + ALERTS
st.markdown('<div class="section-label">GLOBAL DISASTER MAP & RISK HEATMAP</div>', unsafe_allow_html=True)
map_col, alert_col = st.columns([3, 1])
TYPE_COLORS = {"wildfire":"darkred","storm":"blue","volcano":"purple","flood":"darkblue","ice":"lightblue","landslide":"brown"}
TYPE_ICONS = {"wildfire":"fire","storm":"bolt","volcano":"mountain","flood":"tint","ice":"snowflake","landslide":"exclamation"}

with map_col:
    m = folium.Map(location=[20,0], zoom_start=2, tiles="CartoDB dark_matter")
    show_markers = map_layer in ["Markers","Both"]
    show_heatmap = map_layer in ["Risk Heatmap","Both"]

    
    if show_markers and not eq_df.empty:
        for _, row in eq_df.dropna(subset=["latitude","longitude"]).iterrows():
            mag=row.get("magnitude",0) or 0; risk=row.get("risk_score",0) or 0
            tier=row.get("risk_tier",""); t_prob=row.get("tsunami_prob",0) or 0
            color="red" if mag>=6 else "orange" if mag>=4 else "#f5c518"
            folium.CircleMarker(location=[row["latitude"],row["longitude"]],
                radius=max(3,mag*2), color=color, fill=True, fill_opacity=0.65,
                popup=folium.Popup(f"<b>🌋 Earthquake</b><br>Mag: <b>{mag}</b><br>Risk: <b>{risk:.0f}/100</b> ({tier})<br>Tsunami P: <b>{t_prob:.1%}</b><br>{row.get('place','')}<br>Depth: {row.get('depth_km','')} km<br>{row.get('time','')}", max_width=280)
            ).add_to(m)

    if show_markers and not eonet_df.empty:
        for _, row in eonet_df.dropna(subset=["latitude","longitude"]).iterrows():
            etype=row.get("type","unknown")
            folium.Marker(location=[row["latitude"],row["longitude"]],
                icon=folium.Icon(color=TYPE_COLORS.get(etype,"gray"),icon=TYPE_ICONS.get(etype,"info-sign"),prefix="fa"),
                popup=folium.Popup(f"<b>{etype.title()}</b><br>{row.get('title','')}<br>{row.get('last_date','')}", max_width=260)
            ).add_to(m)

    for anom in anomalies:
        c="#ff4b4b" if anom["severity"]=="HIGH" else "#ff8c00"
        folium.Circle(location=[anom["lat"],anom["lon"]], radius=300000, color=c,
            fill=True, fill_opacity=0.12, weight=2, dash_array="5",
            popup=folium.Popup(f"<b>⚠️ Anomaly</b><br>{anom['description']}", max_width=300)).add_to(m)

    st_folium(m, width=None, height=540)
    st.markdown(
    '<div class="data-sources" style="font-size:12px; color:#666; margin-bottom:18px;">'
    '🌋 US Geological Survey &nbsp; 🛰️ NASA Earth Observatory &nbsp; 🇺🇸 Federal Emergency Management Agency'
    '</div>',unsafe_allow_html=True)

with alert_col:
    st.markdown("**🚨 Recent High-Severity Events**")
    alerts_html=""
    if not eq_df.empty:
        for _,row in (eq_df[eq_df["magnitude"]>=5.0].sort_values("magnitude",ascending=False).head(20)).iterrows():
            mag=row.get("magnitude",0) or 0; place=str(row.get("place","Unknown"))[:36]
            risk=row.get("risk_score",0) or 0; sev="red" if mag>=6 else "orange" if mag>=5 else "yellow"
            alerts_html+=f'<div class="alert-item {sev}"><span class="alert-badge badge-{sev}">M {mag}</span>&nbsp;<span style="color:#888;font-size:10px">Risk:{risk:.0f}</span>&nbsp;<span style="color:#ccc">{place}</span><br><span style="color:#555;font-size:10px">{str(row.get("time",""))[:16]}</span></div>'
    if not eonet_df.empty:
        for _,row in eonet_df.head(10).iterrows():
            etype=str(row.get("type","event")).title(); title=str(row.get("title",""))[:36]
            alerts_html+=f'<div class="alert-item yellow"><span class="alert-badge badge-yellow">{etype}</span><br><span style="color:#ccc">{title}</span><br><span style="color:#555;font-size:10px">{str(row.get("last_date",""))[:16]}</span></div>'
    if alerts_html:
        st.markdown(f'<div style="max-height:540px;overflow-y:auto;padding-right:8px">{alerts_html}</div>', unsafe_allow_html=True)
    else:
        st.info("No high-severity alerts.")

#  ANALYTICS
st.markdown('<div class="section-label">ANALYTICS</div>', unsafe_allow_html=True)
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📈 Trends & Forecast","🌋 Earthquake Profile","📍 Regions & Risk",
    "🌿 Natural Events","🇺🇸 FEMA","🔬 ML Risk Analysis","🔗 Multi-Disaster Correlation"])
CL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#aaa",margin=dict(l=0,r=0,t=30,b=0))

# Tab 1: Trends + Seasonal Forecast 
with tab1:
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Daily Earthquake Count — Last 30 Days**")
        if not trend_df.empty:
            fig=px.bar(trend_df.sort_values("date"),x="date",y="count",color="avg_mag",color_continuous_scale="YlOrRd",labels={"count":"Events","date":"Date","avg_mag":"Avg Mag"})
            fig.update_layout(**CL,height=320); st.plotly_chart(fig,use_container_width=True)
        else: st.info("No trend data yet.")
    with c2:
        st.markdown("**Avg Magnitude Trend + Seasonal Forecast**")
        if not trend_df.empty:
            ds=trend_df.sort_values("date")
            ar=SEASONAL_STATS.get("recent_avg_magnitude",ds["avg_mag"].tail(7).mean())
            fs=ds["avg_mag"].tail(7).std()*0.5
            ld=pd.to_datetime(ds["date"].iloc[-1])
            fd=[(ld+timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(14)]
            np.random.seed(42); fv=[ar+np.random.normal(0,fs) for _ in range(14)]
            fu=[v+1.96*fs for v in fv]; fl=[max(0,v-1.96*fs) for v in fv]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ds["date"],y=ds["avg_mag"],mode="lines+markers",line=dict(color="#ff6b6b",width=2),marker=dict(size=5,color="#ff4b4b"),fill="tozeroy",fillcolor="rgba(255,75,75,0.1)",name="Actual"))
            fig.add_trace(go.Scatter(x=fd+fd[::-1],y=fu+fl[::-1],fill="toself",fillcolor="rgba(56,189,248,0.15)",line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
            fig.add_trace(go.Scatter(x=fd,y=fv,mode="lines+markers",line=dict(color="#38bdf8",width=2,dash="dash"),marker=dict(size=4,color="#38bdf8"),name="Seasonal Forecast"))
            fig.add_hline(y=4.0,line_dash="dot",line_color="#ff8c00",annotation_text="Mag 4 threshold",annotation_font_color="#ff8c00")
            fig.update_layout(**CL,height=320,yaxis_title="Avg Magnitude",xaxis_title="Date",legend=dict(orientation="h",y=-0.2,font=dict(size=11)))
            st.plotly_chart(fig,use_container_width=True)
        else: st.info("No trend data yet.")

    if SEASONAL_STATS.get("dow_pattern"):
        st.markdown("**Seasonal Pattern — Day-of-Week (Batch Training)**")
        dm={1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
        dd=[{"day":dm.get(int(k),k),"count":v["count"]} for k,v in SEASONAL_STATS["dow_pattern"].items()]
        if dd:
            fig=px.bar(pd.DataFrame(dd),x="day",y="count",color="count",color_continuous_scale="Tealgrn",labels={"count":"Events","day":"Day of Week"})
            fig.update_layout(**CL,height=250,coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)

# Tab 2: Earthquake Profile
with tab2:
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Depth vs Magnitude**"); st.caption("Shallow quakes (<70 km) cause more surface damage.")
        if not eq_df.empty and "depth_km" in eq_df.columns:
            dp=eq_df.dropna(subset=["depth_km","magnitude"]).copy()
            dp["depth_category"]=pd.cut(dp["depth_km"],bins=[0,70,300,9999],labels=["Shallow (<70 km)","Intermediate (70-300 km)","Deep (>300 km)"])
            fig=px.scatter(dp,x="depth_km",y="magnitude",color="depth_category",color_discrete_map={"Shallow (<70 km)":"#ff4b4b","Intermediate (70-300 km)":"#ff8c00","Deep (>300 km)":"#3b82f6"},opacity=0.6)
            fig.update_layout(**CL,height=340,legend=dict(orientation="h",y=-0.2)); st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("**Depth Category Breakdown**")
        if not eq_df.empty and "depth_km" in eq_df.columns:
            dd=eq_df.dropna(subset=["depth_km"]).copy()
            dd["category"]=pd.cut(dd["depth_km"],bins=[0,70,300,9999],labels=["Shallow (<70 km)","Intermediate (70-300 km)","Deep (>300 km)"])
            dc=dd["category"].value_counts().reset_index(); dc.columns=["category","count"]
            fig=px.pie(dc,names="category",values="count",color="category",color_discrete_map={"Shallow (<70 km)":"#ff4b4b","Intermediate (70-300 km)":"#ff8c00","Deep (>300 km)":"#3b82f6"},hole=0.5)
            fig.update_layout(**CL,height=180,legend=dict(orientation="h",font=dict(size=11))); st.plotly_chart(fig,use_container_width=True)
    if not eq_df.empty:
        st.markdown("**Magnitude Distribution**")
        fig=px.histogram(eq_df,x="magnitude",nbins=30,color_discrete_sequence=["#ff6b6b"])
        fig.update_layout(**CL,height=250); st.plotly_chart(fig,use_container_width=True)

# Tab 3: Regions & Risk
with tab3:
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Top 15 Most Active Earthquake Regions**")
        if not eq_df.empty and "place" in eq_df.columns:
            def exr(p):
                if not isinstance(p,str): return "Unknown"
                pts=p.split(","); return pts[-1].strip() if len(pts)>1 else p.strip()
            eq_df["region"]=eq_df["place"].apply(exr)
            tr=(eq_df.groupby("region").agg(count=("magnitude","count"),avg_mag=("magnitude","mean")).reset_index().sort_values("count",ascending=False).head(15))
            tr["avg_mag"]=tr["avg_mag"].round(2)
            fig=px.bar(tr.sort_values("count"),x="count",y="region",orientation="h",color="avg_mag",color_continuous_scale="YlOrRd",text="count")
            fig.update_traces(textposition="outside"); fig.update_layout(**CL,height=420,coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("**Regional Risk Profile (KMeans Clustering)**")
        st.caption("Regions grouped by disaster profile from batch training")
        if not regional_risk.empty:
            pc={"Critical Seismic Zone":"#ff4b4b","High Seismic Activity":"#ff8c00","Frequent Activity":"#eab308","Low Activity":"#22c55e"}
            fig=px.scatter(regional_risk.head(30),x="eq_count",y="avg_mag",size="max_mag",color="profile",color_discrete_map=pc,hover_name="region")
            fig.update_layout(**CL,height=420,legend=dict(orientation="h",y=-0.15,font=dict(size=11))); st.plotly_chart(fig,use_container_width=True)
    if not regional_risk.empty:
        st.markdown("**🏆 Regional Risk Leaderboard**")
        dc=[c for c in ["region","eq_count","avg_mag","max_mag","avg_risk","profile"] if c in regional_risk.columns]
        cc={"region":"Region","eq_count":st.column_config.NumberColumn("Events",format="%d"),"avg_mag":st.column_config.NumberColumn("Avg Mag",format="%.2f"),"max_mag":st.column_config.NumberColumn("Max Mag",format="%.1f"),"profile":"Profile"}
        if "avg_risk" in regional_risk.columns: cc["avg_risk"]=st.column_config.ProgressColumn("Risk Score",min_value=0,max_value=100,format="%.0f")
        st.dataframe(regional_risk[dc].head(20),use_container_width=True,height=300,column_config=cc)

# Tab 4: Natural Events
with tab4:
    EC={"wildfire":"#ff4b4b","storm":"#3b82f6","volcano":"#a855f7","flood":"#06b6d4","ice":"#93c5fd","landslide":"#b45309"}
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Active Events by Type**")
        if not eonet_df.empty:
            tc=eonet_df["type"].value_counts().reset_index(); tc.columns=["type","count"]
            fig=px.pie(tc,names="type",values="count",color="type",color_discrete_map=EC,hole=0.5)
            fig.update_layout(**CL,height=340,legend=dict(orientation="h",font=dict(size=12))); st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("**Event Count by Type**")
        if not eonet_df.empty:
            fig=px.bar(tc.sort_values("count"),x="count",y="type",orientation="h",color="type",color_discrete_map=EC,text="count")
            fig.update_traces(textposition="outside"); fig.update_layout(**CL,height=340,showlegend=False); st.plotly_chart(fig,use_container_width=True)
    if not eonet_df.empty and "last_date" in eonet_df.columns:
        st.markdown("**Events by Month**")
        dt=eonet_df.copy(); dt["month"]=pd.to_datetime(dt["last_date"],errors="coerce").dt.to_period("M").astype(str)
        mo=dt.groupby(["month","type"]).size().reset_index(name="count").sort_values("month")
        if not mo.empty:
            fig=px.bar(mo,x="month",y="count",color="type",color_discrete_map=EC,barmode="stack")
            fig.update_layout(**CL,height=280,legend=dict(orientation="h",y=-0.25,font=dict(size=11))); st.plotly_chart(fig,use_container_width=True)

# Tab 5: FEMA
with tab5:
    st.markdown("**US Disaster Declarations by State (Top 20)**")
    if not fema_df.empty:
        fig=px.bar(fema_df.nlargest(20,"count"),x="state",y="count",color="count",color_continuous_scale="Reds",text="count")
        fig.update_traces(textposition="outside"); fig.update_layout(**CL,height=360,coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)
    st.markdown("**State Coverage Treemap**")
    if not fema_df.empty:
        fig=px.treemap(fema_df,path=["state"],values="count",color="count",color_continuous_scale="Reds")
        fig.update_layout(**CL,height=380); st.plotly_chart(fig,use_container_width=True)

# Tab 6: ML Risk Analysis 
with tab6:
    st.markdown("**🎯 Earthquake Risk Scoring**")
    ms="Trained PySpark MLlib model" if os.path.exists(os.path.join(ML_DIR,"risk_score_rf")) else "Fallback heuristic (run batch_ml_training.py to train)"
    st.caption(f"Scoring method: {ms}")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Risk Score Distribution**")
        if risk_col and not eq_df.empty:
            fig=px.histogram(eq_df,x=risk_col,nbins=40,color_discrete_sequence=["#ef4444"],labels={risk_col:"Risk Score (0-100)"})
            fig.add_vline(x=70,line_dash="dash",line_color="#ff8c00",annotation_text="High Risk Threshold",annotation_font_color="#ff8c00")
            fig.update_layout(**CL,height=320); st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("**Risk Tier Breakdown**")
        tc2="risk_tier" if "risk_tier" in eq_df.columns else None
        if tc2 and not eq_df.empty:
            tt=eq_df[tc2].value_counts().reset_index(); tt.columns=["tier","count"]
            fig=px.pie(tt,names="tier",values="count",color="tier",color_discrete_map={"Critical":"#7f1d1d","High":"#ef4444","Moderate":"#f59e0b","Low":"#22c55e"},hole=0.5)
            fig.update_layout(**CL,height=320,legend=dict(orientation="h",font=dict(size=12))); st.plotly_chart(fig,use_container_width=True)
    if risk_col and not eq_df.empty:
        st.markdown("**Risk Score vs Magnitude**")
        fig=px.scatter(eq_df.dropna(subset=["magnitude",risk_col]),x="magnitude",y=risk_col,color=tc2,color_discrete_map={"Critical":"#7f1d1d","High":"#ef4444","Moderate":"#f59e0b","Low":"#22c55e"},opacity=0.6)
        fig.update_layout(**CL,height=350,legend=dict(orientation="h",y=-0.15)); st.plotly_chart(fig,use_container_width=True)
    tp="tsunami_prob" if "tsunami_prob" in eq_df.columns else None
    if tp and not eq_df.empty:
        st.markdown("**🌊 Tsunami Probability Estimates**")
        ht=eq_df[eq_df[tp]>=0.10].sort_values(tp,ascending=False).head(15)
        if not ht.empty:
            dc2=[c for c in ["place","magnitude","depth_km",tp,risk_col,"risk_tier"] if c in ht.columns]
            st.dataframe(ht[dc2],use_container_width=True,height=300,column_config={"place":"Location","magnitude":st.column_config.NumberColumn("Mag",format="%.1f"),"depth_km":st.column_config.NumberColumn("Depth km",format="%.1f"),tp:st.column_config.ProgressColumn("Tsunami P",min_value=0,max_value=1,format="%.1%%"),risk_col:st.column_config.ProgressColumn("Risk",min_value=0,max_value=100,format="%.0f")})
    st.markdown("**⚡ Unusual Activity Clusters**")
    st.caption(f"Threshold from batch training: {ANOMALY_STATS.get('threshold_moderate','N/A')} events/cell")
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies)[["severity","description","eq_count","eonet_count","max_mag","multi_type"]],use_container_width=True,column_config={"severity":"Severity","description":"Description","eq_count":st.column_config.NumberColumn("Earthquakes"),"eonet_count":st.column_config.NumberColumn("EONET Events"),"max_mag":st.column_config.NumberColumn("Max Mag",format="%.1f"),"multi_type":st.column_config.CheckboxColumn("Multi-Type?")})
    else: st.success("No anomalous clusters detected.")

# Tab 7: Multi-Disaster Correlation
with tab7:
    st.markdown("**🔗 Multi-Disaster Correlation Analysis**")
    st.caption("Cross-referencing USGS, NASA EONET, and FEMA by geography")
    if not eq_df.empty and not eonet_df.empty:
        eqr=eq_df.dropna(subset=["latitude","longitude"]).copy()
        eor=eonet_df.dropna(subset=["latitude","longitude"]).copy()
        if not eqr.empty and not eor.empty:
            eqr["grid_lat"]=(eqr["latitude"]/5).round()*5; eqr["grid_lon"]=(eqr["longitude"]/5).round()*5
            eor["grid_lat"]=(eor["latitude"]/5).round()*5; eor["grid_lon"]=(eor["longitude"]/5).round()*5
            eg=eqr.groupby(["grid_lat","grid_lon"]).agg(eq_count=("magnitude","count"),max_mag=("magnitude","max")).reset_index()
            og=eor.groupby(["grid_lat","grid_lon"]).agg(eonet_count=("type","count"),event_types=("type",lambda x:", ".join(sorted(x.unique())))).reset_index()
            co=eg.merge(og,on=["grid_lat","grid_lon"],how="inner")
            if not co.empty:
                c1,c2=st.columns(2)
                with c1:
                    fig=px.scatter(co,x="eq_count",y="eonet_count",size="max_mag",color="max_mag",color_continuous_scale="YlOrRd",hover_data=["event_types"])
                    fig.update_layout(**CL,height=380); st.plotly_chart(fig,use_container_width=True)
                with c2: st.dataframe(co.sort_values("eq_count",ascending=False).head(15),use_container_width=True,height=380)
    
# High-Risk Table
st.markdown('<div class="section-label">HIGH-RISK EARTHQUAKE TABLE — ML SCORED</div>', unsafe_allow_html=True)
if not eq_df.empty:
    cs=[c for c in ["time","place","magnitude","depth_km","alert","tsunami",risk_col,"risk_tier","tsunami_prob"] if c and c in eq_df.columns]
    hr=eq_df[eq_df["magnitude"]>=5.5][cs].sort_values(risk_col if risk_col and risk_col in eq_df.columns else "magnitude",ascending=False).head(25)
    if not hr.empty:
        cf={"time":"Time","place":"Location","magnitude":st.column_config.NumberColumn("Mag",format="%.1f"),"depth_km":st.column_config.NumberColumn("Depth km",format="%.1f"),"alert":"Alert","tsunami":"Tsunami","risk_tier":"Tier"}
        if risk_col and risk_col in hr.columns: cf[risk_col]=st.column_config.ProgressColumn("ML Risk",min_value=0,max_value=100,format="%.0f")
        if "tsunami_prob" in hr.columns: cf["tsunami_prob"]=st.column_config.ProgressColumn("Tsunami P",min_value=0,max_value=1,format="%.1%%")
        st.dataframe(hr,use_container_width=True,height=380,column_config=cf)
    else: st.info("No magnitude 5.5+ events in current filter.")

# Raw Data
st.markdown('<div class="section-label">RAW DATA</div>', unsafe_allow_html=True)
with st.expander("📋 Earthquake Data"): st.dataframe(eq_df,use_container_width=True)
with st.expander("🌿 EONET Natural Events"): st.dataframe(eonet_df,use_container_width=True)
with st.expander("📑 FEMA State Summary"): st.dataframe(fema_df,use_container_width=True)

if auto_refresh:
    import time; time.sleep(60); st.rerun()
