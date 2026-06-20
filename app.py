"""
EventShield — AI-Powered Event Traffic Management System
Gridlock Hackathon 2.0 | Flipkart × Bengaluru Traffic Police
Theme: Event-Driven Congestion (Planned & Unplanned)

All ML models are validated via 5-fold cross-validation. Metrics shown in-app
are real out-of-fold scores (see metrics.json), not illustrative placeholders.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle, json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EventShield — Traffic Intelligence",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0f172a; }
    .alert-high { background: #fee2e2; border-left: 4px solid #dc2626; padding: 1rem; border-radius: 8px; }
    .alert-med { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px; }
    .alert-low { background: #d1fae5; border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px; }
    .rec-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem; }
    .model-badge { background: #ecfdf5; border: 1px solid #10b981; color: #065f46;
        padding: 0.3rem 0.7rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
    .stTabs [data-baseweb="tab"] { background: #f1f5f9; border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("event_data.csv")
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce', utc=True)
    df['hour'] = df['start_dt'].dt.hour
    df['dow'] = df['start_dt'].dt.dayofweek
    df['month'] = df['start_dt'].dt.month
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 21) | (df['hour'] <= 5)).astype(int)
    df['resolution_min'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60
    df.loc[df['resolution_min'] <= 0, 'resolution_min'] = np.nan
    df['hour_bucket'] = pd.cut(df['hour'], bins=[0, 6, 10, 16, 20, 24],
                               labels=['night', 'morning', 'midday', 'evening', 'lateeve'],
                               include_lowest=True)
    return df


@st.cache_resource
def load_models():
    return pickle.load(open("models.pkl", "rb"))


@st.cache_data
def load_metrics():
    return json.load(open("metrics.json"))


def encode_safe(le, value, default=0):
    return le.transform([value])[0] if value in le.classes_ else default


def predict_event(models, event_cause, corridor, police_station, veh_type,
                  lat, lon, hour, dow, is_weekend, is_night):
    f = [hour, dow, is_weekend, is_night,
         encode_safe(models['le_cause'], event_cause),
         encode_safe(models['le_corr'], corridor),
         encode_safe(models['le_stat'], police_station),
         encode_safe(models['le_veh'], veh_type),
         lat, lon]
    X = np.array([f])
    return {
        'long_duration_prob': float(models['clf_long'].predict_proba(X)[0][1]),
        'high_impact_prob': float(models['clf_hi'].predict_proba(X)[0][1]),
    }


def forecast_events(models, police_station, dow, hour_bucket):
    ps = encode_safe(models['le_ps'], police_station)
    hb = encode_safe(models['le_hb'], hour_bucket)
    X = np.array([[ps, dow, hb]])
    return float(np.expm1(models['reg_count'].predict(X)[0]))


def generate_resource_plan(event_cause, long_prob, high_prob, lat, lon,
                           corridor, is_weekend, hour):
    impact = 0.5 * high_prob + 0.5 * long_prob
    if impact > 0.65:
        base, barricades = 22, 7
    elif impact > 0.45:
        base, barricades = 14, 4
    elif impact > 0.30:
        base, barricades = 8, 2
    else:
        base, barricades = 4, 1
    if is_weekend:
        base = int(base * 0.85)
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        base = int(base * 1.35); barricades += 2
    if event_cause in ['public_event', 'protest', 'procession']:
        base = int(base * 1.4); barricades += 2
    if event_cause == 'vip_movement':
        base = int(base * 1.7); barricades += 3
    traffic_officers = max(2, int(base * 0.4))
    ground_staff = max(1, int(base * 0.3))
    patrol_vehicles = max(1, int(base / 8))
    tow_trucks = 1 if event_cause in ['vehicle_breakdown', 'accident'] else 0
    diversions = []
    if high_prob > 0.5 or event_cause in ['public_event', 'vip_movement', 'protest', 'procession']:
        if corridor != 'Non-corridor':
            diversions.append(f"Primary diversion before {corridor} entry; signage at 500m and 1km")
        diversions.append("Activate Google Maps / Waze incident reporting for live rerouting")
        diversions.append("Coordinate with BMTC for temporary bus route modifications")
        if impact > 0.6:
            diversions.append("Deploy mobile VMS at 3 approach roads")
            diversions.append("Adjust signal timing at adjacent junctions via ATCS")
    checklist = [f"Deploy {barricades} barricade sets at perimeter",
                 f"Assign {traffic_officers} officers to key intersections"]
    if long_prob > 0.5:
        checklist.append("Pre-position relief team — event likely to exceed 2 hours")
    if event_cause in ['public_event', 'procession', 'protest']:
        checklist += ["Coordinate crowd management with organizers", "Ensure ambulance standby"]
    if tow_trucks:
        checklist.append("Dispatch tow truck to standby position")
    checklist.append("Notify control room and update live dashboard")
    return {'total_personnel': base, 'traffic_officers': traffic_officers,
            'ground_staff': ground_staff, 'patrol_vehicles': patrol_vehicles,
            'tow_trucks': tow_trucks, 'barricade_sets': barricades,
            'diversions': diversions, 'checklist': checklist, 'impact_index': impact}


df = load_data()
models = load_models()
metrics = load_metrics()

st.sidebar.markdown("## 🛡️ EventShield")
st.sidebar.caption("AI-Powered Event Traffic Management")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard", "🔮 Impact Predictor", "📡 Hotspot Forecast",
    "👮 Resource Planner", "📈 Analytics", "🧪 Model Card"
], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data:** {len(df):,} Bengaluru events")
st.sidebar.markdown("**Period:** Nov 2023 – Apr 2024")
st.sidebar.markdown("**Source:** ASTRAM / BTP")


if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">🛡️ EventShield Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Operational intelligence for Bengaluru event-driven congestion")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events", f"{len(df):,}")
    c2.metric("Active Now", f"{len(df[df['status']=='active'])}")
    c3.metric("Road Closures", f"{int(df['requires_road_closure'].sum())}")
    c4.metric("Planned Events", f"{len(df[df['event_type']=='planned'])}")
    c5.metric("Median Clear Time", f"{df['resolution_min'].median():.0f}m")
    st.markdown("---")
    col_map, col_feed = st.columns([3, 2])
    with col_map:
        st.subheader("🗺️ Event Map")
        causes = st.multiselect("Filter by cause", sorted(df['event_cause'].unique()),
                                default=['accident', 'public_event', 'procession', 'vip_movement', 'protest'])
        mdf = df[df['event_cause'].isin(causes)] if causes else df
        fig = px.scatter_mapbox(mdf.head(2000), lat='latitude', lon='longitude',
                                color='event_cause', zoom=10.5,
                                center={"lat": 12.97, "lon": 77.59},
                                mapbox_style="carto-positron", height=520,
                                hover_data=['address', 'priority', 'corridor'])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True)
    with col_feed:
        st.subheader("📋 Recent Events")
        for _, r in df.sort_values('start_dt', ascending=False).head(12).iterrows():
            cls = 'alert-high' if r['priority'] == 'High' else 'alert-med'
            addr = str(r['address'])[:70] if pd.notna(r['address']) else 'Unknown'
            t = r['start_dt'].strftime('%b %d, %H:%M') if pd.notna(r['start_dt']) else ''
            st.markdown(f"""<div class="{cls}" style="margin-bottom:8px;">
                <strong>{r['event_cause'].replace('_',' ').title()}</strong>
                {"🚧" if r['requires_road_closure'] else ""}<br>
                <small>{addr}</small><br><small>🕐 {t} | 📍 {r['corridor']}</small></div>""",
                unsafe_allow_html=True)


elif page == "🔮 Impact Predictor":
    st.markdown('<h1 class="main-header">🔮 Event Impact Predictor</h1>', unsafe_allow_html=True)
    st.caption("Predict whether an event will be high-impact or prolonged — before deploying resources")
    st.markdown(f"""<span class="model-badge">✓ Prolonged model: AUC {metrics['long_duration']['roc_auc']} · F1 {metrics['long_duration']['f1']}</span>
                &nbsp; <span class="model-badge">✓ High-impact model: AUC {metrics['high_impact']['roc_auc']} · F1 {metrics['high_impact']['f1']}</span>""",
                unsafe_allow_html=True)
    st.markdown("---")
    ci, cr = st.columns(2)
    with ci:
        st.subheader("Event Details")
        ec = st.selectbox("Event Type", sorted(df['event_cause'].unique()))
        corr = st.selectbox("Corridor", sorted(df['corridor'].dropna().unique()))
        ps = st.selectbox("Police Station", sorted(df['police_station'].unique()))
        vt = st.selectbox("Vehicle Type (if applicable)",
                          ['Unknown'] + sorted([v for v in df['veh_type'].dropna().unique()]))
        d1, d2 = st.columns(2)
        edate = d1.date_input("Date", datetime.now())
        ehour = d2.slider("Hour", 0, 23, 10)
        l1, l2 = st.columns(2)
        lat = l1.number_input("Latitude", value=12.97, format="%.4f")
        lon = l2.number_input("Longitude", value=77.59, format="%.4f")
        go_btn = st.button("🔮 Predict Impact", type="primary", use_container_width=True)
    with cr:
        if go_btn:
            dow = edate.weekday()
            isw = 1 if dow >= 5 else 0
            isn = 1 if ehour >= 21 or ehour <= 5 else 0
            p = predict_event(models, ec, corr, ps, vt, lat, lon, ehour, dow, isw, isn)
            st.subheader("Prediction")
            m1, m2 = st.columns(2)
            with m1:
                lp = p['long_duration_prob']
                lvl = "Likely" if lp > 0.5 else "Unlikely"
                clr = "#dc2626" if lp > 0.5 else "#10b981"
                st.markdown(f"""<div style="background:{clr}15;border:2px solid {clr};
                    border-radius:12px;padding:1.2rem;text-align:center;">
                    <p style="margin:0;color:#64748b;font-size:0.85rem;">PROLONGED (&gt;2h)</p>
                    <h2 style="margin:0;color:{clr};">{lp:.0%}</h2>
                    <p style="margin:0;color:{clr};font-weight:600;">{lvl}</p></div>""",
                    unsafe_allow_html=True)
            with m2:
                hp = p['high_impact_prob']
                lvl = "High Impact" if hp > 0.5 else "Routine"
                clr = "#ea580c" if hp > 0.5 else "#10b981"
                st.markdown(f"""<div style="background:{clr}15;border:2px solid {clr};
                    border-radius:12px;padding:1.2rem;text-align:center;">
                    <p style="margin:0;color:#64748b;font-size:0.85rem;">IMPACT LEVEL</p>
                    <h2 style="margin:0;color:{clr};">{hp:.0%}</h2>
                    <p style="margin:0;color:{clr};font-weight:600;">{lvl}</p></div>""",
                    unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=(0.5 * hp + 0.5 * lp) * 100,
                title={'text': "Combined Impact Index"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#7c3aed"},
                       'steps': [{'range': [0, 30], 'color': '#d1fae5'},
                                 {'range': [30, 60], 'color': '#fef3c7'},
                                 {'range': [60, 100], 'color': '#fee2e2'}]}))
            fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.session_state['pred'] = {'event_cause': ec, 'corridor': corr,
                'lat': lat, 'lon': lon, 'hour': ehour, 'is_weekend': isw,
                'long_prob': lp, 'high_prob': hp}
            st.info("→ See **Resource Planner** for the deployment recommendation.")
        else:
            st.info("Enter event details and click Predict.")


elif page == "📡 Hotspot Forecast":
    st.markdown('<h1 class="main-header">📡 Spatial-Temporal Hotspot Forecast</h1>', unsafe_allow_html=True)
    st.caption("Predict event load per police station and time window — for proactive resource pre-positioning")
    st.markdown(f"""<span class="model-badge">✓ Forecast model: R² {metrics['event_forecast']['r2']} ·
                {metrics['event_forecast']['improvement_pct']}% better than baseline ·
                MAE {metrics['event_forecast']['mae']} events</span>""", unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Forecast Window")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_name = st.selectbox("Day of Week", days)
        dow = days.index(dow_name)
        hb_name = st.selectbox("Time Window",
            ['night (12-6am)', 'morning (6-10am)', 'midday (10am-4pm)',
             'evening (4-8pm)', 'lateeve (8pm-12am)'])
        hb = hb_name.split(' ')[0]
        st.markdown(f"**Forecasting:** {dow_name}, {hb_name}")
    stations = sorted(df['police_station'].unique())
    rows = []
    for s in stations:
        cnt = forecast_events(models, s, dow, hb)
        sample = df[df['police_station'] == s].iloc[0]
        rows.append({'station': s, 'predicted_events': cnt,
                     'lat': sample['latitude'], 'lon': sample['longitude']})
    fc = pd.DataFrame(rows).sort_values('predicted_events', ascending=False)
    with c2:
        st.subheader("🔥 Predicted Hotspot Map")
        fig = px.scatter_mapbox(fc, lat='lat', lon='lon', size='predicted_events',
                                color='predicted_events', color_continuous_scale='YlOrRd',
                                size_max=25, zoom=10, hover_name='station',
                                center={"lat": 12.97, "lon": 77.59},
                                mapbox_style="carto-positron", height=400)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("📊 Top 15 Predicted Hotspots")
    top = fc.head(15)
    fig2 = px.bar(top, x='predicted_events', y='station', orientation='h',
                  color='predicted_events', color_continuous_scale='OrRd')
    fig2.update_layout(height=400, showlegend=False, yaxis={'autorange': 'reversed'})
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Stations with the highest predicted load should receive pre-positioned patrol units for this window.")


elif page == "👮 Resource Planner":
    st.markdown('<h1 class="main-header">👮 Resource Deployment Planner</h1>', unsafe_allow_html=True)
    st.caption("Rule-based deployment engine, driven by validated impact predictions")
    if 'pred' not in st.session_state:
        st.warning("Run the **Impact Predictor** first to generate a plan.")
        st.stop()
    pr = st.session_state['pred']
    plan = generate_resource_plan(pr['event_cause'], pr['long_prob'], pr['high_prob'],
                                  pr['lat'], pr['lon'], pr['corridor'], pr['is_weekend'], pr['hour'])
    st.markdown(f"**Event:** {pr['event_cause'].replace('_',' ').title()} at {pr['corridor']} "
                f"| **Impact index:** {plan['impact_index']:.2f} "
                f"| **Prolonged risk:** {pr['long_prob']:.0%}")
    st.info("Personnel and barricade counts are configurable operational defaults adjusted by "
            "predicted impact — not ML outputs. The predictions driving them are validated (see Model Card).")
    st.markdown("---")
    st.subheader("📦 Recommended Allocation")
    rc = st.columns(5)
    rc[0].metric("👮 Personnel", plan['total_personnel'])
    rc[1].metric("🚦 Officers", plan['traffic_officers'])
    rc[2].metric("👷 Ground Staff", plan['ground_staff'])
    rc[3].metric("🚔 Patrol Vehicles", plan['patrol_vehicles'])
    rc[4].metric("🚧 Barricades", plan['barricade_sets'])
    st.markdown("---")
    cd, cc = st.columns(2)
    with cd:
        st.subheader("🔀 Diversion Plan")
        if plan['diversions']:
            for i, d in enumerate(plan['diversions'], 1):
                st.markdown(f"""<div class="rec-card"><strong>D{i}.</strong> {d}</div>""", unsafe_allow_html=True)
        else:
            st.success("No diversion required at this impact level.")
    with cc:
        st.subheader("✅ Pre-Event Checklist")
        for item in plan['checklist']:
            st.checkbox(item, key=f"chk_{item[:25]}")


elif page == "📈 Analytics":
    st.markdown('<h1 class="main-header">📈 Event Analytics</h1>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["⏰ Temporal", "📍 Hotspots", "📊 Causes", "🛣️ Corridors"])
    dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    with t1:
        st.subheader("Events by Hour")
        h = df.groupby('hour').size().reset_index(name='count')
        st.plotly_chart(px.bar(h, x='hour', y='count', color='count',
                        color_continuous_scale='YlOrRd').update_layout(height=320, showlegend=False),
                        use_container_width=True)
        st.subheader("Heatmap: Hour × Day")
        hd = df.groupby(['dow', 'hour']).size().reset_index(name='count')
        piv = hd.pivot(index='dow', columns='hour', values='count').fillna(0)
        piv.index = [dow_map[i] for i in piv.index]
        st.plotly_chart(px.imshow(piv, color_continuous_scale='YlOrRd', aspect='auto',
                        labels=dict(x="Hour", y="Day", color="Events")).update_layout(height=300),
                        use_container_width=True)
    with t2:
        st.subheader("Event Density Hotspots")
        st.plotly_chart(px.density_mapbox(df, lat='latitude', lon='longitude', radius=12,
                        zoom=10.5, center={"lat": 12.97, "lon": 77.59},
                        mapbox_style="carto-positron", color_continuous_scale='YlOrRd',
                        height=550).update_layout(margin=dict(l=0, r=0, t=0, b=0)),
                        use_container_width=True)
        st.subheader("Top Police Stations")
        ps = df['police_station'].value_counts().head(10).reset_index()
        ps.columns = ['station', 'count']
        st.plotly_chart(px.bar(ps, x='count', y='station', orientation='h', color='count',
                        color_continuous_scale='Reds').update_layout(height=350, showlegend=False,
                        yaxis={'autorange': 'reversed'}), use_container_width=True)
    with t3:
        st.subheader("Event Cause Breakdown")
        cd2 = df['event_cause'].value_counts().reset_index()
        cd2.columns = ['cause', 'count']
        st.plotly_chart(px.treemap(cd2, path=['cause'], values='count', color='count',
                        color_continuous_scale='RdYlBu_r').update_layout(height=400),
                        use_container_width=True)
        st.subheader("Road Closure Rate by Cause")
        rc2 = df.groupby('event_cause').agg(total=('id', 'count'),
             cl=('requires_road_closure', 'sum')).reset_index()
        rc2['rate'] = rc2['cl'] / rc2['total']
        rc2 = rc2.sort_values('rate')
        st.plotly_chart(px.bar(rc2, x='rate', y='event_cause', orientation='h', color='rate',
                        color_continuous_scale='OrRd').update_layout(height=450, showlegend=False,
                        xaxis_tickformat='.0%'), use_container_width=True)
    with t4:
        st.subheader("Events by Corridor")
        co = df['corridor'].value_counts().head(15).reset_index()
        co.columns = ['corridor', 'count']
        st.plotly_chart(px.bar(co, x='corridor', y='count', color='count',
                        color_continuous_scale='Viridis').update_layout(height=350,
                        showlegend=False, xaxis_tickangle=-45), use_container_width=True)


elif page == "🧪 Model Card":
    st.markdown('<h1 class="main-header">🧪 Model Card</h1>', unsafe_allow_html=True)
    st.caption("Honest performance reporting — all metrics are 5-fold cross-validated out-of-fold scores")
    st.markdown("""EventShield ships three validated models. Every number below is computed on held-out
    folds the model never saw during training — no metric here is hand-picked or illustrative.""")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        m = metrics['long_duration']
        st.markdown(f"""<div class="rec-card"><h4>⏱️ Prolonged-Event Classifier</h4>
            <p>Predicts whether an event takes <strong>&gt;2 hours</strong> to clear.</p>
            <ul><li>ROC-AUC: <strong>{m['roc_auc']}</strong></li>
            <li>PR-AUC: <strong>{m['pr_auc']}</strong></li>
            <li>F1: <strong>{m['f1']}</strong></li>
            <li>Base rate: {m['positive_rate']:.0%} | n={m['n']:,}</li></ul></div>""",
            unsafe_allow_html=True)
    with c2:
        m = metrics['high_impact']
        st.markdown(f"""<div class="rec-card"><h4>⚠️ High-Impact Classifier</h4>
            <p>Flags events needing serious response.</p>
            <ul><li>ROC-AUC: <strong>{m['roc_auc']}</strong></li>
            <li>PR-AUC: <strong>{m['pr_auc']}</strong></li>
            <li>F1: <strong>{m['f1']}</strong></li>
            <li>Base rate: {m['positive_rate']:.0%} | n={m['n']:,}</li></ul></div>""",
            unsafe_allow_html=True)
    with c3:
        m = metrics['event_forecast']
        st.markdown(f"""<div class="rec-card"><h4>📡 Hotspot Forecaster</h4>
            <p>Predicts event count per station per window.</p>
            <ul><li>R²: <strong>{m['r2']}</strong></li>
            <li>MAE: <strong>{m['mae']}</strong> events</li>
            <li>vs baseline: <strong>{m['improvement_pct']}%</strong> better</li>
            <li>n={m['n']:,} cells</li></ul></div>""",
            unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Feature Importance — High-Impact Classifier")
    fi = metrics['high_impact']['feature_importance']
    fi_df = pd.DataFrame(fi, columns=['feature', 'importance']).sort_values('importance')
    st.plotly_chart(px.bar(fi_df, x='importance', y='feature', orientation='h', color='importance',
                    color_continuous_scale='Purples').update_layout(height=350, showlegend=False),
                    use_container_width=True)
    st.markdown("---")
    st.subheader("⚠️ Known Limitations")
    st.markdown("""We report these openly rather than overstate the system's reach:

- **Dataset skew:** 94% of records are unplanned incidents. Only ~640 records are true planned
  events (public events, processions, VIP movements, protests), so predictions for large planned
  gatherings (festivals, stadium events) are **lower-confidence**.
- **Resolution-time regression was dropped.** A direct duration regressor scored R²≈0.03 (no better
  than the median) because clear-times are extremely heavy-tailed. We reframed it as a **binary
  prolonged-event classifier (>2h)** — more robust (AUC 0.88) and more operationally useful.
- **Resource numbers are heuristics**, not learned. They are configurable operational defaults scaled
  by validated impact predictions — labeled as such throughout the app.
- **Feature-signal caution:** lat/lon and vehicle type carry strong signal; in production these should
  be monitored to ensure the model generalizes to new locations.""")


st.markdown("---")
st.markdown("""<div style="text-align:center;color:#94a3b8;font-size:0.8rem;">
    EventShield v2.0 | Gridlock Hackathon 2.0 | Validated ML · Honest metrics
    <br>Built with Streamlit, Plotly, scikit-learn</div>""", unsafe_allow_html=True)
