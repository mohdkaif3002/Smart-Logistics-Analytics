
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Smart Logistics v2.0",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0D1B2A, #0891B2);
    padding: 20px; border-radius: 10px;
    color: white; text-align: center; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_model.pkl")
    encoders = {}
    encoder_files = {
        "Route"             : "models/le_route.pkl",
        "Truck_Type"        : "models/le_truck_type.pkl",
        "Weather_Condition" : "models/le_weather_condition.pkl",
        "Road_Condition"    : "models/le_road_condition.pkl",
        "Cargo_Type"        : "models/le_cargo_type.pkl",
        "Exp_Category"      : "models/le_exp_category.pkl",
        "Season"            : "models/le_season.pkl",
    }
    for col, path in encoder_files.items():
        try:
            encoders[col] = joblib.load(path)
        except:
            pass
    with open("models/model_config_v2.json") as f:
        config = json.load(f)
    return model, encoders, config

@st.cache_data
def load_data():
    df            = pd.read_csv("data/india_logistics.csv")
    best_routes   = pd.read_csv("data/best_routes.csv")
    route_metrics = pd.read_csv("data/route_metrics_final.csv")
    return df, best_routes, route_metrics

model, encoders, config = load_model()
df, best_routes, route_metrics = load_data()

# Sidebar
st.sidebar.markdown("""
<div style="text-align:center; padding:10px;
     background:#0D1B2A; border-radius:8px;">
<h2 style="color:#0891B2;">🚛 Smart Logistics</h2>
<p style="color:#94a3b8; font-size:12px;">
India Heavy Freight Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background:#1a2744; border-radius:10px; padding:14px;">
<p style="color:#0891B2; font-weight:700; font-size:14px; margin:0 0 12px 0;">👥 Team — Group 82</p>
<table style="width:100%; border-collapse:collapse;">
<tr><td style="color:white; font-size:12px; padding:4px 0;">🔹 <b>Mohammad Kaif</b></td></tr>
<tr><td style="color:#94a3b8; font-size:11px; padding:0 0 8px 18px;">SAP: 1000018249</td></tr>
<tr><td style="color:white; font-size:12px; padding:4px 0;">🔹 <b>Harshita Hoiyani</b></td></tr>
<tr><td style="color:#94a3b8; font-size:11px; padding:0 0 8px 18px;">SAP: 1000018489</td></tr>
<tr><td style="color:white; font-size:12px; padding:4px 0;">🔹 <b>Ansh Mittal</b></td></tr>
<tr><td style="color:#94a3b8; font-size:11px; padding:0 0 4px 18px;">SAP: 1000018268</td></tr>
</table>
<hr style="border-color:#2d3748; margin:10px 0;">
<p style="color:#64748b; font-size:11px; text-align:center; margin:0;">UCF 439 | Capstone | JAN–MAY 2026</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("UCF 439 | Capstone | JAN-MAY 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔮 Delay Predictor",
    "🗺️ Route Recommender",
    "📈 Model Performance",
    "💡 Insights"
])

# ── PAGE 1: DASHBOARD ──
if page == "📊 Dashboard":
    st.markdown("""
    <div class="main-header">
    <h1>🚛 Smart Logistics Analytics System</h1>
    <p>India Heavy Freight — Performance Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Total Shipments", f"{len(df):,}")
    col2.metric("Delayed",
                f"{df['Delay_Flag'].sum():,}",
                f"{df['Delay_Flag'].mean()*100:.1f}%")
    col3.metric("Avg Freight Cost",
                f"Rs.{df['Freight_Cost_INR'].mean():,.0f}")
    col4.metric("Avg Distance",
                f"{df['Distance_km'].mean():,.0f} km")
    col5.metric("Routes", f"{df['Route'].nunique()}")

    st.markdown("---")

    # Filters
    col1,col2,col3 = st.columns(3)
    with col1:
        sel_year = st.selectbox("Year",
            ["All"]+sorted(df["Year"].unique().tolist()))
    with col2:
        sel_truck = st.selectbox("Truck Type",
            ["All"]+df["Truck_Type"].unique().tolist())
    with col3:
        sel_route = st.selectbox("Route",
            ["All"]+sorted(df["Route"].unique().tolist()))

    df_f = df.copy()
    if sel_year  != "All":
        df_f = df_f[df_f["Year"]==int(sel_year)]
    if sel_truck != "All":
        df_f = df_f[df_f["Truck_Type"]==sel_truck]
    if sel_route != "All":
        df_f = df_f[df_f["Route"]==sel_route]

    st.markdown(f"**Showing {len(df_f):,} shipments**")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Delay Rate by Truck Type")
        truck_d2 = df_f.groupby("Truck_Type")["Delay_Flag"].mean()*100
        truck_d2 = truck_d2.sort_values(ascending=False).reset_index()
        truck_d2.columns = ["Truck_Type","Delay_Rate"]
        truck_colors = {"Heavy Truck":"#e74c3c",
                        "Medium Truck":"#f39c12",
                        "Light Van":"#2ecc71"}
        fig_t = px.bar(
            truck_d2, x="Truck_Type", y="Delay_Rate",
            color="Truck_Type",
            color_discrete_map=truck_colors,
            text=truck_d2["Delay_Rate"].round(1).astype(str)+"%",
            labels={"Truck_Type":"Truck Type","Delay_Rate":"Delay Rate (%)"})
        fig_t.update_traces(textposition="outside")
        fig_t.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            font_color="white", showlegend=False,
            margin=dict(t=10,b=20,l=10,r=10))
        st.plotly_chart(fig_t, use_container_width=True)
    with col2:
        st.subheader("Monthly Delay Trend")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df_f.groupby("Month_Name")["Delay_Flag"].mean()*100
        monthly = monthly.reindex(
            [m for m in month_order if m in monthly.index])
        st.line_chart(monthly)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Delayed Routes")
        route_d = df_f.groupby("Route")["Delay_Flag"].mean()*100
        route_top = route_d.sort_values(ascending=False).head(5).reset_index()
        route_top.columns = ["Route","Delay_Rate"]
        fig_rt = px.bar(
            route_top, x="Route", y="Delay_Rate",
            color="Delay_Rate",
            color_continuous_scale="RdYlGn_r",
            text=route_top["Delay_Rate"].round(1).astype(str)+"%",
            labels={"Route":"Route","Delay_Rate":"Delay Rate (%)"})
        fig_rt.update_traces(textposition="outside")
        fig_rt.update_coloraxes(showscale=False)
        fig_rt.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            font_color="white", showlegend=False,
            margin=dict(t=10,b=60,l=10,r=10))
        st.plotly_chart(fig_rt, use_container_width=True)
    with col2:
        st.subheader("Delay by Weather Condition")
        weather_d = df_f.groupby(
            "Weather_Condition")["Delay_Flag"].mean()*100
        weather_d2 = weather_d.sort_values(ascending=False).reset_index()
        weather_d2.columns = ["Weather","Delay_Rate"]
        weather_color_map = {"Clear":"#3498db","Fog":"#95a5a6",
                             "Rain":"#2471a3","Storm":"#e74c3c"}
        fig_w = px.bar(
            weather_d2, x="Weather", y="Delay_Rate",
            color="Weather",
            color_discrete_map=weather_color_map,
            text=weather_d2["Delay_Rate"].round(1).astype(str)+"%",
            labels={"Weather":"Weather","Delay_Rate":"Delay Rate (%)"})
        fig_w.update_traces(textposition="outside")
        fig_w.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            font_color="white", showlegend=False,
            margin=dict(t=10,b=20,l=10,r=10))
        st.plotly_chart(fig_w, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Festival Month Impact")
        fest_data = df_f.groupby(
            "Is_Festival_Month")["Delay_Flag"].mean()*100
        fest_df = pd.DataFrame({
            "Period": ["Non-Festival","Festival Month"],
            "Delay Rate %": [
                fest_data.get(0, 0),
                fest_data.get(1, 0)]
        })
        fig_fest = px.bar(
            fest_df, x="Period", y="Delay Rate %",
            color="Period",
            color_discrete_map={
                "Non-Festival":"#2ecc71",
                "Festival Month":"#e74c3c"},
            text=fest_df["Delay Rate %"].round(1).astype(str)+"%")
        fig_fest.update_traces(textposition="outside")
        fig_fest.update_layout(
            plot_bgcolor="#0D1B2A", paper_bgcolor="#0D1B2A",
            font_color="white", showlegend=False, bargap=0.5,
            margin=dict(t=10,b=20,l=10,r=10))
        st.plotly_chart(fig_fest, use_container_width=True)
    with col2:
        st.subheader("Delay by Cargo Type")
        cargo_d2 = df_f.groupby(
            "Cargo_Type")["Delay_Flag"].mean()*100
        cargo_d2 = cargo_d2.sort_values(ascending=False)
        cargo_d2 = cargo_d2.reset_index()
        cargo_d2.columns = ["Cargo_Type","Delay_Rate"]
        fig_cargo = px.bar(
            cargo_d2, x="Cargo_Type", y="Delay_Rate",
            color="Delay_Rate",
            color_continuous_scale="RdYlGn_r",
            text=cargo_d2["Delay_Rate"].round(1).astype(str)+"%",
            labels={"Cargo_Type":"Cargo Type","Delay_Rate":"Delay Rate (%)"})
        fig_cargo.update_traces(textposition="outside")
        fig_cargo.update_coloraxes(showscale=False)
        fig_cargo.update_layout(
            plot_bgcolor="#0D1B2A",
            paper_bgcolor="#0D1B2A",
            font_color="white",
            showlegend=False,
            margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_cargo, use_container_width=True)

# ── PAGE 2: DELAY PREDICTOR ──
elif page == "🔮 Delay Predictor":
    st.title("🔮 AI-Powered Delivery Delay Predictor")
    st.markdown("Powered by **XGBoost** — 85.6% Recall")
    st.markdown("---")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.subheader("📦 Shipment")
        route = st.selectbox("Route",
            sorted(df["Route"].unique()))
        truck_type = st.selectbox("Truck Type",
            ["Heavy Truck","Medium Truck","Light Van"])
        cargo_type = st.selectbox("Cargo Type",
            ["Electronics","FMCG","Auto Parts",
             "Pharma","Food","Cement","Steel","Textile"])
        distance = st.number_input("Distance (km)",
            100, 2000, value=500, step=50)
        weight = st.number_input("Weight (kg)",
            500, 40000, value=5000, step=500)
        num_stops = st.slider("Number of Stops", 1, 8, 2)

    with col2:
        st.subheader("💰 Cost & Load")
        freight_cost = st.number_input("Freight Cost (Rs.)",
            5000, 500000, value=50000, step=1000)
        cost_per_km  = round(freight_cost/distance, 2)
        st.metric("Auto Cost per KM", f"Rs.{cost_per_km}")
        load_util     = st.slider("Load Utilization (%)", 55, 100, 75)
        expected_days = st.selectbox("Expected Days", [1,2,3,4,5])
        truck_age     = st.slider("Truck Age (years)", 1, 15, 3)

    with col3:
        st.subheader("🌦️ Conditions")
        month = st.selectbox("Month", list(range(1,13)),
            format_func=lambda x:
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        weather    = st.selectbox("Weather",
            ["Clear","Rain","Fog","Storm"])
        road       = st.selectbox("Road Condition",
            ["Good","Average","Poor"])
        driver_exp = st.slider("Driver Experience (yrs)", 1, 20, 5)
        driver_age = st.slider("Driver Age", 22, 58, 30)
        congestion = st.slider("Traffic Congestion (%)", 0, 100, 40)

    st.markdown("---")

    if st.button("🔮 PREDICT DELIVERY STATUS",
                  use_container_width=True, type="primary"):

        season_map = {
            12:"Winter",1:"Winter",2:"Winter",
            3:"Summer",4:"Summer",5:"Summer",
            6:"Monsoon",7:"Monsoon",
            8:"Monsoon",9:"Monsoon",
            10:"Autumn",11:"Autumn"}
        season      = season_map[month]
        is_festival = 1 if month in [1,3,4,8,10,11] else 0
        overload    = 1 if load_util > 90 else 0
        old_truck   = 1 if truck_age > 8 else 0
        high_cong   = 1 if congestion > 70 else 0
        long_route  = 1 if distance > 800 else 0
        heavy_load  = 1 if weight > 15000 else 0
        exp_cat     = ("Junior" if driver_exp < 3 else
                       "Mid" if driver_exp < 8 else "Senior")

        try:
            route_enc   = encoders["Route"].transform([route])[0]
            truck_enc   = encoders["Truck_Type"].transform(
                [truck_type])[0]
            weather_enc = encoders["Weather_Condition"].transform(
                [weather])[0]
            road_enc    = encoders["Road_Condition"].transform(
                [road])[0]
            cargo_enc   = encoders["Cargo_Type"].transform(
                [cargo_type])[0]
            season_enc  = encoders["Season"].transform([season])[0]
        except:
            route_enc=0; truck_enc=0; weather_enc=0
            road_enc=0; cargo_enc=0; season_enc=0

        input_data = np.array([[
            distance, weight, freight_cost, cost_per_km,
            expected_days, load_util, driver_exp,
            driver_age, truck_age, congestion, num_stops,
            is_festival, month, route_enc, truck_enc,
            weather_enc, road_enc, cargo_enc, season_enc,
            overload, old_truck, high_cong,
            long_route, heavy_load
        ]])

        proba      = model.predict_proba(input_data)[0][1]
        prediction = int(proba >= config["threshold"])

        st.markdown("---")
        col1,col2,col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH DELAY RISK")
                st.markdown(
                    f"### Delay Probability: {proba*100:.1f}%")
                st.markdown("**Recommendations:**")
                if weather in ["Storm","Rain"]:
                    st.markdown("- Consider delaying shipment")
                if load_util > 90:
                    st.markdown("- Reduce load utilization")
                if truck_age > 8:
                    st.markdown("- Use a newer truck")
                if congestion > 70:
                    st.markdown("- Avoid peak traffic hours")
                if is_festival:
                    st.markdown("- Plan extra buffer days")
            else:
                st.success("✅ LOW DELAY RISK")
                st.markdown(
                    f"### On-Time Probability: {(1-proba)*100:.1f}%")
                st.markdown("Shipment looks good to go!")

        with col2:
            st.info(f"**Route:** {route}")
            st.info(f"**Truck:** {truck_type}")
            st.info(f"**Cargo:** {cargo_type}")
            st.info(f"**Distance:** {distance} km")
            st.info(f"**Stops:** {num_stops}")

        with col3:
            st.info(f"**Weight:** {weight:,} kg")
            st.info(f"**Cost/km:** Rs.{cost_per_km}")
            st.info(f"**Weather:** {weather}")
            st.info(f"**Season:** {season}")
            st.info(f"**Festival Month:** "
                    f"{'Yes' if is_festival else 'No'}")

# ── PAGE 3: ROUTE RECOMMENDER ──
elif page == "🗺️ Route Recommender":
    st.title("🗺️ Smart Route Recommender")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        sel_month = st.selectbox("Select Month",
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"])
        st.markdown("---")
        st.subheader(f"Best Route for {sel_month}")
        month_data = best_routes[
            best_routes["Month"]==sel_month]
        if not month_data.empty:
            row = month_data.iloc[0]
            st.success(f"### {row['Best_Route']}")
            c1,c2 = st.columns(2)
            c1.metric("Delay Rate", f"{row['Delay_%']:.1f}%")
            c2.metric("Cost/km",    f"Rs.{row['Cost_per_km']:.0f}")
            c1.metric("Avg Days",   f"{row['Avg_Days']:.1f}")
            c2.metric("Score",      f"{row['Score']:.4f}")
        st.markdown("---")
        st.subheader("Monthly Best Routes")
        st.dataframe(best_routes, use_container_width=True)

    with col2:
        st.subheader("All Routes Ranked")
        st.dataframe(
            route_metrics.reset_index()[["Route","Rank","Delay_Rate_%",
                           "Avg_Cost_per_km",
                           "Avg_Delivery_Days",
                           "Efficiency_Score"]
                         ].reset_index(),
            use_container_width=True)
        st.markdown("---")
        st.subheader("Efficiency Score Formula")
        st.code("""
Score = 0.35 x Delay_Rate
      + 0.30 x Cost_per_km
      + 0.20 x Avg_Delivery_Days
      + 0.15 x Avg_Congestion

Lower Score = Better Route
        """)

# ── PAGE 4: MODEL PERFORMANCE ──
elif page == "📈 Model Performance":
    st.title("📈 ML Model Performance")
    st.markdown("---")

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Best Model",   "XGBoost")
    col2.metric("Recall",       f"{config['recall']}%")
    col3.metric("F1 Score",     f"{config['f1_score']}%")
    col4.metric("AUC",          f"{config['auc']}")
    col5.metric("Dataset Rows", "30,000")

    st.markdown("---")
    st.subheader("Model Comparison Table")
    comparison = pd.DataFrame({
        "Model"    : ["Logistic Regression",
                      "Random Forest + SMOTE",
                      "LightGBM + SMOTE",
                      "CatBoost + SMOTE",
                      "Voting Ensemble",
                      "XGBoost + SMOTE (Best)"],
        "Recall %"  : [48.86, 64.5, 34.1, 35.5, 63.2, 85.6],
        "F1 %"      : [22.05, 36.7, 33.5, 34.8, 38.6, 38.4],
        "AUC"       : [0.500, 0.598, 0.636, 0.644, 0.632, 0.617],
        "Status"    : ["Baseline","Good","Average",
                       "Average","Good","BEST ✅"]
    })
    st.dataframe(comparison, use_container_width=True)

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Why XGBoost Won")
        st.markdown("""
        - **85.6% Recall** — catches 85 out of 100 delays
        - Handles class imbalance well with SMOTE
        - Built-in feature importance
        - Industry standard for tabular data
        - Faster training than ensemble approaches
        """)
    with col2:
        st.subheader("Training Details")
        st.markdown(f"""
        - **Dataset:** 30,000 shipments
        - **Features:** 24 (including new ones)
        - **Split:** 80% train / 20% test
        - **Imbalance fix:** SMOTE oversampling
        - **Threshold:** {config["threshold"]}
        - **New features:** Festival month, congestion,
          cargo type, truck age, driver age, num stops
        """)

    st.markdown("---")
    st.subheader("New Features in v2.0")
    new_feats = {
        "Is_Festival_Month" : "Flags Oct/Nov/Jan high-demand months",
        "Traffic_Congestion_%": "Highway congestion level",
        "Cargo_Type"        : "Electronics/FMCG/Pharma etc.",
        "Truck_Age_Yrs"     : "Older trucks = more delays",
        "Driver_Age"        : "Age-based risk factor",
        "Num_Stops"         : "More stops = higher delay risk",
        "Overload_Flag"     : "Load utilization > 90%",
        "Old_Truck_Flag"    : "Truck age > 8 years",
        "High_Congestion"   : "Traffic congestion > 70%",
        "Long_Route_Flag"   : "Distance > 800km",
        "Season"            : "Winter/Summer/Monsoon/Autumn",
    }
    feat_df = pd.DataFrame(
        list(new_feats.items()),
        columns=["Feature","Description"])
    st.dataframe(feat_df, use_container_width=True)

# ── PAGE 5: INSIGHTS ──
elif page == "💡 Insights":
    st.title("💡 Key Business Insights")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("What Causes Delays?")
        st.markdown("""
        1. **Traffic Congestion** — Biggest delay driver
        2. **Storm Weather** — 15%+ higher delays
        3. **Overloaded Trucks** — Load >90% = risky
        4. **Old Trucks (>8 yrs)** — Maintenance issues
        5. **Festival Months** — Oct/Nov/Jan demand spikes
        6. **Poor Road Condition** — Highway quality matters
        7. **Long Distance (>800km)** — More risk exposure
        8. **Inexperienced Drivers** — <3 yrs = higher risk
        """)
        st.subheader("Best Routes")
        st.markdown("""
        - **#1 Delhi-Amritsar** — 12% delay, Rs.84/km
        - **#2 Surat-Mumbai** — 14% delay, Rs.84/km
        - **#3 Delhi-Jaipur** — 12% delay, Rs.86/km
        """)

    with col2:
        st.subheader("Worst Routes")
        st.markdown("""
        - **#15 Chennai-Kolkata** — 19% delay, 3.4 days
        - **#14 Delhi-Mumbai** — 15% delay, Rs.88/km
        - **#13 Nagpur-Mumbai** — 19% delay
        """)
        st.subheader("Seasonal Patterns")
        st.markdown("""
        - **Monsoon (Jun-Sep)** — Highest delays
        - **June** — Peak delay month
        - **February** — Best month overall
        - **October** — Best: Bangalore-Hyderabad
        """)
        st.subheader("v2.0 New Insights")
        st.markdown("""
        - Festival months add **+7% delay risk**
        - Congestion >70% adds **+8% delay risk**
        - Trucks >8 yrs add **+6% delay risk**
        - Storm weather adds **+15% delay risk**
        - Drivers <3 yrs exp add **+5% delay risk**
        """)

    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Avg Cost/km",      "Rs.85")
    col2.metric("Best Route Score", "0.128")
    col3.metric("Worst Route Score","0.887")
    col4.metric("Overall Delay",    "14.7%")

    st.markdown("---")
    st.markdown("""
    *Built with ❤️ by Team Group 82 — UCF Capstone 2026*
    *Mohammad Kaif · Harshita Hoiyani · Ansh Mittal*
    """)
