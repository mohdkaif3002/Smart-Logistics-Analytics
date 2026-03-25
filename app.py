
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Smart Logistics Analytics",
    page_icon="🚛",
    layout="wide"
)

# ── Path helper (works locally AND on Streamlit Cloud) ──
BASE = os.path.dirname(os.path.abspath(__file__))

def p(folder, filename):
    return os.path.join(BASE, folder, filename)

@st.cache_resource
def load_model():
    model      = joblib.load(p("models", "random_forest_model.pkl"))
    le_route   = joblib.load(p("models", "le_route.pkl"))
    le_truck   = joblib.load(p("models", "le_truck.pkl"))
    le_weather = joblib.load(p("models", "le_weather.pkl"))
    le_road    = joblib.load(p("models", "le_road.pkl"))
    with open(p("models", "model_config.json")) as f:
        config = json.load(f)
    return model, le_route, le_truck, le_weather, le_road, config

@st.cache_data
def load_data():
    df            = pd.read_csv(p("data", "india_logistics_clean.csv"))
    best_routes   = pd.read_csv(p("data", "best_routes_monthly.csv"))
    route_metrics = pd.read_csv(p("data", "route_metrics.csv"))
    return df, best_routes, route_metrics

model, le_route, le_truck, le_weather, le_road, config = load_model()
df, best_routes, route_metrics = load_data()

# ── Sidebar ──
st.sidebar.title("🚛 Smart Logistics")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Team — Group 82")
st.sidebar.markdown("""
🔹 Mohammad Kaif
🔹 Harshita Hoiyani
🔹 Ansh Mittal
""")
st.sidebar.markdown("---")
st.sidebar.caption("UCF 439 | Capstone | JAN-MAY 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔮 Delay Predictor",
    "🗺️ Route Recommender",
    "💡 Insights"
])

def show_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**👥 Team — Group 82**")
    col2.markdown("Mohammad Kaif | Harshita Hoiyani | Ansh Mittal")
    col3.markdown("UCF 439 | Capstone Project | JAN-MAY 2026")

# ── PAGE 1: DASHBOARD ──
if page == "📊 Dashboard":
    st.title("🚛 Smart Logistics Analytics System")
    st.markdown("**India Heavy Freight — Performance Dashboard**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments",  f"{len(df):,}")
    col2.metric("Delayed",          f"{df['Delay_Flag'].sum():,}",
                                    f"{df['Delay_Flag'].mean()*100:.1f}%")
    col3.metric("Avg Freight Cost", f"Rs.{df['Freight_Cost_INR'].mean():,.0f}")
    col4.metric("Avg Distance",     f"{df['Distance_km'].mean():,.0f} km")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Delay Rate by Truck Type")
        st.bar_chart(df.groupby("Truck_Type")["Delay_Flag"].mean()*100)
    with col2:
        st.subheader("Monthly Delay Trend")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df.groupby("Month_Name")["Delay_Flag"].mean()*100
        monthly = monthly.reindex([m for m in month_order if m in monthly.index])
        st.line_chart(monthly)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Most Delayed Routes")
        route_delay = df.groupby("Route")["Delay_Flag"].mean()*100
        st.bar_chart(route_delay.sort_values(ascending=False).head(5))
    with col2:
        st.subheader("Shipments by Truck Type")
        st.bar_chart(df["Truck_Type"].value_counts())
    show_footer()

# ── PAGE 2: DELAY PREDICTOR ──
elif page == "🔮 Delay Predictor":
    st.title("🔮 Delivery Delay Predictor")
    st.markdown("Enter shipment details to predict delay risk")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Shipment Details")
        route      = st.selectbox("Route", sorted(df["Route"].unique()))
        truck_type = st.selectbox("Truck Type",
                        ["Heavy Truck","Medium Truck","Light Van"])
        distance   = st.number_input("Distance (km)", 100, 2000, 500, 50)
        weight     = st.number_input("Weight (kg)",   500, 40000, 5000, 500)

    with col2:
        st.subheader("Cost & Load")
        freight_cost  = st.number_input("Freight Cost (Rs.)", 5000, 500000, 50000, 1000)
        cost_per_km   = round(freight_cost / distance, 2)
        st.metric("Cost per KM", f"Rs.{cost_per_km}")
        load_util     = st.slider("Load Utilization (%)", 55, 100, 75)
        expected_days = st.number_input("Expected Delivery Days", 1, 5, 2)

    with col3:
        st.subheader("Conditions")
        month      = st.selectbox("Month", list(range(1,13)),
                        format_func=lambda x:
                        ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        weather    = st.selectbox("Weather", ["Clear","Rain","Fog","Storm"])
        road       = st.selectbox("Road Condition", ["Good","Average","Poor"])
        driver_exp = st.slider("Driver Experience (years)", 1, 20, 5)

    st.markdown("---")
    if st.button("🔮 PREDICT DELIVERY STATUS", use_container_width=True):
        route_enc   = le_route.transform([route])[0]
        truck_enc   = le_truck.transform([truck_type])[0]
        weather_enc = le_weather.transform([weather])[0]
        road_enc    = le_road.transform([road])[0]

        input_data = np.array([[
            distance, weight, freight_cost, cost_per_km,
            expected_days, load_util, driver_exp, month,
            route_enc, truck_enc, weather_enc, road_enc
        ]])

        proba      = model.predict_proba(input_data)[0][1]
        prediction = int(proba >= config["threshold"])

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.error("⚠️ LIKELY DELAYED")
                st.markdown(f"### Delay Probability: {proba*100:.1f}%")
            else:
                st.success("✅ ON TIME DELIVERY")
                st.markdown(f"### On-Time Probability: {(1-proba)*100:.1f}%")
        with col2:
            st.info(f"**Route:** {route}")
            st.info(f"**Truck:** {truck_type}")
            st.info(f"**Distance:** {distance} km")
        with col3:
            st.info(f"**Weight:** {weight:,} kg")
            st.info(f"**Cost/km:** Rs.{cost_per_km}")
            st.info(f"**Weather:** {weather}")
    show_footer()

# ── PAGE 3: ROUTE RECOMMENDER ──
elif page == "🗺️ Route Recommender":
    st.title("🗺️ Best Route Recommender")
    st.markdown("Best route per month based on historical performance")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        selected_month = st.selectbox("Select Month",
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"])
        st.markdown("---")
        st.subheader(f"Best Route for {selected_month}")
        month_data = best_routes[best_routes["Month"] == selected_month]
        if not month_data.empty:
            row = month_data.iloc[0]
            st.success(f"### 🏆 {row['Best_Route']}")
            st.metric("Delay Rate",  f"{row['Delay_%']:.1f}%")
            st.metric("Cost per KM", f"Rs.{row['Cost_per_km']:.0f}")
            st.metric("Avg Days",    f"{row['Avg_Days']:.1f}")
            st.metric("Score",       f"{row['Score']:.4f}")
    with col2:
        st.subheader("All Routes Ranked")
        st.dataframe(route_metrics[[
            "Rank","Delay_Rate_%",
            "Avg_Cost_per_km",
            "Avg_Delivery_Days",
            "Efficiency_Score"]],
            use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Best Route Table")
    st.dataframe(best_routes, use_container_width=True)
    show_footer()

# ── PAGE 4: INSIGHTS ──
elif page == "💡 Insights":
    st.title("💡 Key Business Insights")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 What Causes Delays?")
        st.markdown("""
        1. **Cost per KM (0.134)** — Expensive routes = more delays
        2. **Weight (0.128)** — Heavier loads = higher delay risk
        3. **Load Utilization (0.114)** — Overloaded trucks delay more
        4. **Freight Cost (0.110)** — High cost routes are risky
        5. **Month (0.103)** — June worst (monsoon season)
        """)
        st.subheader("🏆 Best Routes")
        st.markdown("""
        - **#1 Delhi–Amritsar** — 12% delay | Rs.84/km
        - **#2 Surat–Mumbai** — 14% delay | Rs.84/km
        - **#3 Delhi–Jaipur** — 12% delay | Rs.86/km
        """)
    with col2:
        st.subheader("❌ Worst Routes")
        st.markdown("""
        - **#15 Chennai–Kolkata** — 19% delay | 3.4 days
        - **#14 Delhi–Mumbai** — 15% delay | Rs.88/km
        - **#13 Nagpur–Mumbai** — 19% delay | Rs.85/km
        """)
        st.subheader("📅 Seasonal Patterns")
        st.markdown("""
        - **June** — Worst month (22%) — Monsoon
        - **February** — Best month (10%)
        - **October** — Best for Bangalore–Hyderabad (0% delay)
        """)

    st.markdown("---")
    st.subheader("🤖 ML Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",            "Random Forest")
    col2.metric("Recall",           "63.64%")
    col3.metric("Training Samples", "4,092")
    col4.metric("Features Used",    "12")
    show_footer()
