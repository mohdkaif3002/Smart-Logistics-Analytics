# 🚛 Smart Logistics Analytics System

> End-to-end India heavy freight logistics analytics with ML-based delay prediction and route optimization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 👥 Team — Group 82 | UCF 439 Capstone | JAN–MAY 2026

| Name | Contribution |
|------|-------------|
| Mohammad Kaif | ML Model & Route Optimization |
| Harshita Hoiyani | EDA & Data Processing |
| Ansh Mittal | Streamlit App & Visualization |

---

## 📌 Problem Statement

Indian logistics companies face:
- 🚚 Frequent delivery delays on major highway routes
- 💸 High and unpredictable freight costs
- ❌ No data-driven route selection system
- 📉 Poor visibility into what actually causes delays

**Our Solution:** A system that predicts delivery delays, scores route efficiency monthly, and recommends the optimal route — all from real operational data.

---

## 🏗️ Architecture
```
Data → EDA → Route Optimization → ML Model → Streamlit App
```

---

## 📊 Dataset

- 3,000 India-specific truck shipments (2020–2024)
- 15 major Indian highway routes
- 23 features: distance, weight, freight cost, weather, road condition, driver experience
- 14.7% real-world delay rate

---

## 📈 Key Findings

| Insight | Finding |
|---------|---------|
| Overall delay rate | 14.7% |
| Worst route | Chennai–Kolkata (19%) |
| Best route | Delhi–Amritsar (12%) |
| Worst month | June — Monsoon (22%) |
| Best month | February (10%) |

---

## 🗺️ Route Optimization

Weighted efficiency score formula:
```
Score = 0.40 × Delay_Rate + 0.35 × Cost_per_km + 0.25 × Avg_Delivery_Days
```

| Rank | Route | Score | Delay | Cost/km |
|------|-------|-------|-------|---------|
| 🥇 1 | Delhi–Amritsar | 0.1286 | 12% | ₹84 |
| 🥈 2 | Surat–Mumbai | 0.1972 | 14% | ₹84 |
| 🥉 3 | Delhi–Jaipur | 0.2237 | 12% | ₹86 |
| ❌ 15 | Chennai–Kolkata | 0.8870 | 19% | ₹86 |

---

## 🤖 Machine Learning

**Target:** Predict `Delay_Flag` (0 = On Time, 1 = Delayed)

| Model | Accuracy | Recall | F1 |
|-------|----------|--------|----|
| Logistic Regression | 49.33% | 48.86% | 22.05% |
| Random Forest | 84.33% | 3.41% | 6.00% |
| RF + SMOTE | 67.67% | 35.23% | 24.22% |
| **RF + SMOTE + Tuned** | **48.67%** | **63.64%** | **26.67%** |

**Top Delay Causes:**
1. Cost per KM (0.134)
2. Weight (0.128)
3. Load Utilization (0.114)
4. Freight Cost (0.110)
5. Month/Season (0.103)

---

## 📸 Charts

![Delay Overview](charts/01_delay_overview.png)
![Route Analysis](charts/02_route_analysis.png)
![Feature Importance](charts/11_feature_importance.png)

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn, SMOTE |
| App | Streamlit |
| Environment | Google Colab |

---

## 🚀 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/Smart-Logistics-Analytics.git
cd Smart-Logistics-Analytics
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📁 Structure
```
Smart-Logistics-Analytics/
├── data/               # Processed datasets
├── charts/             # All EDA & ML charts
├── models/             # Model config
├── notebooks/          # Colab notebook
├── app/                # Streamlit app
├── requirements.txt
└── README.md
```

---
*Built with ❤️ by Team Group 82 — UCF Capstone 2026*
