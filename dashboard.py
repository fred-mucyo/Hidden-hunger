# dashboard.py

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.features import GeoJsonTooltip
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
df = pd.read_csv("outputs/policy_briefs_by_risk_socio.csv")



# Ensure correct formatting
df["Root_Causes"] = df["Root_Causes"].str.replace("['\"]", "")
df["Health_Rec"] = df["Health_Rec"].str.replace("['\"]", "")
df["Agri_Rec"] = df["Agri_Rec"].str.replace("['\"]", "")
df["Edu_Rec"] = df["Edu_Rec"].str.replace("['\"]", "")

# -----------------------------
# 2. Streamlit sidebar filters
# -----------------------------
st.sidebar.title("Filters")
risk_levels = st.sidebar.multiselect(
    "Select Risk Levels",
    options=df["Risk_Level"].unique(),
    default=df["Risk_Level"].unique()
)

province_filter = st.sidebar.text_input("Filter by Province (optional)")

# Apply filters
df_filtered = df[df["Risk_Level"].isin(risk_levels)]

# -----------------------------
# 3. Map
# -----------------------------
st.title("Rwanda Malnutrition Risk Dashboard")

# Base Folium map
m = folium.Map(location=[-1.95, 30.06], zoom_start=8)

# Add districts to map
for idx, row in df_filtered.iterrows():
    tooltip_text = f"""
    <b>District:</b> {row['District']}<br>
    <b>Risk Level:</b> {row['Risk_Level']}<br>
    <b>Root Causes:</b> {row['Root_Causes']}<br>
    <b>Health Recommendations:</b> {row['Health_Rec']}<br>
    <b>Agriculture Recommendations:</b> {row['Agri_Rec']}<br>
    <b>Education Recommendations:</b> {row['Edu_Rec']}
    """
    color = "#FF0000" if row["Risk_Level"] == "High" else "#FFA500" if row["Risk_Level"] == "Medium" else "#90EE90"
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=folium.Tooltip(tooltip_text, sticky=True)
    ).add_to(m)

st.subheader("Rwanda Districts Map")
st_folium(m, width=700, height=500)

# -----------------------------
# 4. Top 5 High-Risk Districts
# -----------------------------
st.subheader("Top 5 High-Risk Districts")
top5 = df_filtered[df_filtered["Risk_Level"]=="High"].sort_values(by="pred_prob", ascending=False).head(5)
st.dataframe(top5[["District", "pred_prob", "Root_Causes", "Health_Rec", "Agri_Rec", "Edu_Rec"]])

# -----------------------------
# 5. Feature Importance Plot
# -----------------------------
st.subheader("Feature Importance (Random Forest Model)")
feat_imp = pd.read_csv("outputs/feature_importance_smote.csv", index_col=0, header=None, names=["Feature", "Importance"])
feat_imp = feat_imp.sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(6,4))
ax.barh(feat_imp.index, feat_imp["Importance"], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
st.pyplot(fig)
