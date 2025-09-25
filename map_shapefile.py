# import geopandas as gpd
# import pandas as pd
# import folium

# # -----------------------------
# # 1. Load Malnutrition Data
# # -----------------------------
# malnutrition_data = pd.read_csv("data/malnutrition_sample.csv")

# # Optional: preview
# print("Malnutrition Data Preview:")
# print(malnutrition_data.head())

# # -----------------------------
# # 2. Load Rwanda District Shapefile
# # -----------------------------
# shapefile_path = "data/rwanda_districts_shapefile/District_Boundaries.shp"
# rwanda_gdf = gpd.read_file(shapefile_path)

# print("\nShapefile Columns:")
# print(rwanda_gdf.columns)

# # -----------------------------
# # 3. Drop datetime / unnecessary columns
# # -----------------------------
# datetime_cols = ['created_da', 'last_edite', 'last_edi_1']
# rwanda_gdf = rwanda_gdf.drop(columns=datetime_cols, errors='ignore')

# # -----------------------------
# # 4. Merge shapefile with malnutrition data
# # -----------------------------
# # Use the shapefile 'district' column and CSV 'District' column
# rwanda_gdf = rwanda_gdf.merge(
#     malnutrition_data,
#     left_on='district',
#     right_on='District'
# )

# # -----------------------------
# # 5. Create Folium Map
# # -----------------------------
# # Center map on Rwanda
# m = folium.Map(location=[-1.944, 30.061], zoom_start=7)

# # Choropleth: number of Stunted children
# folium.Choropleth(
#     geo_data=rwanda_gdf,
#     data=rwanda_gdf,
#     columns=['district', 'Stunted'],
#     key_on='feature.properties.district',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name="Number of Stunted Children"
# ).add_to(m)

# # Add popups for all malnutrition indicators
# for _, row in rwanda_gdf.iterrows():
#     popup_text = (
#         f"<b>{row['district']}</b><br>"
#         f"Children Under 5: {row['Children_Under5']}<br>"
#         f"Stunted: {row['Stunted']}<br>"
#         f"Underweight: {row['Underweight']}<br>"
#         f"Wasted: {row['Wasted']}<br>"
#         f"Vitamin A Deficiency: {row['VitaminA_Deficiency']}<br>"
#         f"Iodine Deficiency: {row['Iodine_Deficiency']}"
#     )
#     folium.Marker(
#         location=[row['Latitude'], row['Longitude']],
#         popup=folium.Popup(popup_text, max_width=300)
#     ).add_to(m)
# print("Total districts in CSV:", len(malnutrition_data))
# print(rwanda_gdf['district'].tolist())
# print(malnutrition_data['District'].tolist())


# # -----------------------------
# # 6. Save map
# # -----------------------------
# m.save("malnutrition_map.html")
# print("\nMap created! Open 'malnutrition_map.html' in your browser.")













import geopandas as gpd
import pandas as pd
import folium

# -----------------------------
# 1. Load Malnutrition Data
# -----------------------------
malnutrition_data = pd.read_csv("data/malnutrition_sample.csv")
print("Malnutrition Data Preview:")
print(malnutrition_data.head())

# -----------------------------
# 2. Load Rwanda Shapefile
# -----------------------------
shapefile_path = "data/rwanda_districts_shapefile/District_Boundaries.shp"
districts = gpd.read_file(shapefile_path)
print("\nShapefile Columns:", districts.columns)

# -----------------------------
# 3. Standardize Names
# -----------------------------
# Rename shapefile column to match CSV naming
districts = districts.rename(columns={"district": "District"})

# Convert both CSV and shapefile District names to lowercase + strip spaces
districts["District"] = districts["District"].str.strip().str.lower()
malnutrition_data["District"] = malnutrition_data["District"].str.strip().str.lower()

# -----------------------------
# 3b. Convert Non-geometry Columns to Serializable
# -----------------------------
for col in districts.columns:
    if col != "geometry" and not pd.api.types.is_numeric_dtype(districts[col]):
        districts[col] = districts[col].astype(str)

# -----------------------------
# 4. Merge Data
# -----------------------------
districts = districts.merge(malnutrition_data, on="District", how="left")
print("\nMerged Data Preview:")
print(districts[["District", "Children_Under5", "Stunted"]].head())

# -----------------------------
# 5. Create Interactive Map
# -----------------------------
# Center the map roughly on Rwanda
m = folium.Map(location=[-1.94, 29.87], zoom_start=8)

# -----------------------------
# 6. Add Choropleth Layer for Stunting
# -----------------------------
folium.Choropleth(
    geo_data=districts,
    name="Stunting",
    data=districts,
    columns=["District", "Stunted"],
    key_on="feature.properties.District",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Stunted Children (under 5)"
).add_to(m)

# -----------------------------
# 7. Add Hover Tooltips
# -----------------------------
folium.GeoJson(
    districts,
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=["District", "Children_Under5", "Stunted", "Underweight", "Wasted", "VitaminA_Deficiency", "Iodine_Deficiency"],
        aliases=["District", "Children Under 5", "Stunted", "Underweight", "Wasted", "Vitamin A Deficiency", "Iodine Deficiency"],
        localize=True
    )
).add_to(m)

# -----------------------------
# 8. Optional: Add CSV Point Markers
# -----------------------------
for _, row in malnutrition_data.iterrows():
    popup_text = f"""
    <b>{row['District'].title()}</b><br>
    Children Under 5: {row['Children_Under5']}<br>
    Stunted: {row['Stunted']}<br>
    Underweight: {row['Underweight']}<br>
    Wasted: {row['Wasted']}<br>
    Vitamin A Deficiency: {row['VitaminA_Deficiency']}<br>
    Iodine Deficiency: {row['Iodine_Deficiency']}
    """
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(m)

# -----------------------------
# 9. Save Map
# -----------------------------
m.save("rwanda_malnutrition_map.html")
print("\n✅ Map has been saved to 'rwanda_malnutrition_map.html'. Open it in a browser.")







# import geopandas as gpd
# import pandas as pd
# import folium
# from branca.colormap import LinearColormap

# # -----------------------------
# # 1. Load Malnutrition Data
# # -----------------------------
# malnutrition_data = pd.read_csv("data/malnutrition_sample.csv")

# # -----------------------------
# # 2. Load Rwanda Shapefile
# # -----------------------------
# shapefile_path = "data/rwanda_districts_shapefile/District_Boundaries.shp"
# districts = gpd.read_file(shapefile_path)

# # -----------------------------
# # 3. Standardize Names
# # -----------------------------
# districts = districts.rename(columns={"district": "District"})
# districts["District"] = districts["District"].str.strip().str.lower()
# malnutrition_data["District"] = malnutrition_data["District"].str.strip().str.lower()

# # -----------------------------
# # 4. Merge Data
# # -----------------------------
# districts = districts.merge(malnutrition_data, on="District", how="left")

# # -----------------------------
# # 5. Ensure Indicator Columns Are Numeric
# # -----------------------------
# indicators = {
#     "Stunted": "Stunted Children",
#     "Underweight": "Underweight Children",
#     "Wasted": "Wasted Children",
#     "VitaminA_Deficiency": "Vitamin A Deficiency",
#     "Iodine_Deficiency": "Iodine Deficiency"
# }

# for col in indicators.keys():
#     districts[col] = pd.to_numeric(districts[col], errors='coerce').fillna(0)

# # -----------------------------
# # 6. Create Base Map
# # -----------------------------
# m = folium.Map(location=[-1.94, 29.87], zoom_start=8)

# # -----------------------------
# # 7. Add Choropleth Layers per Indicator
# # -----------------------------
# for col, label in indicators.items():
#     fg = folium.FeatureGroup(name=label)

#     # Create color map
#     min_val = districts[col].min()
#     max_val = districts[col].max()
#     colormap = LinearColormap(['green', 'yellow', 'red'], vmin=min_val, vmax=max_val, caption=label)

#     # Add GeoJson polygons with correct lambda capture
#     def style_function(feature, colormap=colormap, col=col):
#         value = feature['properties'].get(col, 0)
#         if pd.isnull(value):
#             value = 0
#         return {
#             'fillColor': colormap(value),
#             'color': 'black',
#             'weight': 0.5,
#             'fillOpacity': 0.7
#         }

#     folium.GeoJson(
#         districts,
#         style_function=style_function,
#         tooltip=folium.GeoJsonTooltip(
#             fields=["District", "Children_Under5", "Stunted", "Underweight", "Wasted", "VitaminA_Deficiency", "Iodine_Deficiency"],
#             aliases=["District", "Children Under 5", "Stunted", "Underweight", "Wasted", "Vitamin A Deficiency", "Iodine Deficiency"],
#             localize=True
#         )
#     ).add_to(fg)

#     # Add legend
#     colormap.add_to(fg)
#     fg.add_to(m)

# # -----------------------------
# # 8. Add CSV Point Markers
# # -----------------------------
# for _, row in malnutrition_data.iterrows():
#     popup_text = f"""
#     <b>{row['District'].title()}</b><br>
#     Children Under 5: {row['Children_Under5']}<br>
#     Stunted: {row['Stunted']}<br>
#     Underweight: {row['Underweight']}<br>
#     Wasted: {row['Wasted']}<br>
#     Vitamin A Deficiency: {row['VitaminA_Deficiency']}<br>
#     Iodine Deficiency: {row['Iodine_Deficiency']}
#     """
#     folium.CircleMarker(
#         location=[row['Latitude'], row['Longitude']],
#         radius=6,
#         color="blue",
#         fill=True,
#         fill_opacity=0.7,
#         popup=popup_text
#     ).add_to(m)

# # -----------------------------
# # 9. Add Layer Control
# # -----------------------------
# folium.LayerControl(collapsed=False).add_to(m)

# # -----------------------------
# # 10. Save Map
# # -----------------------------
# m.save("rwanda_malnutrition_multi_map.html")
# print("\n✅ Map saved as 'rwanda_malnutrition_multi_map.html'. Open it in a browser.")




















# import geopandas as gpd
# import pandas as pd
# import folium
# from branca.colormap import LinearColormap

# # -----------------------------
# # 1. Load Malnutrition Data
# # -----------------------------
# malnutrition_data = pd.read_csv("data/malnutrition_sample.csv")

# # -----------------------------
# # 2. Load Rwanda Shapefile
# # -----------------------------
# shapefile_path = "data/rwanda_districts_shapefile/District_Boundaries.shp"
# districts = gpd.read_file(shapefile_path)

# # -----------------------------
# # 3. Standardize Names
# # -----------------------------
# districts = districts.rename(columns={"district": "District"})
# districts["District"] = districts["District"].str.strip().str.lower()
# malnutrition_data["District"] = malnutrition_data["District"].str.strip().str.lower()

# # -----------------------------
# # 4. Merge Data
# # -----------------------------
# districts = districts.merge(malnutrition_data, on="District", how="left")

# # -----------------------------
# # 4b. Fix Timestamp Columns for Folium Compatibility
# # -----------------------------
# for col in districts.columns:
#     if col != "geometry" and pd.api.types.is_datetime64_any_dtype(districts[col]):
#         districts[col] = districts[col].astype(str)

# # -----------------------------
# # 5. Ensure Indicator Columns Are Numeric
# # -----------------------------
# indicators = {
#     "Stunted": "Stunted Children",
#     "Underweight": "Underweight Children",
#     "Wasted": "Wasted Children",
#     "VitaminA_Deficiency": "Vitamin A Deficiency",
#     "Iodine_Deficiency": "Iodine Deficiency"
# }

# for col in indicators.keys():
#     districts[col] = pd.to_numeric(districts[col], errors='coerce').fillna(0)

# # -----------------------------
# # 6. Create Base Map
# # -----------------------------
# m = folium.Map(location=[-1.94, 29.87], zoom_start=8)

# # -----------------------------
# # 7. Add Choropleth Layers per Indicator
# # -----------------------------
# for col, label in indicators.items():
#     fg = folium.FeatureGroup(name=label)

#     # Create color map
#     min_val = districts[col].min()
#     max_val = districts[col].max()
#     colormap = LinearColormap(['green', 'yellow', 'red'], vmin=min_val, vmax=max_val, caption=label)

#     # Add GeoJson polygons with correct lambda capture
#     def style_function(feature, colormap=colormap, col=col):
#         value = feature['properties'].get(col, 0)
#         if pd.isnull(value):
#             value = 0
#         return {
#             'fillColor': colormap(value),
#             'color': 'black',
#             'weight': 0.5,
#             'fillOpacity': 0.7
#         }

#     folium.GeoJson(
#         districts,
#         style_function=style_function,
#         tooltip=folium.GeoJsonTooltip(
#             fields=["District", "Children_Under5", "Stunted", "Underweight", "Wasted", "VitaminA_Deficiency", "Iodine_Deficiency"],
#             aliases=["District", "Children Under 5", "Stunted", "Underweight", "Wasted", "Vitamin A Deficiency", "Iodine Deficiency"],
#             localize=True
#         )
#     ).add_to(fg)

#     # Add legend
#     colormap.add_to(fg)
#     fg.add_to(m)

# # -----------------------------
# # 8. Add CSV Point Markers
# # -----------------------------
# for _, row in malnutrition_data.iterrows():
#     popup_text = f"""
#     <b>{row['District'].title()}</b><br>
#     Children Under 5: {row['Children_Under5']}<br>
#     Stunted: {row['Stunted']}<br>
#     Underweight: {row['Underweight']}<br>
#     Wasted: {row['Wasted']}<br>
#     Vitamin A Deficiency: {row['VitaminA_Deficiency']}<br>
#     Iodine Deficiency: {row['Iodine_Deficiency']}
#     """
#     folium.CircleMarker(
#         location=[row['Latitude'], row['Longitude']],
#         radius=6,
#         color="blue",
#         fill=True,
#         fill_opacity=0.7,
#         popup=popup_text
#     ).add_to(m)

# # -----------------------------
# # 9. Add Layer Control
# # -----------------------------
# folium.LayerControl(collapsed=False).add_to(m)

# # -----------------------------
# # 10. Save Map
# # -----------------------------
# m.save("rwanda_malnutrition_multi_map.html")
# print("\n✅ Map saved as 'rwanda_malnutrition_multi_map.html'. Open it in a browser.")
