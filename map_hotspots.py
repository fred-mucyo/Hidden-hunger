import pandas as pd
import folium

# Load the CSV
data = pd.read_csv("data/malnutrition_sample.csv")

# Create a base map centered roughly in Rwanda
rwanda_map = folium.Map(location=[-1.94, 29.87], zoom_start=7)

# Add districts as circle markers
for index, row in data.iterrows():
    # Total malnutrition score (sum of all indicators)
    total_risk = row['Stunted'] + row['Underweight'] + row['Wasted'] + row['VitaminA_Deficiency'] + row['Iodine_Deficiency']
    
    # Add circle marker
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=total_risk/50,  # scale radius for visibility
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"{row['District']}\nTotal Risk: {total_risk}"
    ).add_to(rwanda_map)

# Save map to HTML
rwanda_map.save("malnutrition_map.html")
print("Map created! Open malnutrition_map.html in your browser.")
