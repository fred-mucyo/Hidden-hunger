import geopandas as gpd

rwanda_gdf = gpd.read_file("data/rwanda_districts_shapefile/District_Boundaries.shp")
print(rwanda_gdf.columns)
