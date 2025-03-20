# **EY Open Science AI & Data Challenge 2025: Cooling Urban Heat Islands**  

![Urban Heat Island Effect](./assets/uhi_analysis_plots.png)  

This repository contains my submission for the **2025 EY Open Science AI and Data Challenge**, which focuses on predicting Urban Heat Island (UHI) hotspots in New York City using machine learning. The goal is to develop a regression model that predicts UHI Index values and identifies key contributing factors without directly using latitude and longitude as features.  

## **Project Overview**  

Urban Heat Islands (UHIs) are a major environmental challenge, causing increased temperatures in densely built-up urban areas. This project leverages satellite imagery and geospatial datasets to analyse urban heat distribution in the Bronx and Manhattan regions. By processing Sentinel-2 satellite data and building footprint information, I train a predictive model that helps identify UHI hotspots and understand the underlying causes.  

## **Repository Contents**  

- **`Sentinel2_GeoTIFF.ipynb`** – Extracts spectral band data from Sentinel-2 satellite imagery using the Microsoft Planetary Computer. These bands provide crucial information about land cover, vegetation, and surface materials that influence heat retention.  
- **`Building_Footprint.ipynb`** – Processes the `Building_Footprint.kml` dataset to extract urban structure data, which can impact temperature variations.  
- **`UHI Experiment Sample Benchmark Notebook V5.ipynb`** – Implements machine learning models to predict UHI Index values, evaluates model performance, and identifies key contributing factors.  

This repository provides a structured approach to addressing the UHI challenge, incorporating geospatial feature extraction and advanced modeling techniques.

## 1. Sentinel-2 Data Processing (`Sentinel2_GeoTIFF.ipynb`)
This notebook handles satellite imagery processing from Sentinel-2:
```python
# Define the bounding box for the area of interest
lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)

# Define the time window
time_window = "2021-06-01/2021-09-01"

# Search for Sentinel-2 imagery with low cloud cover
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = stac.search(
    bbox=bounds, 
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)
```
Key features:

- Connects to Microsoft Planetary Computer's STAC API
- Searches for Sentinel-2 imagery with low cloud cover (<30%)
- Loads spectral bands (Red, Green, Blue, NIR, SWIR) for analysis
- Creates visualizations of different spectral indices:
    - NDVI (Normalized Difference Vegetation Index) for vegetation coverage
    - NDBI (Normalized Difference Buildup Index) for urban areas
    - NDWI (Normalized Difference Water Index) for water bodies
- Exports data as GeoTIFF files for further analysis

## 2. Building Footprint Analysis (`Building_Footprint.ipynb`)
This notebook focuses on analysing building characteristics and their impact on urban heat islands:
```python
def analyze_building_footprints_for_uhi(df, output_file=None, plot=False, calculate_advanced_metrics=True):
    # Calculate basic building features
    gdf = calculate_building_features(df)
    
    # Calculate density-based metrics (100m and 500m radii)
    gdf = calculate_density_metrics(gdf, radius_m=100)
    gdf = calculate_density_metrics(gdf, radius_m=500)
    
    # Calculate advanced metrics if requested
    if calculate_advanced_metrics:
        # Add morphological analysis
        gdf = calculate_morphological_metrics(gdf)
        
        # Add ventilation potential analysis
        gdf = calculate_ventilation_potential(gdf)
    
    # Prepare features for modeling
    gdf = prepare_features_for_modeling(gdf)
    
    # Generate plots if requested
    if plot:
        # Create visualization plots...
```

Key features:

- Processes KML building footprint data to extract building geometries
- Calculates numerous UHI-relevant metrics:
    - Building physical properties (area, perimeter, compactness)
    - Urban density indices at multiple scales (100m, 500m)
    - Morphological metrics like rugosity and building alignment
    - Ventilation potential and wind exposure estimates
- Generates standardized features for UHI modeling
- Creates visualisation plots of building metrics


## 3. UHI Prediction Model (`UHI Experiment Sample Benchmark Notebook V5.ipynb`)
This notebook implements the machine learning pipeline for UHI prediction:

```python
def map_satellite_data_with_buffer(tiff_path, csv_path, buffer_radius=700, 
                                  weighting='gaussian', land_cover_mask=None, 
                                  outlier_threshold=3, min_valid_pixels=5,
                                  adaptive_weighting=False, drop_invalid=True,
                                  handle_out_of_bounds='nearest'):
    # Extracts satellite band values with advanced buffer techniques
    # ...

# Create features at multiple buffer distances
buffer_distances = [250, 500, 1000]
multi_scale_features = []

for buffer in buffer_distances:
    buffer_data = map_satellite_data_with_buffer(
        tiff_path='../data/S2_sample.tiff',
        csv_path='../data/Training_data_uhi_index_2025-02-18.csv',
        buffer_radius=buffer,
        adaptive_weighting=True,
        outlier_threshold=2.5,
        drop_invalid=False,
        handle_out_of_bounds='knn'
    )
    # ...
```

Key features:

- Extracts Sentinel-2 spectral information at various buffer distances
- Applies sophisticated spatial interpolation methods
- Derives 100+ spectral indices related to urban heat
- Joins building footprint features with spectral data
- Implements feature selection to identify key predictors
- Evaluates multiple ML models:
    - RandomForest, GradientBoosting, ExtraTrees
    - Linear models (ElasticNet, Ridge)
    - SVR, KNN, XGBoost, LightGBM


## Contributions

- Fork the repository: Create your own copy of the project
- Set up your environment: Install the required dependencies (see `requirements.txt`)
- Create a branch: Make your changes in a new branch
- Test your changes: Ensure your code runs correctly and maintains model performance
- Submit a pull request: Share your improvements with the community