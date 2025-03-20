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
