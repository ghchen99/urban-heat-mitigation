# The 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands

The 2025 AI & data challenge is focused on a phenomenon known as the urban heat island effect, a situation that occurs due to the high density of buildings and lack of green space and water bodies in urban areas. Temperature variations between rural and urban environments can exceed 10-degrees Celsius in some cases and cause significant health-, social- and energy-related issues. Those particularly vulnerable to heat-related problems include young children, older adults, outdoor workers, and low-income populations.

All output from the challenge can help bring cooling relief to vulnerable communities, but entrants with top scores will take home cash prizes and receive an invitation to an exciting awards celebration.

## Problem Statement

The goal of the challenge is to develop a machine learning model to predict heat island hotspots in an urban location. Additionally, the model should be designed to discern and highlight the key factors that contribute significantly to the development of these hotspots within city environments.

Participants will be given near-surface air temperature data in an index format, which was collected on 24 July 2021 using a ground traverse in the Bronx and Manhattan region of New York City. This dataset constitutes traverse points (latitude and longitude) and their corresponding UHI (Urban Heat Island) index values. Participants will use this dataset to build a regression model to predict UHI Index values for a given set of locations.

It is important to understand that the UHI Index at any given location is indicative of the relative temperature difference at that specific point when compared to the city's average temperature. This index serves as a crucial metric for assessing the intensity of heat within different urban zones.

## Data Description

### Target Dataset `Training_data_uhi_index.csv`

Near-surface air temperature data in an index format was collected on 24 July 2021 across the Bronx and Manhattan regions of New York City in the United States. The data was collected in the afternoon between 3:00 pm and 4:00 pm. This dataset includes time stamps, traverse points (latitude and longitude) and the corresponding Urban Heat Island (UHI) Index values for 11229 data points. These UHI Index values are the target parameters for your model.

Note: Participants are strictly prohibited from using Longitude and Latitude values as features in building their machine learning models. Submissions that employ longitude and latitude values as model features will be disqualified. These values should only be utilized for understanding the attributes and characteristics of the locations.

### Feature Datasets

Participants can leverage many datasets to consider for their models. Their ability to analyze which datasets and parameters are the most important for model development will determine the model performance. The following are the recommended satellite datasets:

* European Sentinel-2 optical satellite data
* NASA Landsat optical satellite data

These datasets can be extracted from Microsoft Planetary Computer Portal's data catalog. Please see the sample notebooks for more details: `Landsat_LST.ipynb` and `Sentinel2_GeoTIFF.ipynb`.

### Additional Datasets

Participants can also explore the following datasets in their model development journey:

* Building footprints of the Bronx and Manhattan regions `Building_Footprint.kml`
* Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021, collected every 5 minutes `NY_Mesonet_Weather.xlsx`