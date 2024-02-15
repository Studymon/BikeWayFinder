# BikeWayFinder

Our App <a target="_blank" href="https://bikewayfinder.streamlit.app/">BikeWayFinder</a> helps cyclists find the most bicycle-friendly route between points A and B. The routing algorithm takes into consideration several criteria (features) which are used to compute a composite bikeability-score. According to this score, the app displays the most cyclist-friendly route between the start and destination locations. In addition, the user can also choose to display the shortest route or compare both routes. For the comparison, values such as distance, estimated time needed to reach the destination, as well as the average bikeability-score are displayed, allowing users to make an informed decisions on the routing options. Last but not least, users also have the option to customize the bike journey by selecting different feature weighting schemes (e.g. preference for riding in nature), further enhancing the cycling experience.

## Data Acquisition
1) Query OpenStreetMap (OSM) data to retrieve the following features: 
road type (e.g. highways, residentail roads, etc.), pavement type, road length, road width, bicycle-parking areas, and bus stops

2) Retrieve elevation data from NASA Earthdata

3) Using API to scrape Street View Images (SVI) from Google Maps, and exploit deep learning models to segment features such as greenery and vehicle frequencies, and to detect features
such as street lights and rail tracks

4) Set up survey to determine subjective perception of bicycle-friendliness of a specific SVI

## Calculation of Bikeability Score
The Bikeability Score can be obtained by combining all features (data) mentioned above, each with a specific weight assigned. For details on the weighting schemes, 
please see `combined_scores.py` in the `src` folder.

## Project Organization


    ├── README.md          <- The top-level README for developers using this   project.
    │
    ├── data_visible
    │   │
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models / link to used models
    │
    ├── notebooks          <- Jupyter notebooks (mainly containing deep learning   scripts outsourced to Google Colab).
    │
    ├── references         <- Literature
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   │
    │   │   ├── image_sampling.py
    │   │   ├── missing_cluster_handling.py
    │   │   └── svi_extraction.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for   modeling and deployment
    │   │   │
    │   │   ├── elevation.py
    │   │   ├── image_features.py
    │   │   └── osm_features_extraction.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained   models to make predictions
    │   │   │                 
    │   │   ├── survey_model.py
    │   │   │
    │   │   └── image_segmentation <- Contains script and custom packages for   image segmentation of SVI
    │   │
    │   └── combined_scores.py
    │
    │
    ├── requirements.txt   <- The requirements file for hosting the App on   Streamlit
    │
    └── all_requirements.txt   <- The requirements file for reproducing the   entire analysis environment

## Execution Guide

### Part 1: OSM Feature Pipeline

`/src/features/osm_features_extraction.py`  <- Queries OSM features needed for bikeability score calculation

`/src/features/elevation.py`  <- Uses elevation raster information to extract elevation data and calculate elevation gain per edge

### Part 2: Image Feature Pipeline

**Important:** The execution guide explains the logical order of executing scripts/notebooks. Sample data is provided in `/data_visible/`. However, some of the scripts can't be executed properly without the full data or required API keys. Some notebooks require execution on Google Colab (most notably the image segmentation) and uploading the required custom packages there.

2.1. `/src/data/svi_extraction.py`  <- Imports OSM data, clusters edges, retrieves SVI from API (requires API key!)

2.2. `/notebooks/SVI_sorting.ipynb`  <- Sorts the retrieved SVI into usable and unusable for image segmentation

*Extra Step*: Ideally, go through images manually to improve quality of sorting and therefore images used for the features

2.3. `/notebooks/segmentation_main_colab.ipynb` or `/src/models/image_segmentation_main.ipy`  <- Does the image segmentation. Best executed on cluster Google Colab via the notebook, else `image_segmentation_main.ipy` allows testing with sample images in `data_visible`

*Note:* Both execution locally and on Colab requires the three custom packages stored in `/src/models/image_segmentation`. the `models` and `modules` packages were taken https://github.com/mapillary/inplace_abn while we created `segmentation_setup` customizing code from their repo to work for our data.

2.4. `/src/data/missing_cluster_handling.py`  <- Analyzes and handles clusters with missing images

2.5. `/src/models/survey_model.py`  <- Predicts survey score for the clusters

2.6. `/features/`image_features.py`  <- Uses the tensors created during image segmentation to extract image features. Merges with the survey predictions and outputs the final edges with all image features

### Part 3: Deployment

3.1. `/src/combined_scores.py`  <- Defines scoring, weights, and combines processed OSM, DEM, and SVI feature data. Applies scoring functions to compute composite bikeability scores used for the routing algorithm

3.2. `app.py` (Note: Doesn't require execution, this is permanently run on the Streamlit servers)


### Scripts/notebooks not mentioned in the pipeline and their purpose:

- `/notebooks/train_SVI_sorting.ipynb`  <- Executed on Google Colab. Notebook for creating and training the model sorting SVI into usable or unusable using the manually sorted SVI used for the perception score survey.
- `notebooks/survey_model_quality.ipynb`  <- Notebook survey model training, graph output, and sanity checks of survey model


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
