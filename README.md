## Project Overview

This repository focuses on analyzing wildfire detection delays using satellite-based datasets, particularly the Canadian Fire Spread Dataset (CFSDS) from 2023. The goal is to explore how environmental and remoteness-related factors (e.g., slope, fuel type, road and hydro density) affect the time it takes to detect wildfires.

Through a series of exploratory data analyses (EDA), hypothesis testing, and machine learning models (e.g., Multiple Linear Regression, Decision Tree Regressor), we aim to identify patterns or bottlenecks that contribute to late detection — especially in remote regions. While most traditional remoteness metrics show weak or insignificant influence, terrain features like slope and wetness index appear to play a more relevant role.

## Folder Structure
``` bash
wildfire_detection/
│
├── data/                   # Raw input datasets (CFSDS, MODIS, VIIRS, etc.)
│   ├── Firegrowth_pts_v1_1_2023.csv
│   ├── modis_2023_Canada.csv
│   └── ...
│
├── document/              # Reference PDFs, academic articles, dataset metadata
│
├── output/                # Processed and cleaned outputs
│   ├── df_cfsds_pts_2023_clean.csv
│   └── fire_events_df_2023.csv
│
├── src/                   # Source code (Jupyter notebooks + helper modules)
│   ├── eda_p1_missingValues.ipynb
│   ├── eda_p2_dataIntegration.ipynb
│   ├── eda_p3_hypothesisTesting.ipynb
│   └── myLib.py
│
├── requirements.txt       # Python environment dependency list
└── README.md              # This file
```
- `data/`: The data files can be found in [cc_data](https://stuconestogacon.sharepoint.com/:f:/s/CC_CasestudiesinAIandML/Em6MtIi2ffxLgnDiUN_y0swBKBIvVVMIBGp0mE9OUFaLLw?e=hFu8ek)
- `output/`: The current output files can be found in [cc_output](https://stuconestogacon.sharepoint.com/:f:/s/CC_CasestudiesinAIandML/Eo9wwYZR63lOurxnzeSW04cBwkWhAQZP__I1-fU_CjiGhA?e=D5GfpX)
- `src/`: covering:
    - Missing value analysis (eda_p1_missingValues.ipynb)
    - Dataset integration and processing (eda_p2_dataIntegration.ipynb)
    - Statistical testing and hypothesis evaluation (eda_p3_hypothesisTesting.ipynb)


## Setup Instructions
To set up the environment and install all required dependencies, run the following command in your terminal (ensure you have Python and pip installed):
``` bash
pip install -r requirements.txt
```
If you're using a virtual environment, you can set it up like this:
``` bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```
If you prefer to use Conda, you can create a new environment and install dependencies with:
``` bash
conda create -n wildfire_env python=3.11
conda activate wildfire_env
pip install -r requirements.txt

```






