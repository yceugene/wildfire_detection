# Project Overview

This repository focuses on analyzing wildfire detection delays using satellite-based datasets, particularly the Canadian Fire Spread Dataset (CFSDS) from 2023. The goal is to explore how environmental and remoteness-related factors (e.g., slope, fuel type, road and hydro density) affect the time it takes to detect wildfires.

Through a series of exploratory data analyses (EDA), hypothesis testing, and machine learning models (e.g., Multiple Linear Regression, Decision Tree Regressor), we aim to identify patterns or bottlenecks that contribute to late detection — especially in remote regions. While most traditional remoteness metrics show weak or insignificant influence, terrain features like slope and wetness index appear to play a more relevant role.

## folders
### `data`
The data files can be found in [cc_data](https://stuconestogacon.sharepoint.com/:f:/s/CC_CasestudiesinAIandML/Em6MtIi2ffxLgnDiUN_y0swBKBIvVVMIBGp0mE9OUFaLLw?e=hFu8ek)

### `output`
The current output files can be found in [cc_output](https://stuconestogacon.sharepoint.com/:f:/s/CC_CasestudiesinAIandML/Eo9wwYZR63lOurxnzeSW04cBwkWhAQZP__I1-fU_CjiGhA?e=D5GfpX)

### `src`
The repo is structured in Jupyter notebooks under the src/ directory, covering:
- Missing value analysis (eda_p1_missingValues.ipynb)
- Dataset integration and processing (eda_p2_dataIntegration.ipynb)
- Statistical testing and hypothesis evaluation (eda_p3_hypothesisTesting.ipynb)



