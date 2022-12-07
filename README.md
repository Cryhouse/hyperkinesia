# Parkinson classification tools

This repo contains the code for my master's thesis that aimed to determine the hyperkinesia in people with Parkinson's disease.

Link to the thesis will be inserted when it is available online.

## Installation
### Prerequisites
python 3.10<br>
Python package `virtualenv`

### 1. Make virtual environment
`virtualenv <virtualenv-name>`

### 2. Install requirements
`pip install -r requirements.txt`

### 3. Install package
`pip install --editable .`

## Usage
If you want to use the package the way I have used it, including parsing raw data, preprocessing it, parsing labels, and training different models, testing different features, I recommend running `pct/report_figures/regression/regression_resample.py` and try to understand how it works. The rough pipeline is: 1. Parse labels. 2. Using the information in the labels, parse and preprocess raw accelerometer data. 3. Train model with the parsed data. 4. If you want to engineer features, go to `pct/pipeline/feature_ext.py` and change what the functions `extract_regressor_features` and `extract_bc_features` return.

If you want to check how the models actually work, check out `pct/pipeline/pipeline.py`. The self engineered features are located in `pct/pipeline/feature_ext.py` and are calculated through the function `get_X_full`, found in `pct/pipeline/pipeline.py`.

If you want to continue the work and perhaps want access to the data used in the thesis, please contact me at `gustaf.grothusen@gmail.com`.