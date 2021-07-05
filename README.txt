# Create directories
mkdir data
mkdir models
mkdir results

Download supplementary material file tx9b00338_si_001.xls with Table S1 from
https://doi.org/10.1021/acs.chemrestox.9b00338
and store in data directory

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate autogluon

# Start Jupyter notebook server
jupyter notebook

1. Prepare the input data using the Jupyter notebook:
prepare-pih-data.ipynb
2. Build an Autogluon model using the pre-defined splits in the Set column:
build-autogluon-pih-model.ipynb
3. Run benchmarks to compare results on datasets with different feature combinations:
python benchmark-autogluon-pih-model.py
