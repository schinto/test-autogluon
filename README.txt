Down supplementary material tx9b00338_si_001.xls with Table S1 from
https://doi.org/10.1021/acs.chemrestox.9b00338
and store file  in data directory

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate autogluon

# Create model directory used to store autogluon models
mkdir models

# Start Jupyter notebook server
jupyter notebook

# Prepare the input data using the Jupyter notebook: prepare_data.ipynb

# Run benchmarks
python benchmark-autogluon.py
