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
