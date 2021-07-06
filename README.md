
# Predict Clinical Photosensitvity using Autogluon Tabular 

Test of Autogluon Tabular to predict the clinical photosensitivity (PIH)
published by  [Schmidt et al Chem. Res. Toxicol. 2019, 32, 2338−2352](https://doi.org/10.1021/acs.chemrestox.9b00338).
The models are trained using different combinations of molecular fingerprints and descriptors.

## Setup
Download supplementary material file `tx9b00338_si_001.xls` with Table S1 from
https://doi.org/10.1021/acs.chemrestox.9b00338

```
# Create directories
mkdir data
mkdir models
mkdir results

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate autogluon

# Start Jupyter notebook server
jupyter notebook
```
Save file `tx9b00338_si_001.xls` in `data` directory

## Running the scripts
1. Prepare the input data using the Jupyter notebook:
   `prepare-pih-data.ipynb`
2. Build an Autogluon model using the splits provided in the `Set` column:
   `build-autogluon-pih-model.ipynb`
3. Run benchmarks to compare results on datasets with different feature combinations:
   `python benchmark-autogluon-pih-model.py`

## Notes
- 9 SMILES in the published PIH data failed clean up steps and were excluded. 