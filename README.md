
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

## Benchmark results
Features|Set|ROC-AUC|Accuracy|Balanced Accuracy|Sensitivity|Specifity|MCC|F1|Precision|Recall
-|-|-|-|-|-|-|-|-|-|-
pih_rdkit|Train|0.9955|0.9599|0.9517|0.9863|0.9446|0.917|0.9475|0.9863|0.9116
pih_rdkit|Test|0.8449|0.7712|0.7314|0.8442|0.7467|0.523|0.65|0.8442|0.5285
pih_rdkit|External|0.8238|0.75|0.7751|0.9|0.6111|0.5303|0.7759|0.9|0.6818
pih_maccs|Train|0.9711|0.8838|0.8661|0.9142|0.8682|0.7568|0.842|0.9142|0.7803
pih_maccs|Test|0.7973|0.7516|0.7084|0.8219|0.7296|0.4794|0.6122|0.8219|0.4878
pih_maccs|External|0.811|0.7308|0.76|0.8958|0.5893|0.5022|0.7544|0.8958|0.6515
pih_flatring_fps|Train|0.9971|0.9639|0.9589|0.9737|0.9579|0.9246|0.9536|0.9737|0.9343
pih_flatring_fps|Test|0.813|0.781|0.7463|0.8333|0.7613|0.5412|0.6763|0.8333|0.5691
pih_flatring_fps|External|0.8569|0.7115|0.7392|0.875|0.5714|0.4622|0.7368|0.875|0.6364
pih_flatring_rdkit_cats|Train|0.9956|0.9549|0.9471|0.9756|0.9428|0.9062|0.9412|0.9756|0.9091
pih_flatring_rdkit_cats|Test|0.8418|0.7712|0.7301|0.8533|0.7446|0.5246|0.6465|0.8533|0.5203
pih_flatring_rdkit_cats|External|0.7899|0.6635|0.6846|0.8163|0.5273|0.3562|0.6957|0.8163|0.6061
pih_flatring_cats|Train|0.9759|0.9108|0.8971|0.9373|0.8964|0.8138|0.8809|0.9373|0.8308
pih_flatring_cats|Test|0.7967|0.7516|0.7177|0.7701|0.7443|0.4733|0.6381|0.7701|0.5447
pih_flatring_cats|External|0.75|0.6731|0.681|0.7963|0.54|0.3489|0.7167|0.7963|0.6515
pih_rdkit_fps|Train|0.9991|0.98|0.976|0.9921|0.9724|0.9583|0.9743|0.9921|0.9571
pih_rdkit_fps|Test|0.8413|0.7876|0.7478|0.8816|0.7565|0.5623|0.6734|0.8816|0.5447
pih_rdkit_fps|External|0.8393|0.75|0.7751|0.9|0.6111|0.5303|0.7759|0.9|0.6818
pih_flatring_rdkit|Train|0.9893|0.9419|0.9302|0.9774|0.9224|0.8799|0.9227|0.9774|0.8737
pih_flatring_rdkit|Test|0.8282|0.7647|0.7206|0.8592|0.7362|0.5126|0.6289|0.8592|0.4959
pih_flatring_rdkit|External|0.8309|0.6827|0.7221|0.8837|0.541|0.4343|0.6972|0.8837|0.5758
pih_flatring_rdkit_fps|Train|0.9985|0.98|0.9769|0.987|0.9755|0.9582|0.9744|0.987|0.9621
pih_flatring_rdkit_fps|Test|0.8394|0.7745|0.7368|0.8375|0.7522|0.5285|0.6601|0.8375|0.5447
pih_flatring_rdkit_fps|External|0.8417|0.75|0.7695|0.8846|0.6154|0.5192|0.7797|0.8846|0.697
pih_flatring_maccs|Train|0.956|0.8577|0.8354|0.8944|0.8402|0.702|0.8022|0.8944|0.7273
pih_flatring_maccs|Test|0.7984|0.7386|0.6895|0.8308|0.7137|0.4542|0.5745|0.8308|0.439
pih_flatring_maccs|External|0.7923|0.6058|0.6615|0.8571|0.4783|0.3291|0.5941|0.8571|0.4545
pih_fps|Train|0.9991|0.989|0.9883|0.9873|0.99|0.977|0.9861|0.9873|0.9848
pih_fps|Test|0.8134|0.781|0.7436|0.85|0.7566|0.5437|0.67|0.85|0.5528
pih_fps|External|0.8242|0.7115|0.7337|0.86|0.5741|0.4504|0.7414|0.86|0.6515
pih_flatring|Train|0.7141|0.6523|0.6167|0.5809|0.6835|0.2484|0.5036|0.5809|0.4444
pih_flatring|Test|0.668|0.6275|0.5819|0.5584|0.6507|0.1851|0.43|0.5584|0.3496
pih_flatring|External|0.6124|0.5288|0.573|0.7297|0.4179|0.1468|0.5243|0.7297|0.4091
pih_cats|Train|0.9262|0.8236|0.8106|0.7957|0.8403|0.6285|0.7708|0.7957|0.7475
pih_cats|Test|0.7652|0.7124|0.6743|0.7108|0.713|0.3843|0.5728|0.7108|0.4797
pih_cats|External|0.7309|0.6346|0.6451|0.7692|0.5|0.2796|0.678|0.7692|0.6061
pih_maccs_fps|Train|0.9981|0.9699|0.9647|0.9841|0.9613|0.9374|0.9612|0.9841|0.9394
pih_maccs_fps|Test|0.8282|0.781|0.745|0.8415|0.7589|0.5424|0.6732|0.8415|0.561
pih_maccs_fps|External|0.8337|0.7115|0.7448|0.8913|0.569|0.4747|0.7321|0.8913|0.6212
