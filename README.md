# RF-PCA-VUVSpectra

Code for:
“A Principal Component Analysis-Integrated Machine Learning Approach for Predicting Gas Phase VUV/UV Absorption Spectra of Molecular Compounds.”

## How to run (quick demo)
1) Install:
   pip install -r requirements.txt
   - RDKit is best via conda:
   - conda install -c conda-forge rdkit

2) Use the sample paths in `main_rf_pca.py` (already set by default):


3) Run:
   python main_rf_pca.py
   

## What this code does
- Loads VUV/UV spectra, denoises, compresses with PCA (99% variance), trains Random Forest (5-fold CV),
  back-projects predictions, reports R², and plots top-N feature importance.

## Files
- `main_rf_pca.py` — main pipeline.
- `Lib_feats_new.py` — RDKit-based featurization (truncated E-state, CDS, literal bag-of-bonds, ABOCH).

## Data availability & provenance

- **Experimental spectra:** Obtained from VUV Analytics, Inc. (https://vuvanalytics.com). 
- **Molecular structures:** SDF files retrieved from PubChem (https://pubchem.ncbi.nlm.nih.gov).
- **Sample data:** A sample dataset is included to demonstrate the pipeline end-to-end.
- **Full processed dataset:** Available from the authors upon reasonable request.

Code for this work is hosted at: https://github.com/matine-g/RF-PCA-VUVSpectra

