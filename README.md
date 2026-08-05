# DigiAra

DigiAra predicts Arabidopsis transcriptional responses under a configured gene
perturbation and microbial condition. The package includes five machine-learning
models, one Transformer model, a six-model Hybrid prediction, and demonstration
data.

## Project layout

Keep the distributed files in the following structure. No model or data paths
need to be entered manually.

```text
DigiAra_Model_and_Demo_Data/
├── Model_and_data/
│   ├── Ara_stand_protein__esmft32_embeddings1.npy
│   ├── Ara_stand_protein_sorted_gl_entry.csv
│   ├── Bagging_Base.joblib
│   ├── Demo_Unperturb_Transcriptomes.tsv
│   ├── ExtraTrees_Large.joblib
│   ├── LGBM.joblib
│   ├── Pst_DC3000_esmft32.mean_expand_27448.npy
│   ├── Allzero_27448M5120.npy
│   ├── Random_forest.joblib
│   ├── Transformer.pt
│   └── XGB.joblib
├── scripts/
│   ├── 00_Prepare_Config.py
│   ├── 01_ML_inference.py
│   ├── 02_Trans_inference.py
│   └── 03_Ensemble.py
├── Prediction_results_Demo/
└── README.md
```

The `Prediction_results/` directory is created automatically when inference is
run.

## Configuration

Edit only `scripts/00_Prepare_Config.py` before running inference:

```python
PERTURBATION_TYPE = "KO"
MUTANT_GENES = ["AT4G33430", "AT2G13790"]
MUTANT_NAME = "bak1_bkk1"
MICROBE = "PstDC3000"
```

- `PERTURBATION_TYPE` accepts `"KO"` or `"OE"`.
- `MUTANT_GENES` accepts one or multiple Arabidopsis locus IDs.
- `MUTANT_NAME` is the label used in output filenames. If it is empty, the locus
  IDs are used automatically.
- `MICROBE` accepts `"PstDC3000"` or `"mock"`.

KO replaces each selected gene value with `-log1p(1e6)`, while OE replaces it
with `+log1p(1e6)`. The unperturbed state is read from
`Model_and_data/Demo_Unperturb_Transcriptomes.tsv`.

## Running inference

Submit the scripts in numerical order:

1. `scripts/01_ML_inference.py`
2. `scripts/02_Trans_inference.py`
3. `scripts/03_Ensemble.py`

Submit `03_Ensemble.py` only after both `01_ML_inference.py` and
`02_Trans_inference.py` have completed successfully.

1. `01_ML_inference.py` runs Bagging, Extra Trees, LightGBM, Random Forest, and
   XGBoost inference. A CPU environment is sufficient.
2. `02_Trans_inference.py` runs Transformer inference. A CUDA 13.0 environment
   is recommended. The script uses CUDA automatically when PyTorch detects an
   available CUDA device; CPU fallback is supported but may be substantially
   slower and more memory intensive.
3. `03_Ensemble.py` verifies that all six predictions are present and calculates
   their arithmetic mean. A CPU environment is sufficient.

After changing `00_Prepare_Config.py`, rerun both inference scripts before
running `03_Ensemble.py`. The Hybrid step intentionally stops with an error if
any of the six required predictions is missing.

## Output files

Individual model predictions are kept separate from the final Hybrid result:

```text
Prediction_results/
├── Individual_Models/
│   ├── DigiAra_<mode>_<mutant>_<microbe>_Bagging_Base.csv
│   ├── DigiAra_<mode>_<mutant>_<microbe>_Bagging_Base.npy
│   ├── DigiAra_<mode>_<mutant>_<microbe>_ExtraTrees_Large.csv
│   ├── DigiAra_<mode>_<mutant>_<microbe>_ExtraTrees_Large.npy
│   ├── DigiAra_<mode>_<mutant>_<microbe>_LGBM.csv
│   ├── DigiAra_<mode>_<mutant>_<microbe>_LGBM.npy
│   ├── DigiAra_<mode>_<mutant>_<microbe>_Random_forest.csv
│   ├── DigiAra_<mode>_<mutant>_<microbe>_Random_forest.npy
│   ├── DigiAra_<mode>_<mutant>_<microbe>_XGB.csv
│   ├── DigiAra_<mode>_<mutant>_<microbe>_XGB.npy
│   ├── DigiAra_<mode>_<mutant>_<microbe>_Transformer.csv
│   └── DigiAra_<mode>_<mutant>_<microbe>_Transformer.npy
├── DigiAra_<mode>_<mutant>_<microbe>_Hybrid_Model.csv
└── DigiAra_<mode>_<mutant>_<microbe>_Hybrid_Model.npy
```

Each CSV contains two columns:

- `Gene`: Arabidopsis gene locus ID.
- `Predicted_Expr_Change`: predicted transcriptional change.

The corresponding NPY file contains the same 27,448 predictions as a float32
vector in the order defined by
`Model_and_data/Ara_stand_protein_sorted_gl_entry.csv`.

## Demonstration result

`Prediction_results_Demo/` contains the Hybrid prediction for the
`bak1_bkk1` KO mutant under the `PstDC3000` condition:

```text
git/DigiAra/Prediction_results_Demo/
├── DigiAra_KO_bak1_bkk1_PstDC3000_Hybrid_Model.csv
└── DigiAra_KO_bak1_bkk1_PstDC3000_Hybrid_Model.npy
```

The CSV is intended for direct inspection and downstream table-based analysis.
The NPY file provides the same result in a compact format for Python workflows.

## Python packages

The inference scripts use the following Python packages:

- NumPy
- pandas
- joblib
- scikit-learn
- XGBoost
- LightGBM
- PyTorch

The supplied joblib and PyTorch files should be treated as trusted model files
and loaded only from the distributed DigiAra package.
