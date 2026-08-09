# DigiAra model and demo data

The large model parameters and demonstration data are hosted on Zenodo and are
not stored in the GitHub repository.

- Version-specific DOI: <https://doi.org/10.5281/zenodo.21824787>
- Zenodo record: <https://zenodo.org/records/21824787>
- Approximate download size: 33.5 GB

From the root of the cloned DigiAra repository, run:

```bash
python download.py
```

The downloader uses only the Python standard library. It downloads each file
into this directory, retains interrupted downloads as `.part` files for later
resumption, and verifies every completed file against the MD5 checksum
published by Zenodo.

Allow at least 40 GB of free disk space before downloading. Do not run the
inference scripts until all files have passed validation.

## Expected files

```text
Allzero_27448M5120.npy
Ara_stand_protein__esmft32_embeddings1.npy
Ara_stand_protein_sorted_gl_entry.csv
Bagging_Base.joblib
Demo_Unperturb_Transcriptomes.tsv
ExtraTrees_Large.joblib
LGBM.joblib
Pst_DC3000_esmft32.mean_expand_27448.npy
Random_forest.joblib
Transformer.pt
XGB.joblib
```

To check files that have already been downloaded without downloading anything:

```bash
python download.py --verify-only
```

If an existing final file fails checksum validation, inspect it before using:

```bash
python download.py --force
```

The `--force` option replaces invalid final files. Correctly validated files are
not downloaded again.

## Access status

The automatic downloader requires the Zenodo files to be publicly accessible.
While the record files remain restricted, Zenodo will reject anonymous download
requests.
