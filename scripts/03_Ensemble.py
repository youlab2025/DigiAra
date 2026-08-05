"""Average the five ML predictions and one Transformer prediction."""

from __future__ import annotations

import gc
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CONFIG_PATH = SCRIPT_DIR / "00_Prepare_Config.py"
DATA_DIR = PROJECT_DIR / "Model_and_data"
OUTPUT_DIR = PROJECT_DIR / "Prediction_results"
INDIVIDUAL_OUTPUT_DIR = OUTPUT_DIR / "Individual_Models"
GENE_FILE = "Ara_stand_protein_sorted_gl_entry.csv"

MODEL_NAMES = (
    "Bagging_Base",
    "ExtraTrees_Large",
    "LGBM",
    "Random_forest",
    "XGB",
    "Transformer",
)

MICROBE_NAMES = {
    "pstdc3000": "PstDC3000",
    "mock": "mock",
}


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def safe_filename_component(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    result = result.strip("._-")
    if not result:
        raise ValueError("An output filename component is empty after sanitization.")
    return result


def load_user_config() -> dict[str, object]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    raw = runpy.run_path(str(CONFIG_PATH))
    raw_mode = raw.get("PERTURBATION_TYPE", "")
    mode = str(raw_mode).strip().upper()
    if mode not in {"KO", "OE"}:
        raise ValueError(
            'PERTURBATION_TYPE must be either "KO" or "OE". '
            f"Read {raw_mode!r} from {CONFIG_PATH}."
        )

    raw_genes = raw.get("MUTANT_GENES", [])
    if isinstance(raw_genes, str):
        raw_genes = [raw_genes]
    if not isinstance(raw_genes, (list, tuple)):
        raise TypeError("MUTANT_GENES must be a list or tuple of gene IDs.")

    genes = []
    for gene in raw_genes:
        gene_id = str(gene).strip().upper()
        if gene_id and gene_id not in genes:
            genes.append(gene_id)

    microbe_key = normalize_key(raw.get("MICROBE", ""))
    if microbe_key not in MICROBE_NAMES:
        supported = ", ".join(MICROBE_NAMES.values())
        raise ValueError(f"Unsupported MICROBE value. Choose one of: {supported}.")

    requested_name = str(raw.get("MUTANT_NAME", "")).strip()
    mutant_name = requested_name or ("_".join(genes) if genes else "WT")
    return {
        "mode": mode,
        "mutant_name": safe_filename_component(mutant_name),
        "microbe_name": MICROBE_NAMES[microbe_key],
    }


def make_prefix(config: dict[str, object]) -> str:
    return "_".join(
        (
            "DigiAra",
            str(config["mode"]),
            str(config["mutant_name"]),
            safe_filename_component(config["microbe_name"]),
        )
    )


def load_gene_table() -> pd.DataFrame:
    path = DATA_DIR / GENE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "GL" not in table.columns:
        raise ValueError(f"{path.name} must contain a 'GL' column.")
    return table


def main() -> None:
    config = load_user_config()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Place the scripts in the project's 'scripts' folder."
        )
    if not INDIVIDUAL_OUTPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Individual-model prediction directory not found: {INDIVIDUAL_OUTPUT_DIR}\n"
            "Run 01_ML_inference.py and 02_Trans_inference.py first."
        )

    gene_table = load_gene_table()
    expected_size = len(gene_table)
    prefix = make_prefix(config)
    input_paths = [
        INDIVIDUAL_OUTPUT_DIR / f"{prefix}_{name}.npy" for name in MODEL_NAMES
    ]
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        missing_names = "\n".join(f"  - {path.name}" for path in missing_paths)
        raise FileNotFoundError(
            "The six-model ensemble requires every expected prediction file. "
            f"Missing files:\n{missing_names}\n"
            "Run 01_ML_inference.py and 02_Trans_inference.py with the current "
            "00_Prepare_Config.py settings."
        )

    ensemble = None
    for index, path in enumerate(input_paths, start=1):
        print(f"[{index}/{len(input_paths)}] Loading: {path.name}")
        current_prediction = np.load(path).astype(np.float32)
        if current_prediction.shape != (expected_size,):
            raise ValueError(
                f"Unexpected shape in {path.name}: {current_prediction.shape}; "
                f"expected ({expected_size},)."
            )
        if not np.all(np.isfinite(current_prediction)):
            raise ValueError(f"{path.name} contains non-finite values.")
        if ensemble is None:
            ensemble = np.zeros_like(current_prediction, dtype=np.float32)
        ensemble += current_prediction
        del current_prediction
        gc.collect()

    ensemble /= len(input_paths)
    output_stem = f"{prefix}_Hybrid_Model"
    npy_path = OUTPUT_DIR / f"{output_stem}.npy"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    np.save(npy_path, ensemble)

    output = pd.DataFrame(
        {
            "Gene": gene_table["GL"],
            "Predicted_Expr_Change": ensemble,
        }
    )
    output.to_csv(csv_path, index=False)

    print(f"Saved: {npy_path.name}")
    print(f"Saved: {csv_path.name}")
    print("Six-model ensemble is complete.")


if __name__ == "__main__":
    main()
