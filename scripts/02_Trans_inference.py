"""Run the DigiAra Transformer model for a configured perturbation."""

from __future__ import annotations

import gc
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CONFIG_PATH = SCRIPT_DIR / "00_Prepare_Config.py"
DATA_DIR = PROJECT_DIR / "Model_and_data"
OUTPUT_DIR = PROJECT_DIR / "Prediction_results" / "Individual_Models"

GENE_FILE = "Ara_stand_protein_sorted_gl_entry.csv"
BASELINE_FILE = "Demo_Unperturb_Transcriptomes.tsv"
PLANT_EMBEDDING_FILE = "Ara_stand_protein__esmft32_embeddings1.npy"
TRANSFORMER_FILE = "Transformer.pt"
MICROBE_FILES = {
    "pstdc3000": ("PstDC3000", "Pst_DC3000_esmft32.mean_expand_27448.npy"),
    "mock": ("mock", "Allzero_27448M5120.npy"),
}

PERTURBATION_VALUES = {
    "KO": -float(np.log1p(1e6)),
    "OE": float(np.log1p(1e6)),
}


class GeneTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(x)
        x = self.transformer(x)
        return self.output_linear(x).squeeze(-1)


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
    if mode not in PERTURBATION_VALUES:
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
    if microbe_key not in MICROBE_FILES:
        supported = ", ".join(item[0] for item in MICROBE_FILES.values())
        raise ValueError(f"Unsupported MICROBE value. Choose one of: {supported}.")
    microbe_name, microbe_file = MICROBE_FILES[microbe_key]

    requested_name = str(raw.get("MUTANT_NAME", "")).strip()
    mutant_name = requested_name or ("_".join(genes) if genes else "WT")

    return {
        "mode": mode,
        "genes": genes,
        "mutant_name": safe_filename_component(mutant_name),
        "microbe_name": microbe_name,
        "microbe_file": microbe_file,
    }


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_gene_table() -> tuple[pd.DataFrame, np.ndarray]:
    path = require_file(DATA_DIR / GENE_FILE)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "GL" not in table.columns:
        raise ValueError(f"{path.name} must contain a 'GL' column.")
    gene_ids = table["GL"].str.strip().str.upper().to_numpy()
    if len(gene_ids) == 0 or np.any(gene_ids == ""):
        raise ValueError(f"{path.name} contains an empty gene list or blank gene IDs.")
    return table, gene_ids


def load_baseline(gene_ids: np.ndarray) -> np.ndarray:
    path = require_file(DATA_DIR / BASELINE_FILE)
    table = pd.read_csv(path, sep="\t", dtype={"GL": str})
    required_columns = {"GL", "meanA_log1p"}
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing_columns)}")

    baseline_gene_ids = table["GL"].str.strip().str.upper().to_numpy()
    if len(baseline_gene_ids) != len(gene_ids) or not np.array_equal(
        baseline_gene_ids, gene_ids
    ):
        raise ValueError(
            f"Gene rows in {path.name} must exactly match the order in {GENE_FILE}."
        )

    values = pd.to_numeric(table["meanA_log1p"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} contains missing or non-finite expression values.")
    return values


def apply_perturbation(
    baseline: np.ndarray,
    gene_ids: np.ndarray,
    target_genes: list[str],
    mode: str,
) -> np.ndarray:
    gene_to_first_index: dict[str, int] = {}
    for index, gene_id in enumerate(gene_ids):
        gene_to_first_index.setdefault(str(gene_id), index)

    missing = [gene for gene in target_genes if gene not in gene_to_first_index]
    if missing:
        raise ValueError(f"Mutant gene IDs not found in the gene list: {', '.join(missing)}")

    state = baseline.copy()
    replacement = np.float32(PERTURBATION_VALUES[mode])
    for gene in target_genes:
        state[gene_to_first_index[gene]] = replacement
    return state


def load_transformer(model_path: Path, input_dim: int, device: torch.device) -> GeneTransformer:
    model = GeneTransformer(input_dim=input_dim).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Transformer.pt must contain a model state dictionary.")

    checkpoint = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def run_transformer(
    model: GeneTransformer,
    plant_embedding: np.ndarray,
    microbe_embedding: np.ndarray,
    mutant_state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    num_genes = plant_embedding.shape[0]
    m_arr = microbe_embedding
    if m_arr.ndim > 1:
        m_arr = np.mean(m_arr, axis=0)
    m_2d = np.tile(m_arr, (num_genes, 1)).astype(np.float32)

    meta_sum_pred = np.zeros(num_genes, dtype=np.float32)
    state_matrix = mutant_state.reshape(-1, 1)
    plant_2d = plant_embedding * state_matrix
    combined_2d = np.concatenate([plant_2d, m_2d], axis=1)
    batch_x_tensor = (
        torch.tensor(combined_2d, dtype=torch.float32).unsqueeze(0).to(device)
    )

    print("Running Transformer encoder inference...")
    with torch.no_grad():
        proj_y_pred = model(batch_x_tensor).cpu().numpy()[0]

    meta_sum_pred += proj_y_pred
    result = meta_sum_pred / float(1)
    del (
        batch_x_tensor,
        combined_2d,
        plant_2d,
        state_matrix,
        m_2d,
        proj_y_pred,
        meta_sum_pred,
    )
    return result


def make_prefix(config: dict[str, object]) -> str:
    return "_".join(
        (
            "DigiAra",
            str(config["mode"]),
            str(config["mutant_name"]),
            safe_filename_component(config["microbe_name"]),
        )
    )


def save_prediction(
    prediction: np.ndarray,
    gene_table: pd.DataFrame,
    output_stem: str,
) -> None:
    npy_path = OUTPUT_DIR / f"{output_stem}.npy"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    np.save(npy_path, prediction.astype(np.float32, copy=False))

    output = pd.DataFrame(
        {
            "Gene": gene_table["GL"],
            "Predicted_Expr_Change": prediction,
        }
    )
    output.to_csv(csv_path, index=False)
    print(f"Saved: {npy_path.name}")
    print(f"Saved: {csv_path.name}")


def main() -> None:
    config = load_user_config()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Place the scripts in the project's 'scripts' folder."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gene_table, gene_ids = load_gene_table()
    baseline = load_baseline(gene_ids)
    mutant_state = apply_perturbation(
        baseline,
        gene_ids,
        config["genes"],
        str(config["mode"]),
    )

    plant_path = require_file(DATA_DIR / PLANT_EMBEDDING_FILE)
    microbe_path = require_file(DATA_DIR / str(config["microbe_file"]))
    model_path = require_file(DATA_DIR / TRANSFORMER_FILE)
    plant_embedding = np.load(plant_path).astype(np.float32)
    microbe_embedding = np.load(microbe_path).astype(np.float32)

    expected_rows = len(gene_ids)
    if plant_embedding.ndim != 2 or plant_embedding.shape[0] != expected_rows:
        raise ValueError(
            f"Unexpected plant embedding shape {plant_embedding.shape}; "
            f"expected ({expected_rows}, feature_dim)."
        )
    if microbe_embedding.shape != plant_embedding.shape:
        raise ValueError(
            f"Microbe embedding shape {microbe_embedding.shape} does not match "
            f"plant embedding shape {plant_embedding.shape}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA is unavailable. Transformer inference on CPU may be slow.")
    print(
        f"Configuration: mode={config['mode']}, mutant={config['mutant_name']}, "
        f"genes={config['genes'] or ['WT']}, microbe={config['microbe_name']}, "
        f"device={device}"
    )

    input_dim = int(plant_embedding.shape[1] * 2)
    print("Loading Transformer model...")
    model = load_transformer(model_path, input_dim, device)
    prediction = run_transformer(
        model,
        plant_embedding,
        microbe_embedding,
        mutant_state,
        device,
    )
    if prediction.size != expected_rows:
        raise ValueError(
            f"Transformer returned {prediction.size} values; expected {expected_rows}."
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Transformer returned non-finite prediction values.")

    save_prediction(prediction, gene_table, f"{make_prefix(config)}_Transformer")
    del model, prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("Transformer prediction is complete.")


if __name__ == "__main__":
    main()
