#!/usr/bin/env python3
"""Run conditional density predictions (Gaussian or Student-t) on the MDCP PovertyMap split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
if _DEFAULT_REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _DEFAULT_REPO_ROOT.as_posix())

from wilds.poverty.train_resnet import (
    GaussianHead,
    GaussianRegressor,
    StudentTHead,
    StudentTRegressor,
    _extend_sys_path,
    build_transforms,
    export_density_parameters,
    filter_indices_by_subset,
    gaussian_nll,
    student_t_nll,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Student-t density on MDCP PovertyMap pending split")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="MDCP repository root")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint (best_model.pth)")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional path to training config.json to infer defaults")
    parser.add_argument("--subset", type=str, choices=["all", "urban", "rural"], default=None, help="Subset to evaluate (defaults to training subset if config is provided)")
    parser.add_argument("--split-csv", type=Path, default=None, help="Path to mdcp_split.csv with train/pending labels")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--density-head", type=str, choices=["gaussian", "student_t"], default=None, help="Density head used by the checkpoint (defaults to training config)")
    parser.add_argument("--scale-floor", type=float, default=None, help="Scale floor added after softplus (overrides config)")
    parser.add_argument("--nu-floor", type=float, default=None, help="Nu floor added after softplus (overrides config; Student-t only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store predictions and summaries")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of pending samples (for smoke tests)")
    return parser.parse_args()


def load_config(config_path: Optional[Path]) -> Dict[str, object]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config path {config_path} does not exist")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} is not a JSON object")
    return data


def resolve_parameter(value: Optional[float], config: Dict[str, object], key: str, default: float) -> float:
    if value is not None:
        return float(value)
    if key in config:
        return float(config[key])
    return default


def resolve_subset(cli_subset: Optional[str], config: Dict[str, object]) -> str:
    if cli_subset is not None:
        return cli_subset
    subset = config.get("subset")
    if isinstance(subset, str):
        return subset
    raise ValueError("Subset must be specified either via --subset or config file")


VALID_DENSITY_HEADS = {"gaussian", "student_t"}


def resolve_head(cli_head: Optional[str], config: Dict[str, object]) -> str:
    if cli_head is not None:
        return cli_head
    head = config.get("density_head")
    if isinstance(head, str) and head in VALID_DENSITY_HEADS:
        return head
    return "student_t"


def extract_head_type(payload: np.lib.npyio.NpzFile) -> Optional[str]:  # type: ignore[attr-defined]
    if not hasattr(payload, "files"):
        return None
    if "head_type" not in payload.files:
        return None
    value = payload["head_type"]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size >= 1:
            return str(value.reshape(-1)[0])
    elif isinstance(value, str):
        return value
    return None


def compute_summary(pred_path: Path, summary_path: Path, expected_head: str) -> None:
    payload = np.load(pred_path)
    payload_head = extract_head_type(payload)
    head_type = payload_head or expected_head
    if head_type not in VALID_DENSITY_HEADS:
        raise ValueError(f"Unsupported head type '{head_type}' for summary computation")
    if payload_head is not None and expected_head is not None and payload_head != expected_head:
        raise ValueError(f"Head type mismatch between file ({payload_head}) and expected ({expected_head})")

    target = payload["target"].astype(float, copy=False)
    mean = payload["mean"].astype(float, copy=False)
    scale = payload["scale"].astype(float, copy=False)

    target_t = torch.from_numpy(target)
    mean_t = torch.from_numpy(mean)
    scale_t = torch.from_numpy(scale)

    if head_type == "student_t":
        nu = payload["nu"].astype(float, copy=False)
        nu_t = torch.from_numpy(nu)
        nll = float(student_t_nll(target_t, mean_t, scale_t, nu_t).mean().item())
    else:
        nll = float(gaussian_nll(target_t, mean_t, scale_t).mean().item())

    rmse = float(np.sqrt(np.mean((mean - target) ** 2)))
    pearson = float(np.corrcoef(mean, target)[0, 1]) if target.size > 1 else float("nan")

    summary = {
        "num_samples": int(target.size),
        "avg_nll": nll,
        "rmse": rmse,
        "pearson": pearson,
        "head_type": head_type,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_csv(pred_path: Path, csv_path: Path, expected_head: str) -> None:
    payload = np.load(pred_path)
    payload_head = extract_head_type(payload)
    if payload_head is not None and payload_head != expected_head:
        raise ValueError(f"Head type mismatch between file ({payload_head}) and expected ({expected_head})")
    header = ["index", "mean", "scale", "nu", "target", "urban"]
    data = np.column_stack([
        payload["indices"],
        payload["mean"],
        payload["scale"],
        payload["nu"],
        payload["target"],
        payload["urban"],
    ])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="", fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.0f"])


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    _extend_sys_path(repo_root)

    config = load_config(args.config_path)
    subset = resolve_subset(args.subset, config)
    head_type = resolve_head(args.density_head, config)

    scale_floor = resolve_parameter(args.scale_floor, config, "scale_floor", 1e-3)
    nu_floor: Optional[float] = None
    if head_type == "student_t":
        nu_floor = resolve_parameter(args.nu_floor, config, "nu_floor", 2.01)

    data_root = repo_root / "data"
    split_csv = args.split_csv or (data_root / "poverty_v1.1" / "mdcp_split.csv")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV {split_csv} not found")

    import pandas as pd  # defer heavy import until needed

    split_df = pd.read_csv(split_csv)
    pending_indices = split_df.loc[split_df["split"] == "pending", "index"].to_numpy(dtype=np.int64)
    if pending_indices.size == 0:
        raise RuntimeError("No pending indices found in split CSV")

    from wilds.datasets.poverty_dataset import PovertyMapDataset  # type: ignore
    from examples.models.resnet_multispectral import ResNet18  # type: ignore

    dataset = PovertyMapDataset(version="1.1", root_dir=data_root.as_posix(), download=False, split_scheme="official")

    metadata_array = dataset.metadata_array
    pending_filtered = filter_indices_by_subset(pending_indices, metadata_array, subset)
    if args.limit is not None:
        pending_filtered = pending_filtered[: args.limit]
    if len(pending_filtered) == 0:
        raise RuntimeError(f"No pending samples remain after applying subset '{subset}'")

    transforms = build_transforms()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    backbone = ResNet18(num_classes=None, num_channels=8)
    if head_type == "gaussian":
        head = GaussianHead(in_features=backbone.d_out, scale_floor=scale_floor)
        model = GaussianRegressor(backbone=backbone, head=head)
    else:
        if nu_floor is None:
            raise ValueError("nu_floor must be provided for Student-t head predictions")
        head = StudentTHead(in_features=backbone.d_out, scale_floor=scale_floor, nu_floor=nu_floor)
        model = StudentTRegressor(backbone=backbone, head=head)
    model.to(device)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint  # assume raw state dict
    else:
        raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_npz_path = output_dir / f"density_params_pending_{subset}.npz"

    export_density_parameters(
        model=model,
        dataset=dataset,
        indices=pending_filtered,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_path=pred_npz_path,
        transform=transforms["val"],
        head_type=head_type,
    )

    csv_path = output_dir / f"density_params_pending_{subset}.csv"
    write_csv(pred_npz_path, csv_path, expected_head=head_type)

    summary_path = output_dir / "prediction_summary.json"
    compute_summary(pred_npz_path, summary_path, expected_head=head_type)

    print(f"Saved NPZ predictions to {pred_npz_path}")
    print(f"Saved CSV predictions to {csv_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
