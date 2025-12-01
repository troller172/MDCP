from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import build_transforms, make_subset
from .models import ArchName, build_model, load_checkpoint


def _build_holdout_loader(dataset, holdout_indices: Sequence[int], batch_size: int, num_workers: int) -> DataLoader:
    transform = build_transforms(train=False)
    subset = make_subset(dataset, holdout_indices, transform, return_index=True)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def run_holdout_prediction(
    dataset,
    holdout_indices: Sequence[int],
    checkpoint_path: Path,
    output_dir: Path,
    arch: Optional[ArchName] = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch_name = arch or checkpoint.get("arch", "densenet121")
    model = build_model(arch=arch_name, num_classes=int(dataset.n_classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    loader = _build_holdout_loader(dataset, holdout_indices, batch_size=batch_size, num_workers=num_workers)

    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    all_indices: list[int] = []
    all_regions: list[int] = []
    metadata_records = []

    region_map = dataset.metadata_map.get("region") if dataset.metadata_map else None

    with torch.no_grad():
        for batch in loader:
            inputs, targets, metadata, indices = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(targets.cpu().tolist())
            all_indices.extend(indices.cpu().tolist())
            all_regions.extend(metadata[:, 0].cpu().tolist())

    probs_arr = np.concatenate(all_probs, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)

    for idx, label, region in zip(all_indices, all_labels, all_regions):
        metadata_idx = int(dataset.full_idxs[int(idx)])
        row = dataset.metadata.iloc[metadata_idx]
        metadata_records.append({
            "dataset_idx": int(idx),
            "metadata_idx": metadata_idx,
            "label": int(label),
            "region_id": int(region),
            "region": region_map[region] if region_map is not None and region < len(region_map) else str(region),
            "timestamp": row["timestamp"],
            "country_code": row.get("country_code", ""),
            "lon": row.get("longitude"),
            "lat": row.get("latitude"),
        })

    metadata_df = pd.DataFrame.from_records(metadata_records)
    preds = probs_arr.argmax(axis=1)
    accuracy = float((preds == metadata_df["label"].to_numpy()).mean())

    metadata_df.to_csv(output_dir / "holdout_metadata.csv", index=False)
    np.save(output_dir / "holdout_probabilities.npy", probs_arr)
    np.save(output_dir / "holdout_logits.npy", logits_arr)

    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "arch": arch_name,
        "n_examples": int(len(metadata_df)),
        "accuracy": accuracy,
    }
    with (output_dir / "prediction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
