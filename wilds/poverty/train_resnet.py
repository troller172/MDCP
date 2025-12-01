#!/usr/bin/env python3
"""Train a ResNet18 multispectral regressor on the MDCP PovertyMap split.

We keep the standard ResNet18 multispectral backbone, 
but instead of leaving its final fully connected layer in place, 
we ask it for the ***penultimate feature vector*** (`backbone.d_out = 512`). 

The new `StudentTHead` takes that feature vector and produces three scalars per sample:
- `μ(x)` — directly from the linear layer,
- `scale(x)` — softplus + floor to ensure positivity,
- `nv(x)` — softplus + floor to keep degrees of freedom > 2.

Those parameters define a Student-t density for `Y|X`. 
- During training we maximise the Student-t log-likelihood, 
  and at inference we save both point metrics and the `{μ, scale, nv}` triplets. 
- So the backbone is unchanged except for bypassing its old single-output head; 
  the StudentTHead handles the density estimation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as T

from wilds._wilds_repo import prepare_wilds_repo


def _extend_sys_path(repo_root: Path) -> Optional[Path]:
    """Ensure the upstream WILDS repo (with examples/) is importable."""

    return prepare_wilds_repo(repo_root, require_examples=True)


@torch.no_grad()
def pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() <= 1 or y.numel() <= 1:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum())
    if denom.item() == 0:
        return float("nan")
    return float((vx * vy).sum() / denom)


def split_train_val(indices: Sequence[int], val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = np.array(indices, dtype=np.int64)
    perm = rng.permutation(idx)
    n_val = max(1, int(math.floor(val_frac * len(perm))))
    n_val = min(n_val, len(perm) - 1)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


@dataclass
class TrainingConfig:
    repo_root: str
    data_root: str
    output_dir: str
    train_indices_path: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_frac: float
    seed: int
    num_workers: int
    betas: Tuple[float, float]
    density_head: str
    subset: str
    scale_floor: float
    nu_floor: Optional[float]


@dataclass
class DensityOutputs:
    mean: torch.Tensor
    scale: torch.Tensor
    nu: Optional[torch.Tensor] = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class StudentTHead(nn.Module):
    """Predict Student-t location, scale, and degrees of freedom."""

    def __init__(self, in_features: int, scale_floor: float, nu_floor: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 3)
        self.softplus = nn.Softplus()
        self.scale_floor = float(scale_floor)
        self.nu_floor = float(nu_floor)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_mean, raw_scale, raw_nu = torch.chunk(self.linear(feats), chunks=3, dim=-1)
        scale = self.softplus(raw_scale) + self.scale_floor
        nu = self.softplus(raw_nu) + self.nu_floor
        return raw_mean.squeeze(-1), scale.squeeze(-1), nu.squeeze(-1)


class StudentTRegressor(nn.Module):
    """Wrap a backbone to emit Student-t parameters."""

    def __init__(self, backbone: nn.Module, head: StudentTHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> DensityOutputs:
        feats = self.backbone(x)
        mean, scale, nu = self.head(feats)
        return DensityOutputs(mean=mean, scale=scale, nu=nu)


class GaussianHead(nn.Module):
    """Predict Gaussian location and scale parameters."""

    def __init__(self, in_features: int, scale_floor: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2)
        self.softplus = nn.Softplus()
        self.scale_floor = float(scale_floor)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_mean, raw_scale = torch.chunk(self.linear(feats), chunks=2, dim=-1)
        scale = self.softplus(raw_scale) + self.scale_floor
        return raw_mean.squeeze(-1), scale.squeeze(-1)


class GaussianRegressor(nn.Module):
    """Wrap a backbone to emit Gaussian parameters."""

    def __init__(self, backbone: nn.Module, head: GaussianHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> DensityOutputs:
        feats = self.backbone(x)
        mean, scale = self.head(feats)
        return DensityOutputs(mean=mean, scale=scale, nu=None)


GAUSSIAN_EFFECTIVE_NU: float = 1e6


def student_t_nll(targets: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Student-t negative log-likelihood."""

    if not (targets.shape == mean.shape == scale.shape == nu.shape):
        raise ValueError("Targets, mean, scale, and nu must share shape")
    scale = torch.clamp(scale, min=1e-6)
    nu = torch.clamp(nu, min=2.0 + 1e-6)

    log_norm = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * (torch.log(nu) + math.log(math.pi))
        - torch.log(scale)
    )
    sq_term = ((targets - mean) / scale) ** 2
    log_inner = torch.log1p(sq_term / nu)
    log_power = -((nu + 1.0) / 2.0) * log_inner
    return -(log_norm + log_power)


def gaussian_nll(targets: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Gaussian negative log-likelihood."""

    if not (targets.shape == mean.shape == scale.shape):
        raise ValueError("Targets, mean, and scale must share shape")
    scale = torch.clamp(scale, min=1e-6)
    var = scale ** 2
    log_term = torch.log(scale) + 0.5 * math.log(2.0 * math.pi)
    sq_term = 0.5 * ((targets - mean) ** 2) / var
    return log_term + sq_term


def filter_indices_by_subset(
    indices: Sequence[int],
    metadata_array: torch.Tensor,
    subset: Literal["all", "urban", "rural"],
) -> List[int]:
    if subset == "all":
        return list(indices)

    if metadata_array.ndim != 2 or metadata_array.size(1) < 1:
        raise ValueError("Metadata array must have urban flag in first column")

    is_urban = metadata_array[:, 0] > 0.5
    mask = is_urban if subset == "urban" else ~is_urban
    filtered = [idx for idx in indices if bool(mask[idx].item())]
    return filtered


def build_transforms() -> Dict[str, T.Compose]:
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])
    val_transform = T.Compose([])
    return {"train": train_transform, "val": val_transform}


def prepare_dataloaders(
    dataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    transforms: Dict[str, T.Compose],
) -> Tuple[DataLoader, DataLoader]:
    from wilds.datasets.wilds_dataset import WILDSSubset  # type: ignore

    train_subset = WILDSSubset(dataset, train_indices, transform=transforms["train"])
    val_subset = WILDSSubset(dataset, val_indices, transform=transforms["val"])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model_state: Dict[str, torch.Tensor], optimizer_state: Dict[str, torch.Tensor], epoch: int, best_metric: float) -> None:
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "best_val_nll": best_metric,
    }
    torch.save(payload, path)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    output_dir: Path,
    loss_fn: Callable[[torch.Tensor, DensityOutputs], torch.Tensor],
) -> List[Dict[str, float]]:
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[Dict[str, float]] = []
    best_val_nll = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        train_nll_sum = 0.0
        train_sq_error_sum = 0.0
        train_samples = 0
        train_preds: List[torch.Tensor] = []
        train_targets: List[torch.Tensor] = []

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().squeeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            batch_nll = loss_fn(targets, outputs)
            loss = batch_nll.mean()
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            train_nll_sum += float(batch_nll.sum().item())
            mean_detached = outputs.mean.detach()
            train_sq_error_sum += float(torch.sum((mean_detached - targets) ** 2).item())
            train_samples += batch_size
            train_preds.append(mean_detached.cpu())
            train_targets.append(targets.detach().cpu())

        train_nll = train_nll_sum / max(1, train_samples)
        train_rmse = math.sqrt(train_sq_error_sum / max(1, train_samples)) if train_samples > 0 else float("nan")
        if train_preds and train_targets:
            train_pearson = pearsonr_torch(torch.cat(train_preds), torch.cat(train_targets))
        else:
            train_pearson = float("nan")

        model.eval()
        val_nll_sum = 0.0
        val_sq_error_sum = 0.0
        val_samples = 0
        val_preds: List[torch.Tensor] = []
        val_targets: List[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float().squeeze(1)
                outputs = model(inputs)
                batch_nll = loss_fn(targets, outputs)
                val_nll_sum += float(batch_nll.sum().item())
                mean_detached = outputs.mean.detach()
                val_sq_error_sum += float(torch.sum((mean_detached - targets) ** 2).item())
                batch_size = targets.size(0)
                val_samples += batch_size
                val_preds.append(mean_detached.cpu())
                val_targets.append(targets.detach().cpu())

        val_nll = val_nll_sum / max(1, val_samples)
        val_rmse = math.sqrt(val_sq_error_sum / max(1, val_samples)) if val_samples > 0 else float("nan")
        if val_preds and val_targets:
            val_pearson = pearsonr_torch(torch.cat(val_preds), torch.cat(val_targets))
        else:
            val_pearson = float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "train_rmse": train_rmse,
                "train_pearson": train_pearson,
                "val_nll": val_nll,
                "val_rmse": val_rmse,
                "val_pearson": val_pearson,
            }
        )

        log_path = output_dir / "training_log.json"
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(
                output_dir / "best_model.pth",
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                best_val_nll,
            )

        scheduler.step()

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_checkpoint(
            output_dir / "best_model.pth",
            model.state_dict(),
            optimizer.state_dict(),
            epochs - 1,
            history[-1]["val_nll"],
        )

    final_ckpt = {
        "model_state": best_state["model_state_dict"],
        "optimizer_state": best_state["optimizer_state_dict"],
        "history": history,
    }
    torch.save(final_ckpt, output_dir / "final_checkpoint.pth")
    return history


def export_density_parameters(
    model: nn.Module,
    dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_path: Path,
    transform: T.Compose,
    head_type: str,
    gaussian_nu: float = GAUSSIAN_EFFECTIVE_NU,
) -> None:
    from wilds.datasets.wilds_dataset import WILDSSubset  # type: ignore

    subset = WILDSSubset(dataset, indices, transform=transform)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataset_indices = np.array(subset.indices, dtype=np.int64)
    cursor = 0
    idx_batches: List[np.ndarray] = []
    mean_batches: List[np.ndarray] = []
    scale_batches: List[np.ndarray] = []
    nu_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    urban_batches: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for inputs, targets, metadata in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().squeeze(1)
            outputs = model(inputs)
            nu_tensor: torch.Tensor
            if outputs.nu is not None:
                nu_tensor = outputs.nu
            else:
                nu_tensor = torch.full_like(outputs.mean, float(gaussian_nu))
            mean = outputs.mean
            scale = outputs.scale

            batch_len = inputs.size(0)
            batch_indices = dataset_indices[cursor:cursor + batch_len]
            cursor += batch_len

            idx_batches.append(batch_indices)
            mean_batches.append(mean.detach().cpu().numpy().astype(np.float32))
            scale_batches.append(scale.detach().cpu().numpy().astype(np.float32))
            nu_batches.append(nu_tensor.detach().cpu().numpy().astype(np.float32))
            target_batches.append(targets.detach().cpu().numpy().astype(np.float32))
            urban_batches.append(metadata[:, 0].detach().cpu().numpy().astype(np.float32))

    if cursor != len(dataset_indices):
        raise RuntimeError("Did not consume all subset indices when exporting density parameters")

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        indices=np.concatenate(idx_batches, axis=0),
        mean=np.concatenate(mean_batches, axis=0),
        scale=np.concatenate(scale_batches, axis=0),
        nu=np.concatenate(nu_batches, axis=0),
        target=np.concatenate(target_batches, axis=0),
        urban=np.concatenate(urban_batches, axis=0),
        head_type=np.array(head_type),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train resnet18_ms on MDCP poverty split")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-indices", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument(
        "--density-head",
        type=str,
        default="gaussian",
        choices=["gaussian", "student_t"],
        help="Conditional density head to use for regression",
    )
    parser.add_argument("--subset", type=str, default="all", choices=["all", "urban", "rural"], help="Subset of data to train on")
    parser.add_argument("--scale-floor", type=float, default=1e-3, help="Minimum scale (added after softplus)")
    parser.add_argument(
        "--nu-floor",
        type=float,
        default=2.01,
        help="Minimum degrees of freedom (added after softplus; used only for Student-t head)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    _extend_sys_path(repo_root)

    from wilds.datasets.poverty_dataset import PovertyMapDataset  # type: ignore
    from examples.models.resnet_multispectral import ResNet18  # type: ignore

    data_root = repo_root / "data"
    default_run_dir = repo_root / "eval_out" / "poverty" / "training" / f"run_{args.density_head}_{args.subset}"
    output_dir = args.output_dir or default_run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices_path = args.train_indices or (repo_root / "eval_out" / "poverty" / "splits" / "train_indices.json")
    train_indices = json.loads(train_indices_path.read_text(encoding="utf-8"))
    if not train_indices:
        raise RuntimeError("No training indices found")

    set_seed(args.seed)

    dataset = PovertyMapDataset(version="1.1", root_dir=data_root.as_posix(), download=False, split_scheme="official")

    metadata_array = dataset.metadata_array
    filtered_indices = filter_indices_by_subset(train_indices, metadata_array, args.subset)
    if len(filtered_indices) < 2:
        raise RuntimeError(f"Subset '{args.subset}' yielded too few samples ({len(filtered_indices)})")

    train_arr, val_arr = split_train_val(filtered_indices, val_frac=args.val_frac, seed=args.seed)
    print(f"Subset '{args.subset}': {len(train_arr)} train / {len(val_arr)} val samples (from {len(filtered_indices)} eligible indices)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = ResNet18(num_classes=None, num_channels=8)
    if args.density_head == "gaussian":
        head = GaussianHead(in_features=backbone.d_out, scale_floor=args.scale_floor)
        model = GaussianRegressor(backbone=backbone, head=head)

        def loss_fn(targets: torch.Tensor, outputs: DensityOutputs) -> torch.Tensor:
            return gaussian_nll(targets, outputs.mean, outputs.scale)

        effective_nu_floor: Optional[float] = None
    elif args.density_head == "student_t":
        if args.nu_floor is None:
            raise ValueError("--nu-floor must be provided for the Student-t head")
        head = StudentTHead(in_features=backbone.d_out, scale_floor=args.scale_floor, nu_floor=args.nu_floor)
        model = StudentTRegressor(backbone=backbone, head=head)

        def loss_fn(targets: torch.Tensor, outputs: DensityOutputs) -> torch.Tensor:
            if outputs.nu is None:
                raise ValueError("Student-t head must return degrees of freedom")
            return student_t_nll(targets, outputs.mean, outputs.scale, outputs.nu)

        effective_nu_floor = float(args.nu_floor)
    else:
        raise ValueError(f"Unsupported density head: {args.density_head}")
    model.to(device)

    transforms = build_transforms()
    train_loader, val_loader = prepare_dataloaders(
        dataset=dataset,
        train_indices=train_arr,
        val_indices=val_arr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms,
    )

    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        output_dir=output_dir,
        loss_fn=loss_fn,
    )

    config = TrainingConfig(
        repo_root=str(repo_root),
        data_root=str(data_root),
        output_dir=str(output_dir),
        train_indices_path=str(train_indices_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        betas=tuple(args.betas),
        density_head=args.density_head,
        subset=args.subset,
        scale_floor=args.scale_floor,
        nu_floor=effective_nu_floor,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    export_density_parameters(
        model=model,
        dataset=dataset,
        indices=train_arr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_path=output_dir / f"density_params_train_{args.subset}.npz",
        transform=transforms["val"],
        head_type=args.density_head,
    )
    export_density_parameters(
        model=model,
        dataset=dataset,
        indices=val_arr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_path=output_dir / f"density_params_val_{args.subset}.npz",
        transform=transforms["val"],
        head_type=args.density_head,
    )

    print(f"Training complete. History length: {len(history)} epochs")
    print(f"Artifacts stored in {output_dir}")


if __name__ == "__main__":
    main()
