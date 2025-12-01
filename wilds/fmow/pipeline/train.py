from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

try:
    from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]
    _autocast_signature = inspect.signature(torch_autocast)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore
    _autocast_signature = None

try:
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]
    _grad_scaler_signature = inspect.signature(TorchGradScaler.__init__)
except (ImportError, AttributeError):
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    _grad_scaler_signature = None

_AUTocast_SUPPORTS_DEVICE_TYPE = _autocast_signature is not None and "device_type" in _autocast_signature.parameters
_GRAD_SCALER_SUPPORTS_DEVICE_TYPE = (
    _grad_scaler_signature is not None and "device_type" in _grad_scaler_signature.parameters
)

from .data import DataLoaders, create_dataloaders
from .models import ArchName, build_model, load_checkpoint


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def _to_device(batch, device: torch.device):
    if len(batch) == 3:
        inputs, targets, metadata = batch
        indices = None
    else:
        inputs, targets, metadata, indices = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    metadata = metadata.to(device, non_blocking=True)
    if indices is not None:
        indices = torch.as_tensor(indices, device=device)
    return inputs, targets, metadata, indices


def _split_for_validation(indices: Sequence[int], val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.array(indices, dtype=np.int64)
    perm = rng.permutation(indices)
    n_val = max(1, int(math.floor(val_frac * len(perm))))
    if n_val >= len(perm):
        n_val = max(1, len(perm) // 5)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _make_grad_scaler(device_type: str, enabled: bool):
    scaler_kwargs = {"enabled": enabled}
    if _GRAD_SCALER_SUPPORTS_DEVICE_TYPE:
        scaler_kwargs["device_type"] = device_type
    return TorchGradScaler(**scaler_kwargs)


class Trainer:
    def __init__(
        self,
        dataset,
        train_indices: Sequence[int],
        holdout_indices: Optional[Sequence[int]],
        output_dir: Path,
        arch: ArchName = "densenet121",
        batch_size: int = 32,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        step_size: int = 20,
        gamma: float = 0.1,
        val_frac: float = 0.1,
        seed: int = 0,
        num_workers: int = 4,
        balance_regions: bool = True,
        use_amp: bool = True,
        resume_checkpoint: Optional[Path] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and self.device.type == "cuda"

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        train_idx, val_idx = _split_for_validation(train_indices, val_frac, seed)
        np.save(self.output_dir / "train_idx_used.npy", train_idx)
        np.save(self.output_dir / "val_idx.npy", val_idx)

        self.model = build_model(arch=arch, num_classes=int(dataset.n_classes), pretrained=True)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        scaler_device_type = self.device.type if self.device.type in {"cuda", "cpu"} else "cuda"
        self.scaler = _make_grad_scaler(device_type=scaler_device_type, enabled=self.use_amp)

        loaders = create_dataloaders(
            dataset=dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            holdout_indices=holdout_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            balance_regions=balance_regions,
        )
        self.loaders = loaders
        self.epochs = epochs
        self.arch = arch

        self.start_epoch = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_acc = 0.0

        if resume_checkpoint is not None and resume_checkpoint.exists():
            self._load_resume_state(resume_checkpoint)

    def _load_resume_state(self, checkpoint_path: Path) -> None:
        checkpoint = load_checkpoint(self.model, checkpoint_path.as_posix(), self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        default_scaler_state = _make_grad_scaler(
            device_type=self.device.type if self.device.type in {"cuda", "cpu"} else "cuda",
            enabled=self.use_amp,
        ).state_dict()
        self.scaler.load_state_dict(checkpoint.get("scaler_state", default_scaler_state))
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))

    def _run_epoch(self, loader: torch.utils.data.DataLoader, training: bool) -> EpochMetrics:
        mode = "train" if training else "eval"
        self.model.train(mode == "train")
        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for batch in loader:
            inputs, targets, _, _ = _to_device(batch, self.device)
            amp_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cuda"
            autocast_kwargs = {"enabled": self.use_amp}
            if _AUTocast_SUPPORTS_DEVICE_TYPE:
                autocast_kwargs["device_type"] = amp_device
            with torch_autocast(**autocast_kwargs):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            if training:
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / max(1, total_samples)
        acc = total_correct / max(1, total_samples)
        return EpochMetrics(loss=avg_loss, accuracy=acc)

    def fit(self) -> Dict[str, float]:
        history: List[Dict[str, float]] = []
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._run_epoch(self.loaders.train, training=True)
            val_metrics = None
            if self.loaders.val is not None:
                val_metrics = self._run_epoch(self.loaders.val, training=False)
                if val_metrics.accuracy > self.best_val_acc:
                    self.best_val_acc = val_metrics.accuracy
                    self.best_state = {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "scaler_state": self.scaler.state_dict() if self.use_amp else None,
                        "epoch": epoch,
                        "best_val_acc": self.best_val_acc,
                        "arch": self.arch,
                    }
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
            }
            if val_metrics is not None:
                record.update({
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "best_val_accuracy": self.best_val_acc,
                })
            history.append(record)
            log_path = self.output_dir / "training_log.json"
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        if self.best_state is None:
            self.best_state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.use_amp else None,
                "epoch": self.epochs - 1,
                "best_val_acc": self.best_val_acc,
                "arch": self.arch,
            }

        checkpoint_path = self.output_dir / "checkpoint_best.pt"
        torch.save(self.best_state, checkpoint_path)
        return {"checkpoint": checkpoint_path.as_posix(), "best_val_accuracy": self.best_val_acc}


def run_training(
    dataset,
    train_indices: Sequence[int],
    holdout_indices: Optional[Sequence[int]],
    output_dir: Path,
    arch: ArchName = "densenet121",
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    step_size: int = 20,
    gamma: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 0,
    num_workers: int = 4,
    balance_regions: bool = True,
    use_amp: bool = True,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, float]:
    trainer = Trainer(
        dataset=dataset,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        output_dir=output_dir,
        arch=arch,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        val_frac=val_frac,
        seed=seed,
        num_workers=num_workers,
        balance_regions=balance_regions,
        use_amp=use_amp,
        resume_checkpoint=resume_checkpoint,
    )
    result = trainer.fit()
    return result
