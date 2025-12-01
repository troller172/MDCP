#!/usr/bin/env python3
"""Evaluate MDCP and baseline conformal methods on PovertyMap pending predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

# Ensure project modules are importable
SCRIPT_DIR = Path(__file__).resolve().parent
POVERTY_ROOT = SCRIPT_DIR.parent
WILDS_ROOT = POVERTY_ROOT.parent
REPO_ROOT = WILDS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "model"))
if str(REPO_ROOT / "notebook") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "notebook"))

from notebook.data_utils import reconstruct_source_data  # type: ignore  # noqa: E402
from notebook.eval_utils import (  # type: ignore  # noqa: E402
    _split_mimic_sets,
    evaluate_mdcp_regression_performance,
    evaluate_regression_performance_with_individual_sets,
    generate_y_grid_regression,
)
from model.MDCP import (  # type: ignore  # noqa: E402
    aggregated_conformal_set_multi,
    fit_lambda_from_sources,
    precompute_calibration_cache,
)
from model.const import RANDOM_SEED, ensure_project_dir, prefer_relative_path  # type: ignore  # noqa: E402

try:  # Prefer SciPy for numerical stability if available
    from scipy.special import gammaln as _gammaln  # type: ignore
except Exception:  # pragma: no cover - fallback when SciPy unavailable
    def _gammaln(x: np.ndarray) -> np.ndarray:
        vec = np.vectorize(math.lgamma, otypes=[float])
        return vec(x)


MIMIC_CAL_RATIO = 0.5


@dataclass(frozen=True)
class ColumnMap:
    """Column indices holding Student-t parameters within the feature matrix."""

    mu: int
    log_scale: int
    log_nu: int


@dataclass(frozen=True)
class EvaluationConfig:
    alpha: float
    train_frac: float
    cal_frac: float
    test_frac: float
    num_trials: int
    random_seed: int
    gamma_values: Tuple[float, ...]
    y_grid_size: int
    y_margin: float


class StudentTPredictionSource:
    """Source model interface backed by precomputed Student-t parameters."""

    def __init__(self, column_map: ColumnMap, name: str = "") -> None:
        self.column_map = column_map
        self.name = name or "student_t_source"

    def _extract_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        mu = X_arr[:, self.column_map.mu]
        scale = np.exp(X_arr[:, self.column_map.log_scale])
        nu = np.exp(X_arr[:, self.column_map.log_nu])
        return mu, np.maximum(scale, 1e-8), np.maximum(nu, 2.0001)

    def predict_mu(self, X: np.ndarray) -> np.ndarray:
        mu, _, _ = self._extract_params(X)
        return mu

    def predict_sigma(self, X: np.ndarray) -> np.ndarray:
        _, scale, nu = self._extract_params(X)
        nu_safe = np.maximum(nu, 2.0001)
        sigma = scale * np.sqrt(nu_safe / np.maximum(nu_safe - 2.0, 1e-6))
        return np.maximum(sigma, 1e-6)

    def joint_pdf_at_pairs(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        mu, scale, nu = self._extract_params(X)
        y_arr = np.asarray(Y, dtype=float)
        if y_arr.ndim == 0:
            y_arr = np.full(mu.shape[0], float(y_arr))
        return _student_t_pdf(y_arr, mu, scale, nu)

    def joint_pdf(self, X: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        mu, scale, nu = self._extract_params(X)
        y = np.asarray(y_grid, dtype=float)
        mu_exp = mu[:, None]
        scale_exp = scale[:, None]
        nu_exp = nu[:, None]
        y_exp = y[None, :]
        return _student_t_pdf(y_exp, mu_exp, scale_exp, nu_exp)

    def marginal_pdf_x(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return np.ones(X_arr.shape[0], dtype=float)


def _student_t_pdf(y: np.ndarray, mu: np.ndarray, scale: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Evaluate the Student-t PDF with broadcasting-friendly inputs."""

    scale_safe = np.maximum(scale, 1e-8)
    nu_safe = np.maximum(nu, 2.0001)
    z = (y - mu) / scale_safe
    log_norm = (
        _gammaln((nu_safe + 1.0) / 2.0)
        - _gammaln(nu_safe / 2.0)
        - 0.5 * (np.log(nu_safe) + np.log(np.pi))
        - np.log(scale_safe)
    )
    log_power = -((nu_safe + 1.0) / 2.0) * np.log1p((z ** 2) / nu_safe)
    log_pdf = log_norm + log_power
    return np.exp(np.clip(log_pdf, a_min=-700.0, a_max=700.0))


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    scores_sorted = np.sort(np.asarray(scores, dtype=float))
    n = scores_sorted.size
    if n == 0:
        raise ValueError("Calibration scores are empty.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = int(np.clip(k, 1, n))
    q_level = (k - 1) / n
    try:
        tau = float(np.quantile(scores_sorted, q_level, method="higher"))
    except TypeError:  # NumPy < 1.22 fallback
        tau = float(np.quantile(scores_sorted, q_level, interpolation="higher"))
    return tau


def format_float(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


def compute_baseline_metrics(
    sources: Sequence[StudentTPredictionSource],
    X_sources_cal: Sequence[np.ndarray],
    Y_sources_cal: Sequence[np.ndarray],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
    source_test: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    thresholds: List[float] = []
    per_source_sets: List[List[np.ndarray]] = []
    single_source_metrics: Dict[str, Dict[str, object]] = {}
    for src_model, X_cal_j, Y_cal_j in zip(sources, X_sources_cal, Y_sources_cal):
        mu_cal = src_model.predict_mu(X_cal_j)
        sigma_cal = src_model.predict_sigma(X_cal_j)
        scores = np.abs(np.asarray(Y_cal_j, dtype=float) - mu_cal) / np.maximum(sigma_cal, 1e-6)
        tau = conformal_threshold(scores, alpha)
        thresholds.append(tau)

    intervals_lower = []
    intervals_upper = []
    for tau, src_model in zip(thresholds, sources):
        mu_test = src_model.predict_mu(X_test)
        sigma_test = src_model.predict_sigma(X_test)
        lower = mu_test - tau * sigma_test
        upper = mu_test + tau * sigma_test
        intervals_lower.append(lower)
        intervals_upper.append(upper)
        per_source_sets.append([np.array([lo, hi], dtype=float) for lo, hi in zip(lower, upper)])

        single_metrics = evaluate_regression_performance_with_individual_sets(
            {"lower": lower, "upper": upper},
            Y_test,
            individual_info=None,
            source_test=source_test,
        )
        compact_single = {
            "coverage": float(single_metrics.get("coverage", np.nan)),
            "avg_width": float(single_metrics.get("avg_width", np.nan)),
            "individual_coverage": np.asarray(single_metrics.get("individual_coverage", []), dtype=float),
            "individual_widths": np.asarray(single_metrics.get("individual_widths", []), dtype=float),
            "unique_sources": np.asarray(single_metrics.get("unique_sources", []), dtype=int),
        }
        single_source_metrics[src_model.name] = compact_single

    lower_stack = np.vstack(intervals_lower)
    upper_stack = np.vstack(intervals_upper)
    direct_union = {
        "lower": np.min(lower_stack, axis=0),
        "upper": np.max(upper_stack, axis=0),
    }
    individual_info = {"individual_sets_all": per_source_sets}
    metrics = evaluate_regression_performance_with_individual_sets(
        direct_union,
        Y_test,
        individual_info=individual_info,
        source_test=source_test,
    )
    metrics["thresholds"] = thresholds
    return metrics, single_source_metrics


def allocate_counts(total: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    weights = np.asarray(fractions, dtype=float)
    if not np.all(weights >= 0):
        raise ValueError("Split fractions must be non-negative.")
    if np.sum(weights) == 0:
        raise ValueError("At least one split fraction must be positive.")
    normalized = weights / np.sum(weights)
    raw = normalized * total
    counts = np.floor(raw).astype(int)
    remainder = int(total - np.sum(counts))
    if remainder > 0:
        fractional = raw - counts
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def to_serializable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def summarize_trials(per_trial: Sequence[Dict[str, object]], metric_keys: Sequence[str]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for key in metric_keys:
        collected: List[np.ndarray] = []
        for entry in per_trial:
            if key not in entry:
                continue
            value = entry[key]
            arr = np.asarray(value, dtype=float)
            collected.append(arr)
        if not collected:
            continue
        stacked = np.stack(collected, axis=0)
        mean_val = stacked.mean(axis=0)
        std_val = stacked.std(axis=0, ddof=0)
        summary[key] = {
            "mean": to_serializable(mean_val),
            "std": to_serializable(std_val),
        }
    return summary


def build_dataset(
    rural_on_rural: Path,
    urban_on_rural: Path,
    rural_on_urban: Path,
    urban_on_urban: Path,
) -> Dict[str, object]:
    npz_paths = {
        "rural_model_rural_data": rural_on_rural,
        "urban_model_rural_data": urban_on_rural,
        "rural_model_urban_data": rural_on_urban,
        "urban_model_urban_data": urban_on_urban,
    }

    payloads: Dict[str, Mapping[str, np.ndarray]] = {}
    head_types: List[str] = []
    for label, path in npz_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        raw_payload = dict(np.load(path))
        payloads[label] = raw_payload
        head_arr = raw_payload.get("head_type")
        if head_arr is not None:
            if isinstance(head_arr, np.ndarray):
                if head_arr.shape == ():
                    head_types.append(str(head_arr.item()))
                elif head_arr.size >= 1:
                    head_types.append(str(head_arr.reshape(-1)[0]))
            elif isinstance(head_arr, str):
                head_types.append(head_arr)

    if head_types:
        unique_heads = {ht for ht in head_types if ht}
        if len(unique_heads) > 1:
            raise ValueError(f"Mismatched head types across prediction files: {sorted(unique_heads)}")
        resolved_head = unique_heads.pop() if unique_heads else "student_t"
    else:
        resolved_head = "student_t"

    records: MutableMapping[int, Dict[str, float]] = {}

    def ingest(model_label: str, payload: Mapping[str, np.ndarray]) -> None:
        indices = payload["indices"].astype(int)
        target = payload["target"].astype(float)
        urban_flag = payload["urban"].astype(int)
        mean = payload["mean"].astype(float)
        scale = payload["scale"].astype(float)
        nu = payload["nu"].astype(float)
        for idx, y, urb, mu, sc, df in zip(indices, target, urban_flag, mean, scale, nu):
            entry = records.setdefault(idx, {})
            if "target" in entry and not math.isclose(entry["target"], float(y), rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Target mismatch for index {idx} across prediction files")
            entry["target"] = float(y)
            urb_int = int(urb)
            if "urban" in entry and entry["urban"] != urb_int:
                raise ValueError(f"Urban flag mismatch for index {idx} across prediction files")
            entry["urban"] = urb_int
            key_prefix = model_label
            entry[f"{key_prefix}_mean"] = float(mu)
            entry[f"{key_prefix}_scale"] = float(sc)
            entry[f"{key_prefix}_nu"] = float(df)

    ingest("rural", payloads["rural_model_rural_data"])
    ingest("urban", payloads["urban_model_rural_data"])
    ingest("rural", payloads["rural_model_urban_data"])
    ingest("urban", payloads["urban_model_urban_data"])

    indices = np.array(sorted(records), dtype=int)
    n = indices.size
    if n == 0:
        raise RuntimeError("No records were constructed from the supplied prediction files.")

    features = np.zeros((n, 6), dtype=float)
    targets = np.zeros(n, dtype=float)
    sources = np.zeros(n, dtype=int)

    for row, idx in enumerate(indices):
        entry = records[idx]
        required = {"rural_mean", "rural_scale", "rural_nu", "urban_mean", "urban_scale", "urban_nu"}
        if not required.issubset(entry):
            raise RuntimeError(f"Incomplete predictions for index {idx}.")
        targets[row] = entry["target"]
        sources[row] = entry["urban"]
        features[row, 0] = entry["rural_mean"]
        features[row, 1] = math.log(entry["rural_scale"])
        features[row, 2] = math.log(entry["rural_nu"])
        features[row, 3] = entry["urban_mean"]
        features[row, 4] = math.log(entry["urban_scale"])
        features[row, 5] = math.log(entry["urban_nu"])

    return {
        "indices": indices,
        "targets": targets,
        "sources": sources,
        "features": features,
        "column_maps": {
            "rural": ColumnMap(mu=0, log_scale=1, log_nu=2),
            "urban": ColumnMap(mu=3, log_scale=4, log_nu=5),
        },
        "source_order": ("rural", "urban"),
        "head_type": resolved_head,
    }


def run_trial(
    trial_idx: int,
    dataset: Mapping[str, object],
    config: EvaluationConfig,
) -> Dict[str, object]:
    rng = np.random.default_rng(config.random_seed + trial_idx)

    features = np.asarray(dataset["features"], dtype=float)
    targets = np.asarray(dataset["targets"], dtype=float)
    sources = np.asarray(dataset["sources"], dtype=int)
    column_maps: Mapping[str, ColumnMap] = dataset["column_maps"]  # type: ignore[assignment]
    source_order: Sequence[str] = dataset["source_order"]  # type: ignore[assignment]

    n_sources = len(source_order)
    per_source_indices = [np.where(sources == j)[0] for j in range(n_sources)]
    train_idx_list: List[np.ndarray] = []
    cal_idx_list: List[np.ndarray] = []
    test_idx_list: List[np.ndarray] = []

    for src_id, src_indices in enumerate(per_source_indices):
        if src_indices.size == 0:
            raise ValueError(f"Source {src_id} has no samples in the dataset.")
        permuted = rng.permutation(src_indices)
        train_count, cal_count, test_count = allocate_counts(
            permuted.size,
            (config.train_frac, config.cal_frac, config.test_frac),
        )
        if cal_count == 0 or test_count == 0:
            raise ValueError(
                f"Source {src_id} split yields insufficient calibration/test samples "
                f"(cal={cal_count}, test={test_count}). Adjust fractions."
            )
        start_cal = train_count
        start_test = train_count + cal_count
        train_idx_list.append(np.sort(permuted[:train_count]))
        cal_idx_list.append(np.sort(permuted[start_cal:start_test]))
        test_idx_list.append(np.sort(permuted[start_test:start_test + test_count]))

    train_idx = np.concatenate(train_idx_list)
    cal_idx = np.concatenate(cal_idx_list)
    test_idx = np.concatenate(test_idx_list)

    X_train = features[train_idx]
    Y_train = targets[train_idx]
    source_train = sources[train_idx]

    X_cal = features[cal_idx]
    Y_cal = targets[cal_idx]
    source_cal = sources[cal_idx]

    X_test = features[test_idx]
    Y_test = targets[test_idx]
    source_test = sources[test_idx]

    X_sources_train, Y_sources_train = reconstruct_source_data(
        X_train, Y_train, source_train, n_sources
    )
    X_sources_cal, Y_sources_cal = reconstruct_source_data(
        X_cal, Y_cal, source_cal, n_sources
    )

    source_models = [
        StudentTPredictionSource(column_maps[name], name=name)
        for name in source_order
    ]

    train_sizes = np.array([len(X) for X in X_sources_train], dtype=float)
    if train_sizes.sum() <= 0:
        raise ValueError("Training split is empty; cannot fit lambda model.")
    source_weights = train_sizes / train_sizes.sum()

    y_grid_true = generate_y_grid_regression(
        Y_sources_train + Y_sources_cal,
        n_grid_points=config.y_grid_size,
        margin_factor=config.y_margin,
    )

    mimic_seed = int(rng.integers(0, 2**32 - 1))
    mimic_components: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = None
    mimic_error: Optional[str] = None
    try:
        mimic_components = _split_mimic_sets(
            X_train,
            Y_train,
            source_train,
            MIMIC_CAL_RATIO,
            mimic_seed,
            stratify=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        mimic_error = str(exc)

    X_sources_mimic_cal: Sequence[np.ndarray]
    Y_sources_mimic_cal: Sequence[np.ndarray]
    y_grid_mimic: Optional[np.ndarray]
    X_mimic_test: Optional[np.ndarray]
    Y_mimic_test: Optional[np.ndarray]
    source_mimic_test: Optional[np.ndarray]
    if mimic_components is not None:
        (
            X_mimic_cal,
            X_mimic_test_raw,
            Y_mimic_cal,
            Y_mimic_test_raw,
            source_mimic_cal,
            source_mimic_test_raw,
        ) = mimic_components
        X_sources_mimic_cal, Y_sources_mimic_cal = reconstruct_source_data(
            X_mimic_cal,
            Y_mimic_cal,
            source_mimic_cal,
            n_sources,
        )
        y_grid_mimic = generate_y_grid_regression(
            Y_sources_train + Y_sources_mimic_cal,
            n_grid_points=config.y_grid_size,
            margin_factor=config.y_margin,
        )
        X_mimic_test = np.asarray(X_mimic_test_raw, dtype=float)
        Y_mimic_test = np.asarray(Y_mimic_test_raw, dtype=float)
        source_mimic_test = np.asarray(source_mimic_test_raw, dtype=int)
    else:
        X_sources_mimic_cal = tuple()
        Y_sources_mimic_cal = tuple()
        y_grid_mimic = None
        X_mimic_test = None
        Y_mimic_test = None
        source_mimic_test = None

    baseline_metrics, single_source_metrics = compute_baseline_metrics(
        source_models,
        X_sources_cal,
        Y_sources_cal,
        X_test,
        Y_test,
        config.alpha,
        source_test,
    )

    if (
        mimic_components is not None
        and X_mimic_test is not None
        and Y_mimic_test is not None
        and source_mimic_test is not None
        and X_mimic_test.shape[0] > 0
    ):
        baseline_mimic_metrics, single_source_mimic_metrics = compute_baseline_metrics(
            source_models,
            X_sources_mimic_cal,
            Y_sources_mimic_cal,
            X_mimic_test,
            Y_mimic_test,
            config.alpha,
            source_mimic_test,
        )
    else:
        baseline_mimic_metrics = None
        single_source_mimic_metrics = None

    def evaluate_mdcp_split(
        lambda_model: object,
        X_cal_list: Sequence[np.ndarray],
        Y_cal_list: Sequence[np.ndarray],
        X_eval: np.ndarray,
        Y_eval: np.ndarray,
        y_grid_eval: np.ndarray,
        source_eval: np.ndarray,
    ) -> Dict[str, object]:
        calibration_cache = precompute_calibration_cache(
            lambda_model.lambda_at_x,
            source_models,
            X_cal_list,
            Y_cal_list,
        )
        lam_values = np.asarray(lambda_model.lambda_at_x(X_eval))
        if lam_values.ndim == 1:
            lam_values = lam_values.reshape(-1, len(source_models))
        mdcp_payload = []
        for idx, x_point in enumerate(X_eval):
            mdcp_payload.append(
                aggregated_conformal_set_multi(
                    lam_model=lambda_model.lambda_at_x,
                    sources=source_models,
                    X_cal_list=X_cal_list,
                    Y_cal_list=Y_cal_list,
                    X_test=x_point,
                    Y_test=Y_eval[idx],
                    y_grid=y_grid_eval,
                    alpha=config.alpha,
                    randomize_ties=True,
                    calibration_cache=calibration_cache,
                    lam_x=lam_values[idx],
                )
            )
        return evaluate_mdcp_regression_performance(
            mdcp_payload,
            Y_eval,
            y_grid_eval,
            config.alpha,
            extend_points=1,
            source_test=source_eval,
        )

    mdcp_results: Dict[str, Dict[str, object]] = {}
    mdcp_mimic_results: Dict[str, Dict[str, object]] = {}
    mdcp_mimic_errors: Dict[str, str] = {}
    for gamma in config.gamma_values:
        spline_kwargs = {
            "gamma1": gamma,
            "gamma2": gamma,
            "gamma3": 0.0,
            "n_splines": 5,
            "degree": 2,
        }
        lambda_model = fit_lambda_from_sources(
            source_models,
            "spline",
            X_train,
            Y_train,
            alpha=config.alpha,
            spline_kwargs=spline_kwargs,
            verbose=False,
            source_weights=source_weights,
        )

        if (
            mimic_components is not None
            and y_grid_mimic is not None
            and X_mimic_test is not None
            and Y_mimic_test is not None
            and source_mimic_test is not None
            and X_mimic_test.shape[0] > 0
        ):
            try:
                mdcp_mimic_results[f"{gamma:g}"] = evaluate_mdcp_split(
                    lambda_model,
                    X_sources_mimic_cal,
                    Y_sources_mimic_cal,
                    X_mimic_test,
                    Y_mimic_test,
                    y_grid_mimic,
                    source_mimic_test,
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                mdcp_mimic_errors[f"{gamma:g}"] = str(exc)

        mdcp_results[f"{gamma:g}"] = evaluate_mdcp_split(
            lambda_model,
            X_sources_cal,
            Y_sources_cal,
            X_test,
            Y_test,
            y_grid_true,
            source_test,
        )

    per_source_counts = {}
    for name, train_ids, cal_ids, test_ids in zip(
        source_order, train_idx_list, cal_idx_list, test_idx_list
    ):
        per_source_counts[name] = {
            "train": int(train_ids.size),
            "cal": int(cal_ids.size),
            "test": int(test_ids.size),
        }

    return {
        "trial": trial_idx,
        "counts": {
            "train_total": int(train_idx.size),
            "cal_total": int(cal_idx.size),
            "test_total": int(test_idx.size),
            "per_source": per_source_counts,
        },
        "baseline": baseline_metrics,
        "single_baseline": single_source_metrics,
        "baseline_mimic": baseline_mimic_metrics,
        "single_baseline_mimic": single_source_mimic_metrics,
        "mdcp": mdcp_results,
        "mdcp_mimic": mdcp_mimic_results,
        "mdcp_mimic_errors": mdcp_mimic_errors,
        "mimic_metadata": {
            "seed": mimic_seed,
            "available": mimic_components is not None,
            "error": mimic_error,
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDCP vs. baseline evaluation on PovertyMap pending predictions",
    )
    base_pred_dir = REPO_ROOT / "eval_out" / "poverty" / "predictions"
    parser.add_argument(
        "--rural-on-rural",
        type=Path,
        default=base_pred_dir / "run_1242327_rural_on_rural" / "density_params_pending_rural.npz",
        help="NPZ predictions: rural-trained model on rural pending subset.",
    )
    parser.add_argument(
        "--urban-on-rural",
        type=Path,
        default=base_pred_dir / "run_1242327_urban_on_rural" / "density_params_pending_rural.npz",
        help="NPZ predictions: urban-trained model on rural pending subset.",
    )
    parser.add_argument(
        "--rural-on-urban",
        type=Path,
        default=base_pred_dir / "run_1242327_rural_on_urban" / "density_params_pending_urban.npz",
        help="NPZ predictions: rural-trained model on urban pending subset.",
    )
    parser.add_argument(
        "--urban-on-urban",
        type=Path,
        default=base_pred_dir / "run_1242327_urban_on_urban" / "density_params_pending_urban.npz",
        help="NPZ predictions: urban-trained model on urban pending subset.",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Target miscoverage level.")
    parser.add_argument(
        "--cal-frac",
        type=float,
        default=0.375,
        help="Fraction of samples per source allocated to calibration split.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.5,
        help="Fraction of samples per source allocated to test split.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of randomized trials (different calibration/test splits).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Base random seed controlling split permutations.",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=(0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        help="List of gamma values evaluated by MDCP lambda spline fitting.",
    )
    parser.add_argument(
        "--y-grid-size",
        type=int,
        default=512,
        help="Number of grid points used when discretizing Y for MDCP evaluation.",
    )
    parser.add_argument(
        "--y-margin",
        type=float,
        default=0.05,
        help="Relative range extension applied when constructing the Y grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval_out" / "poverty" / "mdcp",
        help="Directory where per-trial results and summary will be stored (defaults to eval_out/poverty/mdcp).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> Dict[str, object]:
    args = parse_args(argv)

    train_frac = max(0.0, 1.0 - float(args.cal_frac) - float(args.test_frac))
    config = EvaluationConfig(
        alpha=float(args.alpha),
        train_frac=train_frac,
        cal_frac=float(args.cal_frac),
        test_frac=float(args.test_frac),
        num_trials=int(args.num_trials),
        random_seed=int(args.seed),
        gamma_values=tuple(float(g) for g in args.gamma_grid),
        y_grid_size=int(args.y_grid_size),
        y_margin=float(args.y_margin),
    )

    dataset = build_dataset(
        Path(args.rural_on_rural),
        Path(args.urban_on_rural),
        Path(args.rural_on_urban),
        Path(args.urban_on_urban),
    )
    dataset_head_type = str(dataset.get("head_type", "student_t"))  # type: ignore[arg-type]

    output_dir = ensure_project_dir(args.output_dir)
    trial_records: List[Dict[str, object]] = []
    for trial in range(config.num_trials):
        result = run_trial(trial, dataset, config)
        trial_records.append(result)
        baseline_cov = result["baseline"].get("coverage")  # type: ignore[index]
        mdcp_cov = {
            gamma: metrics.get("coverage")  # type: ignore[index]
            for gamma, metrics in result["mdcp"].items()
        }
        print(
            f"Trial {trial:03d}: baseline coverage={format_float(baseline_cov)}, "
            + ", ".join(
                f"gamma={gamma}: cov={format_float(cov)}"
                for gamma, cov in mdcp_cov.items()
            )
        )

    baseline_metrics: List[Dict[str, object]] = []
    baseline_mimic_metrics: List[Dict[str, object]] = []
    mdcp_metrics: Dict[str, List[Dict[str, object]]] = {}
    mdcp_mimic_metrics: Dict[str, List[Dict[str, object]]] = {}
    mdcp_mimic_errors: Dict[str, List[str]] = {}
    single_baseline_metrics: Dict[str, List[Dict[str, object]]] = {}
    single_baseline_mimic_metrics: Dict[str, List[Dict[str, object]]] = {}
    mimic_error_messages: List[str] = []
    mimic_available_count = 0
    for record in trial_records:
        baseline_metrics.append(record["baseline"])
        baseline_mimic = record.get("baseline_mimic")
        if isinstance(baseline_mimic, dict):
            baseline_mimic_metrics.append(baseline_mimic)

        for gamma, metrics in record["mdcp"].items():
            mdcp_metrics.setdefault(gamma, []).append(metrics)
        for gamma, metrics in record.get("mdcp_mimic", {}).items():
            mdcp_mimic_metrics.setdefault(gamma, []).append(metrics)
        for gamma, error_msg in record.get("mdcp_mimic_errors", {}).items():
            mdcp_mimic_errors.setdefault(gamma, []).append(error_msg)

        for name, metrics in record.get("single_baseline", {}).items():
            single_baseline_metrics.setdefault(name, []).append(metrics)
        for name, metrics in record.get("single_baseline_mimic", {}).items():
            single_baseline_mimic_metrics.setdefault(name, []).append(metrics)

        mimic_meta = record.get("mimic_metadata", {})
        if mimic_meta.get("available"):
            mimic_available_count += 1
        if mimic_meta.get("error"):
            mimic_error_messages.append(str(mimic_meta["error"]))

    summary = {
        "config": {
            "alpha": config.alpha,
            "train_frac": config.train_frac,
            "cal_frac": config.cal_frac,
            "test_frac": config.test_frac,
            "num_trials": config.num_trials,
            "random_seed": config.random_seed,
            "gamma_values": config.gamma_values,
            "y_grid_size": config.y_grid_size,
            "y_margin": config.y_margin,
            "head_type": dataset_head_type,
        },
        "head_type": dataset_head_type,
        "baseline": summarize_trials(
            baseline_metrics,
            metric_keys=("coverage", "avg_width", "individual_coverage", "individual_widths"),
        ),
        "single_baseline": {
            name: summarize_trials(
                metrics,
                metric_keys=("coverage", "avg_width", "individual_coverage", "individual_widths"),
            )
            for name, metrics in single_baseline_metrics.items()
        },
        "mdcp": {
            gamma: summarize_trials(
                metrics,
                metric_keys=("coverage", "avg_width", "individual_coverage", "individual_widths"),
            )
            for gamma, metrics in mdcp_metrics.items()
        },
    }

    if baseline_mimic_metrics:
        summary["baseline_mimic"] = summarize_trials(
            baseline_mimic_metrics,
            metric_keys=("coverage", "avg_width", "individual_coverage", "individual_widths"),
        )
    if single_baseline_mimic_metrics:
        summary["single_baseline_mimic"] = {
            name: summarize_trials(
                metrics,
                metric_keys=("coverage", "avg_width", "individual_coverage", "individual_widths"),
            )
            for name, metrics in single_baseline_mimic_metrics.items()
        }
    if mdcp_mimic_metrics:
        summary["mdcp_mimic"] = {
            gamma: summarize_trials(
                metrics,
                metric_keys=("coverage", "avg_width", "individual_coverage", "individual_widths"),
            )
            for gamma, metrics in mdcp_mimic_metrics.items()
        }
    if mdcp_mimic_errors:
        summary["mdcp_mimic_errors"] = mdcp_mimic_errors
    summary["mimic_overview"] = {
        "trials_with_mimic": mimic_available_count,
        "total_trials": len(trial_records),
        "unique_errors": sorted(set(mimic_error_messages)),
    }

    trials_path = output_dir / "trial_results.json"
    trials_path.write_text(json.dumps(to_serializable(trial_records), indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")

    print(f"Per-trial results saved to {prefer_relative_path(trials_path)}")
    print(f"Summary saved to {prefer_relative_path(summary_path)}")

    return {
        "trials_path": trials_path,
        "summary_path": summary_path,
        "trial_records": trial_records,
        "summary": summary,
    }


if __name__ == "__main__":
    main()
