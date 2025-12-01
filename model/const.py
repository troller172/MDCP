from pathlib import Path
from typing import Union

RANDOM_SEED = 3

TRUE_TRAIN_RATIO = 0.375
TRUE_CAL_RATIO = 0.125
TRUE_TEST_RATIO = 0.5

MIMIC_CALIBRATION_RATIO = 0.5
MIMIC_TEST_RATIO = 0.5

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_OUT_BASE = Path("..") / "eval_out"


PathLike = Union[Path, str]


def _to_path(path: PathLike) -> Path:
	return path if isinstance(path, Path) else Path(path)


def resolve_project_path(path: PathLike) -> Path:
	path_obj = _to_path(path)
	if path_obj.is_absolute():
		return path_obj
	return (BASE_DIR / path_obj).resolve()


def ensure_project_dir(path: PathLike) -> Path:
	target = resolve_project_path(path)
	target.mkdir(parents=True, exist_ok=True)
	return target


def prefer_relative_path(path: PathLike) -> Path:
	path_obj = resolve_project_path(path)
	try:
		relative = path_obj.relative_to(PROJECT_ROOT)
		return Path("..") / relative
	except ValueError:
		return path_obj


def eval_out_relative(*parts: str) -> Path:
	return EVAL_OUT_BASE.joinpath(*parts)


def eval_out_absolute(*parts: str) -> Path:
	return resolve_project_path(eval_out_relative(*parts))



LAMBDA_CLS_FILE = "lambda_values_cls_seed{random_seed}_alpha{alpha}_sources{n_sources}_classes{n_classes}_temperature{temperature}"
LAMBDA_REG_FILE = "lambda_values_reg_seed{random_seed}_alpha{alpha}_sources{n_sources}_temperature{temperature}"

ITERATIVE_EVAL_FOLDER = eval_out_relative("linear")
ITERATIVE_EVAL_LAMBDA_FOLDER = eval_out_relative("linear", "lambda")

NONLINEAR_FOLDER = eval_out_relative("nonlinear")
NONLINEAR_CLASSIFICATION_FOLDER = NONLINEAR_FOLDER / "classification"
NONLINEAR_REGRESSION_FOLDER = NONLINEAR_FOLDER / "regression"
NONLINEAR_LAMBDA_FOLDER = NONLINEAR_FOLDER / "lambda"

DATA_DENSITY_FOLDER = eval_out_relative("data_density")
