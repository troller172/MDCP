from __future__ import annotations

from pathlib import Path

from ..._wilds_repo import prepare_wilds_repo


def import_wilds(wilds_repo: Path | None):
    project_root = Path(__file__).resolve().parents[3]
    prepare_wilds_repo(project_root, wilds_repo, optional=(wilds_repo is None))
    try:
        from wilds import get_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import the WILDS package. Clone it via python -m wilds.setup_wilds_repo or install the pip package."
        ) from exc
    return get_dataset


def load_fmow_dataset(root_dir: Path, wilds_repo: Path | None = None, download: bool = False):
    get_dataset = import_wilds(wilds_repo)
    root_dir = root_dir.resolve()
    version_suffix = 'fmow_v1.1'
    candidate_dir = root_dir / version_suffix
    metadata_file = root_dir / 'rgb_metadata.csv'
    if not candidate_dir.exists() and metadata_file.exists():
        # root_dir already points at extracted fmow_v1.1 folder; use its parent per WILDS expectation
        root_dir = root_dir.parent
    return get_dataset(
        dataset="fmow",
        root_dir=root_dir.as_posix(),
        download=download,
        split_scheme="official",
    )
