"""Helper wrapper to download WILDS datasets when the host certificate is invalid.

The upstream CodaLab mirror occasionally ships an expired TLS certificate,
which breaks urllib's default HTTPS verification. This loader disables
certificate checks before delegating to wilds.download_datasets' CLI so we can
still fetch the archives. Use only on trusted mirrors.
"""

import ssl
from pathlib import Path

from wilds._wilds_repo import prepare_wilds_repo


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
prepare_wilds_repo(_PROJECT_ROOT, optional=True)

from wilds.download_datasets import main as wilds_main  # type: ignore

# Disable HTTPS verification explicitly for urllib.
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    wilds_main()
