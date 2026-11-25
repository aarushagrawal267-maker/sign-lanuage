"""Application configuration.

All paths derive from ``SLR_DATA_DIR`` (default ``<repo>/data``) so tests and
alternative setups can point the whole application at an isolated directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

WLASL_REPO_RAW = "https://raw.githubusercontent.com/dxli94/WLASL/master"
WLASL_ANNOTATION_URL = f"{WLASL_REPO_RAW}/start_kit/WLASL_v0.3.json"
WLASL_NSLT100_URL = f"{WLASL_REPO_RAW}/code/I3D/preprocess/nslt_100.json"

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

VOCAB_SIZE = 100


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SLR_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    assets_dir: Path = REPO_ROOT / "assets"
    frontend_origin: str = "http://localhost:5173"
    # Torch device preference: "auto" uses CUDA when available, otherwise CPU.
    device: str = "auto"
    torch_threads: int = 0  # 0 = let torch decide
    download_workers: int = 4
    download_timeout_s: int = 60
    # Public read-only demo (e.g. Hugging Face Space): disables downloads, training and deletions.
    public_demo: bool = False
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    # Register the pretrained WLASL100 model in models/ on first start so live recognition works
    # straight after setup (no dataset download or training needed).
    import_bundled_model: bool = True
    # Sign-language packs (docs/ISL_DESIGN.md section 4.1): extra pack roots (os.pathsep-separated), searched
    # after models/packs (tracked) and <artifacts>/packs (local builds); later roots override earlier ones.
    packs_dirs: str = ""
    default_language: str = "asl-wlasl100"
    # Data profile used by ISL build commands by default (commercial-safe | research).
    isl_profile: str = "commercial-safe"

    @property
    def annotations_dir(self) -> Path:
        return self.data_dir / "annotations"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def landmarks_dir(self) -> Path:
        return self.data_dir / "landmarks"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def my_signs_dir(self) -> Path:
        return self.data_dir / "my_signs"

    @property
    def holistic_model_path(self) -> Path:
        return self.assets_dir / "mediapipe" / "holistic_landmarker.task"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.annotations_dir,
            self.videos_dir,
            self.landmarks_dir,
            self.checkpoints_dir,
            self.reports_dir,
            self.holistic_model_path.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def resolve_checkpoint(path: str | Path | None) -> Path | None:
    """Checkpoint paths are stored absolute; fall back to the checkpoints dir when the
    project has been moved (e.g. into a container)."""
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    alt = get_settings().checkpoints_dir / p.name
    return alt if alt.exists() else None


def resolve_device(pref: str | None = None) -> str:
    import torch

    pref = (pref or get_settings().device).lower()
    if pref == "cpu":
        return "cpu"
    if pref.startswith("cuda"):
        return pref if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
