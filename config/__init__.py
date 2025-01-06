from pathlib import Path
import os

from omegaconf import DictConfig, OmegaConf

from .settings import settings, setup_langsmith_tracing


os.environ["WORKING_DIR"] = settings.WORKING_DIR
ROOT_PATH: Path = Path(__file__).parent.parent
config: DictConfig = OmegaConf.load(ROOT_PATH / "config/config.yaml")

__all__ = ["ROOT_PATH", "config", "settings", "setup_langsmith_tracing"]
