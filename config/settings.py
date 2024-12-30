from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


env_path: Path = Path(__file__).parent / ".env"


class Settings(BaseSettings):

    model_config: SettingsConfigDict = SettingsConfigDict(env_file=env_path)

    OPENAI_API_KEY: str | None = None
    IS_SUCCESSFULLY_LOADED: bool = False


settings: Settings = Settings()
