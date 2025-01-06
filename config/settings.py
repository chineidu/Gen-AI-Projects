import os
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


env_path: Path = Path(__file__).parent / ".env"


class Settings(BaseSettings):

    model_config: SettingsConfigDict = SettingsConfigDict(env_file=env_path)

    # OpenAI
    OPENAI_API_KEY: SecretStr | None = None

    # LangChain/LangSmith
    LANGCHAIN_TRACING_V2: SecretStr | None = None
    LANGCHAIN_ENDPOINT: SecretStr | None = None
    LANGCHAIN_API_KEY: SecretStr | None = None
    LANGCHAIN_PROJECT: str | None = None

    # Other
    IS_SUCCESSFULLY_LOADED: bool = False
    WORKING_DIR: str = str(Path(__file__).parent.parent)


def setup_langsmith_tracing() -> None:
    """
    Set up LangSmith for tracing and feedback.

    This function configures the LangSmith environment variables required for
    tracing and feedback functionality in LangChain applications.

    Environment Variables Set
    -----------------------
    LANGCHAIN_TRACING_V2 : str
        Enable/disable LangChain tracing v2
    LANGCHAIN_ENDPOINT : str
        LangSmith API endpoint URL
    LANGCHAIN_API_KEY : str
        Authentication key for LangSmith API
    LANGCHAIN_PROJECT : str
        Project name for organizing traces

    Returns
    -------
    None
    """
    os.environ["LANGCHAIN_TRACING_V2"] = (
        settings.LANGCHAIN_TRACING_V2.get_secret_value()
    )
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT.get_secret_value()
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    return None


settings: Settings = Settings()
