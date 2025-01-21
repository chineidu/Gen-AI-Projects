import os
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


env_path: Path = Path(__file__).parent / ".env"


class Settings(BaseSettings):

    model_config: SettingsConfigDict = SettingsConfigDict(env_file=env_path)

    # OpenAI
    OPENAI_API_KEY: SecretStr | None = None

    # LangChain/LangSmith
    LANGCHAIN_TRACING_V2: SecretStr | None = Field(default=None,description="Whether it is Langchain version 2")
    LANGCHAIN_ENDPOINT: SecretStr | None = Field(default=None, description="The URL to Langchain's project")
    LANGCHAIN_API_KEY: SecretStr | None = Field(default=None, description="The Langchain's API key name")
    LANGCHAIN_PROJECT: str | None = Field(default=None, description="The Langchain's project name")

    # Vector DB
    QDRANT_HOST: str = Field(default="0.0.0.0", description="Qdrant host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant port")
    QDRANT_API_KEY: SecretStr | None = Field(default=None, description="Qdrant API key")

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
