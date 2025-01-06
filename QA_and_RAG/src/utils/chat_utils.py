import logging
from typing import Optional, Any
from functools import wraps

from QA_and_RAG.src.chatbot import Chatbot, ChatType


logger = logging.getLogger(__name__)


def handle_model_errors(func):
    """Decorator to handle model-related errors."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Model error in {func.__name__}: {str(e)}")
            raise ModelException(f"Error in {func.__name__}: {str(e)}") from e

    return wrapper


class ModelException(Exception):
    """Custom exception for model-related errors."""

    pass


class ModelManager:
    """A singleton class for managing ML model and its dependencies.

    This class ensures only one instance of the model and its dependencies
    are loaded in memory at any time.

    Attributes
    ----------
    _instance : ModelManager | None
        Singleton instance of the class
    _model : Chatbot | None
        The model instance
    _config : dict
        Model configuration
    """

    _instance: Optional["ModelManager"] = None
    _model: Optional[Chatbot] = None

    def __new__(cls) -> "ModelManager":
        """Create a new instance of ModelManager if one doesn't exist.

        Returns
        -------
        ModelManager
            The singleton instance of ModelManager
        """
        if cls._instance is None:
            logger.info("Creating new ModelManager instance")
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the ModelManager instance."""
        self._load_model()

    @handle_model_errors
    def _load_model(self) -> None:
        """Load the model and its dependencies.

        Raises
        ------
        ModelException
            If there is an error loading the model or dependencies
        """
        logger.info("Loading chatbot model")
        try:
            self._model = Chatbot()
            logger.info("Chatbot model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load chatbot model: {e}")
            raise ModelException("Failed to load chatbot model") from e

    @handle_model_errors
    def get_response(
        self,
        chatbot: list[tuple[str, str]],
        message: str,
        chat_type: ChatType,
        app_functionality: str,
        db_path: str | None = None,
    ) -> tuple[str, list[tuple[str, str]]] | None:
        """Get response from the chatbot.

        Parameters
        ----------
        message : str
            Input message to the chatbot
        chat_type : ChatType
            Type of chat interaction (stored SQL DB, stored flat file SQL DB, or uploaded flat file SQL DB).

        Returns
        -------
        str
            Response from the chatbot

        Raises
        ------
        ModelException
            If there is an error getting the response
        """
        if self._model is None:
            logger.warning("Model not loaded, attempting to load")
            self._load_model()

        return self._model.get_response(
            chatbot, message, chat_type, app_functionality, db_path
        )

    @handle_model_errors
    def clear_cache(self) -> None:
        """Clear the cached chatbot and reload it."""
        logger.info("Clearing model cache")
        self._model = None
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded.

        Returns
        -------
        bool
            True if the model is loaded, False otherwise
        """
        return self._model is not None
