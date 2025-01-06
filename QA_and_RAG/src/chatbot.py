from enum import Enum
import os

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from QA_and_RAG.src.utils.utilities import delete_folder_contents, get_file_paths
from config import config, settings


llm: BaseChatModel = ChatOpenAI(
    model=config.QA_and_RAG.llm.model,
    api_key=settings.OPENAI_API_KEY,
    temperature=config.QA_and_RAG.llm.temperature,
)


class ChatType(str, Enum):
    """Enumeration of available chat types for Q&A interactions.

    Parameters
    ----------
    str
        Base string type for enum values

    Attributes
    ----------
    QA_WITH_STORED_SQL_DB : str
        Chat type for stored Chinook SQL database
    QA_WITH_STORED_FLAT_FILE_SQL_DB : str
        Chat type for stored flat file SQL database
    QA_WITH_UPLOADED_FLAT_FILE_SQL_DB : str
        Chat type for uploaded flat file SQL database

    Returns
    -------
    ChatType
        Enum member representing the chat type
    """

    QA_WITH_STORED_SQL_DB = "q&a-with-stored-chinook-sql-db"
    QA_WITH_STORED_FLAT_FILE_SQL_DB = "q&a-with-stored-flat-file-sql-db"
    QA_WITH_UPLOADED_FLAT_FILE_SQL_DB = "q&a-with-uploaded-flat-file-sql-db"


class Chatbot(BaseModel):

    @staticmethod
    def get_database(db_path: str) -> SQLDatabase:
        """Create SQLDatabase instance from database path.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file

        Returns
        -------
        SQLDatabase
            Database instance for querying
        """
        db: SQLDatabase = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        print(f"DB Dialect: {db.dialect}")
        return db

    def get_response(
        self,
        chatbot: list[tuple[str, str]],
        message: str,
        chat_type: ChatType,
        app_functionality: str,
        db_path: str | None = None,
    ) -> tuple[str, list[tuple[str, str]]] | None:
        """Process user message and generate a response using SQL database.

        Parameters
        ----------
        chatbot : list[tuple[str, str]]
            The conversation history.
        message : str
            The user's input message to process.
        chat_type : Literal
            Type of chat interaction (stored SQL DB, stored flat file SQL DB, or uploaded flat file SQL DB).
        app_functionality : str
            Type of application functionality (e.g. chat).

        Returns
        -------
        tuple[str, list[tuple[str, str]]] | None
            A tuple containing an empty string and the updated conversation history,
            or None if processing fails.
        """
        if app_functionality.lower() == "chat":
            if chat_type == ChatType.QA_WITH_STORED_SQL_DB:
                if db_path is None:
                    db_path = config.QA_and_RAG.chinook_db_path
                if os.path.exists(db_path):
                    db: SQLDatabase = self.get_database(db_path)

                else:
                    chatbot.append(
                        (
                            message,
                            f"SQL DB does not exist. Please first create the `chinook` db",
                        )
                    )
            elif chat_type == ChatType.QA_WITH_STORED_FLAT_FILE_SQL_DB:
                if db_path is None:
                    db_path = config.QA_and_RAG.stored_sql_db_path
                if os.path.exists(db_path):
                    db: SQLDatabase = self.get_database(db_path)

                else:
                    chatbot.append(
                        (
                            message,
                            f"SQL DB from does not exist. Please first create the `chinook` db",
                        )
                    )
            elif chat_type == ChatType.QA_WITH_UPLOADED_FLAT_FILE_SQL_DB:
                file_path: str = str(
                    get_file_paths(config.QA_and_RAG.uploaded_files_path)[0]
                )

                if os.path.exists(file_path):
                    db_path: str = config.QA_and_RAG.uploaded_db_path
                    db: SQLDatabase = self.get_database(db_path)

                else:
                    chatbot.append(
                        (
                            message,
                            f"SQL DB from does not exist. Please upload a valid `CSV`/`PARQUET` file.",
                        )
                    )

            agent_executor = create_sql_agent(
                llm=llm, db=db, agent_type="openai-tools", verbose=False
            )
            response: str = agent_executor.invoke(message)["output"]
            chatbot.append({"role": "user", "content": message})
            chatbot.append({"role": "assistant", "content": response})

            return "", chatbot


chatbot = Chatbot.get_response(
    app_functionality="chat",
    chat_type=ChatType.QA_WITH_STORED_FLAT_FILE_SQL_DB,
    message="How many men were there in the Titanic?",
    chatbot=[],
)
print(chatbot)
