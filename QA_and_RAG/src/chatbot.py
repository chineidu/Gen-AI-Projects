from enum import Enum
import os
from typing import Annotated, Any, TypedDict

import langchain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict

from QA_and_RAG.src.utils.utilities import delete_folder_contents, get_file_paths
from config import config, settings


langchain.debug = False

llm: BaseChatModel = ChatOpenAI(
    model=config.QA_and_RAG.llm.model,
    api_key=settings.OPENAI_API_KEY,
    temperature=config.QA_and_RAG.llm.temperature,
)


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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def get_response(
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
                    db: SQLDatabase = get_database(db_path)

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
                    db: SQLDatabase = get_database(db_path)

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
                    db: SQLDatabase = get_database(db_path)

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
            message = message.strip()
            response: str = agent_executor.invoke(message)["output"]
            chatbot.append({"role": "user", "content": message})
            chatbot.append({"role": "assistant", "content": response})

            return "", chatbot


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


class SQLChatbot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(
        config.QA_and_RAG.sql_agent_prompt
    )

    def write_query(self, state: State, db: SQLDatabase, llm: BaseChatModel):
        """Generate SQL query to fetch information."""
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}

    @staticmethod
    def execute_query(state: State, db: SQLDatabase):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}

    @staticmethod
    def generate_answer(state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    def get_sql_agent_response(
        self, message: str, db: SQLDatabase, llm: BaseChatModel
    ) -> str:
        state_data: State = State({"question": message})
        query: str = self.write_query(state=state_data, db=db, llm=llm)["query"]
        state_data["query"] = query

        query_result: dict[str, Any] = self.execute_query(state=state_data, db=db)[
            "result"
        ]
        state_data["result"] = query_result

        answer: str = self.generate_answer(state=state_data)["answer"]
        return answer

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
        global llm

        if app_functionality.lower() == "chat":
            if chat_type == ChatType.QA_WITH_STORED_SQL_DB:
                if db_path is None:
                    db_path = config.QA_and_RAG.chinook_db_path
                if os.path.exists(db_path):
                    db: SQLDatabase = get_database(db_path)

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
                    db: SQLDatabase = get_database(db_path)

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
                    db: SQLDatabase = get_database(db_path)

                else:
                    chatbot.append(
                        (
                            message,
                            f"SQL DB from does not exist. Please upload a valid `CSV`/`PARQUET` file.",
                        )
                    )
            message = message.strip()
            response: str = self.get_sql_agent_response(message=message, db=db, llm=llm)
            chatbot.append({"role": "user", "content": message})
            chatbot.append({"role": "assistant", "content": response})

            return "", chatbot


# chatbot = Chatbot()
# chatbot.get_response(
#     app_functionality="chat",
#     chat_type=ChatType.QA_WITH_STORED_FLAT_FILE_SQL_DB,
#     message="How many men were there in the Titanic?",
#     chatbot=[],
# )
# print(chatbot)
