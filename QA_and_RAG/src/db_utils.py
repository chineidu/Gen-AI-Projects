import os
from pathlib import Path
from typing import Any
from sqlalchemy import create_engine, inspect, Inspector
import polars as pl
from pydantic import BaseModel, computed_field, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from config import config


class SQLFromTabularData(BaseModel):
    """A class to handle SQL database operations from tabular data files.

    Parameters
    ----------
    file_path : str
        Path to the input CSV or Parquet file
    db_path : str
        Path to the SQLite database file
    table_name : str
        Name of the table to create/update in the database

    Attributes
    ----------
    full_db_path : str
        Complete SQLite connection string

    Example
    -------
    >>> db = SQLFromTabularData(
    ...     file_path="data/input_data.csv",
    ...     db_path="data/my_db.db",
    ...     table_name="my_table"
    ... )
    >>> db.run()

    """

    file_path: str = Field(pattern=r"\.(csv|parquet)")
    db_path: str = Field(pattern=r"\.db")
    table_name: str

    @computed_field
    @property
    def full_db_path(self) -> str:
        """Generate full SQLite database path.

        Returns
        -------
        str
            SQLite connection string
        """
        return f"sqlite:///{self.db_path}"

    def _read_data(self) -> pl.DataFrame | None:
        """Read data from CSV or Parquet file.

        Returns
        -------
        pl.DataFrame | None
            Polars DataFrame containing the data, or None if error occurs
        """
        try:
            df: pl.DataFrame = (
                pl.read_csv(self.file_path)
                if self.file_path.endswith(".csv")
                else pl.read_parquet(self.file_path)
            )
            return df
        except Exception as e:
            print(f"Error loading the data: {e}")
            return None

    def _create_connection(self) -> Any:
        """Create SQLAlchemy engine connection.

        Returns
        -------
        Any
            SQLAlchemy Engine instance
        """
        return create_engine(self.full_db_path)

    def _create_database(self) -> None:
        """Create or update database table from the input data.

        Returns
        -------
        None
        """
        try:
            data: pl.DataFrame = self._read_data()
            data.write_database(
                table_name=self.table_name, connection=self.full_db_path
            )
            print(
                f"DB at {self.full_db_path} successfully "
                f"created/updated with {self.table_name} table."
            )

        except Exception as e:
            print(f"Error creating/updating the DB: {e}")

        return None

    def _validate_db(self) -> None:
        """Validate database by inspecting available tables.

        Returns
        -------
        None
        """
        try:
            conn = self._create_connection()
            insp: Inspector = inspect(conn)

            table_names: list[str] = insp.get_table_names()
            print(
                f"DB Path: {self.full_db_path}"
                "\n================================"
                f"\nAvailable table Names: {table_names}"
                "\n================================"
            )
        except Exception as e:
            print(f"Error validating the DB: {e}")

    def run(self) -> None:
        """Execute database creation and validation.

        Returns
        -------
        None
        """
        self._create_database()
        self._validate_db()


def run_sql_query(query: str, connection: Any) -> pl.DataFrame | None:
    """Execute SQL query and return results as a Polars DataFrame.

    Parameters
    ----------
    query : str
        SQL query string to execute
    connection : Any
        SQLAlchemy connection object

    Returns
    -------
    pl.DataFrame | None
        Query results as a Polars DataFrame, or None if error occurs
    """
    try:
        result: pl.DataFrame = pl.read_database(
            query=query, connection=connection.connect()
        )
        return result
    except Exception as e:
        print(f"Error running the query: {e}")
        return None


class VectorDBManager(BaseModel):
    """

    # Example
    # -------
    # >>> process_files = ProcessFiles(
    # ...     files_dir="data/uploads/",
    # ...     chatbot=[],
    # ... )
    # >>> process_files.run()
    """

    files_dir: str | Path | list[str | Path] = Field(
        default_factory=lambda: Path(config.QA_and_RAG.uploaded_files_path)
    )
    chatbot: list[str] = Field(default_factory=list)
    db_path: str = Field(default_factory=lambda: config.QA_and_RAG.uploaded_db_path)

    @computed_field
    @property
    def full_files_dir(self) -> list[str]:
        """Get list of file paths in the files directory.

        Returns
        -------
        list[str]
            List of absolute file paths as strings for all files in the directory
        """
        if isinstance(self.files_dir, str | Path):
            files_dir: Path = Path(self.files_dir).absolute()
            file_paths: list[Path] = list(files_dir.glob("*"))

            return [str(fp) for fp in file_paths if fp.is_file()]

    # @computed_field
    # @property
    # def full_db_path(self) -> str:
    #     """Get the full SQLite database connection string.

    #     Returns
    #     -------
    #     str
    #         SQLite connection string in format 'sqlite:///path/to/db'
    #     """
    #     return f"sqlite:///{self.db_path}"

    def _process_data(self) -> tuple[str, list[str]] | None:
        """Process CSV and Parquet files into SQLite database.

        Returns
        -------
        tuple[str, list[str]] | None
            Empty string and chatbot messages list if successful, None if error occurs
        """
        if isinstance(self.files_dir, list):
            file_paths: list[str] = self.files_dir
        else:
            file_paths: list[str] = self.full_files_dir
        try:
            for file_path in file_paths:
                file_names_with_extensions: str = os.path.basename(file_path)
                file_name: str
                file_extension: str
                file_name, file_extension = os.path.splitext(file_names_with_extensions)

                df: pl.DataFrame = (
                    pl.read_csv(file_path)
                    if file_extension == ".csv"
                    else (
                        pl.read_parquet(file_path)
                        if file_extension == ".parquet"
                        else None
                    )
                )
                if df is None:
                    raise ValueError(f"Unsupported file format: {file_extension}")
                self._create_database(data=df, table_name=file_name)
            print("==============================")
            print("All csv/parquet files are saved into the sql database.")
            self.chatbot.append(
                {
                    "role": "assistant",
                    "content": "Uploaded files are ready. Please ask your question",
                }
            )

            return ("", self.chatbot)

        except Exception as e:
            print(f"Error loading the data: {e}")
            return None

    def _create_connection(self) -> QdrantClient | None:
        client = QdrantClient(url="http://localhost:6333")
        
        try:
            client.get_collections().collections
            print("Qdrant server is running.")
            return client
        
        except ResponseHandlingException as e:
            print(f"Qdrant server is not running. Error: {e}")
            return None

    def _create_database(self, data: pl.DataFrame, table_name: str) -> None:
        """Create or update database table from the input data.

        Parameters
        ----------
        data : pl.DataFrame
            Polars DataFrame containing the data to write, shape (n_rows, n_columns)
        table_name : str
            Name of the table to create/update

        Returns
        -------
        None
        """
        try:
            data.write_database(
                table_name=table_name,
                connection=self.full_db_path,
                if_table_exists="replace",
            )
            print(
                f"DB at {self.full_db_path} successfully "
                f"created/updated with {table_name} table."
            )

        except Exception as e:
            print(f"Error creating/updating the DB: {e}")

        return None

    def _validate_db(self) -> None:
        """Validate database by inspecting available tables.

        Returns
        -------
        None
            Prints database path and available table names
        """
        try:
            conn: Engine = self._create_connection()
            insp: Inspector = inspect(conn)

            table_names: list[str] = insp.get_table_names()
            print(
                "\nValidating database: "
                "\n================================"
                f"\nAvailable table Names: {table_names}"
                "\n================================"
            )
        except Exception as e:
            print(f"Error validating the DB: {e}")

    def run(self) -> None:
        """Execute database creation and validation.

        Returns
        -------
        None
            Processes data and validates database
        """
        input_txt, chatbot = self._process_data()
        self._validate_db()

        return input_txt, chatbot

