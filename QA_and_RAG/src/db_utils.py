import os
from pathlib import Path
import time
from typing import Any, ClassVar
from sqlalchemy import create_engine, inspect, Inspector
import polars as pl
from pydantic import BaseModel, computed_field, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from sentence_transformers import SentenceTransformer

from config import config, settings


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


class VectorDBManager(BaseModel):
    """A class to manage vector database operations including file processing and data storage.

    Attributes
    ----------
    files_dir : str | Path | list[str | Path]
        Directory path or list of file paths containing data files
    chatbot : list[str]
        List to store chat messages
    db_path : str
        Path to the vector database
    model_name : str
        Name of the encoder model to use

    Examples
    --------
    >>> vector_db = VectorDBManager(
    ...     files_dir="data/uploads/",
    ...     chatbot=[],
    ... )
    >>> vector_db.run()
    """

    _client: ClassVar[QdrantClient | None] = None
    _max_retries: ClassVar[int] = 3
    files_dir: str | Path | list[str | Path] = Field(
        default_factory=lambda: Path(config.QA_and_RAG.uploaded_files_path)
    )
    chatbot: list[str] = Field(default_factory=list)
    db_path: str = Field(default_factory=lambda: config.QA_and_RAG.uploaded_db_path)
    model_name: str = Field(
        default_factory=lambda: config.QA_and_RAG.encoder_model.model
    )

    @classmethod
    def _get_client(cls) -> QdrantClient | None:
        """Get the QdrantClient instance."""
        if cls._client is None:
            for attempt in range(cls._max_retries):
                try:
                    cls._client: QdrantClient = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                        api_key=settings.QDRANT_API_KEY.get_secret_value(),
                        https=False,
                    )
                    # Verify connection
                    cls._client.get_collections().collections
                    print("Qdrant server is running.")
                    return cls._client

                except ResponseHandlingException as e:
                    if attempt == cls._max_retries - 1:
                        print(f"Qdrant server is not running. Error: {e}")
                        return None
                    else:
                        print(
                            f"Failed to connect to Qdrant server. Retrying... ({attempt + 1}/{cls._max_retries})"
                        )
                        time.sleep(1.2)

        # Return the existing client
        return cls._client

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

    @computed_field
    @property
    def encoder(self) -> Any:
        """Get the sentence transformer encoder model.

        Returns
        -------
        SentenceTransformer
            Initialized sentence transformer model
        """
        return SentenceTransformer(
            self.model_name,
            cache_folder=config.QA_and_RAG.encoder_model.cache_folder,
        )

    def _process_data(self) -> tuple[str, list[str]] | None:
        """Process data files and store them in vector database.

        Returns
        -------
        tuple[str, list[str]] | None
            Tuple containing empty string and chatbot messages if successful,
            None if processing fails
        """
        if isinstance(self.files_dir, list):
            file_paths: list[str] = self.files_dir
        else:
            file_paths: list[str] = self.full_files_dir
        try:
            for file_path in file_paths:
                print(f"Processing file: {file_path}")
                file_names_with_extensions: str = os.path.basename(file_path)
                file_name: str
                file_extension: str
                file_name, file_extension = os.path.splitext(file_names_with_extensions)
                if file_extension not in [".csv", ".parquet"]:
                    continue

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
                self._create_collection(collection_name=file_name)
                self._upsert_data(data=df, collection_name=file_name)
            print("==============================")
            print("All csv/parquet files are saved into the vector database.")
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

    def _create_collection(self, collection_name: str) -> None:
        """Create a new collection in the vector database.

        Parameters
        ----------
        collection_name : str
            Name of the collection to create

        Returns
        -------
        None
        """
        client: QdrantClient = self._get_client()
        if client is None:
            return None

        if client.collection_exists(collection_name=collection_name):
            print(f"Collection '{collection_name}' already exists.")

        else:
            print(f"Creating collection '{collection_name}'.")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=config.QA_and_RAG.encoder_model.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        return None

    def _get_collection_names(self) -> list[str]:
        """Get the names of all collections in the vector database.

        Returns
        -------
        list[str]
            List of collection names
        """
        client: QdrantClient = self._get_client()
        if client is None:
            return []
        return [collection.name for collection in client.get_collections().collections]

    def _upsert_data(self, data: pl.DataFrame, collection_name: str) -> None:
        """Insert or update data in the vector database collection.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame containing the data to upsert
        collection_name : str
            Name of the collection to upsert data into

        Returns
        -------
        None
        """
        client: QdrantClient = self._get_client()
        if client is None:
            return None

        documents: list[dict[str, Any]] = data.to_dicts()
        print(f"Upserting data into '{collection_name}' collection.")
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx, vector=self.embed_document(doc["description"]), payload=doc
                )
                for idx, doc in enumerate(documents)
            ],
        )
        print(f"Data from '{collection_name}' is saved into the vector database.")
        return None

    def embed_document(self, document: str) -> list[float]:
        """Embed a document using an embedding model.

        Parameters
        ----------
        document : str
            Text document to embed

        Returns
        -------
        list[float]
            Document embedding vector
        """
        return self.encoder.encode(document).tolist()

    def run(self) -> tuple[str, list[str]] | None:
        """Execute the data processing pipeline.

        Returns
        -------
        tuple[str, list[str]] | None
            Tuple containing empty string and chatbot messages if successful,
            None if processing fails
        """
        input_txt, chatbot = self._process_data()
        return input_txt, chatbot


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
