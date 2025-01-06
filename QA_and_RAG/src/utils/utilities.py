import os
from pathlib import Path
import shutil
from sqlalchemy import Engine, Inspector, create_engine, inspect

import gradio as gr
import polars as pl
from pydantic import BaseModel, Field, computed_field

from config import config


class ProcessFiles(BaseModel):
    """A class to process CSV and Parquet files into a SQLite database.

    Parameters
    ----------
    files_dir : str | Path
        Directory containing CSV and Parquet files
    chatbot : list[str]
        List to store chatbot messages
    db_path : str
        Path to SQLite database file

    Example
    -------
    >>> process_files = ProcessFiles(
    ...     files_dir="data/uploads/",
    ...     chatbot=[],
    ... )
    >>> process_files.run()
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

    @computed_field
    @property
    def full_db_path(self) -> str:
        """Get the full SQLite database connection string.

        Returns
        -------
        str
            SQLite connection string in format 'sqlite:///path/to/db'
        """
        return f"sqlite:///{self.db_path}"

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
            for file_dir in file_paths:
                file_names_with_extensions: str = os.path.basename(file_dir)
                file_name: str
                file_extension: str
                file_name, file_extension = os.path.splitext(file_names_with_extensions)

                df: pl.DataFrame = (
                    pl.read_csv(file_dir)
                    if file_extension == ".csv"
                    else (
                        pl.read_parquet(file_dir)
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

    def _create_connection(self) -> Engine:
        """Create SQLAlchemy engine connection.

        Returns
        -------
        Engine
            SQLAlchemy Engine instance for database connection
        """
        return create_engine(self.full_db_path)

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


def upload_file(
    files_dir: str, chatbot: tuple[str, str], chatbot_functionality: str
) -> tuple[str, tuple[str, str]]:
    """
    Upload and process files based on the specified chatbot functionality.

    Parameters
    ----------
    files_dir : str
        Directory path containing the files to be processed.
    chatbot : tuple[str, str]
        Current state of the chatbot containing message history.
    chatbot_functionality : str
        String indicating the functionality to be executed (e.g., "Process files").

    Returns
    -------
    tuple[str, tuple[str, str]]
        A tuple containing:
        - input_txt : str
            Processed text from the files
        - chatbot : tuple[str, str]
            Updated chatbot state after processing
    """
    if chatbot_functionality.lower() == "chat":
        process_files: ProcessFiles = ProcessFiles(files_dir=files_dir, chatbot=chatbot)
        input_txt, chatbot = process_files.run()
        return input_txt, chatbot
    else:
        pass  # Other functionalities can be implemented here.


def delete_folder_contents(
    folder_path: Path | str, delete_subfolders: bool = False
) -> None:
    """
    Delete all files and subdirectories within the specified folder.

    Parameters
    ----------
    folder_path : Path | str
        The path to the folder whose contents should be deleted.
    delete_subfolders : bool, optional
        If True, also delete subdirectories within the folder. Default is False.

    Returns
    -------
    None
    """
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)

    # Iterate over all files in the folder and delete them
    for file_path in folder_path.iterdir():
        if file_path.is_file() or file_path.is_symlink():
            file_path.unlink()
        if delete_subfolders and file_path.is_dir():
            shutil.rmtree(file_path)


def get_file_paths(folder_path: Path | str) -> list[Path]:
    """
    Get a list of file paths from a specified folder.

    Parameters
    ----------
    folder_path : Path | str
        The path to the folder to search for files.

    Returns
    -------
    list[Path]
        A list of Path objects representing the files (not directories) in the folder.
    """
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    file_paths: list[Path] = list(folder_path.glob("*"))

    # Select only files (not directories)
    file_paths = [path for path in file_paths if path.is_file()]

    return file_paths


class UISettings:

    @staticmethod
    def toggle_sidebar(state):
        state = not state
        return gr.update(visible=state), state

    @staticmethod
    def feedback(data: gr.LikeData):
        if data.liked:
            print(f"You upvoted this response: {data.value}")
        else:
            print(f"You downvoted this response: {data.value}")
