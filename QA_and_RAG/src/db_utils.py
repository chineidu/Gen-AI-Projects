from typing import Any
from sqlalchemy import create_engine, inspect, Inspector
import polars as pl
from pydantic import BaseModel, computed_field, Field


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
