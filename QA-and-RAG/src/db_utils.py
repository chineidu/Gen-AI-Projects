from typing import Any
from sqlalchemy import create_engine, inspect, Inspector
import polars as pl
from pydantic import BaseModel, computed_field, Field


class SQLFromTabularData(BaseModel):

    file_path: str = Field(pattern=r"\.(csv|parquet)")
    db_path: str = Field(pattern=r"\.db")
    table_name: str

    @computed_field
    @property
    def full_db_path(self) -> str:
        return f"sqlite:///{self.db_path}"

    def _read_data(self) -> pl.DataFrame | None:
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
        return create_engine(self.full_db_path)

    def _create_database(self) -> None:
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
        self._create_database()
        self._validate_db()
