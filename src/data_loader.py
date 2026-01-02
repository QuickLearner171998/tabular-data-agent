"""Data loading and management utilities."""

import pandas as pd
import duckdb
from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.data_loader")


class DataLoader:
    """Handles loading and managing tabular data from various sources."""
    
    def __init__(self):
        self.settings = get_settings()
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.conn = duckdb.connect(":memory:")
        self._schema_cache: dict[str, dict] = {}
        logger.info("DataLoader initialized with in-memory DuckDB connection")
    
    def load_csv(self, file_path: str | Path, name: str | None = None) -> str:
        """
        Load a CSV file into memory.
        
        Args:
            file_path: Path to the CSV file.
            name: Optional name for the dataset. Defaults to filename.
        
        Returns:
            The name assigned to the dataset.
        """
        file_path = Path(file_path)
        name = name or file_path.stem
        
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        self.dataframes[name] = df
        self.conn.register(name, df)
        self._schema_cache[name] = self._extract_schema(df, name)
        
        logger.info(f"Loaded dataset '{name}': {len(df):,} rows, {len(df.columns)} columns, {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        return name
    
    def load_dataframe(self, df: pd.DataFrame, name: str) -> str:
        """
        Load a pandas DataFrame directly.
        
        Args:
            df: The DataFrame to load.
            name: Name for the dataset.
        
        Returns:
            The name assigned to the dataset.
        """
        self.dataframes[name] = df
        self.conn.register(name, df)
        self._schema_cache[name] = self._extract_schema(df, name)
        logger.info(f"Loaded DataFrame '{name}': {len(df):,} rows, {len(df.columns)} columns")
        return name
    
    def _extract_schema(self, df: pd.DataFrame, name: str) -> dict[str, Any]:
        """Extract schema information from a DataFrame."""
        schema = {
            "name": name,
            "columns": {},
            "row_count": len(df),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["min"] = float(df[col].min()) if not df[col].isna().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isna().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                if col_info["unique_count"] <= 20:
                    col_info["sample_values"] = df[col].dropna().unique()[:10].tolist()
            
            schema["columns"][col] = col_info
        
        return schema
    
    def get_schema(self, name: str) -> dict[str, Any]:
        """Get the schema for a loaded dataset."""
        if name not in self._schema_cache:
            raise ValueError(f"Dataset '{name}' not found")
        return self._schema_cache[name]
    
    def get_all_schemas(self) -> dict[str, dict]:
        """Get schemas for all loaded datasets."""
        return self._schema_cache.copy()
    
    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Get a loaded DataFrame by name."""
        if name not in self.dataframes:
            raise ValueError(f"Dataset '{name}' not found")
        return self.dataframes[name]
    
    def execute_sql(self, query: str) -> pd.DataFrame:
        """Execute a SQL query against loaded datasets."""
        logger.debug(f"Executing SQL: {query[:100]}...")
        result = self.conn.execute(query).df()
        logger.debug(f"SQL query returned {len(result):,} rows")
        return result
    
    def list_datasets(self) -> list[str]:
        """List all loaded dataset names."""
        return list(self.dataframes.keys())
    
    def get_sample(self, name: str, n: int = 5) -> pd.DataFrame:
        """Get a sample of rows from a dataset."""
        df = self.get_dataframe(name)
        return df.head(n)
    
    def get_schema_description(self) -> str:
        """Get a formatted string description of all loaded data schemas."""
        if not self._schema_cache:
            return "No datasets loaded."
        
        descriptions = []
        for name, schema in self._schema_cache.items():
            desc = [f"Dataset: {name}"]
            desc.append(f"  Rows: {schema['row_count']:,}")
            desc.append(f"  Memory: {schema['memory_usage_mb']:.2f} MB")
            desc.append("  Columns:")
            
            for col_name, col_info in schema["columns"].items():
                col_desc = f"    - {col_name} ({col_info['dtype']})"
                if col_info["null_count"] > 0:
                    col_desc += f" [nulls: {col_info['null_count']}]"
                if "sample_values" in col_info:
                    samples = ", ".join(str(v) for v in col_info["sample_values"][:5])
                    col_desc += f" [values: {samples}...]"
                desc.append(col_desc)
            
            descriptions.append("\n".join(desc))
        
        return "\n\n".join(descriptions)
