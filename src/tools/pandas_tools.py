"""Pandas execution tools for data manipulation and analysis."""

from typing import Any
import pandas as pd
from langchain_core.tools import tool

from src.data_loader import DataLoader
from src.utils.serialization import convert_to_serializable
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.tools.pandas")


class PandasTools:
    """Tools for executing pandas operations on loaded datasets."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def execute_pandas_code(self, code: str, dataset_name: str) -> dict[str, Any]:
        """
        Execute pandas code in a controlled environment.
        
        Args:
            code: Python/pandas code to execute.
            dataset_name: Name of the dataset to operate on.
        
        Returns:
            Dictionary with result, type, and any error information.
        """
        try:
            logger.debug(f"Executing pandas code on dataset '{dataset_name}'")
            df = self.data_loader.get_dataframe(dataset_name)
            
            safe_globals = {
                "pd": pd,
                "df": df,
                "__builtins__": {
                    "len": len,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "True": True,
                    "False": False,
                    "None": None,
                },
            }
            
            local_vars: dict[str, Any] = {}
            exec(code, safe_globals, local_vars)
            
            result = local_vars.get("result", None)
            
            if result is None:
                for var_name, var_value in reversed(list(local_vars.items())):
                    if isinstance(var_value, (pd.DataFrame, pd.Series)):
                        result = var_value
                        break
            
            if isinstance(result, pd.DataFrame):
                logger.debug(f"Pandas execution returned DataFrame: {result.shape}")
                return convert_to_serializable({
                    "success": True,
                    "result_type": "dataframe",
                    "data": result.head(100).to_dict("records"),
                    "columns": list(result.columns),
                    "shape": list(result.shape),
                    "summary": f"DataFrame with {result.shape[0]} rows and {result.shape[1]} columns",
                })
            elif isinstance(result, pd.Series):
                return convert_to_serializable({
                    "success": True,
                    "result_type": "series",
                    "data": result.head(100).to_dict(),
                    "name": str(result.name) if result.name else None,
                    "length": len(result),
                    "summary": f"Series '{result.name}' with {len(result)} values",
                })
            elif result is not None:
                return convert_to_serializable({
                    "success": True,
                    "result_type": "value",
                    "data": result,
                    "summary": str(result),
                })
            else:
                return {
                    "success": True,
                    "result_type": "none",
                    "data": None,
                    "summary": "Code executed successfully but produced no result",
                }
                
        except Exception as e:
            logger.warning(f"Pandas execution failed: {e}")
            return {
                "success": False,
                "result_type": "error",
                "error": str(e),
                "summary": f"Error executing code: {str(e)}",
            }
    
    def get_dataframe_info(self, dataset_name: str) -> dict[str, Any]:
        """Get information about a DataFrame."""
        df = self.data_loader.get_dataframe(dataset_name)
        return convert_to_serializable({
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": list(df.shape),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "null_counts": df.isnull().sum().to_dict(),
        })
    
    def describe_dataframe(self, dataset_name: str) -> dict[str, Any]:
        """Get statistical description of a DataFrame."""
        df = self.data_loader.get_dataframe(dataset_name)
        return convert_to_serializable({
            "numeric_stats": df.describe().to_dict(),
            "non_numeric_stats": df.describe(include=["object", "category"]).to_dict()
            if df.select_dtypes(include=["object", "category"]).shape[1] > 0
            else {},
        })
    
    def filter_dataframe(
        self, dataset_name: str, conditions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Filter a DataFrame based on conditions.
        
        Args:
            dataset_name: Name of the dataset.
            conditions: Dictionary of column -> condition pairs.
        
        Returns:
            Filtered DataFrame information.
        """
        df = self.data_loader.get_dataframe(dataset_name)
        
        for col, condition in conditions.items():
            if col not in df.columns:
                return {"success": False, "error": f"Column '{col}' not found"}
            
            if isinstance(condition, dict):
                op = condition.get("op", "eq")
                value = condition.get("value")
                
                if op == "eq":
                    df = df[df[col] == value]
                elif op == "ne":
                    df = df[df[col] != value]
                elif op == "gt":
                    df = df[df[col] > value]
                elif op == "gte":
                    df = df[df[col] >= value]
                elif op == "lt":
                    df = df[df[col] < value]
                elif op == "lte":
                    df = df[df[col] <= value]
                elif op == "in":
                    df = df[df[col].isin(value)]
                elif op == "contains":
                    df = df[df[col].str.contains(value, na=False)]
            else:
                df = df[df[col] == condition]
        
        return convert_to_serializable({
            "success": True,
            "result_type": "dataframe",
            "data": df.head(100).to_dict("records"),
            "shape": list(df.shape),
            "summary": f"Filtered to {df.shape[0]} rows",
        })
    
    def aggregate_dataframe(
        self,
        dataset_name: str,
        group_by: list[str],
        aggregations: dict[str, str],
    ) -> dict[str, Any]:
        """
        Aggregate a DataFrame with groupby operations.
        
        Args:
            dataset_name: Name of the dataset.
            group_by: Columns to group by.
            aggregations: Dictionary of column -> aggregation function.
        
        Returns:
            Aggregated result.
        """
        df = self.data_loader.get_dataframe(dataset_name)
        
        try:
            result = df.groupby(group_by).agg(aggregations).reset_index()
            return convert_to_serializable({
                "success": True,
                "result_type": "dataframe",
                "data": result.head(100).to_dict("records"),
                "columns": list(result.columns),
                "shape": list(result.shape),
                "summary": f"Aggregated to {result.shape[0]} groups",
            })
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_pandas_tools(data_loader: DataLoader) -> list:
    """Create LangChain tools from PandasTools."""
    pandas_tools = PandasTools(data_loader)
    
    @tool
    def execute_pandas(code: str, dataset_name: str) -> dict:
        """
        Execute pandas code on a dataset.
        The code should use 'df' to refer to the dataset.
        Store the final result in a variable called 'result'.
        
        Args:
            code: Python/pandas code to execute
            dataset_name: Name of the dataset to query
        
        Returns:
            Dictionary with execution result
        """
        return pandas_tools.execute_pandas_code(code, dataset_name)
    
    @tool
    def get_dataframe_info(dataset_name: str) -> dict:
        """
        Get information about a dataset including columns, types, and shape.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary with DataFrame information
        """
        return pandas_tools.get_dataframe_info(dataset_name)
    
    @tool
    def describe_data(dataset_name: str) -> dict:
        """
        Get statistical summary of a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary with statistical descriptions
        """
        return pandas_tools.describe_dataframe(dataset_name)
    
    return [execute_pandas, get_dataframe_info, describe_data]
