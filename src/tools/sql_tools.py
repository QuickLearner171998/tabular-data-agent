"""SQL query tools using DuckDB for data analysis."""

import re
from typing import Any
from langchain_core.tools import tool

from src.data_loader import DataLoader
from src.utils.serialization import convert_to_serializable
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.tools.sql")

MAX_ROWS_DEFAULT = 50  # Default limit for query results


class SQLTools:
    """Tools for executing SQL queries against loaded datasets using DuckDB."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def _ensure_limit(self, query: str, max_rows: int = MAX_ROWS_DEFAULT) -> str:
        """
        Ensure query has a LIMIT clause to prevent large result sets.
        Does not add LIMIT to queries that already have one or use aggregations.
        """
        query_upper = query.upper().strip()
        
        # If query already has LIMIT, don't modify
        if 'LIMIT' in query_upper:
            return query
        
        # Check if it's an aggregation query (likely to return small result)
        has_group_by = 'GROUP BY' in query_upper
        has_agg = any(agg in query_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])
        
        # If aggregation with GROUP BY, add a reasonable limit
        if has_group_by or has_agg:
            return f"{query.rstrip(';')} LIMIT {max_rows}"
        
        # For SELECT without aggregation, always add limit
        if query_upper.startswith('SELECT'):
            return f"{query.rstrip(';')} LIMIT {max_rows}"
        
        return query
    
    def execute_sql(self, query: str, save_as: str | None = None) -> dict[str, Any]:
        """
        Execute a SQL query against loaded datasets.
        
        Args:
            query: SQL query string. Table names match dataset names.
            save_as: Optional name to save results as a new dataset (for viz)
        
        Returns:
            Dictionary with query result.
        """
        try:
            # Ensure query has limits for safety
            safe_query = self._ensure_limit(query)
            logger.info(f"SQL Query: {safe_query}")
            
            result_df = self.data_loader.execute_sql(safe_query)
            row_count = len(result_df)
            logger.info(f"SQL Result: {row_count} rows, {len(result_df.columns)} columns")
            
            # Save result as temp dataset for visualization
            result_dataset = save_as or "_last_query_result"
            self.data_loader.load_dataframe(result_df, result_dataset)
            logger.info(f"Result saved as dataset: {result_dataset}")
            
            # Further limit display data
            display_rows = min(row_count, MAX_ROWS_DEFAULT)
            
            return convert_to_serializable({
                "success": True,
                "result_type": "dataframe",
                "data": result_df.head(display_rows).to_dict("records"),
                "columns": list(result_df.columns),
                "shape": [row_count, len(result_df.columns)],
                "summary": f"Query returned {row_count} rows",
                "result_dataset": result_dataset,  # Tell viz agent where data is
                "truncated": row_count > display_rows,
            })
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return {
                "success": False,
                "result_type": "error",
                "error": str(e),
                "summary": f"SQL Error: {str(e)}",
            }
    
    def get_table_schema(self, table_name: str) -> dict[str, Any]:
        """Get schema information for a table."""
        try:
            df = self.data_loader.get_dataframe(table_name)
            return convert_to_serializable({
                "success": True,
                "table": table_name,
                "columns": [
                    {"name": col, "type": str(df[col].dtype)}
                    for col in df.columns
                ],
                "row_count": len(df),
            })
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_tables(self) -> dict[str, Any]:
        """List all available tables."""
        datasets = self.data_loader.list_datasets()
        return {
            "success": True,
            "tables": datasets,
            "count": len(datasets),
        }
    
    def sample_table(self, table_name: str, n: int = 5) -> dict[str, Any]:
        """Get sample rows from a table."""
        try:
            df = self.data_loader.get_sample(table_name, n)
            return convert_to_serializable({
                "success": True,
                "data": df.to_dict("records"),
                "columns": list(df.columns),
            })
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_sql_tools(data_loader: DataLoader) -> list:
    """Create LangChain tools from SQLTools."""
    sql_tools = SQLTools(data_loader)
    
    @tool
    def execute_sql_query(query: str) -> dict:
        """
        Execute a SQL query against loaded datasets.
        Table names in the query should match the loaded dataset names.
        Results are automatically limited to prevent large outputs.
        
        Example: SELECT category, SUM(revenue) as total FROM transactions GROUP BY category ORDER BY total DESC
        
        Args:
            query: SQL query to execute
        
        Returns:
            Dictionary with query results (max 50 rows)
        """
        return sql_tools.execute_sql(query)
    
    @tool
    def get_table_info(table_name: str) -> dict:
        """
        Get schema information for a specific table/dataset.
        
        Args:
            table_name: Name of the table/dataset
        
        Returns:
            Dictionary with column names and types
        """
        return sql_tools.get_table_schema(table_name)
    
    @tool
    def list_available_tables() -> dict:
        """
        List all available tables/datasets that can be queried.
        
        Returns:
            Dictionary with list of table names
        """
        return sql_tools.list_tables()
    
    @tool
    def get_sample_data(table_name: str, num_rows: int = 5) -> dict:
        """
        Get sample rows from a table to understand its structure.
        
        Args:
            table_name: Name of the table/dataset
            num_rows: Number of sample rows to return (default 5)
        
        Returns:
            Dictionary with sample data
        """
        return sql_tools.sample_table(table_name, num_rows)
    
    return [execute_sql_query, get_table_info, list_available_tables, get_sample_data]
