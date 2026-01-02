"""Data context manager for providing schema and data information to agents."""

from typing import Any
import pandas as pd

from src.data_loader import DataLoader


class DataContext:
    """
    Manages data context for the agent system.
    
    Provides schema information, sample data, and metadata
    that helps agents understand the available data.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self._context_cache: dict[str, Any] = {}
    
    def get_compact_context(self) -> str:
        """
        Get a compact context string with only essential schema info.
        This is optimized to stay within LLM context limits.
        
        Returns:
            A compact formatted string with schema summary.
        """
        if not self.data_loader.list_datasets():
            return "No datasets are currently loaded."
        
        context_parts = ["=== AVAILABLE DATA ===\n"]
        
        for name in self.data_loader.list_datasets():
            schema = self.data_loader.get_schema(name)
            context_parts.append(f"Table: {name} ({schema['row_count']:,} rows)")
            
            columns_summary = []
            for col_name, col_info in schema["columns"].items():
                dtype = col_info["dtype"]
                # Simplify dtype
                if "int" in dtype:
                    dtype = "int"
                elif "float" in dtype:
                    dtype = "float"
                elif dtype == "object":
                    dtype = "str"
                columns_summary.append(f"  - {col_name}: {dtype}")
            
            context_parts.extend(columns_summary)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_full_context(self) -> str:
        """
        Get a comprehensive context string for all loaded datasets.
        WARNING: This can be large - use get_compact_context() for LLM calls.
        
        Returns:
            A formatted string containing schema and sample information.
        """
        if not self.data_loader.list_datasets():
            return "No datasets are currently loaded."
        
        context_parts = [
            "=== DATA CONTEXT ===",
            "",
            "Available Datasets:",
        ]
        
        for name in self.data_loader.list_datasets():
            context_parts.append(f"  - {name}")
        
        context_parts.append("")
        context_parts.append("Schema Information:")
        context_parts.append(self.data_loader.get_schema_description())
        
        return "\n".join(context_parts)
    
    def get_dataset_context(self, dataset_name: str) -> str:
        """
        Get context for a specific dataset including schema and samples.
        
        Args:
            dataset_name: Name of the dataset.
        
        Returns:
            Formatted context string for the dataset.
        """
        if dataset_name not in self.data_loader.list_datasets():
            return f"Dataset '{dataset_name}' not found."
        
        schema = self.data_loader.get_schema(dataset_name)
        sample = self.data_loader.get_sample(dataset_name, n=3)
        
        context_parts = [
            f"=== Dataset: {dataset_name} ===",
            f"Rows: {schema['row_count']:,}",
            f"Memory: {schema['memory_usage_mb']:.2f} MB",
            "",
            "Columns:",
        ]
        
        for col_name, col_info in schema["columns"].items():
            line = f"  - {col_name}: {col_info['dtype']}"
            if col_info["null_count"] > 0:
                pct = col_info["null_count"] / schema["row_count"] * 100
                line += f" ({pct:.1f}% null)"
            context_parts.append(line)
        
        context_parts.append("")
        context_parts.append("Sample Data (first 3 rows):")
        context_parts.append(sample.to_string())
        
        return "\n".join(context_parts)
    
    def get_column_info(self, dataset_name: str, column_name: str) -> dict[str, Any]:
        """
        Get detailed information about a specific column.
        
        Args:
            dataset_name: Name of the dataset.
            column_name: Name of the column.
        
        Returns:
            Dictionary with column statistics and information.
        """
        schema = self.data_loader.get_schema(dataset_name)
        
        if column_name not in schema["columns"]:
            raise ValueError(f"Column '{column_name}' not found in dataset '{dataset_name}'")
        
        col_info = schema["columns"][column_name].copy()
        col_info["name"] = column_name
        col_info["dataset"] = dataset_name
        
        return col_info
    
    def suggest_analysis_type(self, query: str) -> list[str]:
        """
        Suggest relevant analysis types based on the query.
        
        Args:
            query: The user's natural language query.
        
        Returns:
            List of suggested analysis types.
        """
        query_lower = query.lower()
        suggestions = []
        
        viz_keywords = ["plot", "chart", "graph", "visual", "show", "display", "trend", "compare"]
        if any(kw in query_lower for kw in viz_keywords):
            suggestions.append("visualization")
        
        stats_keywords = ["average", "mean", "sum", "count", "total", "max", "min", "correlation"]
        if any(kw in query_lower for kw in stats_keywords):
            suggestions.append("aggregation")
        
        filter_keywords = ["where", "filter", "only", "between", "greater", "less", "equal"]
        if any(kw in query_lower for kw in filter_keywords):
            suggestions.append("filtering")
        
        group_keywords = ["by", "per", "each", "group", "breakdown", "segment"]
        if any(kw in query_lower for kw in group_keywords):
            suggestions.append("grouping")
        
        time_keywords = ["trend", "over time", "monthly", "weekly", "daily", "yearly", "forecast"]
        if any(kw in query_lower for kw in time_keywords):
            suggestions.append("time_series")
        
        if not suggestions:
            suggestions.append("exploration")
        
        return suggestions
    
    def get_relevant_columns(self, query: str, dataset_name: str) -> list[str]:
        """
        Identify columns that might be relevant to a query.
        
        Args:
            query: The user's natural language query.
            dataset_name: Name of the dataset.
        
        Returns:
            List of potentially relevant column names.
        """
        schema = self.data_loader.get_schema(dataset_name)
        query_lower = query.lower()
        relevant = []
        
        for col_name in schema["columns"]:
            col_lower = col_name.lower().replace("_", " ")
            if col_lower in query_lower or any(word in query_lower for word in col_lower.split()):
                relevant.append(col_name)
        
        return relevant
