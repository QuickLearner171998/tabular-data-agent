"""Statistical analysis tools for data analysis."""

from typing import Any
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from src.data_loader import DataLoader
from src.utils.serialization import convert_to_serializable
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.tools.stats")


class StatsTools:
    """Tools for performing statistical analysis on datasets."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def basic_statistics(self, dataset_name: str, column: str | None = None) -> dict[str, Any]:
        """
        Get basic statistics for a dataset or specific column.
        
        Args:
            dataset_name: Name of the dataset.
            column: Optional specific column to analyze.
        
        Returns:
            Dictionary with statistical measures.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            if column:
                if column not in df.columns:
                    return {"success": False, "error": f"Column '{column}' not found"}
                
                series = df[column]
                
                if pd.api.types.is_numeric_dtype(series):
                    stats = {
                        "count": int(series.count()),
                        "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                        "std": float(series.std()) if not pd.isna(series.std()) else None,
                        "min": float(series.min()) if not pd.isna(series.min()) else None,
                        "25%": float(series.quantile(0.25)) if not pd.isna(series.quantile(0.25)) else None,
                        "50%": float(series.quantile(0.50)) if not pd.isna(series.quantile(0.50)) else None,
                        "75%": float(series.quantile(0.75)) if not pd.isna(series.quantile(0.75)) else None,
                        "max": float(series.max()) if not pd.isna(series.max()) else None,
                        "null_count": int(series.isna().sum()),
                        "unique_count": int(series.nunique()),
                    }
                else:
                    value_counts = series.value_counts().head(10)
                    stats = {
                        "count": int(series.count()),
                        "unique_count": int(series.nunique()),
                        "top_value": str(series.mode().iloc[0]) if not series.mode().empty else None,
                        "null_count": int(series.isna().sum()),
                        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
                    }
                
                return convert_to_serializable({"success": True, "column": column, "statistics": stats})
            else:
                numeric_stats = df.describe().to_dict()
                return convert_to_serializable({
                    "success": True,
                    "statistics": numeric_stats,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                })
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def correlation_analysis(
        self, 
        dataset_name: str, 
        columns: list[str] | None = None,
        method: str = "pearson",
    ) -> dict[str, Any]:
        """
        Calculate correlation between numeric columns.
        
        Args:
            dataset_name: Name of the dataset.
            columns: Optional list of columns to correlate.
            method: Correlation method ('pearson', 'spearman', 'kendall').
        
        Returns:
            Dictionary with correlation matrix.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            if columns:
                df = df[columns]
            
            numeric_df = df.select_dtypes(include=["number"])
            
            if numeric_df.empty:
                return {"success": False, "error": "No numeric columns available"}
            
            corr_matrix = numeric_df.corr(method=method)
            
            strong_correlations = []
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i+1:]:
                    corr_value = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                        strong_correlations.append({
                            "column1": str(col1),
                            "column2": str(col2),
                            "correlation": round(float(corr_value), 4),
                        })
            
            return convert_to_serializable({
                "success": True,
                "method": method,
                "correlation_matrix": corr_matrix.round(4).to_dict(),
                "strong_correlations": sorted(
                    strong_correlations, 
                    key=lambda x: abs(x["correlation"]), 
                    reverse=True
                ),
            })
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def group_statistics(
        self,
        dataset_name: str,
        group_by: list[str],
        value_column: str,
        aggregations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate statistics grouped by categories.
        
        Args:
            dataset_name: Name of the dataset.
            group_by: Columns to group by.
            value_column: Column to calculate statistics for.
            aggregations: List of aggregation functions to apply.
        
        Returns:
            Dictionary with grouped statistics.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            if aggregations is None:
                aggregations = ["count", "mean", "sum", "min", "max"]
            
            agg_dict = {value_column: aggregations}
            result = df.groupby(group_by).agg(agg_dict).reset_index()
            
            result.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] 
                for col in result.columns
            ]
            
            return convert_to_serializable({
                "success": True,
                "grouped_statistics": result.head(100).to_dict("records"),
                "group_count": len(result),
                "columns": list(result.columns),
            })
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def time_series_analysis(
        self,
        dataset_name: str,
        date_column: str,
        value_column: str,
        freq: str = "D",
    ) -> dict[str, Any]:
        """
        Perform basic time series analysis.
        
        Args:
            dataset_name: Name of the dataset.
            date_column: Column containing dates.
            value_column: Column with values to analyze.
            freq: Frequency for resampling ('D', 'W', 'M', 'Q', 'Y').
        
        Returns:
            Dictionary with time series analysis results.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column).sort_index()
            
            resampled = df[value_column].resample(freq).agg(["sum", "mean", "count"])
            
            total = float(df[value_column].sum())
            mean = float(df[value_column].mean())
            
            first_half = df.iloc[:len(df)//2][value_column].mean()
            second_half = df.iloc[len(df)//2:][value_column].mean()
            trend = "increasing" if second_half > first_half else "decreasing"
            trend_pct = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
            
            return convert_to_serializable({
                "success": True,
                "summary": {
                    "total": total,
                    "mean": mean,
                    "trend": trend,
                    "trend_percentage": round(float(trend_pct), 2),
                    "date_range": {
                        "start": str(df.index.min()),
                        "end": str(df.index.max()),
                    },
                },
                "resampled_data": resampled.reset_index().to_dict("records"),
            })
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def outlier_detection(
        self,
        dataset_name: str,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> dict[str, Any]:
        """
        Detect outliers in a numeric column.
        
        Args:
            dataset_name: Name of the dataset.
            column: Column to analyze.
            method: Detection method ('iqr' or 'zscore').
            threshold: Threshold for outlier detection.
        
        Returns:
            Dictionary with outlier information.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found"}
            
            series = df[column].dropna()
            
            if method == "iqr":
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            else:
                mean = float(series.mean())
                std = float(series.std())
                z_scores = (series - mean) / std
                outliers = series[abs(z_scores) > threshold]
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            
            return convert_to_serializable({
                "success": True,
                "method": method,
                "threshold": float(threshold),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                },
                "outlier_count": int(len(outliers)),
                "outlier_percentage": round(float(len(outliers) / len(series) * 100), 2),
                "outlier_values": [float(x) for x in outliers.head(20).tolist()],
            })
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def distribution_analysis(
        self,
        dataset_name: str,
        column: str,
    ) -> dict[str, Any]:
        """
        Analyze the distribution of a numeric column.
        
        Args:
            dataset_name: Name of the dataset.
            column: Column to analyze.
        
        Returns:
            Dictionary with distribution information.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found"}
            
            series = df[column].dropna()
            
            if not pd.api.types.is_numeric_dtype(series):
                return {"success": False, "error": f"Column '{column}' is not numeric"}
            
            skewness = float(series.skew())
            kurtosis = float(series.kurtosis())
            
            percentiles = {
                f"p{p}": float(series.quantile(p/100))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
            
            if abs(skewness) < 0.5:
                distribution_shape = "approximately symmetric"
            elif skewness > 0:
                distribution_shape = "right-skewed (positive skew)"
            else:
                distribution_shape = "left-skewed (negative skew)"
            
            mode_val = series.mode()
            mode = float(mode_val.iloc[0]) if not mode_val.empty else None
            
            return convert_to_serializable({
                "success": True,
                "column": column,
                "statistics": {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "mode": mode,
                    "std": float(series.std()),
                    "variance": float(series.var()),
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                },
                "percentiles": percentiles,
                "distribution_shape": distribution_shape,
            })
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_stats_tools(data_loader: DataLoader) -> list:
    """Create LangChain tools from StatsTools."""
    stats_tools = StatsTools(data_loader)
    
    @tool
    def get_statistics(dataset_name: str, column: str = None) -> dict:
        """
        Get basic statistics for a dataset or specific column.
        
        Args:
            dataset_name: Name of the dataset
            column: Optional specific column to analyze
        
        Returns:
            Dictionary with statistical measures
        """
        return stats_tools.basic_statistics(dataset_name, column)
    
    @tool
    def analyze_correlation(dataset_name: str, method: str = "pearson") -> dict:
        """
        Calculate correlation between numeric columns in a dataset.
        
        Args:
            dataset_name: Name of the dataset
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Dictionary with correlation matrix and strong correlations
        """
        return stats_tools.correlation_analysis(dataset_name, method=method)
    
    @tool
    def calculate_grouped_stats(
        dataset_name: str,
        group_by_columns: str,
        value_column: str,
    ) -> dict:
        """
        Calculate statistics grouped by categories.
        
        Args:
            dataset_name: Name of the dataset
            group_by_columns: Comma-separated column names to group by
            value_column: Column to calculate statistics for
        
        Returns:
            Dictionary with grouped statistics
        """
        group_by = [col.strip() for col in group_by_columns.split(",")]
        return stats_tools.group_statistics(dataset_name, group_by, value_column)
    
    @tool
    def detect_outliers(dataset_name: str, column: str, method: str = "iqr") -> dict:
        """
        Detect outliers in a numeric column.
        
        Args:
            dataset_name: Name of the dataset
            column: Column to analyze for outliers
            method: Detection method ('iqr' or 'zscore')
        
        Returns:
            Dictionary with outlier information
        """
        return stats_tools.outlier_detection(dataset_name, column, method)
    
    @tool
    def analyze_distribution(dataset_name: str, column: str) -> dict:
        """
        Analyze the distribution of a numeric column.
        
        Args:
            dataset_name: Name of the dataset
            column: Column to analyze
        
        Returns:
            Dictionary with distribution information including skewness, kurtosis
        """
        return stats_tools.distribution_analysis(dataset_name, column)
    
    return [
        get_statistics,
        analyze_correlation,
        calculate_grouped_stats,
        detect_outliers,
        analyze_distribution,
    ]
