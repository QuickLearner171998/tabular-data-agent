"""Tests for the CPG Data Analysis Agent components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_loader import DataLoader
from src.core.data_context import DataContext


def test_data_loader():
    """Test basic data loading functionality."""
    loader = DataLoader()
    
    df = pd.DataFrame({
        "product": ["A", "B", "C"],
        "category": ["Food", "Beverage", "Food"],
        "revenue": [100.0, 200.0, 150.0],
        "quantity": [10, 20, 15],
    })
    
    name = loader.load_dataframe(df, "test_data")
    assert name == "test_data"
    assert "test_data" in loader.list_datasets()
    
    schema = loader.get_schema("test_data")
    assert schema["row_count"] == 3
    assert len(schema["columns"]) == 4
    
    sample = loader.get_sample("test_data", n=2)
    assert len(sample) == 2
    
    result = loader.execute_sql("SELECT category, SUM(revenue) as total FROM test_data GROUP BY category")
    assert len(result) == 2
    
    print("test_data_loader passed")


def test_data_context():
    """Test data context functionality."""
    loader = DataLoader()
    
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "product": ["A", "B", "A", "C", "B"],
        "sales": [100, 200, 150, 300, 250],
    })
    loader.load_dataframe(df, "sales_data")
    
    context = DataContext(loader)
    
    full_context = context.get_full_context()
    assert "sales_data" in full_context
    assert "columns" in full_context.lower() or "Columns" in full_context
    
    dataset_context = context.get_dataset_context("sales_data")
    assert "sales_data" in dataset_context
    
    suggestions = context.suggest_analysis_type("show me the sales trend over time")
    assert "visualization" in suggestions or "time_series" in suggestions
    
    print("test_data_context passed")


def test_load_cpg_sample_data():
    """Test loading the generated CPG sample data."""
    loader = DataLoader()
    data_dir = Path(__file__).parent.parent / "data" / "cpg_sample"
    
    if not data_dir.exists():
        print("Skipping test_load_cpg_sample_data - sample data not generated")
        return
    
    for csv_file in data_dir.glob("*.csv"):
        name = csv_file.stem
        loader.load_csv(csv_file, name)
    
    datasets = loader.list_datasets()
    assert len(datasets) > 0, "No datasets loaded"
    
    if "transactions" in datasets:
        schema = loader.get_schema("transactions")
        assert schema["row_count"] > 0
        print(f"Loaded transactions with {schema['row_count']} rows")
    
    if "products" in datasets:
        schema = loader.get_schema("products")
        assert schema["row_count"] > 0
        print(f"Loaded products with {schema['row_count']} rows")
    
    print("test_load_cpg_sample_data passed")


if __name__ == "__main__":
    test_data_loader()
    test_data_context()
    test_load_cpg_sample_data()
    print("\nAll tests passed!")
