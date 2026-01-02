"""
Script to download CPG datasets for testing the agent.

Supports multiple data sources:
1. Kaggle datasets (requires kaggle CLI configured)
2. Direct URL downloads
3. Sample data generation
"""

import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def download_kaggle_dataset(dataset: str, output_dir: Path) -> bool:
    """
    Download a dataset from Kaggle using the kaggle CLI.
    Requires: pip install kaggle + ~/.kaggle/kaggle.json configured.
    """
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir), "--unzip"],
            check=True,
            capture_output=True,
        )
        print(f"Successfully downloaded {dataset}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Kaggle download failed: {e}")
        return False


def download_url(url: str, output_path: Path) -> bool:
    """Download a file from a URL."""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"URL download failed: {e}")
        return False


def generate_sample_cpg_data(output_dir: Path) -> None:
    """
    Generate a synthetic CPG dataset for testing when real data is unavailable.
    Creates data that mimics retail/grocery sales patterns.
    """
    np.random.seed(42)
    
    categories = [
        "Beverages", "Snacks", "Dairy", "Frozen Foods", "Personal Care",
        "Household", "Bakery", "Meat & Seafood", "Produce", "Canned Goods"
    ]
    
    brands = [
        "NatureFresh", "ValueChoice", "PremiumSelect", "EcoLife", "DailyBasics",
        "GourmetPlus", "FamilyFavorite", "OrganicHarvest", "QuickBite", "CleanHome"
    ]
    
    retailers = [
        "MegaMart", "FreshGrocers", "ValueStore", "PremiumMarket", "QuickShop",
        "FamilyMart", "UrbanGrocery", "SuburbanStore"
    ]
    
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    
    n_products = 500
    products = pd.DataFrame({
        "product_id": range(1, n_products + 1),
        "product_name": [f"Product_{i}" for i in range(1, n_products + 1)],
        "category": np.random.choice(categories, n_products),
        "brand": np.random.choice(brands, n_products),
        "unit_price": np.round(np.random.uniform(1.99, 29.99, n_products), 2),
        "unit_cost": np.round(np.random.uniform(0.99, 19.99, n_products), 2),
    })
    products["unit_cost"] = np.minimum(products["unit_cost"], products["unit_price"] * 0.7)
    
    n_stores = 200
    stores = pd.DataFrame({
        "store_id": range(1, n_stores + 1),
        "store_name": [f"Store_{i}" for i in range(1, n_stores + 1)],
        "retailer": np.random.choice(retailers, n_stores),
        "region": np.random.choice(regions, n_stores),
        "store_size_sqft": np.random.choice([5000, 10000, 25000, 50000, 100000], n_stores),
    })
    
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    n_transactions = 100000
    
    transactions = pd.DataFrame({
        "transaction_id": range(1, n_transactions + 1),
        "date": np.random.choice(dates, n_transactions),
        "store_id": np.random.choice(stores["store_id"], n_transactions),
        "product_id": np.random.choice(products["product_id"], n_transactions),
        "quantity": np.random.choice([1, 1, 1, 2, 2, 3, 4, 5, 6], n_transactions),
    })
    
    transactions = transactions.merge(products[["product_id", "unit_price", "unit_cost", "category"]], on="product_id")
    transactions = transactions.merge(stores[["store_id", "retailer", "region"]], on="store_id")
    
    transactions["revenue"] = transactions["quantity"] * transactions["unit_price"]
    transactions["cost"] = transactions["quantity"] * transactions["unit_cost"]
    transactions["profit"] = transactions["revenue"] - transactions["cost"]
    
    month_factors = {1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
                     7: 1.15, 8: 1.1, 9: 1.0, 10: 1.05, 11: 1.2, 12: 1.4}
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["month"] = transactions["date"].dt.month
    transactions["quantity"] = (transactions["quantity"] * 
                                transactions["month"].map(month_factors)).astype(int).clip(lower=1)
    transactions["revenue"] = transactions["quantity"] * transactions["unit_price"]
    transactions["cost"] = transactions["quantity"] * transactions["unit_cost"]
    transactions["profit"] = transactions["revenue"] - transactions["cost"]
    
    transactions = transactions.drop(columns=["month"])
    transactions["date"] = transactions["date"].dt.strftime("%Y-%m-%d")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    products.to_csv(output_dir / "products.csv", index=False)
    stores.to_csv(output_dir / "stores.csv", index=False)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
    
    summary = transactions.groupby(["date", "category", "retailer", "region"]).agg({
        "quantity": "sum",
        "revenue": "sum",
        "profit": "sum",
        "transaction_id": "count"
    }).reset_index()
    summary.columns = ["date", "category", "retailer", "region", "units_sold", "revenue", "profit", "num_transactions"]
    summary.to_csv(output_dir / "daily_sales_summary.csv", index=False)
    
    print(f"Generated sample CPG data in {output_dir}")
    print(f"  - products.csv: {len(products)} products")
    print(f"  - stores.csv: {len(stores)} stores")
    print(f"  - transactions.csv: {len(transactions)} transactions")
    print(f"  - daily_sales_summary.csv: {len(summary)} daily summaries")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CPG Dataset Download Script")
    print("=" * 60)
    
    print("\nOption 1: Trying to download Instacart dataset from Kaggle...")
    kaggle_success = download_kaggle_dataset(
        "psparks/instacart-market-basket-analysis",
        DATA_DIR / "instacart"
    )
    
    if not kaggle_success:
        print("\nKaggle download failed. This could be because:")
        print("  - Kaggle CLI not installed (pip install kaggle)")
        print("  - Kaggle credentials not configured (~/.kaggle/kaggle.json)")
        print("\nOption 2: Generating synthetic CPG dataset instead...")
        generate_sample_cpg_data(DATA_DIR / "cpg_sample")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
