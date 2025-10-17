#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "pandas>=2.0.0",
#   "requests>=2.31.0",
# ]
# ///
"""
Preprocess Instacart dataset and convert to Vespa JSONL format.

This script merges product data with aisles and departments, and processes
order data with product associations, outputting in Vespa-compatible JSONL format.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests


def check_and_download_dataset(dataset_dir: Path) -> bool:
    """Check if dataset exists and prompt user to download if not."""
    required_files = [
        "products.csv",
        "aisles.csv",
        "departments.csv",
        "orders.csv",
        "order_products__train.csv",
    ]

    # Check if all required files exist
    missing_files = [f for f in required_files if not (dataset_dir / f).exists()]

    if not missing_files:
        return True

    print("⚠ Dataset not found or incomplete!")
    print(f"Missing files: {', '.join(missing_files)}")
    print()
    print("The Instacart dataset needs to be downloaded from Kaggle.")
    print(
        "URL: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset"
    )
    print()

    response = input("Would you like to download it now? (y/n): ").strip().lower()

    if response not in ["y", "yes"]:
        print("Download cancelled. Please download the dataset manually and try again.")
        return False

    print()
    print("Downloading dataset...")
    print("Note: This requires the Kaggle API to be configured.")
    print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
    print()

    # Create a temporary directory for download
    download_dir = Path.home() / "Downloads"
    zip_file = download_dir / "instacart-online-grocery-basket-analysis-dataset.zip"

    # Download using requests
    download_url = "https://www.kaggle.com/api/v1/datasets/download/yasserh/instacart-online-grocery-basket-analysis-dataset"

    try:
        print(f"Downloading to {zip_file}...")

        # Make the request with streaming to handle large files
        response = requests.get(download_url, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Get total file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Ensure download directory exists
        download_dir.mkdir(parents=True, exist_ok=True)

        # Write the file in chunks with progress indication
        downloaded = 0
        chunk_size = 8192
        with open(zip_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="", flush=True)

        print()  # New line after progress
        print("✓ Download complete!")
        print()

        # Extract the zip file
        print(f"Extracting to {dataset_dir}...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        print("✓ Extraction complete!")
        print()

        # Verify all files are now present
        missing_after_extract = [
            f for f in required_files if not (dataset_dir / f).exists()
        ]

        if missing_after_extract:
            print(
                f"⚠ Warning: Some files are still missing: {', '.join(missing_after_extract)}"
            )
            print("You may need to check the extracted files manually.")
            return False

        print("✓ All required files are now present!")
        print()
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print()
        print("Please ensure you have an active internet connection")
        return False
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def process_products(output_dir: Path, full: bool = False, limit: int = 10) -> None:
    """Process product data and merge with aisles and departments."""
    print("Processing products...")

    df_products = pd.read_csv("dataset/products.csv")
    df_aisles = pd.read_csv("dataset/aisles.csv")
    df_departments = pd.read_csv("dataset/departments.csv")

    # Merge products with aisles and departments to get names instead of IDs
    df_merged = df_products.merge(df_aisles, on="aisle_id", how="left").merge(
        df_departments, on="department_id", how="left"
    )

    # Write to JSONL in Vespa format
    output_file = output_dir / "products.jsonl"

    with open(output_file, "w") as f:
        for i, (_, product) in enumerate(df_merged.iterrows()):
            doc = {
                "put": f"id:products:product::{product['product_id']}",
                "fields": {
                    "product_id": int(product["product_id"]),
                    "product_name": product["product_name"],
                    "aisle": product["aisle"],
                    "department": product["department"],
                },
            }
            f.write(json.dumps(doc) + "\n")

            if not full and i >= limit - 1:
                break

    total_products = len(df_merged) if full else min(limit, len(df_merged))
    print(f"✓ Wrote {total_products} products to {output_file}")


def process_users(output_dir: Path, full: bool = False, limit: int = 10) -> None:
    """Process user data by aggregating orders and building user profiles."""
    print("Processing users...")

    df_order = pd.read_csv("dataset/orders.csv")

    # Read order_products from train set to build user profiles
    df_order_products_train = pd.read_csv("dataset/order_products__train.csv")

    # Merge order_products with orders to get order metadata
    df_order_products_with_metadata = df_order_products_train.merge(
        df_order[["order_id", "user_id", "order_dow", "order_hour_of_day"]],
        on="order_id",
        how="left",
    )

    # Group by user_id to create user documents
    user_groups = df_order_products_with_metadata.groupby("user_id")

    output_file = output_dir / "users.jsonl"

    with open(output_file, "w") as f:
        for i, (user_id, user_purchases_df) in enumerate(user_groups):
            # Build user profile (weightedset of product_ids based on frequency)
            product_counts = user_purchases_df["product_id"].value_counts()
            user_profile = {
                int(prod_id): int(count) for prod_id, count in product_counts.items()
            }

            # Build user_purchases array - flatten each product into a separate purchase
            purchases_array = []
            for _, purchase in user_purchases_df.iterrows():
                purchase_struct = {
                    "order_dow": int(purchase["order_dow"]),
                    "order_hour_of_day": int(purchase["order_hour_of_day"]),
                    "order_id": int(purchase["order_id"]),
                    "product_id": int(purchase["product_id"]),
                }
                purchases_array.append(purchase_struct)

            doc = {
                "put": f"id:users:user::{int(user_id)}",
                "fields": {
                    "user_id": int(user_id),
                    "user_profile": user_profile,
                    "user_purchases": purchases_array,
                },
            }
            f.write(json.dumps(doc) + "\n")

            if not full and i >= limit - 1:
                break

    total_users = len(user_groups) if full else min(limit, len(user_groups))
    print(f"✓ Wrote {total_users} users to {output_file}")


def process_orders(
    output_dir: Path, eval_sets: list[str], full: bool = False, limit: int = 10
) -> None:
    """Process order data and aggregate products into arrays."""
    print(f"Processing orders for eval_set(s): {', '.join(eval_sets)}...")

    df_order = pd.read_csv("dataset/orders.csv")

    for eval_set in eval_sets:
        # Filter orders by eval_set
        df_order_filtered = df_order[df_order["eval_set"] == eval_set].copy()

        if len(df_order_filtered) == 0:
            print(f"⚠ No orders found for eval_set '{eval_set}', skipping...")
            continue

        # Read the corresponding order_products file
        order_products_file = f"dataset/order_products__{eval_set}.csv"
        try:
            df_order_products = pd.read_csv(order_products_file)
        except FileNotFoundError:
            print(f"⚠ File not found: {order_products_file}, skipping {eval_set}...")
            continue

        # Sort order_products by add_to_cart_order to maintain the order
        df_order_products_sorted = df_order_products.sort_values(
            ["order_id", "add_to_cart_order"]
        )

        # Group by order_id and aggregate product_ids into an array
        df_products_aggregated = (
            df_order_products_sorted.groupby("order_id")
            .agg({"product_id": lambda x: list(x)})
            .reset_index()
        )

        # Rename the column to match Vespa schema
        df_products_aggregated.rename(
            columns={"product_id": "product_ids"}, inplace=True
        )

        # Merge with orders dataframe
        df_orders_merged = df_order_filtered.merge(
            df_products_aggregated, on="order_id", how="left"
        )

        # Select only the fields needed for Vespa
        df_orders_vespa = df_orders_merged[
            [
                "order_id",
                "user_id",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
                "product_ids",
            ]
        ]

        # Write to JSONL in Vespa format
        output_file = output_dir / f"orders_{eval_set}.jsonl"

        with open(output_file, "w") as f:
            for i, (_, order) in enumerate(df_orders_vespa.iterrows()):
                doc = {
                    "put": f"id:orders:order::{order['order_id']}",
                    "fields": {
                        "order_id": int(order["order_id"]),
                        "user_id": int(order["user_id"]),
                        "order_dow": int(order["order_dow"]),
                        "order_hour_of_day": int(order["order_hour_of_day"]),
                        "days_since_prior_order": (
                            float(order["days_since_prior_order"])
                            if pd.notna(order["days_since_prior_order"])
                            else None
                        ),
                        "product_ids": order["product_ids"],
                    },
                }
                f.write(json.dumps(doc) + "\n")

                if not full and i >= limit - 1:
                    break

        total_orders = (
            len(df_orders_vespa) if full else min(limit, len(df_orders_vespa))
        )
        print(f"✓ Wrote {total_orders} orders to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Instacart dataset and convert to Vespa JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data (10 items) for train set only (default)
  %(prog)s
  
  # Generate full dataset for train set
  %(prog)s --full
  
  # Generate sample data for eval set
  %(prog)s --eval-set eval
  
  # Generate full dataset for both train and eval sets
  %(prog)s --eval-set both --full
  
  # Generate custom sample size
  %(prog)s --limit 100
        """,
    )

    parser.add_argument(
        "--eval-set",
        choices=["train", "eval", "both"],
        default="train",
        help="Which evaluation set(s) to generate data for (default: train)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate full dataset instead of sample (default: False, generates 10 items)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of items to generate when not using --full (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output directory for generated files (default: dataset/)",
    )

    args = parser.parse_args()

    # Validate limit
    if args.limit < 1:
        print("Error: --limit must be at least 1", file=sys.stderr)
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("Vespa Dataset Preprocessor")
    print("=" * 60)
    if args.full:
        print("Mode: Full dataset generation")
    else:
        print(f"Mode: Sample generation (limit: {args.limit} items)")
    print(f"Eval set(s): {args.eval_set}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print()

    # Check if dataset exists and prompt to download if needed
    dataset_dir = Path("dataset")
    if not check_and_download_dataset(dataset_dir):
        print("Exiting: Dataset is required to proceed.")
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process products
    process_products(args.output_dir, args.full, args.limit)
    print()

    # Process users
    process_users(args.output_dir, args.full, args.limit)
    print()

    # Determine which eval sets to process
    eval_sets = []
    if args.eval_set == "both":
        eval_sets = ["train", "eval"]
    else:
        eval_sets = [args.eval_set]

    # Process orders
    process_orders(args.output_dir, eval_sets, args.full, args.limit)

    print()
    print("=" * 60)
    print("✓ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
