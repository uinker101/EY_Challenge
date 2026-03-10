"""
Extract ALL 14 TerraClimate variables for training and validation datasets.
OPTIMIZED: Uses xarray spatial slicing instead of converting entire global grid.
Outputs: terraclimate_features_training.csv and terraclimate_features_validation.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import pystac_client
import planetary_computer as pc
from tqdm import tqdm
import os
import time

os.chdir("/Users/uinker/Desktop/EY Challenge/Snowflake Notebooks Package")

# All 14 TerraClimate variables
TC_VARS = ["aet", "def", "pet", "ppt", "q", "soil", "srad", "swe",
           "tmax", "tmin", "vap", "vpd", "ws", "PDSI"]

# South Africa bounding box
LAT_MIN, LAT_MAX = -35.18, -21.72
LON_MIN, LON_MAX = 14.97, 32.79


def load_terraclimate_dataset():
    print("Loading TerraClimate dataset from Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])

    print("Dataset loaded successfully.")
    return ds


def extract_var_fast(ds, var):
    """
    Extract a TerraClimate variable for South Africa region, 2011-2015.
    Uses xarray native slicing (much faster than converting entire global grid).
    Returns a DataFrame with columns: Latitude, Longitude, Sample Date, <var>
    """
    print(f"  Selecting spatial/temporal subset for {var}...")
    # Use xarray sel to subset BEFORE converting to DataFrame
    subset = ds[var].sel(
        time=slice("2011-01-01", "2015-12-31"),
        lat=slice(LAT_MAX, LAT_MIN),  # lat is typically descending
        lon=slice(LON_MIN, LON_MAX)
    )

    # If lat slice returned empty (lat may be ascending), try the other direction
    if subset.sizes.get('lat', 0) == 0:
        subset = ds[var].sel(
            time=slice("2011-01-01", "2015-12-31"),
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX)
        )

    print(f"  Subset shape: {dict(subset.sizes)}")
    print(f"  Loading into memory...")

    # Load the subset into memory and convert to DataFrame
    df = subset.to_dataframe().reset_index()
    df = df.rename(columns={'lat': 'Latitude', 'lon': 'Longitude', 'time': 'Sample Date'})
    df['Sample Date'] = pd.to_datetime(df['Sample Date'])

    print(f"  DataFrame shape: {df.shape}")
    return df


def assign_nearest_climate_fast(sa_df, climate_df, var_name):
    """
    Map nearest climate variable values to sampling locations.
    OPTIMIZED: Uses vectorized groupby instead of row-by-row loop.
    """
    sa_df = sa_df.reset_index(drop=True)
    sa_dates = pd.to_datetime(sa_df['Sample Date'], dayfirst=True, errors='coerce')

    # Get unique grid points from climate data
    climate_coords = climate_df[['Latitude', 'Longitude']].drop_duplicates()
    climate_coords_rad = np.radians(climate_coords.values)

    # Find nearest grid point for each sample
    sa_coords_rad = np.radians(sa_df[['Latitude', 'Longitude']].values)
    tree = cKDTree(climate_coords_rad)
    _, idx = tree.query(sa_coords_rad, k=1)

    nearest_lats = climate_coords.iloc[idx]['Latitude'].values
    nearest_lons = climate_coords.iloc[idx]['Longitude'].values

    # Pre-index climate data by (lat, lon) for fast lookup
    climate_df = climate_df.copy()
    climate_df['Sample Date'] = pd.to_datetime(climate_df['Sample Date'])
    climate_grouped = climate_df.groupby(['Latitude', 'Longitude'])

    # Build a dict of (lat, lon) -> sorted time series
    point_data = {}
    for (lat, lon), group in climate_grouped:
        sorted_group = group.sort_values('Sample Date')
        point_data[(lat, lon)] = (
            sorted_group['Sample Date'].values,
            sorted_group[var_name].values
        )

    # Vectorized: for each sample, find nearest time in its grid point
    values = np.full(len(sa_df), np.nan)
    for i in tqdm(range(len(sa_df)), desc=f"  Mapping {var_name}", mininterval=5):
        key = (nearest_lats[i], nearest_lons[i])
        if key not in point_data:
            continue
        times, vals = point_data[key]
        sample_date = sa_dates.iloc[i]
        if pd.isna(sample_date):
            continue
        # Find nearest time index
        time_diffs = np.abs(times - np.datetime64(sample_date))
        nearest_time_idx = np.argmin(time_diffs)
        values[i] = vals[nearest_time_idx]

    return values


def extract_for_dataset(ds, samples_df, dataset_name):
    """Extract all TC variables for a given dataset."""
    results = {}

    for var in TC_VARS:
        var_start = time.time()
        print(f"\n--- {var} ({dataset_name}) ---")
        try:
            climate_df = extract_var_fast(ds, var)
            values = assign_nearest_climate_fast(samples_df, climate_df, var)
            results[var] = values
            non_null = np.sum(~np.isnan(values))
            elapsed = time.time() - var_start
            print(f"  Done: {non_null}/{len(values)} non-null ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    start_time = time.time()

    # Load source data
    wq_train = pd.read_csv("water_quality_training_dataset.csv")
    submission = pd.read_csv("submission_template.csv")
    print(f"Training samples: {len(wq_train)}")
    print(f"Validation samples: {len(submission)}")

    # Load TerraClimate dataset once
    ds = load_terraclimate_dataset()

    # ========== TRAINING DATA ==========
    print("\n" + "=" * 60)
    print("EXTRACTING TRAINING DATA")
    print("=" * 60)

    train_results = extract_for_dataset(ds, wq_train, "training")

    terra_train = pd.DataFrame({
        'Latitude': wq_train['Latitude'],
        'Longitude': wq_train['Longitude'],
        'Sample Date': wq_train['Sample Date'],
    })
    for var in TC_VARS:
        if var in train_results:
            terra_train[var] = train_results[var]

    extracted_train = [v for v in TC_VARS if v in train_results]
    terra_train.to_csv('terraclimate_features_training.csv', index=False)
    print(f"\nSaved terraclimate_features_training.csv: {terra_train.shape}")
    print(f"Variables: {extracted_train}")

    # ========== VALIDATION DATA ==========
    print("\n" + "=" * 60)
    print("EXTRACTING VALIDATION DATA")
    print("=" * 60)

    val_results = extract_for_dataset(ds, submission, "validation")

    terra_val = pd.DataFrame({
        'Latitude': submission['Latitude'],
        'Longitude': submission['Longitude'],
        'Sample Date': submission['Sample Date'],
    })
    for var in TC_VARS:
        if var in val_results:
            terra_val[var] = val_results[var]

    extracted_val = [v for v in TC_VARS if v in val_results]
    terra_val.to_csv('terraclimate_features_validation.csv', index=False)
    print(f"\nSaved terraclimate_features_validation.csv: {terra_val.shape}")
    print(f"Variables: {extracted_val}")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Training: {len(extracted_train)}/{len(TC_VARS)} variables")
    print(f"Validation: {len(extracted_val)}/{len(TC_VARS)} variables")


if __name__ == "__main__":
    main()
