"""
Extract red and blue Landsat bands for training and validation datasets.
Uses rasterio with CRS transformation (Landsat is in UTM, not lat/lon).
Processes in batches with timeouts to prevent hanging.
Checkpoints after each batch to preserve progress.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.transform import rowcol
import pystac_client
import planetary_computer as pc
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import time
import os
import signal

os.chdir("/Users/uinker/Desktop/EY Challenge/Snowflake Notebooks Package")

MAX_WORKERS = 6       # Reduced from 8 to avoid connection exhaustion
BATCH_SIZE = 200      # Process in batches for checkpointing
FUTURE_TIMEOUT = 60   # Timeout per sample (seconds)
CHECKPOINT_FILE_TRAIN = "red_blue_checkpoint_train.csv"
CHECKPOINT_FILE_VAL = "red_blue_checkpoint_val.csv"

# Set GDAL timeout options for rasterio
os.environ["GDAL_HTTP_TIMEOUT"] = "30"
os.environ["GDAL_HTTP_MAX_RETRY"] = "2"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"


def extract_red_blue(row_tuple):
    """Extract red and blue bands for a single location using rasterio + CRS transform."""
    idx, row = row_tuple
    lat = row['Latitude']
    lon = row['Longitude']
    sample_date = pd.to_datetime(row['Sample Date'], dayfirst=True, errors='coerce')

    bbox_size = 0.00089831  # ~100m buffer
    bbox = [
        lon - bbox_size / 2, lat - bbox_size / 2,
        lon + bbox_size / 2, lat + bbox_size / 2
    ]

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime="2011-01-01/2015-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
        )

        items = search.item_collection()

        if not items:
            return idx, np.nan, np.nan

        sample_date_utc = sample_date.tz_localize("UTC") if sample_date.tzinfo is None else sample_date.tz_convert("UTC")

        items = sorted(
            items,
            key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc)
        )
        selected_item = pc.sign(items[0])

        red_val = np.nan
        blue_val = np.nan

        def read_band(item, band_name):
            """Read a single band value from a COG file."""
            if band_name not in item.assets:
                return np.nan
            href = item.assets[band_name].href
            with rasterio.open(href) as src:
                xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                r, c = rowcol(src.transform, xs[0], ys[0])
                r, c = int(r), int(c)
                if 0 <= r < src.height and 0 <= c < src.width:
                    r_start = max(0, r - 1)
                    c_start = max(0, c - 1)
                    window = rasterio.windows.Window(c_start, r_start, 3, 3)
                    data = src.read(1, window=window).astype(float)
                    data[data == 0] = np.nan
                    if not np.all(np.isnan(data)):
                        return float(np.nanmedian(data))
            return np.nan

        red_val = read_band(selected_item, "red")
        blue_val = read_band(selected_item, "blue")

        return idx, red_val, blue_val

    except Exception as e:
        return idx, np.nan, np.nan


def load_checkpoint(checkpoint_file, total_rows):
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        ckpt = pd.read_csv(checkpoint_file)
        red_vals = ckpt["red"].tolist()
        blue_vals = ckpt["blue"].tolist()
        processed = len(red_vals)
        print(f"  Loaded checkpoint: {processed}/{total_rows} samples already done")
        return red_vals, blue_vals, processed
    return [], [], 0


def save_checkpoint(checkpoint_file, red_vals, blue_vals):
    """Save checkpoint to file."""
    pd.DataFrame({"red": red_vals, "blue": blue_vals}).to_csv(checkpoint_file, index=False)


def process_dataset(df, name, checkpoint_file):
    """Extract red/blue for an entire dataset using parallel API calls with batching."""
    print(f"\n{'='*60}")
    print(f"Extracting red+blue for {name} ({len(df)} samples, {MAX_WORKERS} workers)")
    print(f"{'='*60}")

    # Load checkpoint
    red_vals, blue_vals, already_done = load_checkpoint(checkpoint_file, len(df))

    if already_done >= len(df):
        print(f"  All {len(df)} samples already extracted!")
        return red_vals, blue_vals

    remaining_rows = list(df.iloc[already_done:].iterrows())
    total_remaining = len(remaining_rows)
    print(f"  Remaining: {total_remaining} samples")

    start = time.time()

    # Process in batches
    for batch_start in range(0, total_remaining, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_remaining)
        batch = remaining_rows[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} samples)")
        batch_results = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_red_blue, row): row[0] for row in batch}

            done_count = 0
            with tqdm(total=len(batch), desc=f"  batch{batch_num}", leave=False) as pbar:
                for future in as_completed(futures, timeout=FUTURE_TIMEOUT * len(batch)):
                    try:
                        idx, red, blue = future.result(timeout=FUTURE_TIMEOUT)
                        batch_results[idx] = (red, blue)
                    except (TimeoutError, Exception) as e:
                        idx = futures[future]
                        batch_results[idx] = (np.nan, np.nan)
                    done_count += 1
                    pbar.update(1)

            # Handle any futures that didn't complete
            for future, idx in futures.items():
                if idx not in batch_results:
                    batch_results[idx] = (np.nan, np.nan)
                    future.cancel()

        # Append batch results in order
        for row_tuple in batch:
            idx = row_tuple[0]
            red, blue = batch_results.get(idx, (np.nan, np.nan))
            red_vals.append(red)
            blue_vals.append(blue)

        # Save checkpoint after each batch
        save_checkpoint(checkpoint_file, red_vals, blue_vals)
        elapsed = time.time() - start
        done_so_far = batch_end
        rate = done_so_far / elapsed if elapsed > 0 else 0
        eta = (total_remaining - done_so_far) / rate if rate > 0 else 0
        print(f"  Checkpoint saved: {already_done + done_so_far}/{len(df)} | "
              f"{rate:.1f} samples/s | ETA: {eta/60:.1f} min")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes ({elapsed/total_remaining:.2f}s per sample)")

    non_null_red = sum(1 for v in red_vals if not (isinstance(v, float) and np.isnan(v)))
    non_null_blue = sum(1 for v in blue_vals if not (isinstance(v, float) and np.isnan(v)))
    print(f"Red:  {non_null_red}/{len(df)} non-null ({non_null_red/len(df)*100:.1f}%)")
    print(f"Blue: {non_null_blue}/{len(df)} non-null ({non_null_blue/len(df)*100:.1f}%)")

    return red_vals, blue_vals


def main():
    start_time = time.time()

    wq_train = pd.read_csv("water_quality_training_dataset.csv")
    submission = pd.read_csv("submission_template.csv")

    # ---- Training ----
    red_train, blue_train = process_dataset(wq_train, "training", CHECKPOINT_FILE_TRAIN)

    landsat_train = pd.read_csv("landsat_features_training.csv")
    landsat_train["red"] = red_train
    landsat_train["blue"] = blue_train

    # Keep all existing columns + new ones
    existing_cols = [c for c in landsat_train.columns if c not in ["red", "blue"]]
    cols = ["Latitude", "Longitude", "Sample Date", "red", "blue"] + \
           [c for c in existing_cols if c not in ["Latitude", "Longitude", "Sample Date"]]
    landsat_train = landsat_train[cols]
    landsat_train.to_csv("landsat_features_training.csv", index=False)
    print(f"\nUpdated landsat_features_training.csv: {landsat_train.shape}")

    # ---- Validation ----
    red_val, blue_val = process_dataset(submission, "validation", CHECKPOINT_FILE_VAL)

    landsat_val = pd.read_csv("landsat_features_validation.csv")
    landsat_val["red"] = red_val
    landsat_val["blue"] = blue_val
    existing_cols = [c for c in landsat_val.columns if c not in ["red", "blue"]]
    cols = ["Latitude", "Longitude", "Sample Date", "red", "blue"] + \
           [c for c in existing_cols if c not in ["Latitude", "Longitude", "Sample Date"]]
    landsat_val = landsat_val[cols]
    landsat_val.to_csv("landsat_features_validation.csv", index=False)
    print(f"\nUpdated landsat_features_validation.csv: {landsat_val.shape}")

    # Clean up checkpoints
    for f in [CHECKPOINT_FILE_TRAIN, CHECKPOINT_FILE_VAL]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed checkpoint: {f}")

    total = time.time() - start_time
    print(f"\nTotal time: {total/60:.1f} minutes")
    print(f"\nRed and blue bands added to Landsat CSVs.")
    print("New indices available: NDVI, EVI, turbidity, BSI, AWEI, NDBI")


if __name__ == "__main__":
    main()
