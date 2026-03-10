"""
Extract ALL 8 Landsat bands for the validation dataset.
Bands: coastal, blue, green, red, nir08, swir16, swir22, lwir11
Also computes derived indices: NDMI, MNDWI, NDVI, NDWI, EVI, BSI, AWEI, MNDWI2, MI2, SWIR_NDI

Uses rasterio with CRS transformation (Landsat is in UTM, not lat/lon).
Processes in batches with checkpointing to preserve progress.

Usage:
    pip install rasterio pystac-client planetary-computer tqdm pandas numpy
    python extract_all_landsat_validation.py
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====================== CONFIG ======================
MAX_WORKERS = 6
BATCH_SIZE = 50           # Smaller batches for 200 samples
FUTURE_TIMEOUT = 90       # Seconds per sample (lwir11 is a larger file)
CHECKPOINT_FILE = "landsat_all_bands_val_checkpoint.csv"
OUTPUT_FILE = "landsat_features_validation.csv"

# All 8 Landsat bands available in Collection 2 Level-2
BANDS = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "lwir11"]

# Reflectance scale factor (for optical bands B1-B7)
REFLECTANCE_SCALE = 0.0000275
REFLECTANCE_OFFSET = -0.2

# Surface temperature scale factor (for lwir11 / B10)
ST_SCALE = 0.00341802
ST_OFFSET = 149.0

# GDAL settings for faster/more reliable HTTP reads
os.environ["GDAL_HTTP_TIMEOUT"] = "30"
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"


def extract_all_bands(row_tuple):
    """Extract all 8 Landsat bands for a single location."""
    idx, row = row_tuple
    lat = row["Latitude"]
    lon = row["Longitude"]
    sample_date = pd.to_datetime(row["Sample Date"], dayfirst=True, errors="coerce")

    result = {band: np.nan for band in BANDS}

    bbox_size = 0.00089831  # ~100m buffer
    bbox = [
        lon - bbox_size / 2, lat - bbox_size / 2,
        lon + bbox_size / 2, lat + bbox_size / 2,
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
            return idx, result

        # Pick the scene closest to the sample date
        sample_date_utc = (
            sample_date.tz_localize("UTC")
            if sample_date.tzinfo is None
            else sample_date.tz_convert("UTC")
        )
        items = sorted(
            items,
            key=lambda x: abs(
                pd.to_datetime(x.properties["datetime"]).tz_convert("UTC")
                - sample_date_utc
            ),
        )
        selected_item = pc.sign(items[0])

        for band in BANDS:
            if band not in selected_item.assets:
                continue
            try:
                href = selected_item.assets[band].href
                with rasterio.open(href) as src:
                    xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                    r, c = rowcol(src.transform, xs[0], ys[0])
                    r, c = int(r), int(c)
                    if 0 <= r < src.height and 0 <= c < src.width:
                        # Read a 3x3 pixel window and take the median
                        r_start = max(0, r - 1)
                        c_start = max(0, c - 1)
                        window = rasterio.windows.Window(c_start, r_start, 3, 3)
                        data = src.read(1, window=window).astype(float)
                        data[data == 0] = np.nan
                        if not np.all(np.isnan(data)):
                            result[band] = float(np.nanmedian(data))
            except Exception:
                pass

        return idx, result

    except Exception:
        return idx, result


def load_checkpoint(total_rows):
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = pd.read_csv(CHECKPOINT_FILE)
        processed = len(ckpt)
        print(f"  Loaded checkpoint: {processed}/{total_rows} samples done")
        records = ckpt.to_dict("records")
        return records, processed
    return [], 0


def save_checkpoint(records):
    """Save checkpoint to file."""
    pd.DataFrame(records).to_csv(CHECKPOINT_FILE, index=False)


def compute_indices(df):
    """Compute derived spectral indices from raw bands.
    Input bands are raw DN values (not yet scaled to reflectance)."""
    out = df.copy()
    eps = 1e-10

    # Scale optical bands to surface reflectance
    optical_bands = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]
    for band in optical_bands:
        if band in out.columns:
            out[band] = out[band] * REFLECTANCE_SCALE + REFLECTANCE_OFFSET

    # Scale thermal band to surface temperature (Kelvin)
    if "lwir11" in out.columns:
        out["lwir11"] = out["lwir11"] * ST_SCALE + ST_OFFSET

    nir = out["nir08"].astype(float)
    green = out["green"].astype(float)
    red = out["red"].astype(float)
    blue = out["blue"].astype(float)
    swir16 = out["swir16"].astype(float)
    swir22 = out["swir22"].astype(float)

    # Normalized Difference Indices
    out["NDVI"] = (nir - red) / (nir + red + eps)
    out["NDMI"] = (nir - swir16) / (nir + swir16 + eps)
    out["MNDWI"] = (green - swir16) / (green + swir16 + eps)
    out["NDWI"] = (green - nir) / (green + nir + eps)
    out["MI2"] = (nir - swir22) / (nir + swir22 + eps)
    out["MNDWI2"] = (green - swir22) / (green + swir22 + eps)
    out["SWIR_NDI"] = (swir16 - swir22) / (swir16 + swir22 + eps)

    # Enhanced Vegetation Index
    evi_denom = nir + 6.0 * red - 7.5 * blue + 1.0
    out["EVI"] = 2.5 * (nir - red) / (evi_denom + eps)

    # Bare Soil Index
    bsi_num = (swir16 + red) - (nir + blue)
    bsi_den = (swir16 + red) + (nir + blue)
    out["BSI"] = bsi_num / (bsi_den + eps)

    # Automated Water Extraction Index
    out["AWEI"] = blue - 2.5 * green - 1.5 * (nir + swir16) - 0.25 * swir22

    # Turbidity proxy (blue/red ratio)
    out["turbidity_proxy"] = blue / (red + eps)

    # Normalized Difference Built-up Index
    out["NDBI"] = (swir16 - nir) / (swir16 + nir + eps)

    return out


def main():
    start_time = time.time()

    submission = pd.read_csv("submission_template.csv")
    total = len(submission)
    print(f"Validation samples to extract: {total}")

    # Load checkpoint
    records, already_done = load_checkpoint(total)

    if already_done >= total:
        print("All samples already extracted!")
    else:
        remaining_rows = list(submission.iloc[already_done:].iterrows())
        total_remaining = len(remaining_rows)
        print(f"Remaining: {total_remaining} samples")

        for batch_start in range(0, total_remaining, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_remaining)
            batch = remaining_rows[batch_start:batch_end]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} samples)")
            batch_results = {}

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(extract_all_bands, row): row[0] for row in batch
                }

                done_count = 0
                with tqdm(total=len(batch), desc=f"  batch{batch_num}", leave=False) as pbar:
                    for future in as_completed(futures, timeout=FUTURE_TIMEOUT * len(batch)):
                        try:
                            idx, band_vals = future.result(timeout=FUTURE_TIMEOUT)
                            batch_results[idx] = band_vals
                        except (TimeoutError, Exception):
                            idx = futures[future]
                            batch_results[idx] = {band: np.nan for band in BANDS}
                        done_count += 1
                        pbar.update(1)

                # Handle stragglers
                for future, idx in futures.items():
                    if idx not in batch_results:
                        batch_results[idx] = {band: np.nan for band in BANDS}
                        future.cancel()

            # Append results in order
            for row_tuple in batch:
                idx = row_tuple[0]
                vals = batch_results.get(idx, {band: np.nan for band in BANDS})
                records.append(vals)

            # Checkpoint
            save_checkpoint(records)
            elapsed = time.time() - start_time
            done_so_far = already_done + batch_end
            rate = (batch_end) / elapsed if elapsed > 0 else 0
            eta = (total_remaining - batch_end) / rate if rate > 0 else 0
            print(
                f"  Checkpoint saved: {done_so_far}/{total} | "
                f"{rate:.1f} samples/s | ETA: {eta / 60:.1f} min"
            )

    # Build the final DataFrame from raw DN values
    bands_df = pd.DataFrame(records, columns=BANDS)

    # Combine with location info
    result_df = pd.DataFrame({
        "Latitude": submission["Latitude"].values,
        "Longitude": submission["Longitude"].values,
        "Sample Date": submission["Sample Date"].values,
    })
    for band in BANDS:
        result_df[band] = bands_df[band].values

    # Compute all derived indices (also applies scale factors)
    result_df = compute_indices(result_df)

    # Reorder columns: id cols, raw bands, thermal, then indices
    id_cols = ["Latitude", "Longitude", "Sample Date"]
    index_cols = [c for c in result_df.columns if c not in id_cols + BANDS]
    result_df = result_df[id_cols + BANDS + index_cols]

    # Save
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {OUTPUT_FILE}: {result_df.shape}")

    # Report
    print("\nColumn completeness:")
    for c in result_df.columns:
        if c in id_cols:
            continue
        nn = result_df[c].notna().sum()
        print(f"  {c:20s}: {nn}/{total} non-null ({nn / total * 100:.1f}%)")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\nRemoved checkpoint: {CHECKPOINT_FILE}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
