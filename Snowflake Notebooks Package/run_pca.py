"""
Re-run PCA with full 14-variable TerraClimate feature set.
Based on PCA_FEATURE_ENGINEERING_NOTEBOOK.ipynb
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

os.chdir("/Users/uinker/Desktop/EY Challenge/Snowflake Notebooks Package")

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

wq_train = pd.read_csv('water_quality_training_dataset.csv')
landsat_train = pd.read_csv('landsat_features_training.csv')
terra_train = pd.read_csv('terraclimate_features_training.csv')
landsat_val = pd.read_csv('landsat_features_validation.csv')
terra_val = pd.read_csv('terraclimate_features_validation.csv')
submission = pd.read_csv('submission_template.csv')

print(f'Water quality:   {wq_train.shape}')
print(f'Landsat train:   {landsat_train.shape}')
print(f'TerraClimate train: {terra_train.shape}')
print(f'Landsat val:     {landsat_val.shape}')
print(f'TerraClimate val:   {terra_val.shape}')
print(f'Submission:      {submission.shape}')
print(f'\nTerraClimate columns: {[c for c in terra_train.columns if c not in ["Latitude","Longitude","Sample Date"]]}')

# ============================================================
# 2. Landsat Feature Engineering
# ============================================================
print("\n" + "=" * 60)
print("2. LANDSAT FEATURE ENGINEERING")
print("=" * 60)

def engineer_landsat_features(df):
    out = df.copy()
    eps = 1e-10
    nir = out['nir'].astype(float)
    green = out['green'].astype(float)
    swir16 = out['swir16'].astype(float)
    swir22 = out['swir22'].astype(float)

    # Normalized Difference Indices
    out['NDWI'] = (green - nir) / (green + nir + eps)
    out['NDMI'] = (nir - swir16) / (nir + swir16 + eps)
    out['MNDWI'] = (green - swir16) / (green + swir16 + eps)
    out['MI2'] = (nir - swir22) / (nir + swir22 + eps)
    out['MNDWI2'] = (green - swir22) / (green + swir22 + eps)
    out['SWIR_NDI'] = (swir16 - swir22) / (swir16 + swir22 + eps)

    # Band Ratios
    out['nir_green_ratio'] = nir / (green + eps)
    out['nir_swir16_ratio'] = nir / (swir16 + eps)
    out['nir_swir22_ratio'] = nir / (swir22 + eps)
    out['green_swir16_ratio'] = green / (swir16 + eps)
    out['green_swir22_ratio'] = green / (swir22 + eps)
    out['swir16_swir22_ratio'] = swir16 / (swir22 + eps)

    # Band Statistics
    band_stack = np.column_stack([nir, green, swir16, swir22])
    out['band_mean'] = np.nanmean(band_stack, axis=1)
    out['band_std'] = np.nanstd(band_stack, axis=1)
    out['band_range'] = np.nanmax(band_stack, axis=1) - np.nanmin(band_stack, axis=1)
    out['band_max'] = np.nanmax(band_stack, axis=1)
    out['band_min'] = np.nanmin(band_stack, axis=1)
    out['band_cv'] = out['band_std'] / (out['band_mean'] + eps)

    # Log-transformed bands
    out['log_nir'] = np.log1p(nir)
    out['log_green'] = np.log1p(green)
    out['log_swir16'] = np.log1p(swir16)
    out['log_swir22'] = np.log1p(swir22)

    # Interaction terms
    out['nir_x_green'] = nir * green
    out['swir16_x_swir22'] = swir16 * swir22
    out['nir_x_swir16'] = nir * swir16
    out['green_x_swir22'] = green * swir22

    return out

landsat_train_fe = engineer_landsat_features(landsat_train)
landsat_val_fe = engineer_landsat_features(landsat_val)
new_cols = [c for c in landsat_train_fe.columns if c not in landsat_train.columns]
print(f'Landsat: {len(landsat_train.columns)} -> {len(landsat_train_fe.columns)} columns ({len(new_cols)} new)')

# ============================================================
# 3. TerraClimate Feature Engineering
# ============================================================
print("\n" + "=" * 60)
print("3. TERRACLIMATE FEATURE ENGINEERING")
print("=" * 60)

def engineer_terraclimate_features(df):
    out = df.copy()
    eps = 1e-10

    if 'tmax' in out.columns and 'tmin' in out.columns:
        out['temp_range'] = out['tmax'] - out['tmin']
        out['temp_mean'] = (out['tmax'] + out['tmin']) / 2

    if 'ppt' in out.columns and 'pet' in out.columns:
        out['aridity_index'] = out['ppt'] / (out['pet'] + eps)
        out['moisture_surplus'] = out['ppt'] - out['pet']

    if 'aet' in out.columns and 'pet' in out.columns:
        out['evap_fraction'] = out['aet'] / (out['pet'] + eps)
        out['water_stress'] = out['pet'] - out['aet']

    if 'ppt' in out.columns and 'q' in out.columns:
        out['runoff_ratio'] = out['q'] / (out['ppt'] + eps)

    if 'vap' in out.columns and 'vpd' in out.columns:
        out['vap_total'] = out['vap'] + out['vpd']
        out['relative_humidity_proxy'] = out['vap'] / (out['vap'] + out['vpd'] + eps)

    if 'srad' in out.columns and 'tmax' in out.columns:
        out['radiation_temp_interaction'] = out['srad'] * out['tmax']

    if 'ws' in out.columns and 'pet' in out.columns:
        out['wind_evap_interaction'] = out['ws'] * out['pet']

    if 'soil' in out.columns and 'ppt' in out.columns:
        out['soil_ppt_ratio'] = out['soil'] / (out['ppt'] + eps)

    return out

terra_train_fe = engineer_terraclimate_features(terra_train)
terra_val_fe = engineer_terraclimate_features(terra_val)
terra_new = [c for c in terra_train_fe.columns if c not in terra_train.columns]
print(f'TerraClimate: {len(terra_train.columns)} -> {len(terra_train_fe.columns)} columns ({len(terra_new)} new)')
print(f'Derived features: {terra_new}')

# ============================================================
# 4. Temporal Features
# ============================================================
print("\n" + "=" * 60)
print("4. TEMPORAL FEATURES")
print("=" * 60)

def add_temporal_spatial_features(df, date_col='Sample Date'):
    out = df.copy()
    dt = pd.to_datetime(out[date_col], dayfirst=True, errors='coerce')
    out['year'] = dt.dt.year
    out['month'] = dt.dt.month
    out['day_of_year'] = dt.dt.dayofyear
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    out['doy_sin'] = np.sin(2 * np.pi * out['day_of_year'] / 365)
    out['doy_cos'] = np.cos(2 * np.pi * out['day_of_year'] / 365)
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out['season'] = out['month'].map(season_map)
    return out

wq_train_fe = add_temporal_spatial_features(wq_train)
submission_fe = add_temporal_spatial_features(submission)
temporal_cols = ['year', 'month', 'day_of_year', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'season']
print(f'Temporal features: {temporal_cols}')

# ============================================================
# 5. Combine All Features
# ============================================================
print("\n" + "=" * 60)
print("5. COMBINING ALL FEATURES")
print("=" * 60)

id_cols = ['Latitude', 'Longitude', 'Sample Date']
target_cols = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

landsat_feature_cols = [c for c in landsat_train_fe.columns if c not in id_cols]
terra_feature_cols = [c for c in terra_train_fe.columns if c not in id_cols]

# Build combined training
train_combined = wq_train_fe.copy()
for col in landsat_feature_cols:
    train_combined[col] = landsat_train_fe[col].values
for col in terra_feature_cols:
    train_combined[col] = terra_train_fe[col].values

# Build combined validation
val_combined = submission_fe.copy()
for col in landsat_feature_cols:
    val_combined[col] = landsat_val_fe[col].values
for col in terra_feature_cols:
    val_combined[col] = terra_val_fe[col].values

# Identify numeric feature columns
all_feature_cols = [c for c in train_combined.columns
                    if c not in id_cols + target_cols + ['Sample Date']
                    and pd.api.types.is_numeric_dtype(train_combined[c])]

print(f'Total features for PCA: {len(all_feature_cols)}')
print(f'\nAll feature columns:')
for i, c in enumerate(all_feature_cols, 1):
    print(f'  {i:2d}. {c}')

# Handle missing values and infinities
train_features = train_combined[all_feature_cols].copy()
train_features = train_features.replace([np.inf, -np.inf], np.nan)
medians = train_features.median()
train_features = train_features.fillna(medians)

val_features = val_combined[all_feature_cols].copy()
val_features = val_features.replace([np.inf, -np.inf], np.nan)
val_features = val_features.fillna(medians)  # Use training medians

print(f'\nTraining features: {train_features.shape}')
print(f'Validation features: {val_features.shape}')
print(f'Any NaN (train): {train_features.isna().any().any()}')
print(f'Any NaN (val):   {val_features.isna().any().any()}')

# ============================================================
# 6. PCA
# ============================================================
print("\n" + "=" * 60)
print("6. PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_val_scaled = scaler.transform(val_features)

# Fit full PCA
pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train_scaled)

print(f'PCA fitted on {X_train_scaled.shape[1]} features')
print(f'Number of components: {pca_full.n_components_}')

# Explained variance
print('\nExplained Variance:')
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
for i in range(min(15, len(cumvar))):
    print(f'  PC{i+1:2d}: {pca_full.explained_variance_ratio_[i]*100:6.2f}%  (cumulative: {cumvar[i]*100:6.2f}%)')

for threshold in [0.80, 0.90, 0.95, 0.99]:
    n = np.argmax(cumvar >= threshold) + 1
    print(f'\n{threshold*100:.0f}% variance explained by {n} components (out of {len(cumvar)})')

# Top loadings per component
print('\n\nTop feature loadings per component:')
n_show = min(5, pca_full.n_components_)
loadings = pd.DataFrame(
    pca_full.components_[:n_show].T,
    index=all_feature_cols,
    columns=[f'PC{i+1}' for i in range(n_show)]
)
for pc_name in loadings.columns:
    top = loadings[pc_name].abs().sort_values(ascending=False).head(5)
    pc_idx = int(pc_name[2:]) - 1
    print(f'\n  {pc_name} ({pca_full.explained_variance_ratio_[pc_idx]*100:.1f}% variance):')
    for feat, val in top.items():
        sign = '+' if loadings.loc[feat, pc_name] > 0 else '-'
        print(f'    {sign} {feat}: {loadings.loc[feat, pc_name]:.4f}')

# ============================================================
# 7. Export PCA Features (95% variance)
# ============================================================
print("\n" + "=" * 60)
print("7. EXPORTING PCA FEATURES")
print("=" * 60)

target_variance = 0.95
n_components_95 = np.argmax(cumvar >= target_variance) + 1
print(f'Using {n_components_95} components for 95% variance')

pca_reduced = PCA(n_components=n_components_95)
X_train_reduced = pca_reduced.fit_transform(X_train_scaled)
X_val_reduced = pca_reduced.transform(X_val_scaled)

pc_cols = [f'PC{i+1}' for i in range(n_components_95)]

train_pca_df = pd.DataFrame(X_train_reduced, columns=pc_cols)
train_pca_df['Latitude'] = wq_train['Latitude'].values
train_pca_df['Longitude'] = wq_train['Longitude'].values
train_pca_df['Sample Date'] = wq_train['Sample Date'].values
train_pca_df['Total Alkalinity'] = wq_train['Total Alkalinity'].values
train_pca_df['Electrical Conductance'] = wq_train['Electrical Conductance'].values
train_pca_df['Dissolved Reactive Phosphorus'] = wq_train['Dissolved Reactive Phosphorus'].values

val_pca_df = pd.DataFrame(X_val_reduced, columns=pc_cols)
val_pca_df['Latitude'] = submission['Latitude'].values
val_pca_df['Longitude'] = submission['Longitude'].values
val_pca_df['Sample Date'] = submission['Sample Date'].values

train_pca_df.to_csv('pca_features_training.csv', index=False)
val_pca_df.to_csv('pca_features_validation.csv', index=False)

print(f'Saved pca_features_training.csv: {train_pca_df.shape}')
print(f'Saved pca_features_validation.csv: {val_pca_df.shape}')

# ============================================================
# 8. Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f'Input features:')
print(f'  Landsat:      4 raw bands + {len(new_cols)} engineered = {len(landsat_feature_cols)}')
print(f'  TerraClimate: 14 raw vars + {len(terra_new)} derived = {len(terra_feature_cols)}')
print(f'  Temporal:     {len(temporal_cols)}')
print(f'  Total:        {len(all_feature_cols)}')
print(f'\nPCA output:')
print(f'  Components for 95% variance: {n_components_95}')
print(f'  Reduction: {len(all_feature_cols)} -> {n_components_95}')
print(f'\nPC columns: {pc_cols}')
