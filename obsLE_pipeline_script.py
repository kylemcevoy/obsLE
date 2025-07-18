# Template script for running the pipeline. The pipeline doesn't allow for the most
# customization, but the individual component functions can be run with slightly
# more freedom.
# The pipeline requires the obsLE package to be in the same directory as the pipeline
# script.
# Pipeline consists of:
# 1. Load Input Data
# 2. Process Input Data (orthogonalize and standardize climate modes, 
#    but not forcings)
# 3. Use MLE to optimize a Box-Cox transformation of the target variable, y.
# 4. Transform y using optimized transform.
# 5. Fit linear models to get coefficients.
# 6. Moving block bootstrap resample residuals.
# 7. Iterated Amplitude Adjusted Fourier Transform (IAAFT) resample climate modes.
# 8. Holding linear model coefficients fixed plug in new modes and residuals to get
#    new Obs-LE members.

import numpy as np
import regionmask # for subsetting to CONUS
import xarray as xr

# Our functions
import obsLE.gen_obsLE

### Output
save_path = '/home/data/projects/conus_precip_extremes/obsLE/pipeline/'

### Climate Mode Parameters
mode_path = '/home/data/projects/conus_precip_extremes/climate_modes/'
start_year = '1920'
end_year = '2020'
# setting (model_mode_list = None) uses all modes: ENSO, PDO, PNA, NAO, AO.
mode_list = ['enso', 'pdo', 'pna', 'nao', 'ao']
fit_seasonal = [mode != 'pdo' for mode in mode_list]
##### Optimization Grid
# Boxcox parameters for optimization
# lambda is the boxcox power and offset is the boxcox shift
# ((y + offset)**lambda - 1) / lambda
lambda_values = np.array([1/4, 1/3, 1/2, 2/3, 3/4, 1])
offset_values = np.array([1e-6])

### Load GPCC
gpcc_path = '/home/data/GPCC/monthly/*_10.nc'
##### lat/lon coord ranges
lat_max = 50
lat_min = 24.5
lon_min = -126
lon_max = -65

gpcc = xr.open_mfdataset(gpcc_path)
gpcc = gpcc['precip']
gpcc = gpcc.sel(time=slice(start_year, end_year))

# pay attention to slicing order
gpcc_na = gpcc.sel(lat=slice(lat_max, lat_min),
                   lon=slice(lon_min, lon_max))

countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
US_mask = countries.mask(gpcc_na.lon, gpcc_na.lat) == 4

gpcc_na = gpcc_na.where(US_mask)
gpcc_na = gpcc_na.load()

obsLE.gen_obsLE.obsLE_pipeline(n_ens_members=10,
                     y=gpcc_na,
                     mode_path=mode_path,
                     start_year=start_year,
                     end_year=end_year,
                     mode_list=mode_list,
                     lambda_values=lambda_values,
                     offset_values=offset_values,
                     fit_seasonal=fit_seasonal,
                     block_size=24,
                     save_path=save_path)
