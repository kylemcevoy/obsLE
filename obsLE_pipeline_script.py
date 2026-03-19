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
# import regionmask # for subsetting to CONUS uncomment if loading GPCC in script
import xarray as xr

# Our functions (this line works if obsLE directory is contained in same directory as
# the pipeline script)
from obsLE import gen_obsLE

# result of 7 10-sided dice
rng = np.random.default_rng(4743105)

### Output directory -- (end with trailing slash)
proj_dir = '/home/data/projects/conus_precip_extremes/'
save_dir = proj_dir + 'obsLE/gpcc_cvdp/'
mode_path = proj_dir + 'climate_modes/'

#using pre-processed gpcc otherwise comment this out and 
# uncomment the Load GPCC code
gpcc_path = proj_dir + 'gpcc/gpcc_mmday.nc'
gpcc_mmday = xr.open_dataarray(gpcc_path)   

### Climate Modes
mode_df = xr.open_dataset(mode_path + 'cvdp_obs_modes.nc')
mode_df = mode_df.to_dataframe()
start_year = '1920'
end_year = '2020'
# setting (model_mode_list = None) uses all modes: ENSO, PDO, PNA, NAO.
mode_list = ['nao']
# the modes that are calculated using multivariate iaaft
mv_mode_list = ['enso', 'pdo', 'pna']
fit_seasonal = [True]
mv_fit_seasonal = [True, False, True]

### Optimization Grid
# Boxcox parameters for optimization
# lambda is the boxcox power and offset is the boxcox shift
# ((y + offset)**lambda - 1) / lambda
lambda_values = np.array([1/4, 1/3, 1/2, 2/3, 3/4, 1])
# Offset is necessary so that Jacobian of the Box-Cox transform is well-defined.
offset = 1e-6

### Load Forcings (if using)
# forcing_dir = '/home/data/projects/conus_precip_extremes/forcings/'
# forcing_ds = xr.open_dataset(forcing_dir + 'forcings.nc')

# forcing_df = forcing_ds.to_pandas()

# ### Load GPCC
# gpcc_path = '/home/data/GPCC/monthly/*_10.nc'
# ##### lat/lon coord ranges
# lat_max = 50
# lat_min = 24.5
# lon_min = -126
# lon_max = -65

# gpcc = xr.open_mfdataset(gpcc_path)
# gpcc = gpcc['precip']
# gpcc = gpcc.sel(time=slice(start_year, end_year))

# # pay attention to slicing order
# gpcc_na = gpcc.sel(lat=slice(lat_max, lat_min),
#                    lon=slice(lon_min, lon_max))

# countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
# US_mask = countries.mask(gpcc_na.lon, gpcc_na.lat) == 4

# gpcc_na = gpcc_na.where(US_mask)
# gpcc_na = gpcc_na.compute().astype('float64')

# days_in_month = gpcc_na.time.dt.days_in_month
# gpcc_mmday = gpcc_na / days_in_month
# gpcc_mmday = gpcc_mmday.rename('precip')



gen_obsLE.obsLE_pipeline(n_ens_members=1000,
                         rng=rng,
                         y=gpcc_mmday,
                         mode_df=mode_df,
                         forcing_df=None,
                         mode_list=mode_list,
                         mv_mode_list=mv_mode_list,
                         lam=lambda_values,
                         offset=offset,
                         fit_seasonal=fit_seasonal,
                         mv_fit_seasonal=mv_fit_seasonal,
                         transform=True,
                         block_size=24,
                         save_resid=False,
                         save_dir=save_dir)
