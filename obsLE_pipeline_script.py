import numpy as np
import pandas as pd
import xarray as xr

# Our functions
import obsLE.obsLE

### Climate Mode Parameters
mode_path = '/home/data/projects/NA_precip_extremes/climate_modes/'
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
lat_max = 60
lat_min = 22
lon_min = -126
lon_max = -65

gpcc = xr.open_mfdataset(gpcc_path)
gpcc = gpcc['precip']
gpcc = gpcc.sel(time=slice(start_year, end_year))

# pay attention to slicing order
gpcc_na = gpcc.sel(lat=slice(lat_max, lat_min),
                   lon=slice(lon_min, lon_max))
gpcc_na = gpcc_na.load()

obsLE.obsLE_pipeline(n_ens_members=10,
                     target_da=gpcc_na,
                     mode_path=mode_path,
                     start_year=start_year,
                     end_year=end_year,
                     mode_list=mode_list,
                     lambda_values=lambda_values,
                     offset_values=offset_values,
                     fit_seasonal=fit_seasonal,
                     block_size=24,
                     save_path='/home/data/projects/NA_precip_extremes/obsLE/pipeline/')
