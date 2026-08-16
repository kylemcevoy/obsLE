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
import pathlib
import xarray as xr

# Our functions (this line works if obsLE directory is contained in same directory as
# the pipeline script)
from obsLE import gen_obsLE

rng = np.random.default_rng(693974)

proj_dir = '/home/data/projects/conus_precip_extremes/'

### Output directory -- (end with trailing slash)
save_dir = proj_dir + 'synthLE/test/'

### Climate Mode Parameters
start_year = '1920'
end_year = '2020'
# mode_list contains modes that are generated using univariate iaaft
# mv_mode_list contains modes that are generated using multivariate iaaft
mode_list = ['nao']
mv_mode_list = ['enso', 'pdo', 'pna']
# whether to match monthly standard deviations of generated modes to observed 
# values
fit_seasonal = [True]
mv_fit_seasonal = [True, False, True]
##### Optimization Grid
# Boxcox parameters for optimization
# lambda is the boxcox power and offset is the boxcox shift
# ((y + offset)**lambda - 1) / lambda
lambda_values = np.array([1/4, 1/3, 1/2, 2/3, 3/4, 1])
# Offset is only necessary if using the log transform.
offset = 1e-6

### Load Forcings (if using)
# forcing_dir = proj_dir + 'forcings/'
# forcing_ds = xr.open_dataset(forcing_dir + 'forcings.nc')

# forcing_df = forcing_ds.to_pandas()

### CESM2 modes
cvdp_dir = '/home/data/projects/conus_precip_extremes/cvdp/'
cvdp_modes = xr.open_dataset(cvdp_dir + 'cesm2_lens_cvdp_modes.nc')

### Load CESM2
cesm2_path = proj_dir + 'cesm2/cesm2_PRECT_processed.nc'
cesm2 = xr.open_dataarray(cesm2_path)
cesm2 = cesm2.rename('precip')

for mem in np.arange(2):
    print(f'member number: {mem}')
    
    save_dir = save_dir + f'mem{mem:02}/'
    pathlib.Path(save_dir).mkdir(exist_ok=True)
    
    cesm2_mem = (cesm2.sel(ens_mem=mem)
                 .squeeze(drop=True))
    
    cvdp_mode_mem = cvdp_modes.sel(ens_mem=mem)
    cvdp_mode_mem = cvdp_mode_mem.drop_vars('ens_mem')
    
    cvdp_mode_df = cvdp_mode_mem.to_dataframe()
    
    gen_obsLE.obsLE_pipeline(n_ens_members=1000,
                            y=cesm2_mem,
                            mode_df=cvdp_mode_df,
                            forcing_df=None,
                            mode_list=mode_list,
                            mv_mode_list=mv_mode_list,
                            lam=lambda_values,
                            offset=offset,
                            fit_seasonal=fit_seasonal,
                            mv_fit_seasonal=mv_fit_seasonal,
                            transform=True,
                            block_size=24,
                            save_dir=save_dir,
                            rng=rng)
