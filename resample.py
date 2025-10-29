# Functions for bootstrap resampling residuals and creating surrogate climate modes
# using IAAFT.
import numpy as np
import xarray as xr

#package defined functions
from . import process_data as data_proc

def iaaft(x, fit_seasonal=False, rng=None):
    """Return a surrogate time series based on IAAFT.

    Parameters
    ----------
    x : 1D numpy array
        Original time series 
    fit_seasonal : bool
        Should the monthly amplitudes be matched? Use True for ENSO.
    rng : Numpy Random Generator as produced by np.random.default_rng().
        If None is passed, the function will create a new generator.
        For details see:
        https://numpy.org/doc/stable/reference/random/generator.html.

    Returns
    -------
    x_new : 1D numpy array
        Surrogate time series. Contains a single np.nan if the algorith did not
        converge.
    iter_count : int
        Number of iterations until convergence was reached. 0 if algorithm did not
        converge.
    """

    # Initialize a new rng if none is passed. For reproducible code pass a 
    # generator instance with a set seed to the function.
    if rng is None:
        rng = np.random.default_rng()
        
    # To maintain the seasonality in ENSO variance, the seasonal cycle in 
    # ENSO variance is stored for rescaling the output surrogate time series
    if fit_seasonal:
        nyrs = x.shape[0] // 12
        x_months = x[:(nyrs*12)].reshape((nyrs, 12))
        # possible to do this with resampling:
        # idx = np.random.choice(np.arange(nyrs), nyrs, replace=True)
        # x_months = x_months[idx, :]
        seasonal_sigma = np.std(x_months, ddof=1, axis=0)
        
    xbar = np.mean(x)
    x = x - xbar  # remove mean
    rank = np.argsort(x)
    x_sort = x[rank]

    I_k = np.abs(np.fft.fft(x))
    x_new = rng.permutation(x)

    delta_criterion = 1
    criterion_new = 100
    max_iters = 1000
    iter_count = 0
    while delta_criterion > 1e-8:
        criterion_old = criterion_new
        # iteration 1: spectral adjustment
        x_old = x_new
        x_fourier = np.fft.fft(x_old)
        
        I_k_new = np.abs(x_fourier)
        # The time series is set to have mean 0, sometimes this results
        # in the 0th frequency of the fft, which is the sum of the time series
        # to be set to 0 rather than some extremely small value. This causes
        # np.nan issues when dividing.
        I_k_new[I_k_new == 0] = I_k_new[I_k_new == 0] + 1e-12
        
        adjusted_coeff = I_k * (x_fourier / I_k_new)
        x_new = np.fft.ifft(adjusted_coeff)

        # iteration 2: amplitude adjustment
        x_old = x_new
        index = np.argsort(np.real(x_old))
        x_new[index] = x_sort
        x_new = np.real(x_new)

        # Rescale the seasonal standard deviations to match original data
        if fit_seasonal:
            this_sigma = np.array([np.std(x_new[mo::12], ddof=1) for mo in range(12)])
            scaling = seasonal_sigma / this_sigma

            for mo in range(12):
                x_new[mo::12] = scaling[mo] * x_new[mo::12]

        criterion_new = (1 / np.std(x)) * np.sqrt((1 / len(x)) * np.sum((I_k - np.abs(x_fourier))**2))
        delta_criterion = np.abs(criterion_new - criterion_old)

        if iter_count > max_iters:
            return np.array(np.nan), 0

        iter_count += 1

    x_new = x_new + xbar
    x_new = np.real(x_new)

    return x_new, iter_count

def create_surrogate_modes(ortho_mode_df,
                           fit_seasonal,
                           n_ens_members,
                           mode_path=None,
                           mode_list=None,
                           max_iters=np.inf,
                           rng=None,
                           save_path=None):
    """Create modes that have been resampled through Iterative
    Adjusted Amplitude Fourier Transform (IAAFT). The modes
    will have the same Fourier amplitudes as the original time
    series with nearly the same power spectra.

    Parameters
    ----------
    mode_path : str
        Path to the netCDF file where the climate modes are saved. See
        process_climate_modes() for details.
    mode_list: list of strings
        list of variable names for which to create surrogate modes.
    fit_seasonal: list of bools
        Must match the length of mode_list. True/False values give
        whether to scale the resulting time series to match the monthly
        cycle of standard deviations in the observed modes.  
    n_ens_members : int
        Number of mode sets to create
    max_iters: numeric or np.inf
        gives a maximum number of loops to perform before aborting IAAFT sampling.
    rng : Numpy random Generator
        Like np.random.default_rng()
    save_path : str
        Path to the directory for saving the modes. Nothing is saved if set to None.
        

    Returns
    -------
    modes_out : numpy.ndarray
        Array (n_ens_members x n_modes x n_time) of surrogate time series
    mode_list : list
        Returns the input list of variable names
    time_indx : pd.DateTimeIndex
        The time index for the modes.
    """

    if rng is None:
        rng = np.random.default_rng()

    if (ortho_mode_df is None) and (mode_path is None):
        raise ValueError('One of ortho_mode_df or mode_path must be supplied.')
    
    if ortho_mode_df is None:
        # Load original modes
        mode_df = xr.open_dataset(mode_path)
        mode_df = mode_df.to_pandas()

        ortho_mode_df = data_proc.orthogonalize_modes(mode_df, mode_list)
        data_proc.check_mode(mode_df)
    
    data_proc.check_ortho_mode(ortho_mode_df)
    
    if mode_list is not None:
        mode_subset = ortho_mode_df[mode_list].values
    else:
        mode_subset = ortho_mode_df.drop('intercept', axis=1)
        mode_list = mode_subset.columns.to_list()
        mode_subset = mode_subset.values
        
    n_time = mode_subset.shape[0]
    n_modes = mode_subset.shape[1]
    time_indx = ortho_mode_df.index

    modes_out = np.empty((n_ens_members, n_modes, n_time))

    for k in range(n_ens_members):
        for j in range(n_modes):
            iter_out = 0
            total_iters = 0
            while iter_out == 0:
                surrogate_ts, iter_out = iaaft(mode_subset[:, j],
                                               rng=rng,
                                               fit_seasonal=fit_seasonal[j])
                total_iters += 1
                if total_iters > max_iters:
                    raise RuntimeError('The IAAFT while loop exceeded max_iters.')
                
            modes_out[k, j] = surrogate_ts

    dim_tuple = ('ens_member', 'time')
    # xarray Dataset constructor uses a tuple of (dimensions, ndarray)
    # to construct variables
    surrogate_list = [(dim_tuple, modes_out[:, i]) for i in range(n_modes)]
    var_dict = dict(zip(mode_list, surrogate_list))
    
    surrogate_ds = xr.Dataset(data_vars=var_dict,
                              coords={'ens_member': np.arange(n_ens_members),
                                      'time': time_indx})

    if save_path is not None:
        surrogate_ds.to_netcdf(save_path + 'surrogate_modes.nc')

    return surrogate_ds

def bootstrap_residuals(residuals_da, block_size, rng=None):
    """Perform moving block bootstrapping of the residuals. The blocks contain the 
    entire spatial field and a number of whole years of data, specified by 
    block_size.

    Parameters
    ----------
    residuals_da: xr.DataArray, dims (time, lat, lon)
        DataArray containing residuals from the linear models fit by 
        fit_model.fit_linear_models(). See fit_model.build_model_ds()

    block_size: int
        Number of consecutive months to include in the blocks for the moving block
        bootstrap. Must be divisible by 12 so that full years are included in the 
        moving block bootstrap. The entire spatial field is kept in each block.

    rng: np.random.Generator
        As constructed by np.random.default_rng().

    Returns
    -------
    boot_residuals: xr.DataArray, dims: (time, lat, lon)
        DataArray containing the moving block bootstrapped residuals. The coordinates
        are equivalent to the coordinates of residuals_da. 
    """
    if block_size % 12 != 0:
        raise ValueError('block_size should contain whole years to preserve ' 
        'seasonality')

    if rng is None:
        rng = np.random.default_rng()
        
    n_time = residuals_da.shape[0]
    num_starts = np.ceil(n_time / block_size).astype('int')

    # create block starting points for each year
    block_starts = np.arange(n_time, step=12)

    rand_blocks = rng.choice(block_starts, size=num_starts, replace=True)

    bootstrap_indx = np.hstack([np.arange(s, s + block_size) for s in rand_blocks])
    
    # loop indices to 0 for circular bootstrap
    bootstrap_indx = bootstrap_indx % n_time
    # keep the size of original time field
    bootstrap_indx = bootstrap_indx[:n_time]

    boot_residuals = residuals_da.isel(time=bootstrap_indx)
    boot_residuals = boot_residuals.assign_coords({'time': residuals_da.time})

    return boot_residuals


