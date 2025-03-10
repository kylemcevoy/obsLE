import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.path as mpath

####### Loading climate modes

def process_climate_modes(path_prefix, start_year='1920', end_year='2020', save_path=None):
    """Process the climate mode files into a single data object
    containing monthly climate mode values.

    Parameters
    ----------
    path_prefix : int
        path to the directory containing the climate mode data
    start_year : str
        four digit string for the first year of data to include
        in the climate mode output. Value should be >= 1920
    end_year : str
        four digit string for last year of data to include
        in the climate mode output. The year should have all 12 months
        of data. Value should be <= 2020.
    save_path : str
        Path to the directory for saving the output. The string should end in / after
        the directory name.
        If None, nothing will be saved. The pd.DataFrame output is converted to an
        xr.DataSet before saving, and is saved at save_path + "climate_mode.nc".

    Returns
    ---------
    mode_df : pd.DataFrame
        DataFrame with columns: [enso, pdo, pna, nao, ao]
        Rows contain monthly values for the modes from January of start_year
        to December of end_year. 
        
        
    """
    
    var_fnames = {'enso': 'nino34.long.anom.nc',
                  'pdo': 'pdo.timeseries.hadisst1-1.nc',
                  'pna_noaa': 'norm.mon.pna.wg.jan1950-current.ascii.table',
                  'pna_20cr': 'pna.20crv2c.long.data',
                  'nao': 'nao_3dp.dat',
                  'ao_noaa': 'monthly.ao.index.b50.current.ascii.table',
                  'ao_20cr': 'ao.20cr.long.data'}

    
    enso_ds = xr.open_dataset(path_prefix + var_fnames['enso'])
    enso_pd = (enso_ds.
                 to_dataframe().
                 rename(columns={'value': 'enso'}).
                 astype(np.float64))
    enso_pd = enso_pd['enso']

    pdo = xr.open_dataset(path_prefix + var_fnames['pdo'])
    pdo_pd = (pdo.
              astype(np.float64).
              to_dataframe())
    pdo_pd = pdo_pd['pdo']

    #noaa data time range: 1950-2023
    pna_noaa = pd.read_csv(path_prefix + var_fnames['pna_noaa'],
                           sep='\\s+',
                           header=None)
    pna_noaa = pna_noaa.rename(columns={0: 'year'})
    pna_noaa = pna_noaa.set_index('year')
    
    #20CR has data from 1920 to 1950
    pna_20cr = pd.read_csv(path_prefix + var_fnames['pna_20cr'],
                           engine='python',
                           skiprows=1,
                           skipfooter=5,
                           sep='\\s+',
                           header=None)
    pna_20cr = pna_20cr.rename(columns={0: 'year'})
    pna_20cr = pna_20cr.set_index('year').loc[slice('1920', '1949')]
    
    pna_df = pd.concat([pna_20cr, pna_noaa])
    pna_df = pna_df.reset_index()
    
    pna_melt = pd.melt(pna_df,
                       id_vars='year',
                       var_name='month',
                       value_name='pna')
    pna_melt['day'] = 1
    pna_melt['time'] = pd.to_datetime(pna_melt[['year', 'month', 'day']])
    pna_melt = pna_melt.set_index('time').sort_index()
    pna_pd = pna_melt['pna']

    
    nao = pd.read_csv(path_prefix + var_fnames['nao'],
                      engine='python',
                      sep='\\s+', 
                      header=None, 
                      na_values='-99.990',
                      skiprows=4,
                      skipfooter=1)
    nao = nao.rename(columns={0: 'year'})
    nao = nao.drop(columns=13)
    
    nao_pd = pd.melt(nao,
                     id_vars='year',
                     var_name='month',
                     value_name='nao')
    
    nao_pd['day'] = 1
    nao_pd['time'] = pd.to_datetime(nao_pd[['year', 'month', 'day']])
    nao_pd = nao_pd.set_index('time').sort_index()['nao']

    
    ao_noaa = pd.read_csv(path_prefix + var_fnames['ao_noaa'],
                     sep='\\s+',
                     names=['year'] + [*range(1, 13)],
                     skiprows=1)
    
    ao_20cr = pd.read_csv(path_prefix + var_fnames['ao_20cr'],
                          engine='python',
                          sep='\\s+',
                          skiprows=1,
                          skipfooter=5,
                          header=None)
    ao_20cr = ao_20cr.rename(columns={0: 'year'})
    ao_20cr = (ao_20cr.
              set_index('year').
              loc[:1949].
              reset_index())
    
    ao = pd.concat([ao_20cr, ao_noaa])
    
    ao_melt = pd.melt(ao,
                      id_vars='year',
                      var_name='month',
                      value_name='ao')
    
    ao_melt['day'] = 1
    ao_melt['time'] = pd.to_datetime(ao_melt[['year', 'month', 'day']])
    ao_melt = ao_melt.set_index('time').sort_index()
    ao_pd = ao_melt['ao']

    
    mode_df = pd.DataFrame({'enso': enso_pd,
                            'pdo': pdo_pd,
                            'pna': pna_pd,
                            'nao': nao_pd,
                            'ao': ao_pd})
    
    mode_df = mode_df.loc[slice(start_year, end_year)]

    if save_path is not None:
        mode_ds = mode_df.to_xarray()
        description = ('dataset containing time series of climate modes for the regression model. '
                       'Created by the process_climate_modes() function in precip_functions.py')
        mode_ds.attrs['description'] = description
        mode_ds.to_netcdf(save_path + 'climate_modes.nc')

    return mode_df

###### For sequentially orthogonalizing the climate mode variables. ######
def gram_schmidt(mode, ortho_basis=None):
    """Perform a gram-schmidt orthogonalization on the input vector with respect to the given ortho basis

    Parameters
    ----------
    mode: 1D numpy array
        Vector of floats containing a time series of input data.
    ortho_basis: 2D numpy array
        The columns of this numpy array contain the orthogonal basis vectors against which orthogonalize the mode.
        ortho_basis.shape[0] should equal mode.shape[0].

    Returns
    -------
    ortho_mode : numpy array
        1D numpy array containing the orthogonalized mode
        
    
    """
    if ortho_basis is None:
        ortho_mode = mode
    else:
        inner_products = mode @ ortho_basis
        norm_sq = np.linalg.norm(ortho_basis, axis=0)**2
        proj_weights = inner_products / norm_sq

        ortho_mode = mode - np.sum(proj_weights * ortho_basis, axis=1)

    return ortho_mode     

def find_ortho_basis(mode_array):
    """Sequentially orthogonalize the columns of the input array with respect to all preceding columns.

    Parameters
    ----------
    mode_array: 2D numpy array of floats

    Returns
    -------
    ortho_basis: 2D numpy array
        Array of same dimension as input whose columns are now orthogonalized
        against each other. The orthogonalization is performed iteratively over
        the columns in order by applying the gram-schmidt procedure. 
        See gram_schmidt() for more details.

    """
    ortho_basis = mode_array[:, [0]]
    if mode_array.ndim > 1:
        for j in range(1, mode_array.shape[1]):
            ortho_mode = gram_schmidt(mode_array[:, j], ortho_basis)[:, np.newaxis]
            ortho_basis = np.hstack([ortho_basis, ortho_mode])
    return ortho_basis

def orthogonalize_modes(mode_df, model_mode_list):
    """Sequentially orthogonalize a DataFrame of climate modes
        using Gram-Schmidt orthogonalization,

    Parameters
    ----------
    mode_df : pd.DataFrame
        DataFrame containing monthly observations of climate modes
        with a pd.DateTime index. As outputted by process_climate_modes().
    model_mode_list : list of strings
        list of variable names from mode_df to sequentially orthogonalize

    Returns
    -------
    ortho_mode_df : pd.DataFrame
        ortho_mode_df will have the same number of rows as mode_df, it will
        have 1 + len(model_mode_list) number of columns with an intercept 
        (all ones) column as the first column. Each column in ortho_mode_df
        is orthogonal to all preceding columns (including the intercept).
    
    """
    X = mode_df[model_mode_list].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    X_orth = find_ortho_basis(X)
    
    X_orth_std = X_orth[:, 1:] / np.std(X_orth[:, 1:], axis=0, ddof=1)
    X_orth_std = np.hstack([X_orth[:, [0]], X_orth_std])

    
    ortho_modes_df = pd.DataFrame(X_orth_std,
                                  index=mode_df.index,
                                  columns=['intercept'] + model_mode_list)

    return ortho_modes_df

####### Fit monthly regressions. 

def fit_linear_models(Y, X):
    """
    Fit OLS models of each individual column (locations) of Y regressed against the design matrix X. The models are fit separately
    for each month by subsetting the rows of Y and X. A total of 12 * #{locations} independent regression models are fit.

    Parameters
    ----------
    Y: 2D numpy array
        Matrix containing n observation rows against l columns containing time series of output data. 
        l is an index for the locations at which the time series are recorded.
    X: 2D numpy array
        Design matrix containing n observation rows against p + 1 variable columns. The columns of X should contain time series
        of monthly data containing only complete years, so that n is divisible by 12. X should contain an intercept column.

    Returns
    -------
    tuple(beta, RSS, residuals, fitted_values)

    beta: numpy array of shape (12, p + 1, l)
        Contains the fitted regression coefficients indexed by (month, variable, location). For each regression model
        fit using Y[I{month_j}] ~ X[I{month_j}], where I{month_j} is an indicator for the rows coming from month j.

    RSS: numpy array of shape (12, l)
        Contains the residual sums of squares of each fitted model
        
    residuals: numpy array of shape (12, n // 12, l)
        Contains the residuals of each fitted model split by month.

    fitted_values: numpy array of shape (12, n // 12, l)
        Contains the fitted values of each regression model split by month.
        
    """
    n = X.shape[0]
    p = X.shape[1]
    
    if Y.ndim > 1:
        l = Y.shape[1]
        
        beta = np.zeros((12, p, l))
        residuals = np.zeros((12, n // 12, l))
        fitted_values = np.zeros((12, n // 12, l))
        RSS = np.zeros((12, l))
    else:
        beta = np.zeros((12, p))
        residuals = np.zeros((12, n // 12))
        fitted_values = np.zeros((12, n // 12))
        RSS = np.zeros((12))
        
    I_month = [(np.arange(n) % 12) == i for i in np.arange(12)]

    for i in range(12):
        beta[i, ...], RSS[i], *_ = np.linalg.lstsq(X[I_month[i]],
                                                    Y[I_month[i]],
                                                    rcond=None)
        fitted_values[i] = (X[I_month[i]] @ beta[i, ...])
        residuals[i] = Y[I_month[i]] - (X[I_month[i]] @ beta[i, ...])

        
    return beta, RSS, residuals, fitted_values

###### Building xarray output objects for betas fit by fit_linear_models

def build_xr(beta_array, nan_mask, var_names, RSS, TSS, lat_coord, lon_coord):
    p = beta_array.shape[0]
    lat_len = lat_coord.shape[0]
    lon_len = lon_coord.shape[0]

    Rsq = 1 - (RSS / TSS)
    
    beta_xr_data = np.empty((p, 12, lat_len, lon_len))
    beta_xr_data.fill(np.nan)
    beta_xr_data[:, :, nan_mask] = beta_array

    Rsq_data = np.empty((12, lat_len, lon_len))
    Rsq_data.fill(np.nan)
    Rsq_data[:, nan_mask] = Rsq

    beta_ds = xr.Dataset(coords={'month': np.arange(1, 13),
                             'lat': lat_coord,
                             'lon': lon_coord})

    for i, var in enumerate(var_names):
        beta_ds[var] = (('month', 'lat', 'lon'),
                        beta_xr_data[i])

    beta_ds['R_squared'] = (('month', 'lat', 'lon'),
                            Rsq_data)
    beta_ds['residual_var'] = 1 - beta_ds['R_squared']
    
    return beta_ds

####### Plotting functions, using ouptut of build_xr as input.

def plot_var(ds,
             var_name,
             month_list,
             cmap,
             proj,
             levels,
             title,
             polar=False,
             robust=True,
             cbar_label='',
             save_fig=False,
             path_prefix='',
             plot_name=None):
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection=proj)
    
    if len(month_list) > 1: 
        (ds[var_name].
                sel(month=month_list).
                mean('month').
                plot(ax=ax,
                     cmap=cmap,
                     levels=levels,
                     robust=robust,
                     transform=ccrs.PlateCarree(),
                     cbar_kwargs={'shrink': 0.7,
                                 'label': cbar_label}))
    else:
        (ds[var_name].
         sel(month=month_list).
         plot(ax=ax,
              cmap=cmap,
              levels = levels,
              robust=robust,
              transform=ccrs.PlateCarree(),
              cbar_kwargs={'shrink': 0.7,
                          'label': cbar_label}))
    ax.coastlines()
    ax.set_title(title)
    
    if polar:
        ax.gridlines(draw_labels=True)
        # Code copied from Pangeo workshop notebook:
        #https://pangeo-data.github.io/escience-2022/examples/notebooks/xesmf_regridding.html

        
        ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax.set_boundary(circle, transform=ax.transAxes)
    
    if save_fig:
        if plot_name is None:
            raise ValueError('input path_prefix and plot_name for saving.')
        else:
            file_path = path_prefix + plot_name + '.png'
            fig.savefig(fname=file_path)

    plt.close(fig)
    return None

def plot_monthly_vars(ds,
                      var_name,
                      cmap,
                      levels,
                      title,
                      proj,
                      polar=False,
                      robust=True,
                      cbar_label='',
                      save_fig=False,
                      path_prefix='',
                      plot_name=None):

    import calendar
    
    for i in range(1, 13):
        month_abrv = calendar.month_abbr[i].lower()
        plot_var(ds,
                 var_name=var_name,
                 month_list=[i],
                 cmap=cmap,
                 proj=proj,
                 levels=levels,
                 polar=polar,
                 title=month_abrv.title() + title,
                 cbar_label=cbar_label,
                 save_fig=save_fig, 
                 path_prefix=path_prefix,
                 plot_name=plot_name + '_' + month_abrv)
    return None

####### Boxcox Transforms and Inverse Transforms.
def boxcox_transform_np(x, offset, lam):
    """Perform boxcox transform on numpy arrays,
    this function is primarily used in the optimization
    of (lambda, offset) where a single value of lambda
    and offset are used to transform an entire numpy array.
    When (lambda, offset) vary over time, use boxcox_transform().


    Parameters
    ----------
    x: numpy ndarray
        x contains data to transform
    offset: numeric scalar
        offset is for constant shifts to x. If x contains 0s, offset
        should be positive.
    lam: non-negative numeric scalar
        boxcox lambda parameter

    Returns
    -------
    y: numpy ndarray
        y is the Box-Cox transform of x with power parameter lam
        and shift parameter offset. This is given by the formula
        y = np.log(x + offset) if lam == 0 and
        y = ((x + offset)^lam - 1) / lam if lam > 0
    """
    if lam == 0:
        y = np.log(x + offset)
    else:
        y = ((x + offset)**lam - 1) / lam
    return y
    
def boxcox_transform(x, offset, lam_da):
    # copy the xarray object, so original xarray is not affected by assignments
    y = x.copy()
    
    if np.isscalar(offset):
        da_flag = False
        y = y + offset
    elif isinstance(offset, xr.DataArray):
        da_flag = True
    else:
        raise ValueError('offset should be a single scalar offset value' +
                         'or a xr.DataArray containing offsets for each location + month')

    #lam was optimized for each month, so perform transform looping over the months 
    for i in range(1, 13):
        month_slice = y.loc[dict(time = (y['time.month'] == i))]
        
        if da_flag:
            month_slice = month_slice + offset.loc[dict(month = i)]
            
        lam_slice = lam_da.loc[dict(month = i)]

        # the use of fmax here is to suppress divide by 0 warnings at locations where
        # lambda == 0, so the log transform will be used instead anyway
        month_transform = xr.where(lam_slice == 0,
                               np.log(month_slice), 
                               (month_slice**lam_slice - 1) / np.fmax(lam_slice, 1e-6))
    
        y.loc[dict(time=(y['time.month'] == i))] = month_transform
        
    return y

def inv_boxcox_transform(y, offset, lam_da):
    # copy the xarray object, so original xarray is not affected by assignments
    x = y.copy()
    
    if np.isscalar(offset):
        da_flag = False
    elif isinstance(offset, xr.DataArray):
        da_flag = True
    else:
        raise ValueError('offset should be a single scalar offset value' +
                         'or a xr.DataArray containing offsets for each location + month')

    #lam was optimized for each month, so perform transform looping over the months 
    for i in range(1, 13):
        month_slice = x.loc[dict(time = (x['time.month'] == i))]

        lam_slice = lam_da.loc[dict(month = i)]

        # the use of fmin below is to prevent overflow warnings on locations where a non-log
        # boxcox transform was used resulting in still quite large value. In these cases,
        # the non-log boxcox-transform will have been applied instead anyway.
        month_transform = xr.where(lam_slice == 0,
                               np.exp(np.fmin(month_slice, 500)), 
                               (month_slice * lam_slice + 1)**(1 / lam_slice))

        if da_flag:
            month_transform = month_transform - offset.loc[dict(month = i)]
        else:
            month_transform = month_transform - offset

        x.loc[dict(time=(x['time.month'] == i))] = month_transform

    return x

###### Optimization Helpers
def np_to_da(data_np, var_name, coord_dict, nan_mask):
    dim_len = []
    for val in coord_dict.values():
        dim_len.append(val.shape[0])
    da_data = np.empty(dim_len)
    da_data.fill(np.nan)
    da_data[..., nan_mask] = data_np
    da = xr.DataArray(data=da_data, coords=coord_dict, name=var_name)
    return da

def create_surrogate_modes(mode_path,
                           mode_list,
                           fit_seasonal,
                           n_ens_members,
                           max_iters=np.inf,
                           rng=None,
                           seed=None,
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
    seed : int
        Integer for using in np.random.default_rng(seed). Only use if rng=None.
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
        if seed is not None:
            rng = np.random.defaul_rng(seed)
        else:
            rng = np.random.default_rng()
    
    # Load original modes
    mode_df = xr.open_dataset(mode_path)
    mode_df = mode_df.to_pandas()
    ortho_mode_df = pr_func.orthogonalize_modes(mode_df, mode_list)

    mode_subset = ortho_mode_df[mode_list].values

    n_time = ortho_mode_df.shape[0]
    n_modes = len(mode_list)
    time_indx = ortho_mode_df.index

    modes_out = np.empty((n_ens_members, n_modes, n_time))

    for k in range(n_ens_members):
        for j in range(n_modes):
            iter_out = 0
            total_iters = 0
            while iter_out == 0:
                surrogate_ts, iter_out = pr_func.iaaft(mode_subset[:, j],
                                                  rng=rng,
                                                  fit_seasonal=fit_seasonal[j])
                total_iters += 1
                if total_iters > max_iters:
                    raise RuntimeError('The IAAFT while loop exceeded max_iters.')
                
            modes_out[k, j] = surrogate_ts

    if save_path is not None:
        dim_tuple = ('member', 'time')
        # xarray Dataset constructor uses a tuple of (dimensions, ndarray)
        # to construct variables
        surrogate_list = [(dim_tuple, modes_out[:, i]) for i in range(n_modes)]
        var_dict = dict(zip(mode_list, surr_list))
        
        surrogate_ds = xr.Dataset(data_vars=var_dict,
                             coords={'member': np.arange(n_ens_members),
                                     'time': time_indx})
        surrogate_ds.to_netcdf(save_path + 'surrogate_modes.nc')

    return modes_out, time_indx


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
        Surrogate time series
    this_iter : int
        Number of iterations until convergence
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
    iter_out = 0
    while delta_criterion > 1e-8:
        criterion_old = criterion_new
        # iteration 1: spectral adjustment
        x_old = x_new
        x_fourier = np.fft.fft(x_old)
        adjusted_coeff = I_k * (x_fourier / np.abs(x_fourier))
        x_new = np.fft.ifft(adjusted_coeff)

        # iteration 2: amplitude adjustment
        x_old = x_new
        index = np.argsort(np.real(x_old))
        x_new[index] = x_sort
        x_new = np.real(x_new)

        # Rescale the seasonal standard deviations to match original data
        if fit_seasonal:
            this_sigma = np.array([np.std(x_new[mo::12]) for mo in range(12)])
            scaling = seasonal_sigma / this_sigma

            for mo in range(12):
                x_new[mo::12] = scaling[mo] * x_new[mo::12]

        criterion_new = (1 / np.std(x)) * np.sqrt((1 / len(x)) * np.sum((I_k - np.abs(x_fourier))**2))
        delta_criterion = np.abs(criterion_new - criterion_old)

        if this_iter > max_iters:
            return np.array(np.nan), 0

        iter_out += 1

    x_new = x_new + xbar
    x_new = np.real(x_new)

    return x_new, iter_out










