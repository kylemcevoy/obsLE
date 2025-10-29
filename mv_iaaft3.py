import numpy as np

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

def mv_iaaft2(mode1, mode2, rng):
    mode2_sort = np.sort(mode2)
    
    fourier_mode1 = np.fft.rfft(mode1)
    mode1_phi = np.angle(fourier_mode1)
    
    mode1_star = iaaft(mode1, fit_seasonal=True, rng=rng)[0]
    fourier_mode1_star = np.fft.rfft(mode1_star)
    mode1_phi_star = np.angle(fourier_mode1_star)
    
    fourier_mode2 = np.fft.rfft(mode2)
    mode2_phi = np.angle(fourier_mode2)
    mode2_mod = np.abs(fourier_mode2)
    
    mode2_star = iaaft(mode2, fit_seasonal=False, rng=rng)[0]
    fourier_mode2_star = np.fft.rfft(mode2_star)
    mode2_phi_star = np.angle(fourier_mode2_star)
    
    Phi_mode12 = mode1_phi - mode2_phi
    Phi_star_mode12 = mode1_phi_star - mode2_phi_star
    
    mode2_phi_shifted = mode2_phi_star + Phi_star_mode12 - Phi_mode12
    
    mode2_s_iter = np.fft.irfft(mode2_mod * np.exp(1j * mode2_phi_shifted))
    mode2_sortindex = np.argsort(mode2_s_iter)
    
    mode2_r_iter = np.empty_like(mode2_sort)
    mode2_r_iter[mode2_sortindex] = mode2_sort
    
    fourier_mode2_star = np.fft.rfft(mode2_r_iter)
    psi_iter = np.angle(fourier_mode2_star)
    s_iter_final = np.fft.irfft(mode2_mod * np.exp(1j * psi_iter))
    
    return np.stack([mode1_star, s_iter_final], axis=0)
    
    
    
        
     
    
    
    