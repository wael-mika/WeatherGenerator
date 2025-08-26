# #!/usr/bin/env -S uv run
# # /// script
# # dependencies = [
# #   "xarray>=2025.6.1",
# #   "dask>=2024.9.1",
# #   "zarr==2.18.4, <3",
# #   "numcodecs<0.16.0",
# #   "anemoi-datasets==0.5.24",
# #   "scipy==1.13.1",
# #   "xhistogram",
# # ]
# # [tool.uv]
# # ///

# import sys
# import time
# import argparse
# from pathlib import Path
# import logging
# import anemoi.datasets as anemoi_datasets
# import dask.array as da
# import numpy as np
# import xarray as xr
# #from scipy.stats import wasserstein_distance
# from xhistogram.xarray import histogram
# import logging

# # Configure logger
# LOG_LEVEL = logging.INFO  # Set desired log level
# fmt_log = "%(asctime)s - %(levelname)s - %(message)s"
# if LOG_LEVEL == logging.DEBUG:
#     fmt_log = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"

# logging.basicConfig(
#     level=LOG_LEVEL,  # Set logging level 
#     format=fmt_log,
#     handlers=[
#         logging.StreamHandler(sys.stdout)  # Log to stdout
#     ]
# )
# logger = logging.getLogger(__name__)

# def histogram_wasserstein(
#     x: xr.DataArray,
#     y: xr.DataArray,
#     bins: int = 100,
#     val_range: tuple = None,
#     dim: str = "flat_dim",
#     clip: bool = True,
#     sigma: float | None = None
# ) -> xr.DataArray:
#     """
#     Approximate the 1D Wasserstein distance between two distributions
#     using histogram-based empirical CDFs.

#     Parameters
#     ----------
#     x, y : xr.DataArray
#         Input data arrays (must be aligned and 1D or stacked to 1D).
#     bins : int
#         Number of histogram bins.
#     val_range : tuple of (float, float), optional
#         Min and max range for histogram bins. If None, inferred from data.
#     dim : str
#         Dimension along which to compute histograms.
#     clip: bool
#         Enable clipping for more effcient binning (ony active if val_range is None)
#     sigma: float
#         standard deviation (required for clipping)
#     Returns
#     -------
#     wasserstein : xr.DataArray
#         Scalar or array of Wasserstein distances.
#     """
#     logger = logging.getLogger(__name__)

#     # Infer bin range if not given
#     if val_range is None:
#         xmin, xmax = x.min().values, x.max().values
#         ymin, ymax = y.min().values, y.max().values
#         if clip:
#             assert sigma is not None, f"sigma must be parsed if clipping is active"
#             min_ = np.min([xmin, max([ymin, xmin - 4*sigma])])
#             max_ = np.max([xmax, min([ymax, xmax + 4*sigma])])

#             y = y.clip(min=min_, max=max_)
#         else:
#             min_ = np.min([xmin, ymin])
#             max_ = np.max([xmin, ymax])
#         val_range = (min_, max_)

#     # Create shared bins
#     bin_edges = np.linspace(val_range[0], val_range[1], bins + 1)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # Compute histograms
#     hist_x = histogram(x, bins=[bin_edges], dim=[dim], density=False,  block_size=None)
#     hist_y = histogram(y, bins=[bin_edges], dim=[dim], density=False,  block_size=None)

#     # Rename dimensions for consistency
#     hist_x = hist_x.rename({str(hist_x.dims[0]): "bins"})
#     hist_y = hist_y.rename({str(hist_y.dims[0]): "bins"})

#     if clip:
#         if hist_y.isel(bins=0).values > 100:
#             msg = f"Warning: y histogram has a very high value in the first bin ({hist_y.isel(bins=0).values})," + \
#                   "which may indicate clipping issues. Consider running without clipping or adjusting the sigma value."
#             logger.warning(msg)

#     assert np.all(hist_x["bins"] == hist_y["bins"]), "Histogram data of x and y must have the same bin edges."

#     # Normalize to empirical probability mass functions
#     hist_x = hist_x / hist_x.sum("bins")
#     hist_y = hist_y / hist_y.sum("bins")

#     # Compute empirical CDFs
#     cdf_x = hist_x.cumsum("bins")
#     cdf_y = hist_y.cumsum("bins")

#     # Compute Wasserstein-1 approximation
#     dx = np.diff(bin_edges)  # Bin widths
#     dx_da = xr.DataArray(dx, coords={"bins": hist_x.bins})

#     wasserstein = (np.abs(cdf_x - cdf_y) * dx_da).sum("bins")

#     return wasserstein


# def wasserstein_to_gaussian(x, clip: bool = False, sigma: float | None = None):
#     """
#     Compute 1D Wasserstein distance between empirical data and fitted Gaussian.
#     """
#     logger = logging.getLogger(__name__)

#     x = x.compute()#.values
#     x = x[~np.isnan(x)]
#     x.name = "real_sample"

#     n_ref_samples = len(x)

#     if clip:
#         assert sigma is not None, f"sigma must be parsed if clipping is active"

#     logger.info("Generating samples from Gauss distribution...")
#     #gaussian_sample = np.random.normal(loc=0., scale=1., size=n_ref_samples)
#     gaussian_sample = xr.DataArray(
#         np.random.normal(loc=0., scale=1., size=n_ref_samples),
#         dims=x.dims, coords=x.coords,
#         name="gaussian_sample"
#      )

#     # calculate Wasserstein distance
#     # Scipy's function is about 4x slower than the histogram method, so we use later.
#     #print("Start calculating Wasserstein distance...")
#     #start_time = time.time()
#     #wd1 = wasserstein_distance(x.values, gaussian_sample.values)
#     #elapsed_time = time.time() - start_time
#     #print(f"Wasserstein distance using scipy calculated in {elapsed_time:.5f}s.")

#     logger.info("Start calculating Wasserstein distance using histogram method...")
#     start_time = time.time()
#     wd = histogram_wasserstein(x, gaussian_sample, dim="flat_dim", bins = 1000, clip=False, sigma=sigma)
#     elapsed_time = time.time() - start_time
#     logger.info(f"Wasserstein distance using histogram method calculated in {elapsed_time:.2f}s.")

#     return wd


# def transform_and_score_wasserstein(x_raw: xr.DataArray, epsilons: list[float]):
#     """
#     Apply log transformation and standard normalization to the input data array,
#     then compute Wasserstein distance to a Gaussian distribution for each epsilon.
    
#     Parameters
#     ----------
#     x_raw : xr.DataArray
#         Input data array to be transformed.
#     epsilons : list of float
#         List of epsilon values for log transformation.
#     Returns
#     -------
#     results : list of tuples
#         Each tuple contains (epsilon, statistics, wasserstein_distance).
#         Statistics is a dict with keys 'mu' and 'sigma'.
#     """
#     logger = logging.getLogger(__name__)

#     results = []
#     for eps in epsilons:
#         logger.info(f"Perform calculations for epsilon: {eps:.1e}")
#         x_trans, stat = log_normalize(x_raw, eps)

#         # Calculate Wasserstein distance to Gaussian
#         wd = wasserstein_to_gaussian(x_trans)

#         results.append((eps, stat, wd))
#     return results


# def log_normalize(x: xr.DataArray, eps: float = 1e-6):
#     """
#     Apply log transformation and subsequent standard normalization to the input data array.

#     Parameters
#     ----------
#     x : xr.DataArray
#         Input data array to be transformed.
#     eps : float
#         Small value to avoid log(0) issues. Default: 1e-6.
#     Returns
#     -------
#     x_transformed : xr.DataArray
#         Log-transformed and standardized data array.
#     """
#     log_x = np.log((x + eps) / eps)

#     # Get Gaussian parameters
#     mu, sigma = get_gaussian_params(log_x)  

#     return (log_x - mu) / sigma, {"mu": mu, "sigma": sigma}


# def get_gaussian_params(x: xr.DataArray):
#     """
#     Calculate mean and standard deviation of the input data array.
    
#     Parameters
#     ----------
#     x : xr.DataArray
#         Input data array for which to compute statistics.
#     Returns
#     -------
#     mu : float
#         Mean of the data.
#     sigma : float
#         Standard deviation of the data.
#     """
#     logger = logging.getLogger(__name__)

#     stats = xr.concat([x.mean(), x.std(ddof=1)], dim="stat")
#     stats = stats.assign_coords(stat=["mean", "std"])

#     stats = stats.compute()
#     mu, sigma = stats.sel(stat="mean"), stats.sel(stat="std")

#     logger.info(f"Statistics calculated. mu: {mu.values:.5f}, sigma: {sigma.values:.5f}")

#     return mu, sigma

# def main(args):

#     logger = logging.getLogger(__name__)

#     zarr_imerg = Path(args.zarr_imerg)
#     if not zarr_imerg.exists():
#         raise FileNotFoundError(f"Zarr file {zarr_imerg} does not exist.")
    
#     chunk_nsamples = args.chunk_nsamples            # chunk size in lazy array
#     nsamples_comp = args.nsamples_comp              # samples over which to compute statistics
#     nsamples_start = args.nsamples_start            # samples to skip at the beginning

#     # Peek meta-information from the dataset
#     ds = anemoi_datasets.open_dataset(zarr_imerg)

#     lats, lons = ds.latitudes, ds.longitudes
#     dates = ds.dates
#     ntimes, npoints = len(ds), len(lats)

#     assert ntimes > nsamples_start + nsamples_comp, f"Not enough time samples in dataset ({ntimes}). nsamples_start must be less than {ntimes - nsamples_comp}."

#     # Read data (lazily)
#     imerg_precip = da.from_zarr(
#         zarr_imerg,
#         component="data/",
#         chunks=(chunk_nsamples, 1, 1, npoints),
#     ).squeeze()  # .flatten()


#     imerg_precip = xr.DataArray(
#         imerg_precip,
#         dims=["time", "ipoint"],
#         coords={
#             "time": dates.astype("datetime64[ns]"),
#             "ipoint": np.arange(npoints),
#             #                                      "lat": ("ipoint", lats),
#             #                                      "lon": ("ipoint", lons)}
#         },
#     )

#     # Slice data
#     imerg_precip = imerg_precip.isel({"time": range(nsamples_start, nsamples_start + nsamples_comp)})

#     imerg_1d = imerg_precip.data.reshape((-1,))
#     imerg_1d = xr.DataArray(imerg_1d, dims=["flat_dim"])

#     wd_results = transform_and_score_wasserstein(imerg_1d, args.epsilons)

#     # Print results
#     logger.info("-" * 40)
#     for eps, stat, score in wd_results:
#         logger.info(f"Results for epsilon: {eps:.1e}")
#         logger.info(f"Wasserstein distance: {score:.5f}")
#         logger.info(f"Mean: {stat['mu']:.5f}, Std: {stat['sigma']:.5f}")
#         logger.info("-" * 40)
#     # TO-DO: create histogram plots incl. info on Wasserstein distance and norm parameters


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compute Wasserstein distance for IMERG data.")
#     parser.add_argument("-z", "--zarr_imerg", type=str,
#                         default=Path("/p/scratch/hclimrep/shared/weather_generator_data/obs/imerg-nasa-grib-n320-1998-2024-6h-v1.zarr"), 
#                         help="Path to the Zarr file containing IMERG data.")
#     parser.add_argument("-c", "--chunk_nsamples", type=int, 
#                         default=100, help="Chunk size in samples for lazy loading.")
#     parser.add_argument("-n", "--nsamples_comp", type=int,
#                         default=200,
#                         help="Number of samples to compute statistics over.")
#     parser.add_argument("-s", "--nsamples_start", type=int, 
#                         default=0,
#                         help="Number of samples to skip at the beginning.")
#     parser.add_argument("-e", "--epsilons", nargs="+", type=float,
#                         default=[1.8e-07, 1.9e-07, 2e-07, 2.1e-07, 2.2e-07, 2.3e-07, 2.4e-07, 2.5e-07, 2.6e-07, 2.7e-07],
#                         help="List of epsilon values for log transformation.")

#     args = parser.parse_args()
#     main(args)

#####################################################################################################
# import sys
# import time
# import argparse
# import logging
# from pathlib import Path
# from functools import lru_cache

# import numpy as np
# import xarray as xr
# import dask.array as da
# from scipy.optimize import minimize_scalar
# from xhistogram.xarray import histogram

# import anemoi.datasets as anemoi_datasets

# # -----------------------------------------------------------------------------
# # Configure logger
# # -----------------------------------------------------------------------------
# LOG_LEVEL = logging.INFO
# fmt_log = "%(asctime)s - %(levelname)s - %(message)s"
# if LOG_LEVEL == logging.DEBUG:
#     fmt_log = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"

# logging.basicConfig(
#     level=LOG_LEVEL,
#     format=fmt_log,
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# logger = logging.getLogger(__name__)


# # -----------------------------------------------------------------------------
# # Core functions: histogram-based Wasserstein & log‐normalization
# # -----------------------------------------------------------------------------
# def histogram_wasserstein(
#     x: xr.DataArray,
#     y: xr.DataArray,
#     bins: int = 100,
#     val_range: tuple | None = None,
#     dim: str = "flat_dim",
#     clip: bool = True,
#     sigma: float | None = None,
# ) -> xr.DataArray:
#     """
#     Approximate the 1D Wasserstein distance between two distributions
#     using histogram-based empirical CDFs.
#     """
#     # infer range
#     if val_range is None:
#         xmin, xmax = x.min().values, x.max().values
#         ymin, ymax = y.min().values, y.max().values
#         if clip:
#             assert sigma is not None
#             min_ = np.min([xmin, max([ymin, xmin - 4*sigma])])
#             max_ = np.max([xmax, min([ymax, xmax + 4*sigma])])
#             y = y.clip(min=min_, max=max_)
#         else:
#             min_, max_ = np.min([xmin, ymin]), np.max([xmax, ymax])
#         val_range = (min_, max_)

#     bin_edges = np.linspace(val_range[0], val_range[1], bins + 1)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # compute histograms
#     hist_x = histogram(x, bins=[bin_edges], dim=[dim], density=False, block_size=None)
#     hist_y = histogram(y, bins=[bin_edges], dim=[dim], density=False, block_size=None)
#     hist_x = hist_x.rename({hist_x.dims[0]: "bins"})
#     hist_y = hist_y.rename({hist_y.dims[0]: "bins"})

#     # normalize to PMFs
#     hist_x = hist_x / hist_x.sum("bins")
#     hist_y = hist_y / hist_y.sum("bins")

#     # empirical CDFs
#     cdf_x = hist_x.cumsum("bins")
#     cdf_y = hist_y.cumsum("bins")

#     # bin widths
#     dx = np.diff(bin_edges)
#     dx_da = xr.DataArray(dx, coords={"bins": hist_x.bins})

#     # W₁ = ∫ |F₁ - F₂|
#     wass = (np.abs(cdf_x - cdf_y) * dx_da).sum("bins")
#     return wass


# def log_normalize(x: xr.DataArray, eps: float = 1e-6):
#     """
#     Apply log-transform x' = log((x + eps) / eps), then standardize to N(0,1).
#     """
#     log_x = np.log((x + eps) / eps)
#     mu = log_x.mean().compute()
#     sigma = log_x.std(ddof=1).compute()
#     x_std = (log_x - mu) / sigma
#     return x_std, {"mu": float(mu.values), "sigma": float(sigma.values)}


# # -----------------------------------------------------------------------------
# # Main: load data, build 1D sample, then optimize epsilon
# # -----------------------------------------------------------------------------
# def main(args):
#     # --- load dataset
#     zarr_imerg = Path(args.zarr_imerg)
#     if not zarr_imerg.exists():
#         raise FileNotFoundError(f"Zarr file {zarr_imerg} not found.")
#     ds = anemoi_datasets.open_dataset(zarr_imerg)
#     dates = ds.dates
#     npoints = ds.latitudes.size
#     ntimes = len(ds)
#     assert ntimes > args.nsamples_start + args.nsamples_comp

#     # lazy-load precipitation into 1D
#     arr = da.from_zarr(
#         zarr_imerg,
#         component="data/",
#         chunks=(args.chunk_nsamples, 1, 1, npoints),
#     ).squeeze()
#     imerg = xr.DataArray(
#         arr,
#         dims=["time", "ipoint"],
#         coords={"time": dates.astype("datetime64[ns]"), "ipoint": np.arange(npoints)},
#     )
#     imerg = imerg.isel(time=range(args.nsamples_start, args.nsamples_start + args.nsamples_comp))
#     imerg_1d = imerg.data.reshape((-1,))
#     imerg_1d = xr.DataArray(imerg_1d, dims=["flat_dim"], coords={"flat_dim": np.arange(imerg_1d.size)})

#     # --- prepare fixed Gaussian reference sample (for reproducibility)
#     n = imerg_1d.size
#     np.random.seed(0)
#     gaussian_ref = xr.DataArray(
#         np.random.normal(loc=0.0, scale=1.0, size=(n,)),
#         dims=imerg_1d.dims,
#         coords=imerg_1d.coords,
#         name="gaussian_ref"
#     )

#     # --- objective: W₁ distance between standardized log(x+ε) and N(0,1)
#     @lru_cache(maxsize=None)
#     def wd_for_eps(eps: float) -> tuple[float, dict]:
#         x_std, stats = log_normalize(imerg_1d, eps)
#         wd = histogram_wasserstein(x_std, gaussian_ref, dim="flat_dim", bins=1000, clip=False)
#         return float(wd.values), stats

#     def wd_scalar(eps: float) -> float:
#         return wd_for_eps(eps)[0]

#     # --- coarse grid search over ε ∈ [1e-8, 1e+2]
#     eps_grid = np.logspace(-8, 2, num=50)
#     wds = [wd_scalar(e) for e in eps_grid]
#     best_idx = int(np.argmin(wds))
#     eps0, wd0 = eps_grid[best_idx], wds[best_idx]
#     logger.info(f"Grid search best: ε = {eps0:.2e}, W₁ = {wd0:.5f}")

#     # --- fine tuning with bounded scalar minimization
#     lo = eps_grid[max(0, best_idx - 1)]
#     hi = eps_grid[min(len(eps_grid) - 1, best_idx + 1)]
#     res = minimize_scalar(
#         wd_scalar,
#         bounds=(lo, hi),
#         method="bounded",
#         options={"xatol": 1e-12}
#     )
#     eps_opt, wd_opt = res.x, res.fun
#     logger.info(f"Optimized ε ≈ {eps_opt:.3e}, W₁ ≈ {wd_opt:.5f}")

#     # --- report final Gaussian stats at optimal ε
#     _, stat_opt = wd_for_eps(eps_opt)
#     logger.info(f"At ε={eps_opt:.3e}: μ={stat_opt['mu']:.5f}, σ={stat_opt['sigma']:.5f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Find optimal ε for log-transform precipitation via W₁ distance.")
#     parser.add_argument("-z", "--zarr_imerg", type=str,
#                         help="Path to IMERG Zarr dataset.")
#     parser.add_argument("-c", "--chunk_nsamples", type=int, default=100,
#                         help="Dask chunk size (time dimension).")
#     parser.add_argument("-n", "--nsamples_comp", type=int, default=200,
#                         help="Number of time samples to analyze.")
#     parser.add_argument("-s", "--nsamples_start", type=int, default=0,
#                         help="Skip this many initial time samples.")
#     args = parser.parse_args()
#     main(args)


# """
# uv run src/weathergen/datasets/eps.py --zarr_imerg /p/scratch/hclimrep/shared/weather_generator_data/anemoi/imerg-nasa-grib-n320-1998-2024-6h-v1.zarr --chunk_nsamples 50 --nsamples_comp 1800 --nsamples_start 0
# """
#####################################################################################################

# import sys
# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import xarray as xr
# import dask.array as da
# from xhistogram.xarray import histogram
# import anemoi.datasets as anemoi_datasets

# LOG_LEVEL = logging.INFO
# fmt_log = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=LOG_LEVEL, format=fmt_log, handlers=[logging.StreamHandler(sys.stdout)])
# logger = logging.getLogger(__name__)

# def histogram_wasserstein(
#     x: xr.DataArray,
#     y: xr.DataArray,
#     bins: int = 1000,
#     val_range: tuple = None,
#     dim: str = "flat_dim"
# ) -> float:
#     """
#     Approximate the 1D Wasserstein distance between two distributions
#     using histogram-based empirical CDFs.
#     """
#     # Infer bin range if not given
#     if val_range is None:
#         xmin, xmax = float(x.min()), float(x.max())
#         ymin, ymax = float(y.min()), float(y.max())
#         min_ = min(xmin, ymin)
#         max_ = max(xmax, ymax)
#         val_range = (min_, max_)
#     bin_edges = np.linspace(val_range[0], val_range[1], bins + 1)
#     # Compute histograms
#     hist_x = histogram(x, bins=[bin_edges], dim=[dim], density=False, block_size=10_000_000)
#     hist_y = histogram(y, bins=[bin_edges], dim=[dim], density=False, block_size=10_000_000)
#     hist_x = hist_x.rename({hist_x.dims[0]: "bins"})
#     hist_y = hist_y.rename({hist_y.dims[0]: "bins"})
#     # Normalize
#     hist_x = hist_x / hist_x.sum("bins")
#     hist_y = hist_y / hist_y.sum("bins")
#     cdf_x = hist_x.cumsum("bins")
#     cdf_y = hist_y.cumsum("bins")
#     dx = np.diff(bin_edges)
#     dx_da = xr.DataArray(dx, coords={"bins": hist_x.bins})
#     wasserstein = (np.abs(cdf_x - cdf_y) * dx_da).sum("bins")
#     return float(wasserstein.values)  # Return float for convenience

# def log_normalize(x: np.ndarray, eps: float):
#     log_x = np.log((x + eps) / eps)
#     mu = np.mean(log_x)
#     sigma = np.std(log_x, ddof=1)
#     return (log_x - mu) / sigma, {"mu": mu, "sigma": sigma}

# def main(args):
#     zarr_imerg = Path(args.zarr_imerg)
#     if not zarr_imerg.exists():
#         raise FileNotFoundError(f"Zarr file {zarr_imerg} does not exist.")
#     ds = anemoi_datasets.open_dataset(zarr_imerg)
#     npoints = ds.latitudes.size
#     dates = ds.dates
#     # Read lazily, slice, and flatten
#     arr = da.from_zarr(
#         zarr_imerg,
#         component="data/",
#         chunks=(args.chunk_nsamples, 1, 1, npoints),
#     ).squeeze()
#     arr = arr[args.nsamples_start : args.nsamples_start + args.nsamples_comp, :]
#     arr = arr.reshape(-1)
#     arr = arr.compute()
#     arr = arr[~np.isnan(arr)]
#     # Generate Gaussian reference once, set name for xhistogram
#     np.random.seed(0)
#     gaussian_ref = np.random.normal(loc=0.0, scale=1.0, size=arr.shape)
#     yarr = xr.DataArray(gaussian_ref, dims=["flat_dim"], name="y")  # <-- name set
#     logger.info(f"Number of data points: {len(arr)}")
#     # Evaluate all requested epsilons
#     results = []
#     for eps in args.epsilons:
#         logger.info(f"Calculating for epsilon={eps:.2e}")
#         x_std, stats = log_normalize(arr, eps)
#         x_std_arr = xr.DataArray(x_std, dims=["flat_dim"], name="x")  # <-- name set
#         wd = histogram_wasserstein(x_std_arr, yarr, bins=1000, dim="flat_dim")
#         logger.info(f"  W₁ = {wd:.5f}   mu={stats['mu']:.5f}  sigma={stats['sigma']:.5f}")
#         results.append((eps, stats, wd))
#     logger.info("Done.")
#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compute Wasserstein distance for IMERG data.")
#     parser.add_argument("-z", "--zarr_imerg", type=str,
#         default="/p/scratch/hclimrep/shared/weather_generator_data/anemoi/imerg-nasa-grib-n320-1998-2024-6h-v1.zarr")
#     parser.add_argument("-c", "--chunk_nsamples", type=int, default=50)
#     parser.add_argument("-n", "--nsamples_comp", type=int, default=1800)
#     parser.add_argument("-s", "--nsamples_start", type=int, default=0)
#     parser.add_argument("-e", "--epsilons", nargs="+", type=float, default=[1.5e-7, 2.0e-7, 2.25e-7, 2.39e-7, 2.5e-7, 2.75e-7, 3e-7, 3.5e-7, 4e-7])
#     args = parser.parse_args()
#     main(args)
# """
# uv run src/weathergen/datasets/eps.py 
# """
#############################################################################################################
#!/usr/bin/env -S uv run
# /// script
dependencies = [
    "xarray>=2025.6.1",
    "dask>=2024.9.1",
    "zarr==2.18.4, <3",
    "numcodecs<0.16.0",
    "anemoi-datasets==0.5.24",
    "scipy==1.13.1",
    "xhistogram",
]
# [tool.uv]
# ///

import argparse
import numpy as np
import xarray as xr
import logging
import warnings
from scipy.stats import boxcox, anderson, gamma
from scipy.stats._distn_infrastructure import FitError

# Configure warnings to ignore specific non-critical messages
warnings.filterwarnings(
    "ignore",
    message="Failed to open Zarr store with consolidated metadata"
)
warnings.filterwarnings(
    "ignore",
    message="The specified chunks separate the stored chunks along dimension"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Degrees of freedom <= 0 for slice"
)

# Configure logger
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Log Transformation
def log_transform(x: np.ndarray, eps: float) -> np.ndarray:
    return np.log((x + eps) / eps)

# Box-Cox Transformation
def boxcox_transform(x: np.ndarray, lmbda: float | None = None) -> tuple[np.ndarray, float]:
    x_positive = x + np.abs(np.nanmin(x)) + 1e-6
    return boxcox(x_positive, lmbda=lmbda)

# Wasserstein Distance using NumPy
def wasserstein_distance(x: np.ndarray, y: np.ndarray, bins: int = 1000) -> float:
    bin_edges = np.linspace(
        min(x.min(), y.min()), max(x.max(), y.max()), bins + 1
    )
    hist_x, _ = np.histogram(x, bins=bin_edges)
    hist_y, _ = np.histogram(y, bins=bin_edges)

    cdf_x = np.cumsum(hist_x) / np.sum(hist_x)
    cdf_y = np.cumsum(hist_y) / np.sum(hist_y)

    dx = np.diff(bin_edges)
    return float(np.sum(np.abs(cdf_x - cdf_y) * dx))

# Anderson-Darling Test with error handling
def anderson_darling_test(x: np.ndarray, dist: str = "norm") -> float:
    try:
        result = anderson(x, dist=dist)
        return float(result.statistic)
    except Exception:
        return np.nan

# Main analysis
def main(args: argparse.Namespace):
    # Open Zarr dataset without consolidated metadata and rechunk
    ds = xr.open_zarr(
        args.zarr_imerg,
        consolidated=False
    )
    ds = ds.chunk({"time": args.chunk_nsamples})

    # Select the first data variable
    var_name = list(ds.data_vars)[0]
    arr = ds[var_name]

    # Flatten data and compute numpy array
    flat = arr.data.reshape(-1)
    x = flat.compute()
    x = x[~np.isnan(x)]

    # Loop over epsilon values
    for eps in args.epsilons:
        # Log transform
        x_log = log_transform(x, eps)

        # Box-Cox transform
        try:
            x_boxcox, fitted_lmbda = boxcox_transform(x)
        except Exception:
            logger.warning(
                "Box-Cox transform failed for eps=%0.1e: data may be constant.", eps
            )
            x_boxcox, fitted_lmbda = x_log, np.nan

        # Reference: Gaussian
        gauss = np.random.normal(0, 1, size=len(x))

        # Reference: Gamma fit
        try:
            gamma_params = gamma.fit(x)
            gamma_sample = gamma.rvs(*gamma_params, size=len(x))
        except FitError:
            logger.warning("Gamma fitting failed for raw data: skipping Gamma metrics.")
            gamma_sample = None

        # Compute distances
        wd_log_gauss = wasserstein_distance(x_log, gauss)
        ad_log_gauss = anderson_darling_test(x_log, "norm")

        if gamma_sample is not None:
            wd_log_gamma = wasserstein_distance(x_log, gamma_sample)
            ad_log_gamma = anderson_darling_test(x_log, "gamma")
        else:
            wd_log_gamma = ad_log_gamma = np.nan

        wd_boxcox_gauss = wasserstein_distance(x_boxcox, gauss)
        ad_boxcox_gauss = anderson_darling_test(x_boxcox, "norm")

        if gamma_sample is not None:
            wd_boxcox_gamma = wasserstein_distance(x_boxcox, gamma_sample)
            ad_boxcox_gamma = anderson_darling_test(x_boxcox, "gamma")
        else:
            wd_boxcox_gamma = ad_boxcox_gamma = np.nan

        # Log results
        logger.info(f"Epsilon={eps:.1e}, Box-Cox λ={fitted_lmbda}")
        logger.info(f"WD(Log→Gauss)={wd_log_gauss:.4f}, AD(Log→Gauss)={ad_log_gauss:.4f}")
        logger.info(f"WD(Log→Gamma)={wd_log_gamma:.4f}, AD(Log→Gamma)={ad_log_gamma:.4f}")
        logger.info(f"WD(BoxCox→Gauss)={wd_boxcox_gauss:.4f}, AD(BoxCox→Gauss)={ad_boxcox_gauss:.4f}")
        logger.info(
            f"WD(BoxCox→Gamma)={wd_boxcox_gamma:.4f}, AD(BoxCox→Gamma)={ad_boxcox_gamma:.4f}"
        )
        logger.info("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimal ε exploration: comparing Log and Box-Cox normalization"
    )
    parser.add_argument(
        "-z", "--zarr_imerg", type=str, required=True,
        help="Path to IMERG Zarr dataset"
    )
    parser.add_argument(
        "-c", "--chunk_nsamples", type=int, default=5000,
        help="Number of time samples per chunk"
    )
    parser.add_argument(
        "-e", "--epsilons", nargs="+", type=float,
        default=[1e-6, 1e-5, 1e-4, 1e-3],
        help="List of epsilon values for log transform"
    )
    args = parser.parse_args()
    main(args)