import datetime
import glob
import logging
import os
import re
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf as oc
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import wilcoxon

from weathergen.common.config import _load_private_conf
from weathergen.evaluate.plotting.plot_utils import (
    DefaultMarkerSize,
)
from weathergen.evaluate.utils.regions import RegionBoundingBox

work_dir = Path(_load_private_conf(None)["path_shared_working_dir"]) / "assets/cartopy"

cartopy.config["data_dir"] = str(work_dir)
cartopy.config["pre_existing_data_dir"] = str(work_dir)
os.environ["CARTOPY_DATA_DIR"] = str(work_dir)

np.seterr(divide="ignore", invalid="ignore")

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_logger.debug(f"Taking cartopy paths from {work_dir}")


class Plotter:
    """
    Contains all basic plotting functions.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path, stream: str | None = None):
        """
        Initialize the Plotter class.

        Parameters
        ----------
        plotter_cfg:
            Configuration dictionary containing basic information for plotting.
            Expected keys are:
                - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
                - dpi_val: DPI value for the saved images
                - fig_size: Size of the figure (width, height) in inches
                - tokenize_spacetime: If True, all valid times will be plotted in one plot
        output_basedir:
            Base directory under which the plots will be saved.
            Expected scheme `<results_base_dir>/<run_id>`.
        stream:
            Stream identifier for which the plots will be created.
            It can also be set later via update_data_selection.
        """

        _logger.info(f"Taking cartopy paths from {work_dir}")

        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.fps = plotter_cfg.get("fps")
        self.regions = plotter_cfg.get("regions")
        self.plot_subtimesteps = plotter_cfg.get(
            "plot_subtimesteps", False
        )  # True if plots are created for each valid time separately
        self.run_id = output_basedir.name

        self.out_plot_basedir = Path(output_basedir) / "plots"

        if not os.path.exists(self.out_plot_basedir):
            _logger.info(f"Creating dir {self.out_plot_basedir}")
            os.makedirs(self.out_plot_basedir, exist_ok=True)

        self.sample = None
        self.stream = stream
        self.fstep = None
        self.select = {}

    def update_data_selection(self, select: dict):
        """
        Set the selection for the plots. This will be used to filter the data for plotting.

        Parameters
        ----------
        select:
            Dictionary containing the selection criteria. Expected keys are:
                - "sample": Sample identifier
                - "stream": Stream identifier
                - "forecast_step": Forecast step identifier
        """
        self.select = select

        if "sample" not in select:
            _logger.warning("No sample in the selection. Might lead to unexpected results.")
        else:
            self.sample = select["sample"]

        if "stream" not in select:
            _logger.warning("No stream in the selection. Might lead to unexpected results.")
        else:
            self.stream = select["stream"]

        if "forecast_step" not in select:
            _logger.warning("No forecast_step in the selection. Might lead to unexpected results.")
        else:
            self.fstep = select["forecast_step"]

        return self

    def clean_data_selection(self):
        """
        Clean the data selection by resetting all selected values.
        """
        self.sample = None
        self.stream = None
        self.fstep = None

        self.select = {}
        return self

    def select_from_da(self, da: xr.DataArray, selection: dict) -> xr.DataArray:
        """
        Select data from an xarray DataArray based on given selectors.

        Parameters
        ----------
        da:
            xarray DataArray to select data from.
        selection:
            Dictionary of selectors where keys are coordinate names and values are the values to
            select.

        Returns
        -------
            xarray DataArray with selected data.
        """
        for key, value in selection.items():
            if key in da.coords and key not in da.dims:
                # Coordinate like 'sample' aligned to another dim
                da = da.where(da[key] == value, drop=True)
            else:
                # Scalar coord or dim coord (e.g., 'forecast_step', 'channel')
                da = da.sel({key: value})
        return da

    def create_histograms_per_sample(
        self,
        target: xr.DataArray,
        preds: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
    ) -> list[str]:
        """
        Plot histogram of target vs predictions for each variable and valid time in the DataArray.

        Parameters
        ----------
        target: xr.DataArray
            Target sample for a specific (stream, sample, fstep)
        preds: xr.DataArray
            Predictions sample for a specific (stream, sample, fstep)
        variables: list
            List of variables to be plotted
        select: dict
            Selection to be applied to the DataArray
        tag: str
            Any tag you want to add to the plot

        Returns
        -------
            List of plot names for the saved histograms.
        """
        plot_names = []

        self.update_data_selection(select)

        # Basic map output directory for this stream
        hist_output_dir = self.out_plot_basedir / self.stream / "histograms"

        if not os.path.exists(hist_output_dir):
            _logger.info(f"Creating dir {hist_output_dir}")
            os.makedirs(hist_output_dir)

        for var in variables:
            select_var = self.select | {"channel": var}

            targ, prd = (
                self.select_from_da(target, select_var),
                self.select_from_da(preds, select_var),
            )

            # Remove NaNs
            targ = targ.dropna(dim="ipoint")
            prd = prd.dropna(dim="ipoint")
            assert targ.size > 0, "Data array must not be empty or contain only NAs"
            assert prd.size > 0, "Data array must not be empty or contain only NAs"

            if self.plot_subtimesteps:
                ntimes_unique = len(np.unique(targ.valid_time))
                _logger.info(
                    f"Creating histograms for {ntimes_unique} valid times of variable {var}."
                )

                groups = zip(targ.groupby("valid_time"), prd.groupby("valid_time"), strict=False)
            else:
                _logger.info(f"Plotting histogram for all valid times of {var}")

                groups = [((None, targ), (None, prd))]  # wrap once with dummy valid_time

            for (valid_time, targ_t), (_, prd_t) in groups:
                if valid_time is not None:
                    _logger.debug(f"Plotting histogram for {var} at valid_time {valid_time}")
                name = self.plot_histogram(targ_t, prd_t, hist_output_dir, var, tag=tag)
                plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def plot_histogram(
        self,
        target_data: xr.DataArray,
        pred_data: xr.DataArray,
        hist_output_dir: Path,
        varname: str,
        tag: str = "",
    ) -> str:
        """
        Plot a histogram comparing target and prediction data for a specific variable.

        Parameters
        ----------
        target_data: xr.DataArray
            DataArray containing the target data for the variable.
        pred_data: xr.DataArray
            DataArray containing the prediction data for the variable.
        hist_output_dir: Path
            Directory where the histogram will be saved.
        varname: str
            Name of the variable to be plotted.
        tag: str
            Any tag you want to add to the plot.

        Returns
        -------
            Name of the saved plot file.
        """

        # Get common bin edges
        vals = np.concatenate([target_data, pred_data])
        bins = np.histogram_bin_edges(vals, bins=50)

        # Plot histograms
        plt.hist(target_data, bins=bins, alpha=0.7, label="Target")
        plt.hist(pred_data, bins=bins, alpha=0.7, label="Prediction")

        # set labels and title
        plt.xlabel(f"Variable: {varname}")
        plt.ylabel("Frequency")
        plt.title(
            f"Histogram of Target and Prediction: {self.stream}, {varname} : "
            f"fstep = {self.fstep:03}"
        )
        plt.legend(frameon=False)

        valid_time = (
            target_data["valid_time"][0]
            .values.astype("datetime64[m]")
            .astype(datetime.datetime)
            .strftime("%Y-%m-%dT%H%M")
        )

        # TODO: make this nicer
        parts = [
            "histogram",
            self.run_id,
            tag,
            str(self.sample),
            valid_time,
            self.stream,
            varname,
            str(self.fstep).zfill(3),
        ]
        name = "_".join(filter(None, parts))

        fname = hist_output_dir / f"{name}.{self.image_format}"
        _logger.debug(f"Saving histogram to {fname}")
        plt.savefig(fname)
        plt.close()

        return name

    def create_maps_per_sample(
        self,
        data: xr.DataArray,
        variables: list,
        select: dict,
        tag: str = "",
        map_kwargs: dict | None = None,
    ) -> list[str]:
        """
        Plot 2D map for each variable and valid time in the DataArray.

        Parameters
        ----------
        data: xr.DataArray
            DataArray for a specific (stream, sample, fstep)
        variables: list
            List of variables to be plotted
        label: str
            Any tag you want to add to the plot
        select: dict
            Selection to be applied to the DataArray
        tag: str
            Any tag you want to add to the plot. Note: This is added to the plot directory.
        map_kwargs: dict
            Additional keyword arguments for the map.
            Known keys are:
                - marker_size: base size of the marker (default is 1)
                - scale_marker_size: if True, the marker size will be scaled based on latitude
                  (default is False)
                - marker: marker style (default is 'o')
            Unknown keys will be passed to the scatter plot function.

        Returns
        -------
            List of plot names for the saved maps.
        """
        self.update_data_selection(select)

        # copy global plotting options, not specific to any variable
        map_kwargs_global = {
            key: value
            for key, value in (map_kwargs or {}).items()
            if not isinstance(value, oc.DictConfig)
        }

        # Basic map output directory for this stream
        map_output_dir = self.get_map_output_dir(tag)

        if not os.path.exists(map_output_dir):
            _logger.info(f"Creating dir {map_output_dir}")
            os.makedirs(map_output_dir)

        for region in self.regions:
            if region != "global":
                bbox = RegionBoundingBox.from_region_name(region)
                reg_data = bbox.apply_mask(data)
            else:
                reg_data = data

            plot_names = []
            for var in variables:
                select_var = self.select | {"channel": var}
                da = self.select_from_da(reg_data, select_var).compute()

                if self.plot_subtimesteps:
                    ntimes_unique = len(np.unique(da.valid_time))
                    _logger.info(
                        f"Creating maps for {ntimes_unique} valid times of variable {var} - {tag}"
                    )
                    if ntimes_unique == 0:
                        _logger.warning(
                            f"No valid times found for variable {var} - {tag}. Skipping."
                        )
                        continue
                    groups = da.groupby("valid_time")
                else:
                    _logger.info(f"Creating maps for all valid times of {var} - {tag}")
                    groups = [(None, da)]  # single dummy group

                for valid_time, da_t in groups:
                    if valid_time is not None:
                        _logger.debug(f"Plotting map for {var} at valid_time {valid_time}")

                    da_t = da_t.dropna(dim="ipoint")
                    assert da_t.size > 0, "Data array must not be empty or contain only NAs"

                    name = self.scatter_plot(
                        da_t,
                        map_output_dir,
                        var,
                        region,
                        tag=tag,
                        map_kwargs=dict(map_kwargs.get(var, {})) | map_kwargs_global,
                        title=f"{self.stream}, {var} : fstep = {self.fstep:03} ({valid_time})",
                    )
                    plot_names.append(name)

        self.clean_data_selection()

        return plot_names

    def scatter_plot(
        self,
        data: xr.DataArray,
        map_output_dir: Path,
        varname: str,
        regionname: str | None,
        tag: str = "",
        map_kwargs: dict | None = None,
        title: str | None = None,
    ):
        """
        Plot a 2D map for a data array using scatter plot.

        Parameters
        ----------
        data: xr.DataArray
            DataArray to be plotted
        map_output_dir: Path
            Directory where the map will be saved
        varname: str
            Name of the variable to be plotted
        regionname: str
            Name of the region to be plotted
        tag: str
            Any tag you want to add to the plot
        map_kwargs: dict | None
            Additional keyword arguments for the map.
        title: str | None
            Title for the plot.

        Returns
        -------
            Name of the saved plot file.
        """
        # check for known keys in map_kwargs
        map_kwargs_save = map_kwargs.copy() if map_kwargs is not None else {}
        marker_size_base = map_kwargs_save.pop(
            "marker_size", DefaultMarkerSize.get_marker_size(self.stream)
        )
        scale_marker_size = map_kwargs_save.pop("scale_marker_size", False)
        marker = map_kwargs_save.pop("marker", "o")
        vmin = map_kwargs_save.pop("vmin", None)
        vmax = map_kwargs_save.pop("vmax", None)
        cmap = plt.get_cmap(map_kwargs_save.pop("colormap", "coolwarm"))

        if isinstance(map_kwargs_save.get("levels", False), oc.listconfig.ListConfig):
            norm = mpl.colors.BoundaryNorm(
                map_kwargs_save.pop("levels", None), cmap.N, extend="both"
            )
        else:
            norm = mpl.colors.Normalize(
                vmin=vmin,
                vmax=vmax,
                clip=False,
            )

        # scale marker size
        marker_size = marker_size_base
        if scale_marker_size:
            marker_size = np.clip(
                marker_size / np.cos(np.radians(data["lat"])) ** 2,
                a_max=marker_size * 10.0,
                a_min=marker_size,
            )

        # Create figure and axis objects
        fig = plt.figure(dpi=self.dpi_val)

        proj = ccrs.PlateCarree()
        if regionname == "global":
            proj = ccrs.Robinson()

        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.coastlines()

        assert data["lon"].shape == data["lat"].shape == data.shape, (
            f"Scatter plot:: Data shape do not match. Shapes: "
            f"lon {data['lon'].shape}, lat {data['lat'].shape}, data {data.shape}."
        )

        scatter_plt = ax.scatter(
            data["lon"],
            data["lat"],
            c=data,
            norm=norm,
            cmap=cmap,
            s=marker_size,
            marker=marker,
            transform=ccrs.PlateCarree(),
            linewidths=0.0,  # only markers, avoids aliasing for very small markers
            **map_kwargs_save,
        )

        plt.colorbar(scatter_plt, ax=ax, orientation="horizontal", label=f"Variable: {varname}")
        plt.title(title)
        if regionname == "global":
            ax.set_global()
        else:
            region_extent = [
                data["lon"].min().item(),
                data["lon"].max().item(),
                data["lat"].min().item(),
                data["lat"].max().item(),
            ]
            ax.set_extent(region_extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)

        # TODO: make this nicer
        parts = ["map", self.run_id, tag]

        if self.sample is not None:
            parts.append(str(self.sample))

        if "valid_time" in data.coords:
            valid_time = data["valid_time"][0].values
            if ~np.isnat(valid_time):
                valid_time = (
                    valid_time.astype("datetime64[m]")
                    .astype(datetime.datetime)
                    .strftime("%Y-%m-%dT%H%M")
                )

                parts.append(valid_time)

        if self.stream:
            parts.append(self.stream)

        parts.append(regionname)
        parts.append(varname)

        if self.fstep is not None:
            parts.extend(["fstep", f"{self.fstep:03d}"])

        name = "_".join(filter(None, parts))
        fname = f"{map_output_dir.joinpath(name)}.{self.image_format}"

        _logger.debug(f"Saving map to {fname}")
        plt.savefig(fname)
        plt.close()

        return name

    def animation(self, samples, fsteps, variables, select, tag) -> list[str]:
        """
        Plot 2D animations for a dataset

        Parameters
        ----------
        samples: list
            List of the samples to be plotted
        fsteps: list
            List of the forecast steps to be plotted
        variables: list
            List of variables to be plotted
        select: dict
            Selection to be applied to the DataArray
        tag: str
            Any tag you want to add to the plot

        Returns
        -------
            List of plot names for the saved animations.

        """

        self.update_data_selection(select)
        map_output_dir = self.get_map_output_dir(tag)

        # Convert FPS to duration in milliseconds
        duration_ms = int(1000 / self.fps) if self.fps > 0 else 400

        for region in self.regions:
            for _, sa in enumerate(samples):
                for _, var in enumerate(variables):
                    _logger.info(f"Creating animation for {var} sample: {sa} - {tag}")
                    image_paths = []
                    for _, fstep in enumerate(fsteps):
                        # TODO: refactor to avoid code duplication with scatter_plot
                        parts = [
                            "map",
                            self.run_id,
                            tag,
                            str(sa),
                            "*",
                            self.stream,
                            region,
                            var,
                            "fstep",
                            str(fstep).zfill(3),
                        ]

                        name = "_".join(filter(None, parts))
                        fname = f"{map_output_dir.joinpath(name)}.{self.image_format}"

                        names = glob.glob(fname)
                        image_paths += names

                    if image_paths:
                        images = [Image.open(path) for path in image_paths]
                        images[0].save(
                            f"{map_output_dir}/animation_{self.run_id}_{tag}_{sa}_{self.stream}_{region}_{var}.gif",
                            save_all=True,
                            append_images=images[1:],
                            duration=duration_ms,
                            loop=0,
                        )

                    else:
                        _logger.warning(f"No images found for animation {var} sample {sa}")

        return image_paths

    def get_map_output_dir(self, tag):
        return self.out_plot_basedir / self.stream / "maps" / tag


class LinePlots:
    def __init__(self, plotter_cfg: dict, output_basedir: str | Path):
        """
        Initialize the LinePlots class.

        Parameters
        ----------
        plotter_cfg:
            Configuration dictionary containing basic information for plotting.
            Expected keys are:
                - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
                - dpi_val: DPI value for the saved images
                - fig_size: Size of the figure (width, height) in inches
                -  plot_ensemble:
                    If True, plot ensemble spread if 'ens' dimension is present. Options are:
                        - False: do not plot ensemble spread
                        - "std": plot mean +/- standard deviation
                        - "minmax": plot min-max range
                        - "members": plot individual ensemble members
        output_basedir:
            Base directory under which the plots will be saved.
            Expected scheme `<results_base_dir>/<run_id>`.
        """

        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.fig_size = plotter_cfg.get("fig_size")
        self.log_scale = plotter_cfg.get("log_scale")
        self.add_grid = plotter_cfg.get("add_grid")
        self.plot_ensemble = plotter_cfg.get("plot_ensemble", False)
        self.baseline = plotter_cfg.get("baseline")
        self.out_plot_dir = Path(output_basedir) / "line_plots"
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

        _logger.info(f"Saving summary plots to: {self.out_plot_dir}")

    def _check_lengths(self, data: xr.DataArray | list, labels: str | list) -> tuple[list, list]:
        """
        Check if the lengths of data and labels match.

        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset

        Returns
        -------
            data_list, label_list - lists of data and labels
        """
        assert isinstance(data, xr.DataArray | list), (
            "Compare::plot - Data should be of type xr.DataArray or list"
        )
        assert isinstance(labels, str | list), (
            "Compare::plot - Labels should be of type str or list"
        )

        # convert to lists

        data_list = [data] if isinstance(data, xr.DataArray) else data
        label_list = [labels] if isinstance(labels, str) else labels

        assert len(data_list) == len(label_list), "Compare::plot - Data and Labels do not match"

        return data_list, label_list

    def print_all_points_from_graph(self, fig: plt.Figure) -> None:
        for ax in fig.get_axes():
            for line in ax.get_lines():
                ydata = line.get_ydata()
                xdata = line.get_xdata()
                label = line.get_label()
                _logger.info(f"Summary for {label} plot:")
                for xi, yi in zip(xdata, ydata, strict=False):
                    xi = xi if isinstance(xi, str) else f"{float(xi):.3f}"
                    yi = yi if isinstance(yi, str) else f"{float(yi):.3f}"
                    _logger.info(f"  x: {xi}, y: {yi}")
                _logger.info("--------------------------")
        return

    def _plot_ensemble(self, data: xr.DataArray, x_dim: str, label: str) -> None:
        """
        Plot ensemble spread for a data array.

        Parameters
        ----------
        data: xr.xArray
            DataArray to be plotted
        x_dim: str
            Dimension to be used for the x-axis.
        label: str
            Label for the dataset
        Returns
        -------
            None
        """
        averaged = data.mean(dim=[dim for dim in data.dims if dim != x_dim], skipna=True).sortby(
            x_dim
        )

        lines = plt.plot(
            averaged[x_dim],
            averaged.values,
            label=label,
            marker="o",
            linestyle="-",
        )
        line = lines[0]
        color = line.get_color()

        ens = data.mean(
            dim=[dim for dim in data.dims if dim not in [x_dim, "ens"]], skipna=True
        ).sortby(x_dim)

        if self.plot_ensemble == "std":
            std_dev = ens.std(dim="ens", skipna=True).sortby(x_dim)
            plt.fill_between(
                averaged[x_dim],
                (averaged - std_dev).values,
                (averaged + std_dev).values,
                label=f"{label} - std dev",
                color=color,
                alpha=0.2,
            )

        elif self.plot_ensemble == "minmax":
            ens_min = ens.min(dim="ens", skipna=True).sortby(x_dim)
            ens_max = ens.max(dim="ens", skipna=True).sortby(x_dim)

            plt.fill_between(
                averaged[x_dim],
                ens_min.values,
                ens_max.values,
                label=f"{label} - min max",
                color=color,
                alpha=0.2,
            )

        elif self.plot_ensemble == "members":
            for j in range(ens.ens.size):
                plt.plot(
                    ens[x_dim],
                    ens.isel(ens=j).values,
                    color=color,
                    alpha=0.2,
                )
        else:
            _logger.warning(
                f"LinePlot:: Unknown option for plot_ensemble: {self.plot_ensemble}. "
                "Skipping ensemble plotting."
            )

    def _preprocess_data(
        self, data: xr.DataArray, x_dim: str | list[str], verbose: bool = True
    ) -> xr.DataArray:
        """
        Average all dimensions except x_dim (which may be a string or list)
        and then sort the result.

        Parameters
        ----------
        data : xr.DataArray
            DataArray to be preprocessed.
        x_dim : str or list of str
            Dimension(s) to be preserved for the x-axis.
        verbose : bool
            Log information about averaging.

        Returns
        -------
        xr.DataArray
            Preprocessed DataArray.
        """

        x_dims = [x_dim] if isinstance(x_dim, str) else list(x_dim)

        non_x_dims = [dim for dim in data.dims if dim not in x_dims]

        if any(data.sizes.get(dim, 1) > 1 for dim in non_x_dims) and verbose:
            logging.info(f"Averaging over dimensions: {non_x_dims}")

        out = data.mean(dim=non_x_dims, skipna=True)

        for xd in x_dims:
            out = out.sortby(xd)

        return out

    def plot(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        tag: str = "",
        x_dim: str = "lead_time",
        y_dim: str = "value",
        print_summary: bool = False,
        plot_ensemble: str | bool = False,
    ) -> None:
        """
        Plot a line graph comparing multiple datasets.

        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset
        tag:
            Tag to be added to the plot title and filename
        x_dim:
            Dimension to be used for the x-axis. The code will average over all other dimensions.
        y_dim:
            Name of the dimension to be used for the y-axis.
        print_summary:
            If True, print a summary of the values from the graph.
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        assert x_dim in data_list[0].dims or x_dim in data_list[0].coords, (
            f"x dimension '{x_dim}' not found in data dimensions "
            f"{data_list[0].dims} or coords {data_list[0].coords}."
        )

        fig = plt.figure(figsize=(12, 6), dpi=self.dpi_val)

        for i, data in enumerate(data_list):
            non_zero_dims = [dim for dim in data.dims if dim != x_dim and data[dim].shape[0] > 1]

            if self.plot_ensemble and "ens" in non_zero_dims:
                _logger.info(f"LinePlot:: Plotting ensemble with option {self.plot_ensemble}.")
                self._plot_ensemble(data, x_dim, label_list[i])
            else:
                averaged = self._preprocess_data(data, x_dim)

                plt.plot(
                    averaged[x_dim],
                    averaged.values,
                    label=label_list[i],
                    marker="o",
                    linestyle="-",
                )

        parts = ["compare", tag]
        name = "_".join(filter(None, parts))
        self._plot_base(fig, name, x_dim, y_dim, print_summary)

    def _plot_base(
        self,
        fig: plt.Figure,
        name: str,
        x_dim: str,
        y_dim: str,
        print_summary: bool = False,
        line: float | None = None,
        vlines: bool = False,
        title: str | None = None,
    ) -> None:
        """
        Apply labels, title, legend, save and optionally print summary.
        Parameters
        ----------
        fig:
            Matplotlib figure to be finalized
        name:
            Name of the plot file
        x_dim:
            Label for the x-axis
        y_dim:
            Label for the y-axis
        print_summary:
            If True, print a summary of the values from the graph.
        line:
            If provided, draw a horizontal line at the given y-value.
        vlines:
            If True, draw vertical lines to separate each group of variables.
        title:
            Title for the plot.
        Returns
        -------
            None
        """
        plt.xlabel("".join(c if c.isalnum() else " " for c in x_dim))
        plt.ylabel("".join(c if c.isalnum() else " " for c in y_dim))
        plt.title(title if title is not None else " ".join(c if c.isalnum() else " " for c in name))
        plt.legend(frameon=False)

        if self.add_grid:
            plt.grid(True, linestyle="--", color="gray", alpha=0.5)

        if self.log_scale:
            plt.yscale("log")

        if print_summary:
            _logger.info(f"Summary values for {name}")
            self.print_all_points_from_graph(fig)

        if line:
            plt.axhline(y=line, color="black", linestyle="--", linewidth=1, zorder=1)

        if vlines:
            vlines = []
            last_prefix = None

            channels = [t.get_text() for t in fig.gca().get_xticklabels() if t.get_text()]

            for idx, ch in enumerate(channels):
                m = re.match(r"([a-zA-Z]+)_\d+", ch)
                prefix = m.group(1) if m else ch
                if last_prefix is not None and prefix != last_prefix:
                    vlines.append(idx - 0.5)
                last_prefix = prefix
            for vl in vlines:
                plt.axvline(x=vl, color="#001f3f", linestyle="-", linewidth=0.5, zorder=1)

        plt.tight_layout()
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")
        plt.close()

    def ratio_plot(
        self,
        data: xr.DataArray | list,
        run_ids: list[str],
        labels: str | list,
        tag: str = "",
        x_dim: str = "forecast_step",
        y_dim: str = "value",
        print_summary: bool = False,
    ) -> None:
        """
        Plot a ratio plot comparing multiple datasets to the first dataset.
        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        run_ids:
            List of run IDs corresponding to each dataset
        labels:
            Label or list of labels for each dataset
        tag:
            Tag to be added to the plot title and filename
        x_dim:
            Dimension to be used for the x-axis. The code will average over all other dimensions.
        y_dim:
            Name of the dimension to be used for the y-axis.
        print_summary:
            If True, print a summary of the values from the graph.
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        if len(data_list) < 2:
            baseline = xr.full_like(data_list[0], 1.0)
            baseline_name = "ones"
            descr = "scores"
        else:
            descr = "ratio_plot"
            baseline_name = self.baseline
            baseline_idx = run_ids.index(self.baseline) if self.baseline in run_ids else None
            if baseline_idx is not None:
                _logger.info(f"Using baseline run ID '{self.baseline}' for ratio plot.")
                baseline = data_list[baseline_idx]

            else:
                baseline_name = run_ids[0]
                baseline = data_list[0]

        ref_raw = self._preprocess_data(baseline, x_dim, verbose=False)

        channel_names = set(ref_raw.channel.values)
        # Merge channels from remaining datasets
        for data in data_list[1:]:
            channel_names.update(data.channel.values)  # add new channels

        # Sort the merged list
        ref_channel_names = sorted(channel_names, key=channel_sort_key)

        ref = align_labels(ref_raw, ref_channel_names, x_dim).reindex(channel=ref_channel_names)

        fig = plt.figure(figsize=(max(12, len(ref_channel_names) * 0.25), 6))

        for data, run_id, lbl in zip(data_list, run_ids, label_list, strict=False):
            if run_id == baseline_name:
                continue  # skip baseline

            num_raw = self._preprocess_data(data, x_dim, verbose=False)
            num = align_labels(num_raw, ref_channel_names, x_dim).reindex(channel=ref_channel_names)

            ratio = num.sel(channel=ref_channel_names) / ref.sel(channel=ref_channel_names)

            plt.plot(
                ref_channel_names,
                ratio.values,
                label=lbl,
                marker="o",
                linestyle="-",
            )

        parts = [descr, tag]
        name = "_".join(filter(None, parts))
        plt.xticks(rotation=90, ha="right")
        plt.grid(True, linestyle="--", color="gray", alpha=0.2)
        title = (
            f"{descr.replace('_', ' ')} {tag.split('_')[0]} -"
            f" {tag.split('_')[-1]} (baseline: {baseline_name})"
        )
        self._plot_base(fig, name, x_dim, y_dim, print_summary, line=1.0, vlines=True, title=title)

    def heat_map(
        self,
        data: xr.DataArray | list,
        labels: str | list,
        metric: str,
        x_dim,
        tag: str = "",
    ) -> None:
        """
        Plot a heat map comparing multiple datasets.
        Parameters
        ----------
        data:
            DataArray or list of DataArrays to be plotted
        labels:
            Label or list of labels for each dataset
        metric:
            Metric for which we are plotting
        x_dim:
            Dimension to be used for the x-axis. The code will average over all other dimensions.
        tag:
            Tag to be added to the plot title and filename
        Returns
        -------
            None
        """

        data_list, label_list = self._check_lengths(data, labels)

        n_runs = len(data_list)

        x_ticks_names = set()

        for data in data_list:
            da = data.isel({x_dim: 0})
            x_ticks_names.update(map(str, da.channel.values))

        ref_ticks_names = sorted(x_ticks_names, key=channel_sort_key)

        fig, axes = plt.subplots(
            1, n_runs, figsize=(8 * n_runs, max(12, len(ref_ticks_names) * 0.25)), squeeze=False
        )

        global_min = float("inf")
        global_max = float("-inf")

        for ax, data, label in zip(axes[0], data_list, labels, strict=False):
            time_steps = sorted(data[x_dim].values)

            # Use the first time step as reference
            ref = data.reindex(channel=ref_ticks_names).sel({x_dim: time_steps[0]})
            ref = self._preprocess_data(ref, "channel", verbose=False)

            if ref.isnull().all():
                _logger.warning(
                    f"Heatmap:: Reference data for metric {metric} and label {label} contains "
                    "only NaNs. Skipping heatmap."
                )
                continue

            # Compute ratio for all time steps
            num = self._preprocess_data(data, [x_dim, "channel"], verbose=False)
            num = num.reindex(channel=ref_ticks_names).sel({x_dim: time_steps})

            heatmap_data = num / ref

            cmap = plt.get_cmap("magma_r") if lower_is_better(metric) else plt.get_cmap("magma")
            global_min = min(global_min, float(heatmap_data.min()))
            global_max = max(global_max, float(heatmap_data.max()))

            last_hm = sns.heatmap(
                heatmap_data.values.T,
                ax=ax,
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
                xticklabels=time_steps,
                yticklabels=ref_ticks_names,
                annot=False,
                fmt=".2f",
                cbar=False,
            )
            ax.set_title(f"Heatmap {metric} – {label}")
            ax.set_xlabel(f"{x_dim.replace('_', ' ').title()} (h)")
            ax.set_ylabel("Variable")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        cbar = fig.colorbar(
            last_hm.collections[0], ax=axes.ravel().tolist(), shrink=0.6, location="right", pad=0.02
        )
        cbar.set_label(rf"{metric} - $t_{{\mathrm{{step}}}}[0] / t_{{\mathrm{{step}}}}[x]$")
        parts = ["heat_map", metric, tag]
        name = "_".join(filter(None, parts))
        plt.savefig(f"{self.out_plot_dir.joinpath(name)}.{self.image_format}")


class ScoreCards:
    """
    Initialize the ScoreCards class.

    Parameters
    ----------
    plotter_cfg:
        Configuration dictionary containing basic information for plotting.
        Expected keys are:
            - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
            - improvement: Size of the figure (width, height) in inches
    output_basedir:
        Base directory under which the score cards will be saved.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.improvement = plotter_cfg.get("improvement_scale", 0.2)
        self.out_plot_dir = Path(output_basedir) / "score_cards"
        self.baseline = plotter_cfg.get("baseline")
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Plot score cards comparing performance between run_ids against a baseline over channels
        of interest.

        Parameters
        ----------
        data:
            List of (xarray) DataArrays with the scores (stream, region and metric specific)
        runs:
            List containing runs (in str format) to be compared (provided in the config)
        metric:
            Metric for which we are plotting
        channels:
            List containing channels (in str format) of interest (provided in the config)
        tag:
            Tag to be added to the plot title and filename
        """
        n_runs = len(runs)

        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]

        common_channels, n_common_channels = self.extract_common_channels(data, channels, n_runs)

        fig, ax = plt.subplots(figsize=(2 * n_runs, 1.2 * n_common_channels))

        baseline = data[0]
        skill_models = []
        for run_index in range(1, n_runs):
            skill_model = 0.0
            for var_index, var in enumerate(common_channels):
                if var not in data[0].channel.values or var not in data[run_index].channel.values:
                    continue
                diff, avg_diff, avg_skill = self.compare_models(
                    data, baseline, run_index, var, metric
                )
                skill_model += avg_skill.values

                # Get symbols based on difference and performance as well as coordinates
                # for the position of the triangles.

                x, y, alt, color, triangle, size = self.get_plot_symbols(
                    run_index, var_index, avg_skill, avg_diff, metric
                )

                ax.scatter(x, y, marker=triangle, color=color, s=size.values, zorder=3)

                # Perform Wilcoxon test
                if len(diff["forecast_step"].values) > 1:
                    stat, p = wilcoxon(diff, alternative=alt)

                    # Draw rectangle border for significance
                    if p < 0.05:
                        lw = 2 if p < 0.01 else 1
                        rect_color = color
                        rect = plt.Rectangle(
                            (x - 0.25, y - 0.25),
                            0.5,
                            0.5,
                            fill=False,
                            edgecolor=rect_color,
                            linewidth=lw,
                            zorder=2,
                        )
                        ax.add_patch(rect)

            skill_models.append(skill_model / n_common_channels)

        # Set axis labels
        ylabels = [
            f"{var}\n({baseline.coords['metric'].item().upper()}={baseline.sel(channel=var).mean().values.squeeze():.3f})"
            for var in common_channels
        ]
        xlabels = [
            f"{model_name}\nSkill: {skill_models[i]:.3f}" for i, model_name in enumerate(runs[1::])
        ]
        ax.set_xticks(np.arange(1, n_runs))
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_yticks(np.arange(n_common_channels) + 0.5)
        ax.set_yticklabels(ylabels, fontsize=10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("center")
            label.set_x(-0.17)
        ax.set_ylabel("Variable", fontsize=14)
        ax.set_title(
            f"Model Scorecard vs. Baseline '{runs[0]}'",
            fontsize=16,
            pad=20,
        )
        for x in np.arange(0.5, n_runs - 1, 1):
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=0, alpha=0.5)
        ax.set_xlim(0.5, n_runs - 0.5)
        ax.set_ylim(0, n_common_channels)

        legend = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="white",
                label=f"{self.improvement * 100:.0f}% improvement",
                markerfacecolor="blue",
                markersize=np.sqrt(200),
            )
        ]
        plt.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        _logger.info(f"Saving scorecards to: {self.out_plot_dir}")

        parts = ["score_card", tag] + runs
        name = "_".join(filter(None, parts))
        plt.savefig(
            f"{self.out_plot_dir.joinpath(name)}.{self.image_format}",
            bbox_inches="tight",
            dpi=self.dpi_val,
        )
        plt.close(fig)

    def extract_common_channels(self, data, channels, n_runs):
        common_channels = []
        for run_index in range(1, n_runs):
            for var in channels:
                if var not in data[0].channel.values or var not in data[run_index].channel.values:
                    continue
                common_channels.append(var)
        common_channels = list(set(common_channels))
        n_vars = len(common_channels)
        return common_channels, n_vars

    def compare_models(
        self,
        data: list[xr.DataArray],
        baseline: xr.DataArray,
        run_index: int,
        var: str,
        metric: str,
        x_dim="forecast_step",
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compare a model with a baseline model and calculate skill scores.

        Parameters
        ----------
        data: list[xr.DataArray]
            List of all scores in xarray format for each model.

        baseline: xarray DataArray
            The baseline scores in xarrays format.

        run_index: int
            The order index over the run_ids.

        var: str
            The specified channel over which we compare.

        xdim: str
            The dimension for which an average will not be calculated.

        Returns
        ----------
        diff: xr.DataArray
            Difference in scores between baseline and model.

        diff.mean(dim="forecast_step"): xr.DataArray
            Average difference in scores over all forecast steps between baseline and model .

        skill.mean(dim="forecast_step"): xr.DataArray
            Average skill scores over all forecast steps between baseline and model .

        """
        baseline_var = baseline.sel({"channel": var})
        data_var = data[run_index].sel({"channel": var})

        baseline_score, model_score = calculate_average_over_dim(x_dim, baseline_var, data_var)
        diff = baseline_score - model_score

        skill = self.get_skill_score(model_score, baseline_score, metric)
        return diff, diff.mean(dim=x_dim), skill.mean(dim=x_dim)

    def get_skill_score(
        self, score_model: xr.DataArray, score_ref: xr.DataArray, metric: str
    ) -> xr.DataArray:
        """
        Calculate skill score comparing a model against a baseline.

        Skill score is defined as: (model_score - baseline_score) / (perfect_score - baseline_score)

        Parameters
        ----------
        score_model : xr.DataArray
            The scores of the model being evaluated
        score_ref : xr.DataArray
            The scores of the reference/baseline model
        metric : str
            The metric name for which to calculate skill score

        Returns
        -------
        xr.DataArray
            Skill scores comparing model to baseline
        """
        perf_score = self.get_perf_score(metric)
        skill_score = (score_model - score_ref) / (perf_score - score_ref)
        return skill_score

    def get_perf_score(self, metric: str) -> float:
        """
        Get the perfect score for a given metric.

        Perfect scores represent ideal performance:
        - Error metrics: 0 (lower is better)
        - Skill/score metrics: 1 (higher is better)
        - PSNR: 100 (higher is better)

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        float
            Perfect score for the specified metric
        """
        # Metrics where lower values indicate better performance (error metrics)
        if lower_is_better(metric):
            return 0.0

        # Metrics where higher values indicate better performance (with specific perfect score)
        elif metric in ["psnr"]:
            return 100.0

        # Metrics where higher values indicate better performance (default perfect score)
        else:
            return 1.0

    def get_plot_symbols(
        self,
        run_index: int,
        var_index: int,
        avg_skill: xr.DataArray,
        avg_diff: xr.DataArray,
        metric: str,
    ) -> tuple[int, float, str, str, str, xr.DataArray]:
        """
        Determine plot symbol properties based on performance difference.

        Parameters
        ----------
        run_index : int
            Index of the model.
        var_index : int
            Index of the variable/channel.
        avg_skill : xr.DataArray
            Average skill score of the model.
        avg_diff : xr.DataArray
            Average difference between baseline and model.
        metric : str
            Metric used for interpretation.

        Returns
        -------
        Tuple[int, float, str, str, str, xr.DataArray]
            x, y coordinates, alternative hypothesis, color, triangle symbol, size.
        """
        # Conservative choice
        alt = "two-sided"
        modus = "different"
        color = "gray"

        # Determine if diff_mean indicates improvement
        is_improvement = (avg_diff > 0 and lower_is_better(metric)) or (
            avg_diff < 0 and not lower_is_better(metric)
        )

        if is_improvement:
            alt = "greater"
            modus = "better"
            color = "blue"
        elif not is_improvement and avg_diff != 0:
            alt = "less"
            modus = "worse"
            color = "red"
        else:
            alt = "two-sided"
            modus = "different"

        triangle = "^" if modus == "better" else "v"

        # Triangle coordinates
        x = run_index
        # First row is model 1 vs model 0
        y = var_index + 0.5

        size = 200 * (1 - (1 / (1 + abs(avg_skill) / self.improvement)))  # Add base size to all

        return x, y, alt, color, triangle, size


class BarPlots:
    """
    Initialize the BarPlots class.

    Parameters
    ----------
    plotter_cfg:
        Configuration dictionary containing basic information for plotting.
        Expected keys are:
            - image_format: Format of the saved images (e.g., 'png', 'pdf', etc.)
            - improvement: Size of the figure (width, height) in inches
    output_basedir:
        Base directory under which the score cards will be saved.
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        self.image_format = plotter_cfg.get("image_format")
        self.dpi_val = plotter_cfg.get("dpi_val")
        self.cmap = plotter_cfg.get("cmap", "bwr")
        self.out_plot_dir = Path(output_basedir) / "bar_plots"
        self.baseline = plotter_cfg.get("baseline")
        _logger.info(f"Saving bar plots to: {self.out_plot_dir}")
        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Plot (ratio) bar plots comparing performance between different run_ids over channels of
        interest.

        Parameters
        ----------
        data:
            List of (xarray) DataArrays with the scores (stream, region and metric specific)
        runs:
            List containing runs (in str format) to be compared (provided in the config)
        metric:
            Metric name
        channels:
            List containing channels (in str format) of interest (provided in the config)
        tag:
            Tag to be added to the plot title and filename
        """

        fig, ax = plt.subplots(
            1,
            len(runs) - 1 if len(runs) > 1 else 1,
            figsize=(5 * len(runs), 2 * len(channels)),
            dpi=self.dpi_val,
            squeeze=False,
        )
        ax = ax.flatten()
        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]
        elif len(runs) < 2:
            _logger.warning(
                "BarPlots:: Less than two runs provided. Generating bar plot against ones."
            )
            ones_array = xr.full_like(data[0], 1.0)
            runs = [""] + runs
            data = [ones_array] + data

        for run_index in range(1, len(runs)):
            score, channels_per_comparison = self.calc_ratio_per_run_id(data, channels, run_index)
            if len(score) > 0:
                ax[run_index - 1].barh(
                    np.arange(len(score)),
                    score,
                    color=self.colors(score, metric),
                    align="center",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax[run_index - 1].set_yticks(np.arange(len(score)), labels=channels_per_comparison)
                ax[run_index - 1].invert_yaxis()

                xlabel = (
                    f"Relative {data[0].coords['metric'].item().upper()}: "
                    f"Target Model ({runs[run_index]}) / Reference Model ({runs[0]})"
                )

                if len(runs) == 2 and runs[0] == "":
                    xlabel = xlabel.replace("Relative ", "")
                    xlabel = xlabel.replace(
                        f"Target Model ({runs[run_index]}) / Reference Model ({runs[0]})",
                        f"Model ({runs[run_index]})",
                    )

                ax[run_index - 1].set_xlabel(xlabel)
            else:
                ax[run_index - 1].set_visible(False)  # or annotate as missing
                # Or show a message:
                ax[run_index - 1].text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax[run_index - 1].transAxes,
                )

        _logger.info(f"Saving bar plots to: {self.out_plot_dir}")
        parts = ["bar_plot", tag] + runs
        name = "_".join(filter(None, parts))
        plt.savefig(
            f"{self.out_plot_dir.joinpath(name)}.{self.image_format}",
            bbox_inches="tight",
            dpi=self.dpi_val,
        )
        plt.close(fig)

    def calc_ratio_per_run_id(
        self,
        data: list[xr.DataArray],
        channels: list[str],
        run_index: int,
        x_dim="channel",
    ) -> tuple[np.array, str]:
        """
        This function calculates the ratio per comparison model for each channel.

        Parameters
        ----------
        data: list[xr.DataArray]
            List of all scores for each model in xarrays format.
        channels: list[str]
            All the available channels.
        run_index: int
            The order index over the run_ids.
        xdim: str
            The dimension for which an average will not be calculated.

        Returns
        ----------
        ratio_score: np.array
            The (ratio) skill over each channel for a specific model
        channels_per_comparison: str
            The common channels over which the baseline and the other model will be compared.

        """
        ratio_score = []
        channels_per_comparison = []

        for _, var in enumerate(channels):
            if var not in data[0].channel.values or var not in data[run_index].channel.values:
                continue
            baseline_var = data[0].sel({"channel": var})
            data_var = data[run_index].sel({"channel": var})
            channels_per_comparison.append(var)

            baseline_score, model_score = calculate_average_over_dim(x_dim, baseline_var, data_var)

            ratio_score.append(model_score / baseline_score)

        if np.allclose(baseline_score, 1.0, atol=1e-6):
            ratio_score = np.array(ratio_score)
        else:
            ratio_score = np.array(ratio_score) - 1

        return ratio_score, channels_per_comparison

    def colors(self, ratio_score: np.array, metric: str) -> list[tuple]:
        """
        This function calculates colormaps based on the skill scores. From negative value blue
        color variations should be given otherwise red color variations should be given.

        Parameters
        ----------
        ratio_score: np.array
            The (ratio) skill for a specific model
        metric: str
            The metric of interest
        Returns
        ----------
        colors: list[tuple]
            The color magnitude (blue to red) of the bars in the plots
        """
        max_val = np.abs(ratio_score).max()
        if lower_is_better(metric):
            cmap = plt.get_cmap("bwr")
        else:
            cmap = plt.get_cmap("bwr_r")
        colors = [cmap(0.5 + v / (2 * max_val)) for v in ratio_score]
        return colors


def calculate_average_over_dim(
    x_dim: str, baseline_var: xr.DataArray, data_var: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate average over xarray dimensions that are larger than 1. Those might be the
    forecast-steps or the samples.

    Parameters
    ----------
    xdim: str
        The dimension for which an average will not be calculated.
    baseline_var: xr.DataArray
        xarray DataArray with the scores of the baseline model for a specific channel/variable
    data_var: xr.DataArray
        xarray DataArray with the scores of the comparison model for a specific channel/variable

    Returns
    -------
    baseline_score: xarray DataArray
        The baseline average scores over the dimensions not specified by xdim
    model_score: xarray DataArray
        The model average scores over the dimensions not specified by xdim
    """
    non_zero_dims = [
        dim for dim in baseline_var.dims if dim != x_dim and baseline_var[dim].shape[0] > 1
    ]

    if non_zero_dims:
        _logger.info(f"Found multiple entries for dimensions: {non_zero_dims}. Averaging...")

    baseline_score = baseline_var.mean(
        dim=[dim for dim in baseline_var.dims if dim != x_dim], skipna=True
    )
    model_score = data_var.mean(dim=[dim for dim in data_var.dims if dim != x_dim], skipna=True)

    return baseline_score, model_score


def lower_is_better(metric: str) -> bool:
    # Determine whether lower or higher is better
    return metric in {"l1", "l2", "mae", "mse", "rmse", "vrmse", "bias", "crps", "spread"}


def compute_offsets(n, spacing=0.11):
    idx = np.arange(n)
    return (idx - (n - 1) / 2.0) * spacing


def align_labels(da: xr.DataArray, labels: list[str], x_dim: str) -> xr.DataArray:
    """
    Reindex a DataArray to include all labels in the canonical order.
    Missing variables are filled with NaN.
    """
    # Convert labels → index format expected by xarray
    labels = np.array(labels, dtype=object)

    # Reindex, inserting NaN for missing labels
    return da.reindex({x_dim: labels})


def channel_sort_key(name: str) -> tuple[int, str, int]:
    """
    Sorting key for channel names like 't_850', 'z_500', etc.
    Splits the name into a prefix and a number suffix for sorting.
    Parameters
    ----------
    name : str
        Channel name to be sorted.
    Returns
    -------
    tuple[int, str, int]
        Sorting key: (0, prefix, number) if pattern matches, else (1,
    """
    m = re.match(r"(.+?)_(\d+)$", name)
    if m:
        prefix, number = m.groups()
        return (0, prefix, int(number))
    else:
        return (1, name, float("inf"))


class HeatmapScoreCard:
    """
    Generate heatmap-style score cards showing variables vs lead times.

    Creates a grid visualization where:
    - Rows: Variables/channels
    - Columns: Lead times (hours)
    - Color intensity: Metric values

    Supports side-by-side comparison of multiple models with diverging
    colormaps when comparing to a baseline (blue=better, red=worse).
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        """
        Initialize the HeatmapScoreCard class.

        Parameters
        ----------
        plotter_cfg : dict
            Configuration dictionary containing:
                - image_format: Format of saved images (e.g., 'png', 'pdf')
                - dpi_val: DPI value for saved images
                - baseline: Run ID to use as baseline for comparison
                - annotate_cells: bool, whether to show numeric values in cells
        output_basedir : str | Path
            Base directory for saving plots
        """
        self.image_format = plotter_cfg.get("image_format", "png")
        self.dpi_val = plotter_cfg.get("dpi_val", 300)
        self.baseline = plotter_cfg.get("baseline")
        self.annotate_cells = plotter_cfg.get("annotate_cells", True)
        self.out_plot_dir = Path(output_basedir) / "heatmap_scorecards"

        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Generate heatmap scorecard visualization.

        Parameters
        ----------
        data : list[xr.DataArray]
            List of DataArrays with scores for each run
        runs : list[str]
            List of run identifiers
        metric : str
            Metric name being visualized
        channels : list[str]
            List of channel/variable names
        tag : str
            Tag for filename
        """
        n_runs = len(runs)
        if n_runs == 0:
            _logger.warning("HeatmapScoreCard: No data to plot.")
            return

        # Reorder so baseline is first if specified
        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            runs = [runs[baseline_idx]] + runs[:baseline_idx] + runs[baseline_idx + 1 :]
            data = [data[baseline_idx]] + data[:baseline_idx] + data[baseline_idx + 1 :]

        # Get common channels sorted
        common_channels = sorted(
            [ch for ch in channels if all(ch in d.channel.values for d in data)],
            key=channel_sort_key,
        )

        if not common_channels:
            _logger.warning("HeatmapScoreCard: No common channels found.")
            return

        # Get lead times from first dataset
        sample_data = data[0]
        if "lead_time" in sample_data.coords:
            lead_times = np.unique(sample_data.lead_time.values)
            x_label = "Lead Time (h)"
            x_dim = "lead_time"
        else:
            lead_times = sample_data.forecast_step.values
            x_label = "Forecast Step"
            x_dim = "forecast_step"

        n_channels = len(common_channels)
        n_times = len(lead_times)

        # Create figure with subplots for each model
        fig_width = max(8, 2 * n_times * n_runs / 10)
        fig_height = max(6, 0.4 * n_channels)
        fig, axes = plt.subplots(
            1, n_runs, figsize=(fig_width, fig_height), squeeze=False, dpi=self.dpi_val
        )
        axes = axes.flatten()

        # Prepare heatmap data for each run
        heatmap_matrices = []
        for run_data in data:
            matrix = self._create_heatmap_data(run_data, common_channels, lead_times, x_dim)
            heatmap_matrices.append(matrix)

        # Determine if we should use comparison mode (ratio to baseline)
        use_comparison = self.baseline is not None and n_runs > 1
        baseline_matrix = heatmap_matrices[0] if use_comparison else None

        # Find global min/max for consistent colorbar
        if use_comparison:
            # Compute ratios for non-baseline runs
            ratio_matrices = []
            for i, matrix in enumerate(heatmap_matrices):
                if i == 0:
                    ratio_matrices.append(np.ones_like(matrix))  # baseline shows as 1.0
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = matrix / baseline_matrix
                        ratio_matrices.append(ratio)

            all_ratios = np.concatenate([m.flatten() for m in ratio_matrices[1:]])
            valid_ratios = all_ratios[np.isfinite(all_ratios)]
            if len(valid_ratios) > 0:
                vmin = max(0.5, np.nanpercentile(valid_ratios, 2))
                vmax = min(2.0, np.nanpercentile(valid_ratios, 98))
            else:
                vmin, vmax = 0.5, 2.0
            vcenter = 1.0
        else:
            all_values = np.concatenate([m.flatten() for m in heatmap_matrices])
            valid_values = all_values[np.isfinite(all_values)]
            if len(valid_values) > 0:
                vmin = np.nanpercentile(valid_values, 2)
                vmax = np.nanpercentile(valid_values, 98)
            else:
                vmin, vmax = 0, 1
            vcenter = None

        # Plot each model
        for idx, (ax, run_id, matrix) in enumerate(zip(axes, runs, heatmap_matrices, strict=False)):
            if use_comparison and idx > 0:
                # Show ratio to baseline
                with np.errstate(divide="ignore", invalid="ignore"):
                    plot_matrix = matrix / baseline_matrix
                cmap = self._get_comparison_colormap(metric)
                norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                title_suffix = f" (vs {runs[0]})"
            elif use_comparison and idx == 0:
                # Baseline: show absolute values
                plot_matrix = matrix
                cmap = plt.get_cmap("magma_r") if lower_is_better(metric) else plt.get_cmap("magma")
                norm = mpl.colors.Normalize(
                    vmin=np.nanpercentile(matrix, 2), vmax=np.nanpercentile(matrix, 98)
                )
                title_suffix = " (baseline)"
            else:
                # No comparison: absolute values
                plot_matrix = matrix
                cmap = plt.get_cmap("magma_r") if lower_is_better(metric) else plt.get_cmap("magma")
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                title_suffix = ""

            # Format x-axis labels (lead times)
            if x_dim == "lead_time":
                x_labels = [f"{int(lt)}h" for lt in lead_times]
            else:
                x_labels = [str(int(lt)) for lt in lead_times]

            im = sns.heatmap(
                plot_matrix,
                ax=ax,
                cmap=cmap,
                norm=norm,
                xticklabels=x_labels,
                yticklabels=common_channels,
                annot=self.annotate_cells,
                fmt=".2f",
                annot_kws={"size": 7},
                cbar=False,
                linewidths=0.5,
                linecolor="white",
            )

            ax.set_title(f"{run_id}{title_suffix}", fontsize=10, fontweight="bold")
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel("Variable" if idx == 0 else "", fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)

        # Add shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        if use_comparison:
            sm = plt.cm.ScalarMappable(cmap=self._get_comparison_colormap(metric), norm=norm)
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(f"{metric.upper()} Ratio", fontsize=9)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(metric.upper(), fontsize=9)

        fig.suptitle(f"Heatmap Scorecard: {metric.upper()}", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        # Save figure
        parts = ["heatmap_scorecard", tag] + runs
        name = "_".join(filter(None, parts))
        save_path = self.out_plot_dir / f"{name}.{self.image_format}"
        _logger.info(f"Saving heatmap scorecard to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi_val)
        plt.close(fig)

    def _create_heatmap_data(
        self,
        data: xr.DataArray,
        channels: list[str],
        lead_times: np.ndarray,
        x_dim: str,
    ) -> np.ndarray:
        """
        Prepare 2D array for heatmap: rows=channels, cols=lead_times.

        Parameters
        ----------
        data : xr.DataArray
            Input data array
        channels : list[str]
            List of channel names for rows
        lead_times : np.ndarray
            Array of lead time values for columns
        x_dim : str
            Dimension name for x-axis ('lead_time' or 'forecast_step')

        Returns
        -------
        np.ndarray
            2D array with shape (n_channels, n_lead_times)
        """
        n_channels = len(channels)
        n_times = len(lead_times)
        matrix = np.full((n_channels, n_times), np.nan)

        # Average over sample and any other dimensions except channel and x_dim
        avg_dims = [d for d in data.dims if d not in ["channel", x_dim, "forecast_step"]]
        if avg_dims:
            data = data.mean(dim=avg_dims, skipna=True)

        # If x_dim is lead_time but data uses forecast_step, swap
        if x_dim == "lead_time" and "lead_time" in data.coords:
            if "forecast_step" in data.dims:
                data = data.swap_dims({"forecast_step": "lead_time"})

        for i, ch in enumerate(channels):
            if ch not in data.channel.values:
                continue
            ch_data = data.sel(channel=ch)

            for j, lt in enumerate(lead_times):
                try:
                    if x_dim in ch_data.dims:
                        val = ch_data.sel({x_dim: lt}, method="nearest")
                    elif x_dim in ch_data.coords:
                        val = ch_data.where(ch_data[x_dim] == lt, drop=True)
                        if val.size > 0:
                            val = val.mean()
                        else:
                            continue
                    else:
                        continue

                    matrix[i, j] = float(val.values) if val.size == 1 else float(val.mean().values)
                except (KeyError, ValueError):
                    continue

        return matrix

    def _get_comparison_colormap(self, metric: str) -> mpl.colors.Colormap:
        """
        Return appropriate diverging colormap for metric comparison.

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        mpl.colors.Colormap
            Colormap where blue=better, red=worse
        """
        # For lower-is-better metrics: ratio < 1 is better (blue), ratio > 1 is worse (red)
        # For higher-is-better metrics: ratio > 1 is better, ratio < 1 is worse
        if lower_is_better(metric):
            return plt.get_cmap("RdBu")  # Red for high (worse), Blue for low (better)
        else:
            return plt.get_cmap("RdBu_r")  # Blue for high (better), Red for low (worse)


class SummaryCard:
    """
    Generate compact summary cards showing key statistics per model.

    Each card displays:
    - Model name and overall mean score
    - Best/worst performing variables
    - Improvement vs baseline (if available)
    - Score trend across lead time (mini sparkline)
    """

    def __init__(self, plotter_cfg: dict, output_basedir: str | Path) -> None:
        """
        Initialize the SummaryCard class.

        Parameters
        ----------
        plotter_cfg : dict
            Configuration dictionary containing:
                - image_format: Format of saved images
                - dpi_val: DPI value
                - baseline: Run ID for baseline comparison
                - cards_per_row: int, number of cards per row in grid layout
                - show_sparklines: bool, show mini trend lines
        output_basedir : str | Path
            Base directory for saving plots
        """
        self.image_format = plotter_cfg.get("image_format", "png")
        self.dpi_val = plotter_cfg.get("dpi_val", 300)
        self.baseline = plotter_cfg.get("baseline")
        self.cards_per_row = plotter_cfg.get("cards_per_row", 3)
        self.show_sparklines = plotter_cfg.get("show_sparklines", True)
        self.out_plot_dir = Path(output_basedir) / "summary_cards"

        if not os.path.exists(self.out_plot_dir):
            _logger.info(f"Creating dir {self.out_plot_dir}")
            os.makedirs(self.out_plot_dir, exist_ok=True)

    def plot(
        self,
        data: list[xr.DataArray],
        runs: list[str],
        metric: str,
        channels: list[str],
        tag: str,
    ) -> None:
        """
        Generate summary card visualization for multiple models.

        Parameters
        ----------
        data : list[xr.DataArray]
            List of DataArrays with scores for each run
        runs : list[str]
            List of run identifiers
        metric : str
            Metric name being visualized
        channels : list[str]
            List of channel/variable names
        tag : str
            Tag for filename
        """
        n_runs = len(runs)
        if n_runs == 0:
            _logger.warning("SummaryCard: No data to plot.")
            return

        # Reorder so baseline is first if specified
        baseline_data = None
        if self.baseline and self.baseline in runs:
            baseline_idx = runs.index(self.baseline)
            baseline_data = data[baseline_idx]

        # Calculate grid layout
        n_cols = min(self.cards_per_row, n_runs)
        n_rows = (n_runs + n_cols - 1) // n_cols

        # Create figure
        card_width = 3.5
        card_height = 3.0 if self.show_sparklines else 2.5
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(card_width * n_cols, card_height * n_rows),
            squeeze=False,
            dpi=self.dpi_val,
        )

        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        # Create each card
        for idx, (ax, run_id, run_data) in enumerate(
            zip(axes_flat[:n_runs], runs, data, strict=False)
        ):
            self._create_single_card(
                ax,
                run_data,
                run_id,
                metric,
                channels,
                baseline_data if run_id != self.baseline else None,
            )

        # Hide unused axes
        for ax in axes_flat[n_runs:]:
            ax.axis("off")

        fig.suptitle(
            f"Model Summary Cards: {metric.upper()}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        # Save figure
        parts = ["summary_cards", tag] + runs
        name = "_".join(filter(None, parts))
        save_path = self.out_plot_dir / f"{name}.{self.image_format}"
        _logger.info(f"Saving summary cards to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi_val)
        plt.close(fig)

    def _create_single_card(
        self,
        ax: plt.Axes,
        data: xr.DataArray,
        run_id: str,
        metric: str,
        channels: list[str],
        baseline_data: xr.DataArray | None = None,
    ) -> None:
        """
        Render a single summary card on the given axes.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to draw on
        data : xr.DataArray
            Score data for this run
        run_id : str
            Run identifier
        metric : str
            Metric name
        channels : list[str]
            List of channel names
        baseline_data : xr.DataArray | None
            Baseline data for comparison
        """
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Compute summary statistics
        stats = self._compute_summary_stats(data, channels, metric)

        # Card background
        from matplotlib.patches import FancyBboxPatch

        card_bg = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f8f9fa",
            edgecolor="#dee2e6",
            linewidth=2,
        )
        ax.add_patch(card_bg)

        # Model name header
        ax.text(
            0.5,
            0.92,
            run_id,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#212529",
        )

        # Mean score
        ax.text(
            0.5,
            0.78,
            f"Mean {metric.upper()}: {stats['mean_score']:.3f}",
            ha="center",
            va="top",
            fontsize=10,
            color="#495057",
        )

        # Divider line
        ax.axhline(y=0.70, xmin=0.1, xmax=0.9, color="#dee2e6", linewidth=1)

        # Best/Worst channels
        ax.text(
            0.1,
            0.62,
            f"Best:  {stats['best_channel']} ({stats['best_score']:.3f})",
            ha="left",
            va="top",
            fontsize=9,
            color="#28a745",
        )
        ax.text(
            0.1,
            0.50,
            f"Worst: {stats['worst_channel']} ({stats['worst_score']:.3f})",
            ha="left",
            va="top",
            fontsize=9,
            color="#dc3545",
        )

        # Improvement vs baseline
        if baseline_data is not None:
            baseline_stats = self._compute_summary_stats(baseline_data, channels, metric)
            improvement, color = self._format_improvement(
                stats["mean_score"], baseline_stats["mean_score"], metric
            )

            ax.axhline(y=0.40, xmin=0.1, xmax=0.9, color="#dee2e6", linewidth=1)
            ax.text(
                0.5,
                0.32,
                f"vs Baseline: {improvement}",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
            sparkline_y = 0.22
        else:
            sparkline_y = 0.32

        # Sparkline showing trend across lead times
        if self.show_sparklines and len(stats["trend_values"]) > 1:
            ax.axhline(y=sparkline_y + 0.08, xmin=0.1, xmax=0.9, color="#dee2e6", linewidth=1)
            self._draw_sparkline(ax, stats["trend_values"], stats["trend_times"], metric, sparkline_y)

    def _compute_summary_stats(
        self,
        data: xr.DataArray,
        channels: list[str],
        metric: str,
    ) -> dict:
        """
        Compute summary statistics for a single run.

        Parameters
        ----------
        data : xr.DataArray
            Score data
        channels : list[str]
            List of channel names
        metric : str
            Metric name

        Returns
        -------
        dict
            Dictionary with summary statistics
        """
        # Filter to available channels
        available_channels = [ch for ch in channels if ch in data.channel.values]
        if not available_channels:
            available_channels = list(data.channel.values)

        # Average over all dimensions except channel for per-channel stats
        non_channel_dims = [d for d in data.dims if d != "channel"]
        if non_channel_dims:
            channel_means = data.sel(channel=available_channels).mean(dim=non_channel_dims, skipna=True)
        else:
            channel_means = data.sel(channel=available_channels)

        # Overall mean
        mean_score = float(channel_means.mean(skipna=True).values)

        # Best/worst channels (depends on metric direction)
        channel_values = {ch: float(channel_means.sel(channel=ch).values) for ch in available_channels}
        if lower_is_better(metric):
            best_channel = min(channel_values, key=channel_values.get)
            worst_channel = max(channel_values, key=channel_values.get)
        else:
            best_channel = max(channel_values, key=channel_values.get)
            worst_channel = min(channel_values, key=channel_values.get)

        best_score = channel_values[best_channel]
        worst_score = channel_values[worst_channel]

        # Trend across lead times
        if "lead_time" in data.coords:
            x_dim = "lead_time"
            if "forecast_step" in data.dims:
                trend_data = data.swap_dims({"forecast_step": "lead_time"})
            else:
                trend_data = data
        elif "forecast_step" in data.dims:
            x_dim = "forecast_step"
            trend_data = data
        else:
            x_dim = None
            trend_data = data

        if x_dim and x_dim in trend_data.dims:
            # Average over all dims except x_dim
            other_dims = [d for d in trend_data.dims if d != x_dim]
            if other_dims:
                trend_means = trend_data.mean(dim=other_dims, skipna=True)
            else:
                trend_means = trend_data
            trend_values = trend_means.values
            trend_times = trend_means[x_dim].values
        else:
            trend_values = np.array([mean_score])
            trend_times = np.array([0])

        return {
            "mean_score": mean_score,
            "best_channel": best_channel,
            "worst_channel": worst_channel,
            "best_score": best_score,
            "worst_score": worst_score,
            "trend_values": trend_values,
            "trend_times": trend_times,
        }

    def _draw_sparkline(
        self,
        ax: plt.Axes,
        values: np.ndarray,
        times: np.ndarray,
        metric: str,
        y_position: float,
    ) -> None:
        """
        Draw a mini sparkline showing score trend across lead times.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on
        values : np.ndarray
            Score values
        times : np.ndarray
            Time values
        metric : str
            Metric name
        y_position : float
            Y position for sparkline (in axes coordinates)
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Create inset axes for sparkline
        inset = inset_axes(
            ax,
            width="70%",
            height="25%",
            loc="lower center",
            bbox_to_anchor=(0, y_position - 0.15, 1, 0.3),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        # Normalize x values for plotting
        x = np.arange(len(values))

        # Determine color based on trend direction
        if len(values) > 1:
            slope = (values[-1] - values[0]) / (len(values) - 1) if len(values) > 1 else 0
            # For lower-is-better: increasing slope is bad (red)
            # For higher-is-better: decreasing slope is bad (red)
            if (lower_is_better(metric) and slope > 0) or (not lower_is_better(metric) and slope < 0):
                color = "#dc3545"  # Red - degrading
            else:
                color = "#28a745"  # Green - improving or stable
        else:
            color = "#6c757d"  # Gray

        inset.plot(x, values, color=color, linewidth=1.5)
        inset.fill_between(x, values, alpha=0.2, color=color)

        # Minimal axis styling
        inset.set_xlim(x[0], x[-1])
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
        inset.spines["left"].set_visible(False)
        inset.spines["bottom"].set_linewidth(0.5)
        inset.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True, labelsize=6)

        # Show first and last time labels
        if len(times) > 1:
            if isinstance(times[0], (np.timedelta64, np.datetime64)):
                first_label = f"{int(times[0].astype('timedelta64[h]').astype(int))}h"
                last_label = f"{int(times[-1].astype('timedelta64[h]').astype(int))}h"
            else:
                first_label = str(int(times[0]))
                last_label = str(int(times[-1]))
            inset.set_xticks([x[0], x[-1]])
            inset.set_xticklabels([first_label, last_label])
        else:
            inset.set_xticks([])

    def _format_improvement(
        self,
        current: float,
        baseline: float,
        metric: str,
    ) -> tuple[str, str]:
        """
        Format improvement value and determine color.

        Parameters
        ----------
        current : float
            Current model's score
        baseline : float
            Baseline model's score
        metric : str
            Metric name

        Returns
        -------
        tuple[str, str]
            Formatted improvement string and color
        """
        if baseline == 0:
            return "N/A", "#6c757d"

        pct_change = ((current - baseline) / abs(baseline)) * 100

        # Determine if improvement or degradation based on metric direction
        if lower_is_better(metric):
            # Lower is better: negative change = improvement
            is_improvement = pct_change < 0
        else:
            # Higher is better: positive change = improvement
            is_improvement = pct_change > 0

        if is_improvement:
            color = "#28a745"  # Green
            arrow = "\u2193" if lower_is_better(metric) else "\u2191"
        else:
            color = "#dc3545"  # Red
            arrow = "\u2191" if lower_is_better(metric) else "\u2193"

        return f"{arrow} {abs(pct_change):.1f}%", color
