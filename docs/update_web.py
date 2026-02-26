"""Script to run the UNET model on SSH data from Copernicus Marine Service and plot the results."""
# /// script
# dependencies = [
#   "copernicusmarine",
#   "cartopy",
#   "cmocean",
#   "matplotlib",
#   "scikit-image",
#   "scipy",
#   "torch",
#   "xarray",
# ]
# ///

import datetime
import logging
import glob
import os
import sys
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import copernicusmarine
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d, map_coordinates
from skimage import measure

sys.path.append("./")  # use function from the test_modes.py file
from test_models import get_best_device, load_model, test_model

device = get_best_device()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mpl.use("agg")  # Use a non-interactive backend for rendering


def add_colorbar(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    var: mpl.cm.ScalarMappable,
    fmt: Optional[str] = None,
    range_limit: Optional[List[float]] = None,
) -> mpl.colorbar.Colorbar:
    """Add a colorbar to the figure and format it.

    Args:
        fig: Matplotlib Figure object.
        ax: Matplotlib Axes object.
        var: The mappable object (e.g., result of pcolormesh).
        fmt: Optional string format for the colorbar.
        range_limit: Optional [min, max] for colorbar limits.

    Returns:
        The created colorbar object.

    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
    cb = fig.colorbar(var, cax=cax, format=fmt)
    if range_limit is not None:
        cb.mappable.set_clim(range_limit)
    cb.ax.tick_params(which="major", labelsize=6, length=3, width=0.5, pad=0.05)
    return cb


def deep_gom_mean(ds: xr.Dataset) -> Tuple[float, np.ndarray]:
    """Calculate the mean of the data for the deep Gulf of Mexico region.

    Args:
        ds: Last week of SSH data as an xarray Dataset.

    Returns:
        Tuple containing the mean value of ADT in the deep GoM and the mask used.

    """
    adt = ds["adt"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # I resample ETOPO 60min on a coarser grid to save space
    # Source:
    # https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/60s/60s_surface_elev_netcdf/catalog.html
    #
    # etopo_coarse = (
    #     xr.open_dataset("ETOPO_2022_v1_60s_N90W180_surface.nc")
    #     .sel(
    #         lon=slice(-100, -74),
    #         lat=slice(17, 33),
    #     )
    #     .coarsen(
    #         lon=8,
    #         lat=8,
    #         boundary="trim",
    #     )
    #     .mean()
    #     .to_netcdf("etopo_gom_coarse.nc")
    # )
    etopo = xr.open_dataset("data/etopo_gom_coarse.nc")
    depth = etopo["z"].interp(lon=lon, lat=lat, method="linear")

    # Deep GoM mask (depth < -200m)
    deep_gom_mask = np.where(depth < -200, 1, 0)

    # Remove Caribbean & Atlantic GoM
    external_bnd = np.array(
        [
            [-87.5, 21.15],
            [-84.15, 22.35],
            [-82.9, 22.9],
            [-81, 22.9],
            [-81, 27],
            [-82.5, 32.5],
            [-74.5, 32.5],
            [-74.5, 16.5],
            [-90, 16.5],
            [-87.5, 21.15],
        ]
    )
    caribbean_path = path.Path(external_bnd)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
    gom_mask = ~caribbean_path.contains_points(points).reshape(lon_grid.shape)

    # inside gom and deep gom intersection
    mask = np.logical_and(deep_gom_mask, gom_mask)
    mean_value = adt[:, mask == 1].mean()

    return mean_value, mask


def normalize(ds: xr.Dataset) -> np.ndarray:
    """Normalize the data by removing the mean of the deep GoM.

    Args:
        ds: Last week of SSH data as an xarray Dataset.

    Returns:
        Normalized ADT data as a numpy array.

    """
    data = ds["adt"].values.copy()
    mean_value, _ = deep_gom_mean(ds)
    data -= mean_value
    return data


def add_eddy_contours(
    ax: mpl.axes.Axes,
    lon1d: np.ndarray,
    lat1d: np.ndarray,
    prediction: np.ndarray,
    color: str = "black",
    linewidth: float = 1.0,
    linestyle: str = "solid",
) -> None:
    """Add smoothed eddy contours to the plot.

    Args:
        ax: Matplotlib Axes object.
        lon1d: 1D longitude array.
        lat1d: 1D latitude array.
        prediction: 2D mask/prediction array.
        color: Line color.
        linewidth: Line width.
        linestyle: Line style.

    """
    mask = np.nan_to_num(prediction)
    contours = measure.find_contours(mask, 0.90)
    lon_grid, lat_grid = np.meshgrid(lon1d, lat1d)

    for contour in contours:
        x = gaussian_filter1d(contour[:, 0], sigma=3)
        y = gaussian_filter1d(contour[:, 1], sigma=3)
        smoothed = np.stack([y, x], axis=1)
        if not np.allclose(smoothed[0], smoothed[-1]):
            smoothed = np.vstack([smoothed, smoothed[0]])
        row = smoothed[:, 1]
        col = smoothed[:, 0]
        coords = np.vstack([row, col])
        lon = map_coordinates(lon_grid, coords, order=1, mode="nearest")
        lat = map_coordinates(lat_grid, coords, order=1, mode="nearest")
        c = np.stack([lon, lat], axis=1)
        ax.plot(
            c[:, 0],
            c[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label="Contours",
        )


def plot_gom_figure(
    day: datetime.datetime,
    ds: xr.open_dataset,
    prediction: np.ndarray,
    lon: List[float],
    lat: List[float],
) -> None:
    """Create and save a figure with SSH data and eddy predictions.

    Args:
        day: Date for the plot.
        ds: Xarray dataset containing SSH data.
        prediction: Model prediction (2D array).
        lon: Longitude range [min, max].
        lat: Latitude range [min, max].

    """
    plot_crs = ccrs.PlateCarree()
    ticks = [4, 4]
    contour_value = 0.55  # fix contour for Loop Current height

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=plot_crs)
    ax.set_title(f"{day.strftime('%Y-%m-%d')}")

    pcm = ax.pcolormesh(
        np.mod(ds["longitude"], 180) - 180,
        ds["latitude"],
        ds["adt"],
        transform=ccrs.PlateCarree(),
        cmap=cmocean.cm.balance,
        shading="gouraud",
    )
    cb = add_colorbar(fig, ax, pcm)
    cb.set_label("Sea Surface Height [m]", fontsize=8)

    # add LC isoline
    ax.contour(
        np.mod(ds["longitude"], 180) - 180,
        ds["latitude"],
        ds["adt"],
        [contour_value],
        colors="black",
        linestyles="solid",
        linewidths=1,
    )

    # add identified eddies contour
    add_eddy_contours(
        ax,
        ds["longitude"].values,
        ds["latitude"].values,
        prediction,
        color="black",
        linestyle="dashed",
    )

    ax.legend(
        handles=[
            mlines.Line2D([], [], linestyle="solid", color="black", label="LC isoline"),
            mlines.Line2D(
                [], [], linestyle="dashed", color="black", label="Identified eddies (UNET)"
            ),
        ],
        loc="upper left",
        fontsize=7,
        frameon=False,
    )

    ax.add_feature(cfeature.LAND, facecolor="dimgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(lon[0], lon[1] + 1e-6, ticks[0]), crs=plot_crs)
    ax.set_yticks(np.arange(lat[0], lat[1], ticks[1]), crs=plot_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xlim(lon)
    ax.set_ylim(lat)

    fig.savefig(f"figs/{day.strftime('%Y-%m-%d')}.png", bbox_inches="tight", dpi=600)


def download_ssh_data(day: datetime.datetime, lon: List[float], lat: List[float]) -> None:
    """Download SSH data from Copernicus Marine Service.

    Args:
        day: Today's date.
        lon: Longitude range [min, max].
        lat: Latitude range [min, max].

    """
    prev_day = day - datetime.timedelta(days=7)
    data_file = f"{day.strftime('%Y-%m-%d')}.nc"

    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        variables=["adt", "ugos", "vgos"],
        minimum_longitude=lon[0],
        maximum_longitude=lon[1],
        minimum_latitude=lat[0],
        maximum_latitude=lat[1],
        start_datetime=prev_day.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=day.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0,
        maximum_depth=0,
        output_filename=data_file,
        output_directory="data/",
    )


def clean_old_images(folder_path, files_to_keep):
    """Identify and remove old image files using glob and sorting by modification time.

    Assumes all .png files in the folder are dated images.
    This version automatically deletes files without confirmation.

    Args:
        folder_path (str): The path to the folder to clean.
        files_to_keep (int): The number of most recent files to keep.

    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return

    print(f"Scanning folder: {folder_path}")

    # Use glob to find all .png files
    # glob.glob returns full paths
    all_png_files = sorted(glob.glob(f"{folder_path}/*.png"), reverse=True)

    # Identify files to keep and files to delete
    files_to_delete = all_png_files[files_to_keep:]

    if not files_to_delete:
        print("No old files found to delete.")
        return

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"  Deleted: {os.path.basename(file_path)}")
        except OSError as e:
            print(f"  Error deleting {os.path.basename(file_path)}: {e}")
    print("\nCleanup complete.")


def run_model(input_data: List[Dict[str, Any]]) -> np.ndarray:
    """Run the model with the specified parameters and return prediction.

    Args:
        input_data: List of dictionaries with input data.

    Returns:
        Model prediction as a numpy array.

    """
    inputs = "only_ssh"
    days_before = 2
    lcv_length = 5
    test_id = 0

    weights_file = f"model_weights_DaysBefore_{days_before:02d}_DaysAfter_{lcv_length:02d}_DaysCoherent_{lcv_length:02d}.pth"
    weights_folder = (
        f"DaysBefore_{days_before:02d}_DaysAfter_{lcv_length:02d}_DaysCoherent_{lcv_length:02d}"
    )
    model_weights_path = join(
        "../model_weights",
        "EddyDetection_ALL_1998-2022_gaps_filled_submean_sst_chlora"
        if inputs == "ssh_sst_chlora"
        else "EddyDetection_ALL_1993-2022_gaps_filled_submean_only_ssh",
        weights_folder,
        weights_file,
    )

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")

    model = load_model(
        model_weights_path,
        input_type=inputs,
        days_before=days_before,
        days_after=lcv_length,
    )

    input_data_cur_date = input_data[test_id]["data"][0]
    prediction = test_model(model, input_data_cur_date[None, ...], device=device)
    return prediction.squeeze()


if __name__ == "__main__":
    # Set Copernicus username and password as environment variables or directly here
    copernicusmarine.login(
        username=os.getenv("COPERNICUS_USER"),
        password=os.getenv("COPERNICUS_PASS"),
        force_overwrite=True,
    )

    # Gulf of Mexico region
    lon = [-99, -75]
    lat = [18, 32]

    for i in range(1, 8):
        day = datetime.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        day = day - datetime.timedelta(days=i)  # for testing, use 2 days before
        data_file = f"data/{day.strftime('%Y-%m-%d')}.nc"

        if not Path(data_file).exists():
            logger.info("Downloading data")
            download_ssh_data(day, lon, lat)

        if Path(data_file).exists():
            logger.info("Data file exists, now running model")

            # prepare input data
            ds = xr.open_dataset(data_file)
            input_data_values = normalize(ds)
            input_data_values[np.isnan(input_data_values)] = 0
            input_data = [{"file": "7 days before today", "data": (input_data_values,)}]

            # run model
            prediction = run_model(input_data)

            # plot
            ds_plot = ds.isel(time=2)  # 2 days before, 5 days after
            day_plot = day - datetime.timedelta(days=5)
            plot_gom_figure(day_plot, ds_plot, prediction, lon, lat)
        else:
            logger.error("Data file does not exist.")

        # cleanup
        clean_old_images("figs/", 7)
