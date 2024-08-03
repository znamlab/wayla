"""Functions to save diagnostic plots for eye tracking"""

import cv2
import flexiznam as flz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import EllipseModel
from tqdm import tqdm
from znamutils import slurm_it

from . import eye_io, utils


def check_cropping(dlc_ds, camera_ds, rotate180=False, conflicts="skip"):
    """Check cropping of DLC dataset

    Plot a frame from the video with the DLC tracking and the cropping area.

    Args:
        dlc_ds (flexiznam.Dataset): DLC dataset
        camera_ds (flexiznam.Dataset): Camera dataset
        rotate180 (bool, optional): Whether to rotate the image 180 degrees. Defaults
            to False.
        conflicts (str, optional): How to deal with existing crop files. Defaults to
            "skip".

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    dlc_results = pd.read_hdf(dlc_file)

    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if not crop_file.exists() or conflicts != "skip":
        print(f"Creating {crop_file}")
        from cottage_analysis.eye_tracking.eye_tracking import create_crop_file

        create_crop_file(camera_ds, dlc_ds, conflicts=conflicts)

    with open(crop_file, "r") as f:
        crop_info = yaml.safe_load(f)

    cam_data = cv2.VideoCapture(str(video_path))
    fid = int(cam_data.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
    cam_data.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cam_data.read()
    assert ret, f"Could not read frame {fid}"
    cam_data.release()
    if dlc_ds.extra_attributes["cropping"] is not None:
        crop = dlc_ds.extra_attributes["cropping"]
        if isinstance(crop, list):
            frame = frame[crop[2] : crop[3], crop[0] : crop[1]]
        else:
            print(f"Cropping is not a list. It is {crop}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(frame)
    ax.set_title(f"{camera_ds.dataset_name}, frame {fid}")
    frame_tracking = dlc_results.median(axis=0)
    frame_tracking.index = frame_tracking.index.droplevel("scorer")
    if "eye_0" not in frame_tracking:
        eye_tracking = frame_tracking["eye_12"]
    else:
        eye_tracking = frame_tracking["eye_0"]
    ax.scatter(eye_tracking.x, eye_tracking.y, s=20, label="Pupil")

    corners = [
        "temporal_eye_corner",
        "nasal_eye_corner",
        "top_eye_lid",
        "bottom_eye_lid",
    ]
    for color, corner in zip("rgbk", corners):
        data = frame_tracking[corner]
        ax.plot(data["x"], data["y"], marker="o", ls="none", color=color, label=corner)

    rec = plt.Rectangle(
        (crop_info["xmin"], crop_info["ymin"]),
        crop_info["xmax"] - crop_info["xmin"],
        crop_info["ymax"] - crop_info["ymin"],
        fill=False,
        color="orange",
        lw=3,
    )
    ax.add_patch(rec)
    if rotate180:
        ax.invert_yaxis()
        ax.invert_xaxis()
    ax.legend()

    target = dlc_ds.path_full / "diagnostic_cropping.png"
    fig.savefig(target, dpi=300)
    return fig


def plot_dlc_tracking(camera_ds, dlc_ds, likelihood_threshold=None):
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cap = cv2.VideoCapture(str(video_path))
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    raise NotImplementedError("This function is not implemented yet")


@slurm_it(
    conda_env="cottage_analysis",
    slurm_options=dict(
        ntasks=1,
        time="72:00:00",
        mem="16G",
        partition="ncpu",
    ),
)
def plot_ellipse_fit(
    camera_ds_name,
    project,
    likelihood_threshold=None,
    start_frame=None,
    duration=None,
    playback_speed=4,
    vmin=None,
    vmax=None,
    plot_reprojection=False,
):
    """Plot ellipse fit for a given camera dataset

    The generated movie
    Args:
        camera_ds_name (str): Name of the camera dataset
        project (str): Name of the project
        likelihood_threshold (float, optional): Likelihood threshold. Defaults to None.
        start_frame (int, optional): Frame to start plotting from.  If None,
            use the middle of the movie. Defaults to None.
        duration (int, optional): Duration of output movie in seconds. Defaults to None.
        playback_speed (int, optional): Playback speed, relative to original speed.
            Defaults to 4 times faster.
        vmin (float, optional): Minimum value for the colormap. Defaults to None.
        vmax (float, optional): Maximum value for the colormap. Defaults to None.
        plot_reprojection (bool, optional): Whether to plot the reprojection of the
            fitted ellipse. Defaults to False.

    Returns:
        None
    """
    flm_sess = flz.get_flexilims_session(project_id=project)
    camera_ds = flz.Dataset.from_flexilims(
        name=camera_ds_name, flexilims_session=flm_sess
    )
    ds_dict = eye_io.get_tracking_datasets(camera_ds, flexilims_session=flm_sess)
    if ds_dict["cropped"] is None:
        raise IOError("No cropped dataset found")
    dlc_ds = ds_dict["cropped"]
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if likelihood_threshold is None:
        if "likelihood_threshold" in dlc_ds.extra_attributes:
            likelihood_threshold = dlc_ds.extra_attributes["likelihood_threshold"]
        else:
            likelihood_threshold = 1
    if start_frame is None:
        start_frame = frame_count // 2 - 30
    plot_movie(
        camera=camera_ds,
        target_file=dlc_ds.path_full / "ellipse_fit.mp4",
        start_frame=start_frame,
        duration=duration,
        dlc_res=None,
        ellipse=None,
        vmax=vmax,
        vmin=vmin,
        adapt_alpha=False,
        playback_speed=playback_speed,
        crop_border=dlc_ds.extra_attributes["cropping"],
        recrop=False,
        likelihood_threshold=likelihood_threshold,
        plot_reproj=plot_reprojection,
    )


def get_example_frame(
    binned_ellipses,
    camera_ds,
    dlc_ds,
    cropping,
    example_frame=None,
    bin_edges_x=None,
    bin_edges_y=None,
):
    """Get an example frame to use as background for plotting

    Args:
        binned_elipses (pd.DataFrame): Binned ellipse parameters, used if any of
            `dlc_ds`, `bin_edges_x` or `bin_edges_y` is None.
        camera_ds (flexiznam.Dataset): Camera dataset
        dlc_ds (flexiznam.Dataset | pd.DataFrame): DLC dataset or DLC results
        cropping (list): Cropping info for image
        example_frame (int, optional): Frame to use as background. Defaults to None.
        bin_edges_x (np.array, optional): Binning edges for x. Defaults to None.
        bin_edges_y (np.array, optional): Binning edges for y. Defaults to None.

    Returns:
        tuple: (gray, reflection, extent)
    """
    if camera_ds is not None:
        # get one frame in the middle of the recording to use as background
        video_file = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
        cam_data = cv2.VideoCapture(str(video_file))
        if example_frame is None:
            example_frame = int(cam_data.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        cam_data.set(cv2.CAP_PROP_POS_FRAMES, example_frame)
        ret, frame = cam_data.read()
        assert ret, f"Could not read frame {example_frame}"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cropping is not None:
            gray = gray[cropping[2] : cropping[3], cropping[0] : cropping[1]]
        cam_data.release()
    else:
        gray = None

    if (dlc_ds is not None) and (example_frame is not None):
        if isinstance(dlc_ds, flz.Dataset):
            dlc_data = pd.read_hdf(
                dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
            )
            dlc_data.columns = dlc_data.columns.droplevel("scorer")
            dlc_data = dlc_data.loc[example_frame]
        elif isinstance(dlc_ds, pd.DataFrame):
            dlc_data = dlc_ds.loc[example_frame]
        else:
            raise ValueError("dlc_ds must be a Dataset or a DataFrame")
        reflection = np.array([dlc_data.reflection.x, dlc_data.reflection.y])
    else:
        reflection = np.array(
            [
                binned_ellipses.reflection_x.median(),
                binned_ellipses.reflection_y.median(),
            ]
        )
    if (binned_ellipses is None) and (bin_edges_x is None) or (bin_edges_y is None):
        extent = None
    else:
        if bin_edges_x is None:
            bin_edges_x = np.arange(
                binned_ellipses.bin_centre_x.min(),
                binned_ellipses.bin_centre_x.max() + 1,
            )
        if bin_edges_y is None:
            bin_edges_y = np.arange(
                binned_ellipses.bin_centre_y.min(),
                binned_ellipses.bin_centre_y.max() + 1,
            )

        extent = (
            bin_edges_x.min() + reflection[0],
            bin_edges_x.max() + reflection[0],
            bin_edges_y.max() + reflection[1],
            bin_edges_y.min() + reflection[1],
        )
    return gray, reflection, extent


def plot_binned_ellipse_params(
    binned_ellipses,
    ns,
    save_folder,
    min_frame_cutoff=10,
    fig_title=None,
    camera_ds=None,
    example_frame=None,
    dlc_ds=None,
    cropping=None,
    col2plot=("angle", "eccentricity", "minor_radius", "major_radius"),
    bin_edges_x=None,
    bin_edges_y=None,
    angl_clim=None,
):
    """Plot binned ellipse parameters

    Args:
        binned_elipses (pd.DataFrame): Binned ellipse parameters
        ns (pd.DataFrame): Number of frames per bin
        save_folder (Path): Folder to save the figure to
        min_frame_cutoff (int, optional): Minimum number of frames to include a bin.
            Defaults to 10.
        fig_title (str, optional): Title of the figure. Defaults to None.
        camera_ds (flexiznam.Dataset, optional): Camera dataset. Defaults to None.
        example_frame (int, optional): Frame to use as background. Defaults to None.
        dlc_ds (flexiznam.Dataset, optional): DLC dataset to find reflection position
            for example frame. If None will take median reflection position. Defaults
            to None.
        cropping (list, optional): Cropping info for image. Defaults to None.
        col2plot (tuple, optional): Columns to plot. Defaults to
            ("angle", "eccentricity", "minor_radius", "major_radius").
        bin_edges_x (np.array, optional): Binning edges for x. Defaults to None.
        bin_edges_y (np.array, optional): Binning edges for y. Defaults to None.
        angl_clim (tuple, optional): CLimits for the angle plot. Defaults to None.

    Returns:
        None
    """

    gray, reflection, extent = get_example_frame(
        binned_ellipses,
        camera_ds,
        dlc_ds,
        cropping,
        example_frame,
        bin_edges_x,
        bin_edges_y,
    )

    enough_frames = binned_ellipses[ns > min_frame_cutoff].copy()
    enough_frames["eccentricity"] = np.sqrt(
        1 - (enough_frames["minor_radius"] ** 2 / enough_frames["major_radius"] ** 2)
    )

    nplots = len(col2plot)
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.ceil(nplots / nrows))
    fig = plt.figure(figsize=(ncols * 3, nrows * 2.5))

    for ip, p in enumerate(col2plot):
        print(p)
        if p == "angle":
            enough_frames["angle_deg"] = np.rad2deg(enough_frames["angle"])
            p = "angle_deg"
            if angl_clim is None:
                lim = (-90, 90)
            else:
                lim = angl_clim
        else:
            lim = None

        mat = mat_from_binned(enough_frames, value_col=p)

        if lim is None:
            lim = np.nanquantile(mat, [0.01, 0.99])

        ax = fig.add_subplot(nrows, ncols, ip + 1)
        divider = make_axes_locatable(ax)
        if camera_ds is not None:
            ax.imshow(gray, cmap="gray")
        cmap = "viridis" if ip else "twilight"
        img = ax.imshow(mat, vmin=lim[0], vmax=lim[1], cmap=cmap, extent=extent)

        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, orientation="vertical")
        ax.set_title(p)
        if camera_ds is not None:
            ax.set_xlim(0, gray.shape[1])
            ax.set_ylim(gray.shape[0], 0)
    fig.suptitle(fig_title)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1, top=0.8)
    fig.savefig(save_folder / "binned_pupilparams.png", dpi=600)
    return fig


def mat_from_binned(binned_df, value_col):
    """Convert a binned dataframe to a matrix

    Args:
        binned_df (pd.DataFrame): Binned dataframe
        value_col (str): Column to use as values

    Returns:
        np.array: Matrix of values
    """
    ind_col = binned_df.index.levels[0]
    ind_row = binned_df.index.levels[1]
    bin_col = np.arange(ind_col.min(), ind_col.max() + 1)
    bin_row = np.arange(ind_row.min(), ind_row.max() + 1)
    mat = np.zeros((len(bin_row), len(bin_col))) + np.nan
    mat[
        binned_df.index.get_level_values(1) - ind_row.min(),
        binned_df.index.get_level_values(0) - ind_col.min(),
    ] = binned_df[value_col]

    return mat


def plot_eye_centre_estimate(
    eye_centre_binned,
    f_z0_binned,
    camera_ds,
    binned_frames,
    cropping,
    save_folder,
    example_frame=1000,
    bin_edges_x=None,
    bin_edges_y=None,
    dlc_ds=None,
):
    """Plot eye centre estimate"""
    gray, reflection, extent = get_example_frame(
        binned_frames,
        camera_ds,
        dlc_ds,
        cropping,
        example_frame,
        bin_edges_x,
        bin_edges_y,
    )

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10 * gray.shape[0] / gray.shape[1])
    divider = make_axes_locatable(ax)
    img = ax.imshow(gray, cmap="gray")

    # add eccentricity color coded
    binned_frames = binned_frames.copy()
    binned_frames["eccentricity"] = np.sqrt(
        1 - (binned_frames["minor_radius"] ** 2 / binned_frames["major_radius"] ** 2)
    )
    mat = mat_from_binned(binned_frames, "eccentricity")
    img = ax.imshow(mat, cmap="viridis_r", alpha=0.9, extent=extent)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, orientation="vertical")

    for i, series in binned_frames.iterrows():
        origin = np.array([series.pupil_x, series.pupil_y])
        ref = np.array([series.reflection_x, series.reflection_y])
        n_v = np.array(
            [np.cos(series.angle + np.pi / 2), np.sin(series.angle + np.pi / 2)]
        )
        rng = np.array([-200, 200])
        ax.plot(
            *[(origin[a] + ref[a] + n_v[a] * rng) for a in range(2)],
            color="purple",
            alpha=0.2,
            lw=2,
        )
    ax.plot(*(eye_centre_binned + ref), color="k", marker="o")
    eye_binned = mpl.patches.Circle(
        xy=(eye_centre_binned + ref),
        radius=f_z0_binned,
        facecolor="none",
        edgecolor="k",
    )
    ax.add_artist(eye_binned)
    ax.set_xlim(0, gray.shape[1])
    _ = ax.set_ylim(gray.shape[0], 0)

    fig.savefig(save_folder / "eye_centre_estimate.png")
    return fig


def plot_reprojection(
    eye_centre,
    f_z0,
    dlc_res,
    fitted_params,
    fitted_model,
    camera_ds,
    cropping,
    target_file=None,
    initial_model=None,
    initial_eye_centre=None,
    initial_f_z0=None,
    example_frame=None,
):
    """Plot reprojection of fitted ellipse on example frame

    The frame is selected to be as close as possible to the fitted eye centre.

    Args:
        eye_centre (np.array): Estimated eye centre
        f_z0 (float): Estimated f/z0
        dlc_res (pd.DataFrame): DLC results
        fitted_params (tuple): Parameters of the fitted ellipse.
        fitted_model (EllipseModel): Output of the fit.
        camera_ds (flexiznam.Dataset): Camera dataset
        cropping (list): Cropping info for image
        target_file (Path, optional): File name to save figure. Defaults to None.
        initial_model (EllipseModel, optional): Initial model. Defaults to None.
        initial_eye_centre (np.array, optional): Initial eye centre. Defaults to None.
        initial_f_z0 (float, optional): Initial f/z0. Defaults to None.
        example_frame (int, optional): Frame to use as background and for DLC res.
            Defaults to None, taking the frame closest to the fitted eye centre.

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # find example frame with eye close to median position
    fitted_params = fitted_params[
        ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
    ]
    dlc_res = dlc_res.copy()
    if "scorer" in dlc_res.columns.names:
        dlc_res.columns = dlc_res.columns.droplevel("scorer")
    if example_frame is None:
        eye_labels = [f"eye_{i}" for i in range(1, 13)]
        ref = dlc_res["reflection"]
        eye_track = dlc_res[eye_labels]
        eye_track.columns = eye_track.columns.droplevel("bodyparts")
        eye_track = eye_track - ref
        eyex = eye_track["x"].median(axis=1)
        eyey = eye_track["y"].median(axis=1)
        dst = (eyex - fitted_params.pupil_x).abs() + (
            eyey - fitted_params.pupil_y
        ).abs()
        example_frame = dst.idxmin()

    gray, reflection, extent = get_example_frame(
        None, camera_ds, dlc_res, cropping, example_frame
    )
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    # background
    ax.imshow(gray, cmap="gray")
    # dlc track
    circ_coord = EllipseModel.predict_xy(
        None, np.arange(0, 2 * np.pi, 0.1), fitted_params
    ) + reflection.reshape(1, 2)
    ax.plot(circ_coord[:, 0], circ_coord[:, 1], label="DLC fit", color="lightblue")

    # plot eye centre
    ax.plot(*(eye_centre + reflection), color="g", marker="o", label="Eye centre")
    eye_binned = mpl.patches.Circle(
        xy=(eye_centre + reflection),
        radius=f_z0,
        facecolor="none",
        edgecolor="g",
        label=r"$\frac{f}{z_0}$" + f" = {f_z0:.2f}",
    )
    ax.add_artist(eye_binned)

    # ellipse fit
    circ_coord = EllipseModel.predict_xy(
        None, np.arange(0, 2 * np.pi, 0.1), fitted_model
    ) + reflection.reshape(1, 2)
    ax.plot(
        circ_coord[:, 0],
        circ_coord[:, 1],
        label="Reprojection",
        color="purple",
        ls="--",
    )
    if initial_model is not None:
        circ_coord = EllipseModel.predict_xy(
            None, np.arange(0, 2 * np.pi, 0.1), initial_model
        ) + reflection.reshape(1, 2)
        ax.plot(
            circ_coord[:, 0],
            circ_coord[:, 1],
            label="Initial fit",
            color="purple",
            ls=":",
        )
    if initial_eye_centre is not None:
        ax.plot(
            *(initial_eye_centre + reflection),
            color="pink",
            marker="o",
            ls=":",
            label="Initial eye centre",
        )
        if initial_f_z0 is not None:
            eye_binned = mpl.patches.Circle(
                xy=(initial_eye_centre + reflection),
                radius=initial_f_z0,
                facecolor="none",
                edgecolor="pink",
                label=r"$\frac{f}{z_0}$",
            )
        ax.add_artist(eye_binned)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if target_file is not None:
        fig.savefig(target_file)
    return fig


def plot_gaze_fit(
    binned_ellipses,
    eye_rotation,
    error=None,
    camera_ds=None,
    dlc_ds=None,
    cropping=None,
    example_frame=None,
    bin_edges_x=None,
    bin_edges_y=None,
    target_file=None,
):
    """Plot gaze fit on a grid

    Args:
        binned_ellipses (pd.DataFrame): Binned ellipse parameters
        eye_rotation (np.array): Estimated eye rotation (same length as binned_ellipses)
        error (np.array, optional): Error in the fit (same length as binned_ellipses)
            Defaults to None.
        camera_ds (flexiznam.Dataset, optional): Camera dataset. Defaults to None.
        dlc_ds (flexiznam.Dataset, optional): DLC dataset to find reflection position
            for example frame. If None will take median reflection position. Defaults
            to None.
        cropping (list, optional): Cropping info for image. Defaults to None.
        example_frame (int, optional): Frame to use as background. Defaults to None.
        bin_edges_x (np.array, optional): Binning edges for x. Defaults to None.
        bin_edges_y (np.array, optional): Binning edges for y. Defaults to None.
        target_file (Path, optional): File name to save figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure object
    """

    gray, reflection, extent = get_example_frame(
        binned_ellipses,
        camera_ds,
        dlc_ds,
        cropping,
        example_frame,
        bin_edges_x,
        bin_edges_y,
    )

    combined = binned_ellipses.copy()
    combined["phi"] = np.rad2deg(eye_rotation[:, 0])
    combined["theta"] = np.rad2deg(eye_rotation[:, 1])
    combined["radius"] = eye_rotation[:, 2]
    labels = ["phi", "theta", "radius"]

    if error is not None:
        combined["error"] = error
        labels.append("error")

    fig = plt.figure(figsize=(15, 4))
    nplots = len(labels)
    for i in range(nplots):
        mat = mat_from_binned(combined, value_col=labels[i])
        ax = plt.subplot(1, nplots, 1 + i)
        if i < 2:
            cmap = "twilight_shifted"
            absmax = np.nanmax(np.abs(mat))
            vmin, vmax = -absmax, absmax
        else:
            cmap = "viridis"
            vmin, vmax = None, None
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if gray is not None:
            ax.imshow(gray, cmap="gray")
        img = ax.imshow(
            mat, cmap=cmap, extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax.set_title(labels[i])
        if gray is not None:
            ax.set_xlim(0, gray.shape[1])
            ax.set_ylim(gray.shape[0], 0)
        fig.colorbar(img, cax=cax, orientation="vertical")
        if i:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.2, top=0.95)
    if target_file is not None:
        fig.savefig(target_file)
    return fig


def plot_movie(
    camera,
    target_file,
    start_frame=0,
    duration=None,
    dlc_res=None,
    reproj_ds=None,
    ellipse=None,
    vmax=None,
    vmin=None,
    playback_speed=4,
    crop_border=None,
    use_original_encoding=False,
    recrop=False,
    likelihood_threshold=0.88,
    adapt_alpha=True,
    plot_tracking=True,
    plot_ellipse=True,
    plot_reproj=False,
    rotate180=False,
):
    """Plot a movie of raw video, video with dlc tracking and video with ellipse fit

    Args:
        camera (flexiznam.schema.camera_data.CameraData): Camera dataset
        target_file (str): Full path to video output
        start_frame (int, optional): First frame to plot. Defaults to 0.
        duration (float, optional): Duration of video, in seconds. If None, plot until
            the end. Defaults to None.
        dlc_res (pandas.DataFrame, optional): DLC results. Will be loaded if None.
            Defaults to None.
        reproj_ds (pandas.DataFrame, optional): Reprojection results. Required if
            plot_reproj is True. Defaults to None.
        ellipse (pandas.DataFrame, optional): Ellipse fit. Will be loaded if None.
            Defaults to None.
        vmax (int, optional): vmax for video grayscale image. Defaults to None
        vmin (int, optional): vmin for video grayscale image. Defaults to None
        playback_speed (float, optional): playback speed, relative to original video
            speed (which might not be real time). Default to 4.
        crop_border (list, optional): Border to crop video. Defaults to None.
        use_original_encoding (bool, optional): Whether to use original video encoding
            (might not be supported by opencv). Defaults to False.
        recrop (bool, optional): Whether to recrop video. Defaults to False.
        likelihood_threshold (float, optional): Threshold on DLC likelihood use for
            scatter color.
        adapt_alpha (bool, optional): Whether to adapt alpha of scatter points based on
            likelihood. Defaults to True.
        plot_tracking (bool, optional): Whether to plot DLC tracking. Defaults to True.
        plot_ellipse (bool, optional): Whether to plot ellipse fit. Defaults to True.
        plot_reproj (bool, optional): Whether to plot ellipse fit. Defaults to
            False.
        rotate180 (bool, optional): Whether to rotate the image 180 degrees. Defaults
            to False.
    """

    if dlc_res is None or ellipse is None:
        dlc_res, ellipse, dlc_ds = eye_io.get_data(
            camera, flexilims_session=camera.flexilims_session
        )
    # Find DLC crop area
    if recrop:
        borders = np.zeros((4, 2))
        for iw, w in enumerate(
            ("left_eye_corner", "right_eye_corner", "top_eye_lid", "bottom_eye_lid")
        ):
            vals = dlc_res.xs(w, level=1, axis=1)
            vals.columns = vals.columns.droplevel("scorer")
            v = np.nanmedian(vals[["x", "y"]].values, axis=0)
            borders[iw, :] = v

        borders = np.vstack([np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)])
        borders += ((np.diff(borders, axis=0) * 0.1).T @ np.array([[-1, 1]])).T
        borders = borders.astype(int)
    video_file = camera.path_full / camera.extra_attributes["video_file"]

    fig = plt.figure()
    n_axes = 1
    if plot_tracking:
        n_axes += 1
    if plot_ellipse:
        n_axes += 1
    if plot_reproj:
        n_axes += 1
    fig.set_size_inches((3 * n_axes, 3))

    img = get_img_from_fig(fig)
    cam_data = cv2.VideoCapture(str(video_file))
    fps = cam_data.get(cv2.CAP_PROP_FPS)
    if use_original_encoding:
        fcc = int(cam_data.get(cv2.CAP_PROP_FOURCC))
        fcc = (
            chr(fcc & 0xFF)
            + chr((fcc >> 8) & 0xFF)
            + chr((fcc >> 16) & 0xFF)
            + chr((fcc >> 24) & 0xFF)
        )
    else:
        fcc = "mp4v"
    output = cv2.VideoWriter(
        str(target_file),
        cv2.VideoWriter_fourcc(*fcc),
        fps * playback_speed,
        (img.shape[1], img.shape[0]),
    )

    if plot_reproj:
        # plot eye centre
        assert (
            reproj_ds is not None
        ), "reproj_ds must be provided if plot_reproj is True"
        eye_param = dict(np.load(reproj_ds.path_full.with_name("eye_parameters.npz")))
        eye_reproj_by_frame = np.load(reproj_ds.path_full)

    if duration is None:
        nframes = cam_data.get(cv2.CAP_PROP_FRAME_COUNT) - start_frame
    else:
        nframes = int(fps * duration)
    cam_data.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    reflection_df = eye_io.get_reflection(camera_ds=camera, project=camera.project)
    for frame_id in tqdm(np.arange(nframes) + start_frame):
        # set figure
        fig.clear()
        ax_img = fig.add_subplot(1, n_axes, 1)
        axes = [ax_img]
        sub_ref = [False]
        iax = 2
        if plot_tracking:
            ax_track = fig.add_subplot(1, n_axes, iax)
            axes.append(ax_track)
            sub_ref.append(False)
            iax += 1
        if plot_ellipse:
            ax_fit = fig.add_subplot(1, n_axes, iax)
            axes.append(ax_fit)
            sub_ref.append(True)
            iax += 1
        if plot_reproj:
            ax_reproj = fig.add_subplot(1, n_axes, iax)
            axes.append(ax_reproj)
            sub_ref.append(True)
            iax += 1
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # get image
        ret, frame = cam_data.read()
        assert ret, f"Could not read frame {frame_id}"
        if crop_border is not None:
            frame = frame[
                crop_border[2] : crop_border[3], crop_border[0] : crop_border[1]
            ]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if vmin is None:
            vmin = np.percentile(gray, 1)
            print(f"Keeping 1% of pixel black. vmin = {vmin}")
        if vmax is None:
            vmax = np.percentile(gray, 95)
            print(f"Saturating 5% of pixels. vmax = {vmax}")
        img = gray[slice(*borders[:, 1]), slice(*borders[:, 0])] if recrop else gray

        # get reflection
        reflection = reflection_df.loc[frame_id, ["x0", "y0"]].values

        # plot image, center on reflection if needed
        for ax, use_ref in zip(axes, sub_ref):
            extent = np.array([0, img.shape[1], img.shape[0], 0], dtype=float)
            if use_ref:
                extent[:2] -= reflection[0]
                extent[2:] -= reflection[1]
            ax.imshow(
                img,
                cmap="gray",
                vmax=vmax,
                vmin=vmin,
                extent=extent,
            )
            ax.set_yticks([])
            ax.set_xticks([])
            if use_ref:
                ax.set_xlim(-img.shape[1] / 2, img.shape[1] / 2)
                ax.set_ylim(img.shape[0] / 2, -img.shape[0] / 2)

        # plot DLC
        if plot_tracking:
            ax_track = plot_dlc_on_frame(
                ax_track,
                frame_id,
                dlc_res,
                origin="cropped",
                likelihood_threshold=likelihood_threshold,
                adapt_alpha=adapt_alpha,
                left_bottom=(borders[0, 0], borders[0, 1]) if recrop else [0, 0],
            )

        # plot ellipse
        if plot_ellipse:
            # params are xc, yc, a, b, theta
            ax_fit.plot(0, 0, marker="o")
            plot_ellipse_on_frame(
                ax_fit,
                frame_id,
                ellipse,
                origin="reflection",
                dlc_res=dlc_res,
                reflection_fit=reflection_df,
            )
        # plot reprojection
        if plot_reproj:
            # plot eye centre
            ax_reproj.plot(
                *(eye_param["eye_centre"] + reflection),
                color="g",
                marker="o",
                label="Eye centre",
            )
            # and the reproj ellipse
            phitheta = eye_reproj_by_frame[frame_id]
            reprellipse = utils.reproj_ellipse(*phitheta, **eye_param)
            circ_coord = utils.predict_xy(np.arange(0, 2 * np.pi, 0.1), *reprellipse)
            ax_reproj.plot(circ_coord[:, 0], circ_coord[:, 1], label="Reprojection")

        if rotate180:
            for ax in axes:
                ax.invert_yaxis()
                ax.invert_xaxis()
        write_fig_to_video(fig, output)

    cam_data.release()
    output.release()
    print(f"Saved in {target_file}")


def plot_ellipse_on_frame(
    ax,
    frame_id,
    ellipse,
    origin="reflection",
    left_bottom=None,
    dlc_res=None,
    reflection_fit=None,
):
    """Plot ellipse fit on a frame

    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        frame_id (int): Frame id to plot
        ellipse (pandas.DataFrame): Ellipse fits
        origin (str, optional): Origin of the frame. One of "uncropped", "cropped",
            "reflection". Defaults to "reflection".
        left_bottom (np.ndarray, optional): Borders of the frame to use if
            origin="cropped". Defaults to None.
        dlc_res (pandas.DataFrame, optional): DLC results to use if origin="reflection".
            Defaults to None.
        reflection_fit (np.array, optional): Reflection fit to use if
            origin="reflection". Defaults to None.


    Returns:
        matplotlib.axes.Axes: Axes with plot
    """
    if origin == "reflection":
        if reflection_fit is None:
            track = dlc_res.loc[frame_id]
            track.index = track.index.droplevel(["scorer"])
            xs = track.loc[("reflection", "x")]
            ys = track.loc[("reflection", "y")]
        else:
            xs, ys = reflection_fit.loc[frame_id, ["x0", "y0"]]
    elif origin == "uncropped":
        xs, ys = 0, 0
    elif origin == "cropped":
        xs, ys = left_bottom
    else:
        raise ValueError(
            f"Unknown origin {origin}. Must be one of 'reflection',"
            " 'uncropped', 'cropped'"
        )
    params = ellipse.loc[
        frame_id, ["centre_x", "centre_y", "major_radius", "minor_radius", "angle"]
    ]
    circ_coord = utils.predict_xy(np.arange(0, 2 * np.pi, 0.1), *params)
    ax.plot(circ_coord[:, 0] - xs, circ_coord[:, 1] - ys)
    return ax


def plot_dlc_on_frame(
    ax,
    frame_id,
    dlc_res,
    origin="reflection",
    likelihood_threshold=0.8,
    left_bottom=None,
    adapt_alpha=True,
):
    """Plot DLC results on a frame

    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        frame_id (int): Frame id to plot
        dlc_res (pandas.DataFrame): DLC results
        origin (str, optional): Origin of the frame. One of "uncropped", "cropped",
            "reflection". Defaults to "reflection".
        likelihood_threshold (float, optional): Threshold on likelihood. Defaults to
            0.8.
        left_bottom (np.ndarray, optional): Borders of the frame to use if
            origin="cropped". Defaults to None.
        adapt_alpha (bool, optional): Whether to adapt alpha to likelihood. Defaults to
            True.

    Returns:
        matplotlib.axes.Axes: Axes with plot
    """
    track = dlc_res.loc[frame_id]
    track.index = track.index.droplevel(["scorer"])
    xdata = track.loc[[(f"eye_{i}", "x") for i in np.arange(1, 13)]]
    ydata = track.loc[[(f"eye_{i}", "y") for i in np.arange(1, 13)]]
    likelihood = track.loc[[(f"eye_{i}", "likelihood") for i in np.arange(1, 13)]]
    if origin == "reflection":
        xs = track.loc[("reflection", "x")]
        ys = track.loc[("reflection", "y")]
    elif origin == "cropped":
        xs, ys = 0, 0
    elif origin == "uncropped":
        xs, ys = left_bottom
    else:
        raise ValueError(
            f"Unknown origin {origin}. Must be one of 'reflection',"
            " 'uncropped', 'cropped'"
        )
    ax.scatter(
        xdata - xs,
        ydata - ys,
        s=likelihood * 10,
        alpha=likelihood if adapt_alpha else 1,
        color=["g" if lh > likelihood_threshold else "r" for lh in likelihood],
    )
    ax.scatter(
        track.loc[("reflection", "x")] - xs,
        track.loc[("reflection", "y")] - ys,
    )
    return ax


def get_img_from_fig(fig):
    """Get the array from a matplotlib figure

    This is particularly useful to generate videos from matplotlib videos

    Args:
        fig (plt.Figure): figure handle

    Returns:
        image_from_plot (np.array): RGB image from figure

    """
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    return image_from_plot


def write_fig_to_video(fig, video_capture):
    """Save the figure as last frame of an opened video capture

    Use cv2.VideoCapture to create

    Args:
        fig (plt.figure): Matplotlib figure to save
        video_capture (cv2.VideoCapture): video capture object, should be created with
            relevant parameters (fps, codecs, etc...)
    """
    img_array = get_img_from_fig(fig)
    # convert RGB to BGR for cv2
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    video_capture.write(img_array)
