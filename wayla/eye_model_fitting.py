"""
Fitting of eye tracking results


Code adapted from the C++ version https://github.com/LeszekSwirski/singleeyefitter
"""

import warnings

import cv2
import numpy as np
import pandas as pd
from cottage_analysis.eye_tracking import diagnostics, eye_io, utils
from numba_progress import ProgressBar
from skimage.measure import EllipseModel
from tqdm import tqdm


def fit_ellipses(dlc_res_file, likelihood_threshold=None):
    """Fit an ellipse to DLC set of points

    This is the first post-dlc step. Simply find the best ellipse through the 12 points
    tracked on the pupil border

    Args:
        dlc_res_file (pandas.DataFrame or str): DLC data or path to the file containing
            them
        likelihood_threshold (float, optional): Threshold on likelihood to include
            points in fit. Defaults to None.

    Returns:
        pandas.DataFrame: Ellipse dataframe with a line per frame. Failed fit have all
            their parameters to NaN
    """
    if isinstance(dlc_res_file, pd.DataFrame):
        dlc_res = dlc_res_file
    else:
        dlc_res = pd.read_hdf(dlc_res_file)
    ellipse = EllipseModel()
    ellipse_fits = []
    for frame_id, track in tqdm(dlc_res.iterrows(), total=len(dlc_res)):
        # remove the model name
        track = track.copy()
        track.index = track.index.droplevel(0)
        xdata = track.loc[[("eye_{0}".format(pos), "x") for pos in range(1, 13)]]
        ydata = track.loc[[("eye_{0}".format(pos), "y") for pos in range(1, 13)]]
        likelihood = track.loc[
            [("eye_{0}".format(pos), "likelihood") for pos in range(1, 13)]
        ]
        if likelihood_threshold is not None:
            ok = likelihood > likelihood_threshold
            xdata = xdata[ok]
            ydata = ydata[ok]

        xy = np.vstack([xdata.values, ydata.values]).T
        success = ellipse.estimate(xy)
        if not success:
            print("Failed to fit %s" % frame_id, flush=True)
            ellipse_fits.append(
                dict(
                    centre_x=np.nan,
                    centre_y=np.nan,
                    angle=np.nan,
                    major_radius=np.nan,
                    minor_radius=np.nan,
                    error=np.nan,
                    rsquare=np.nan,
                )
            )
            continue
        xc, yc, a, b, theta = ellipse.params
        # It's a mess. see:
        # https://github.com/scikit-image/scikit-image/issues/2646
        # but should be fixed by https://github.com/scikit-image/scikit-image/pull/6943
        if a < b:
            warnings.warn(
                "Ellipse major and minor axis are swapped. Update scikit-image"
            )
            a, b = b, a
            theta += np.pi / 2

        residuals = ellipse.residuals(xy)
        ss_res = np.sum(residuals**2)
        error = ss_res / len(residuals)
        ss_tot = np.sum((xy - np.mean(xy, axis=0)) ** 2)
        rsquare = 1 - ss_res / ss_tot
        ellipse_fits.append(
            dict(
                centre_x=xc,
                centre_y=yc,
                angle=theta,
                major_radius=a,
                minor_radius=b,
                error=error,
                rsquare=rsquare,
            )
        )
    return pd.DataFrame(ellipse_fits)


def estimate_eye_centre(binned_frames, verbose=True):
    """Estimate the eye centre and f/z0 from binned ellipse parameters

    Args:
        binned_frames (pandas.DataFrame): Binned ellipse parameters
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        tuple: (eye_centre, f_z0)
    """
    if verbose:
        print("Find eye centre", flush=True)
    p = np.vstack([binned_frames[f"pupil_{a}"].values for a in "xy"])
    n = np.vstack(
        [np.cos(binned_frames.angle.values), np.sin(binned_frames.angle.values)]
    )
    intercept_minor = utils.pts_intersection(p, n)
    n = np.vstack(
        [
            np.cos(binned_frames.angle + np.pi / 2),
            np.sin(binned_frames.angle + np.pi / 2),
        ]
    )
    axes_ratio = binned_frames.minor_radius.values / binned_frames.major_radius.values
    eye_centre_binned = intercept_minor.flatten()

    delta_pts = (
        np.vstack([binned_frames.pupil_x, binned_frames.pupil_y])
        - eye_centre_binned[:, np.newaxis]
    )
    sum_sqrt_ratio = np.sum(
        np.sqrt(1 - axes_ratio**2) * np.linalg.norm(delta_pts, axis=0)
    )
    sum_sq_ratio = np.sum(1 - axes_ratio**2)
    f_z0_binned = sum_sqrt_ratio / sum_sq_ratio
    if verbose:
        print(rf"Eye centre: {eye_centre_binned}. f/z0: {f_z0_binned}")
    return eye_centre_binned, f_z0_binned


def reproject_ellipses(
    camera_ds,
    target_ds,
    phi0=0,
    theta0=0,
    likelihood_threshold=0.88,
    rsquare_threshold=0.99,
    error_threshold=None,
    min_frame_cutoff=10,
    plot=True,
):
    """Run the reproject_eye function on a camera dataset

    DLC and ellipse fitting must have been done first

    Args:
        camera_ds (flexiznam.schema.camera_data.CameraData): Camera dataset
        target_ds (flexiznam.schema.datasets.Dataset): Target dataset
        phi0 (int, optional): Initial guess for the phi angle. Defaults to 0.
        theta0 (float, optional): Initial guess for the theta angle. Defaults to 0.
        likelihood_threshold (float, optional): Threshold on likelihood to include
            points in fit. Defaults to 0.88.
        rsquare_threshold (float, optional): Threshold on rsquare to include
            points in fit. Defaults to 0.99.
        error_threshold (float, optional): Threshold on error to include points in fit.
            If None, use 5 sd. Defaults to None.
        min_frame_cutoff (int, optional): Minimum number of frames in a bin to include
            it in the fit. Defaults to 10.
        plot (bool, optional): Plot results. Defaults to True.
    """

    # GET DATA
    flm_sess = camera_ds.flexilims_session
    dlc_res, data, dlc_ds = eye_io.get_data(
        camera_ds,
        flexilims_session=flm_sess,
        likelihood_threshold=likelihood_threshold,
        rsquare_threshold=rsquare_threshold,
        error_threshold=error_threshold,
    )
    save_folder = target_ds.path_full.parent
    # shit ellipse in the [-pi/2, pi/2] range
    data["angle_original"] = data["angle"].copy()
    data["angle"] = data["angle"] - np.pi * (data["angle"] > np.pi / 2)

    # BIN ELLIPSES BY POSITION
    print("Bin data", flush=True)
    binned_ellipses, bedg_x, bedg_y = bin_ellipse_by_position(data, nbins=(25, 25))
    enough_frames = binned_ellipses[binned_ellipses.n_frames_in_bin > min_frame_cutoff]
    if plot:
        dlc_tracks = eye_io.get_tracking_datasets(camera_ds, flexilims_session=flm_sess)
        dlc_ds = dlc_tracks["cropped"]
        cropping = dlc_ds.extra_attributes["cropping"]

        diagnostics.plot_binned_ellipse_params(
            binned_ellipses,
            binned_ellipses["n_frames_in_bin"],
            save_folder,
            min_frame_cutoff=min_frame_cutoff,
            fig_title=camera_ds.full_name,
            camera_ds=camera_ds,
            cropping=cropping,
            bin_edges_y=bedg_y,
            bin_edges_x=bedg_x,
        )

    # ESTIMATE EYE CENTER
    eye_centre_binned, f_z0_binned = estimate_eye_centre(enough_frames)
    if plot:
        diagnostics.plot_eye_centre_estimate(
            eye_centre_binned,
            f_z0_binned,
            camera_ds,
            binned_frames=enough_frames,
            cropping=cropping,
            save_folder=save_folder,
            example_frame=1000,
            bin_edges_y=bedg_y,
            bin_edges_x=bedg_x,
        )

    # OPTIMISE ROUND1 for all binned positions
    print("Reproject binned data", flush=True)
    p0 = (float(phi0), float(theta0), 1.0)
    eye_rotation_initial = np.zeros((len(enough_frames), 3))
    error_at_bin_initial = np.zeros(len(enough_frames))
    for i_pos, (pos, s) in tqdm(
        enumerate(enough_frames.iterrows()), total=len(enough_frames)
    ):
        ellipse_params = s[
            ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
        ].values
        p, e, all_errors = utils.minimise_reprojection_error(
            ellipse_params,
            p0=p0,
            eye_centre=eye_centre_binned,
            f_z0=f_z0_binned,
            p_range=(np.pi / 2, np.pi / 2, 1),
            grid_size=10,
            niter=4,
            reduction_factor=2,
        )
        eye_rotation_initial[i_pos] = p
        error_at_bin_initial[i_pos] = e

    if plot:
        diagnostics.plot_gaze_fit(
            binned_ellipses=enough_frames,
            eye_rotation=eye_rotation_initial,
            error=error_at_bin_initial,
            target_file=save_folder / "initial_gaze_fit.png",
            camera_ds=camera_ds,
            dlc_ds=dlc_ds,
            bin_edges_y=bedg_y,
            bin_edges_x=bedg_x,
            example_frame=None,
            cropping=cropping,
        )
        # Plot fit of median position
        params_most_frequent_bin = enough_frames.loc[
            enough_frames.n_frames_in_bin.idxmax()
        ]
        fit_most_frequent_bin = eye_rotation_initial[
            enough_frames.n_frames_in_bin.values.argmax()
        ]
        initial_model = utils.reproj_ellipse(
            phi=fit_most_frequent_bin[0],
            theta=fit_most_frequent_bin[1],
            r=fit_most_frequent_bin[2],
            eye_centre=eye_centre_binned,
            f_z0=f_z0_binned,
        )

        diagnostics.plot_reprojection(
            eye_centre_binned,
            f_z0_binned,
            dlc_res,
            fitted_params=params_most_frequent_bin,
            fitted_model=initial_model,
            cropping=cropping,
            camera_ds=camera_ds,
            target_file=save_folder / "initial_reprojection_median_eye_position.png",
        )

    # OPTIMISE EYE PARAMETERS
    print("Optimise eye parameters", flush=True)
    source_ellipses = enough_frames.loc[
        :, ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
    ].values
    (x, y, f_z0), err = utils.optimise_eye_parameters(
        ellipses=source_ellipses,
        gazes=eye_rotation_initial,
        p0=(*eye_centre_binned, f_z0_binned),
        p_range=(100.0, 100.0, 100.0),
        grid_size=10,
        niter=2,
        reduction_factor=5,
        verbose=True,
        prange_inner=(np.pi / 5, np.pi / 5, 1),
        niter_inner=2,
        grid_size_inner=5,
        refit_from_p0=False,
        debug=False,
    )
    eye_centre = np.array([x, y])

    if plot:
        print("Reproject binned data", flush=True)
        eye_rotation_optimised = np.zeros((len(enough_frames), 3))
        error_at_bin = np.zeros(len(enough_frames))
        for i_pos, (pos, s) in tqdm(
            enumerate(enough_frames.iterrows()), total=len(enough_frames)
        ):
            ellipse_params = s[
                ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
            ].values
            p, e, all_errors = utils.minimise_reprojection_error(
                ellipse_params,
                p0=p0,
                eye_centre=eye_centre,
                f_z0=f_z0,
                p_range=(np.pi / 2, np.pi / 2, 1),
                grid_size=10,
                niter=5,
                reduction_factor=3,
            )
            eye_rotation_optimised[i_pos] = p
            error_at_bin[i_pos] = e

        diagnostics.plot_gaze_fit(
            binned_ellipses=enough_frames,
            eye_rotation=eye_rotation_optimised,
            error=error_at_bin,
            target_file=save_folder / "optimised_gaze_fit.png",
            camera_ds=camera_ds,
            dlc_ds=dlc_ds,
            bin_edges_y=bedg_y,
            bin_edges_x=bedg_x,
            example_frame=None,
            cropping=cropping,
        )

        fit_most_frequent_bin_opti = eye_rotation_optimised[
            enough_frames.n_frames_in_bin.values.argmax()
        ]
        optimised_model = utils.reproj_ellipse(
            phi=fit_most_frequent_bin_opti[0],
            theta=fit_most_frequent_bin_opti[1],
            r=fit_most_frequent_bin_opti[2],
            eye_centre=eye_centre,
            f_z0=f_z0,
        )
        diagnostics.plot_reprojection(
            eye_centre,
            f_z0,
            dlc_res,
            fitted_params=params_most_frequent_bin,
            fitted_model=optimised_model,
            cropping=cropping,
            camera_ds=camera_ds,
            target_file=save_folder / "optimised_reprojection_median_eye_position.png",
        )

    # SAVE Eye parameters
    print("Saving eye parameters", flush=True)
    np.savez(
        save_folder / "eye_parameters.npz",
        eye_centre=eye_centre,
        f_z0=f_z0,
    )

    # OPTIMISE FOR ALL FRAMES
    print("Fitting all frames", flush=True)
    # create a list of all ellipse params
    parameters = data[
        ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
    ].values
    with ProgressBar(total=len(parameters)) as progress:
        eye_rotation = utils.minimise_all(
            parameters,
            p0,
            eye_centre,
            f_z0,
            progress,
            p_range=(1.0, 1.0, 0.5),
            grid_size=10,
            niter=3,
            reduction_factor=3,
        )
    np.save(target_ds.path_full, eye_rotation)
    print("Done!")


def bin_ellipse_by_position(data, nbins=(25, 25)):
    """Bin ellipse parameters by position

    Args:
        data (pandas.DataFrame): Ellipse parameters
        nbins (tuple, optional): Number of bins in x and y. Defaults to (25, 25).

    Returns:
        pandas.DataFrame: Binned ellipse parameters
        numpy.array: Bin edges in x
        numpy.array: Bin edges in y
    """
    elli = pd.DataFrame(data[data.valid], copy=True)
    nbins = (25, 25)
    bin_edges_x = np.linspace(elli.pupil_x.min(), elli.pupil_x.max(), nbins[0])
    bin_edges_y = np.linspace(elli.pupil_y.min(), elli.pupil_y.max(), nbins[1])
    bin_width = np.diff(bin_edges_x)[0]
    bin_height = np.diff(bin_edges_y)[0]
    bin_edges_x = np.hstack((bin_edges_x, bin_edges_x[-1] + bin_width))
    bin_edges_y = np.hstack((bin_edges_y, bin_edges_y[-1] + bin_height))
    # find the start of the bin that each ellipse belongs to
    elli["bin_id_x"] = bin_edges_x.searchsorted(elli.pupil_x.values)
    elli["bin_id_y"] = bin_edges_y.searchsorted(elli.pupil_y.values)
    elli["bin_centre_x"] = bin_edges_x[elli.bin_id_x.values] + bin_width / 2
    elli["bin_centre_y"] = bin_edges_y[elli.bin_id_y.values] + bin_height / 2

    binned_ellipses = elli.groupby(["bin_id_x", "bin_id_y"])
    ns = binned_ellipses.valid.aggregate(len)
    binned_ellipses = binned_ellipses.aggregate(np.nanmedian)
    binned_ellipses["n_frames_in_bin"] = ns
    return binned_ellipses, bin_edges_x, bin_edges_y


def convert_to_world(gaze_vec, rvec):
    """Convert gaze vectors from camera to world coordinates

    Convert rvec to matrix and multiply the gaze vector to get the gaze vector in
    aruco == world coordinates

    Args:
        gaze_vec (numpy.array): N x 3 array of gaze in camera coordinate
        rvec (numpy.array): rvec from extrinsics

    Returns:
        numpy array: N x 3 array
    """
    rmat = cv2.Rodrigues(rvec)[0]
    gaze_vec = np.array(gaze_vec, copy=True)
    rotated_gaze_vec = (rmat.T @ gaze_vec.T).T
    return rotated_gaze_vec


def gaze_to_azel(gaze_vector, zero_median=False, worled_is_mirrored=False):
    """Transform gaze vectors in world coordinates to Azimuth and Elevation

    This assumes that the gaze vector come in the aruco reference frame , with y
    pointing in front of the mouse, x to the right and z up
    Except if `world_is_mirrored` is True, in which case the aruco has been wrongly
    oriented and y points to the right, x to the left and z up. We therefore need to
    flip the x and y coordinates and reverse them both.


    Args:
        gaze_vector (numpy.array): N x 3 array of gaze
        zero_median (bool, optional): Subtract the median. Defaults to False.
        worled_is_mirrored (bool, optional): Whether the world is mirrored. Defaults to
            False.

    Returns:
        azimuth (numpy.array): len(N) array of azimuth in radians in the range [-pi, pi]
        elevation (numpy.array): len(N) array of elevation in radians
    """
    if worled_is_mirrored:
        print("Mirrored world")
        gaze_vector[:, :2] *= -1
        gaze_vector = gaze_vector[:, [1, 0, 2]]

    azimuth = np.arctan2(gaze_vector[:, 1], gaze_vector[:, 0])
    elevation = np.arctan2(
        gaze_vector[:, 2], np.sqrt(np.sum(gaze_vector[:, :2] ** 2, axis=1))
    )
    # zero the median pos
    if zero_median:
        azimuth -= np.nanmedian(azimuth)
        elevation -= np.nanmedian(elevation)
        # put back in -pi pi
        azimuth = np.mod(azimuth + np.pi, 2 * np.pi) - np.pi
        elevation = np.mod(elevation + np.pi, 2 * np.pi) - np.pi
    else:
        # rotate by 90 degrees to put azimuth facing the mouse (instead of right)
        azimuth += np.pi / 2
        azimuth = np.mod(azimuth + np.pi, 2 * np.pi) - np.pi

    return azimuth, elevation


def fit_globe(dlc_res, camera_ds, cropping, local_threshold=65, block_size=3):
    """Fit a circle to the eye border

    Args:
        dlc_res (pandas.DataFrame): DLC results
        camera_ds (flexiznam.schema.camera_data.CameraData): Camera dataset
        cropping (tuple): Cropping of the image
        local_threshold (int, optional): Local threshold for segmentation. Defaults to
            65.
        block_size (int, optional): Block size for segmentation. Defaults to 3.

    Returns:
        numpy.array: N x 3 array of circle parameters
        numpy.array: N array of errors
    """
    circle_fit = np.zeros((len(dlc_res), 3)) + np.nan
    errors_fit = np.zeros(len(dlc_res)) + np.nan
    video_file = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cam_data = cv2.VideoCapture(str(video_file))

    for frame_id in tqdm(range(len(dlc_res))):
        ret, frame = cam_data.read()
        assert ret, f"Could not read frame {frame_id}"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped = gray[cropping[2] : cropping[3], cropping[0] : cropping[1]]
        segmented_img = utils.segment_eye_border(
            cropped, local_threshold=local_threshold, block_size=block_size
        )
        if segmented_img is None:
            continue
        y, x = np.where(segmented_img)
        if len(x) < 10:
            continue
        out = utils.fit_circle(x, y)
        circle_fit[frame_id, :] = out[:3]
        errors_fit[frame_id] = out[3] / len(x)
    cam_data.release()
    return circle_fit, errors_fit
