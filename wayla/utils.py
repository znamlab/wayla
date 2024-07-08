"""Functions that help dealing with ellipse and optimisation

This contains the simple math utils and the grid search functions

"""

import math

import cv2
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from scipy.optimize import leastsq


@njit
def reproj_centre(phi, theta, eye_centre, f_z0):
    """Reproject ellipse centre on camera frame

    Wallace and Kerr method.

    There is an extra minus 1 in the y of the centre reprojection compared to their
    methods to have the camera y axis pointing down

    Args:
        phi (float): Vertical angle in radians
        theta (float): Horizontal angle in radians
        eye_centre (numpy.array): x,y position of eye centre
        f_z0 (float): Scale factor

    Returns:
        numpy.array: X, Y of pupil centre in camera coordinates
    """

    return f_z0 * np.array([np.sin(theta), -np.sin(phi) * np.cos(theta)]) + eye_centre


@njit
def reproj_ellipse(phi, theta, r, eye_centre, f_z0):
    """Reproject ellipse on camera frame

    Wallace and Kerr method

    Args:
        phi (float): Vertical angle in radians
        theta (float): Horizontal angle in radians
        r (float): Radius of pupil in units of f_z0
        eye_centre (numpy.array): x,y position of eye centre
        f_z0 (float): Scale factor

    Returns:
        EllipseModel: Ellipse in camera coordinates
    """
    w3 = -np.cos(phi) * np.cos(theta)
    major = r * f_z0
    minor = np.abs(w3) * major
    # from Wallace et al:
    if np.sin(phi) != 0:
        angle = np.arctan(np.tan(theta) / np.sin(phi))
    else:
        angle = np.pi / 2 * np.sign(np.tan(theta))
    centre = reproj_centre(phi, theta, eye_centre, f_z0)
    if False:
        # one could also look at the angle to centre
        vect = centre - eye_centre
        angle = np.arcsin(vect[0] / np.linalg.norm(vect))

    # params are xc, yc, a, b, theta
    ellipse = (centre[0], centre[1], major / 2, minor / 2, angle)
    return ellipse


@njit
def ellipse_distance(model1, model2, ev_pts=None):
    """Compute the distance between two ellipses

    This is done by summing the distances of points along the border

    Args:
        model1 (tuple): First ellipse
        model2 (tuple): Second ellipse
        ev_pts (numpy.array, optional): Angles to use for comparison. If None will do
            a full circle in pi/12 increment. Defaults to None.

    Returns:
        float: Error as sum of distances
    """
    if ev_pts is None:
        ev_pts = np.arange(0, 2 * np.pi, np.pi / 12)
    xc, yc, a, b, angle = model1
    pts1 = predict_xy(ev_pts, xc, yc, a, b, angle)
    xc, yc, a, b, angle = model2
    pts2 = predict_xy(ev_pts, xc, yc, a, b, angle)

    total_dst = 0.0
    for i in range(len(pts1)):
        min_dist = np.inf
        for j in range(len(pts2)):
            dist = np.sqrt(np.sum((pts1[i] - pts2[j]) ** 2))
            if dist < min_dist:
                min_dist = dist
        total_dst += min_dist
    return total_dst


@njit
def _roll(arr, shift):
    """Roll an array along the first axis

    Modified from https://stackoverflow.com/questions/61011294/numba-jit-can-t-compile-np-roll
    """
    b = np.empty_like(arr)
    rows_num = arr.shape[0]
    cols_num = arr.shape[1]
    for i in range(cols_num):
        b[shift:, i] = arr[: rows_num - shift, i]
        b[:shift, i] = arr[rows_num - shift :, i]
    return b


def pts_intersection(pts, normals):
    """Find best intersection of lines in 2D

    See:
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_two_dimensions_2

    Args:
        pts (numpy.array): 2 x N array of points on the lines
        normals (numpy.array): 2 x N array of normals to the lines

    Returns:
        numpy.array: (x, y) of least-square solution
    """
    n_nt = normals.T[:, :, np.newaxis] @ normals.T[:, np.newaxis, :]
    inv_sum = np.linalg.inv(np.sum(n_nt, axis=0))
    direct_sum = np.sum(n_nt @ pts.T[:, :, np.newaxis], axis=0)
    return inv_sum @ direct_sum


@njit
def minimise_reprojection_error(
    ellipse,
    p0,
    eye_centre,
    f_z0,
    p_range=(1.0, 1.0, 0.5),
    grid_size=10,
    niter=3,
    reduction_factor=3,
    debug=False,
):
    """Iterative grid search of best gaze vector to minimize reprojection error

    Args:
        ellipse (tuple): Ellipse to fit, provided either as (x,y, major, minor, angle)
            tuple of parameters
        p0 (tuple): Starting estimates of parameter (phi, theta, radius), centre of grid
        eye_centre (numpy.array): x,y of eye centre in camera coordinate
        f_z0 (float): scale factor
        p_range (tuple, optional): range of grid for the 3 parameters. Defaults to
            (1, 1, 0.5)
        grid_size (int, optional): number of values for each level of the grid.
            Defaults to 10.
        niter (int, optional): number of iteration. Defaults to 3
        reduction_factor (int, optional): reduction of p_range at each iteration.
            Defaults to 5
        verbose (bool, optional): Print progress. Default to True.
        debug (bool, optional): Return debug info. Defaults to False.

    Returns:
        parameters (tuple): Best gaze parameters (phi, theta, radius)
        min_ind (tuple): Index of minimal error in grid for (phi, theta, radius)
        error (numpy array): len(grid_phi) x len(grid_theta) x len(grid_radius) array
            of reprojection errors
    """
    params = tuple(p0)
    errors = []
    for i_iter in range(niter):
        grids = []
        for p, r in zip(params, p_range):
            rng = np.empty(grid_size)
            if grid_size > 1:
                rng[:] = np.linspace(-r, r, grid_size)
            grids.append(rng + p)
        # ensure last parameter is positive
        grids[-1] = grids[-1][grids[-1] > 0]
        params, error, errors_iter = grid_search_best_gaze(
            ellipse,
            eye_centre=eye_centre,
            f_z0=f_z0,
            grid_phi=grids[0],
            grid_theta=grids[1],
            grid_radius=grids[2],
            debug=debug,
        )
        if debug:
            errors.append((grids, errors_iter))

        p_range = (
            p_range[0] / reduction_factor,
            p_range[1] / reduction_factor,
            p_range[2] / reduction_factor,
        )
    return params, error, errors


@njit
def grid_search_best_gaze(
    source_ellipse, eye_centre, f_z0, grid_phi, grid_theta, grid_radius, debug=False
):
    """Grid search of best gaze vector to minimize reprojection error

    Args:
        source_ellipse (tuple): Ellipse to fit, (x,y, major, minor, angle) tuple of
            parameters
        eye_centre (numpy.array): x,y of eye centre in camera coordinate
        f_z0 (float): scale factor
        grid_phi (numpy.array): Values of phi for grid search
        grid_theta (numpy.array): Values of theta for grid search
        grid_radius (numpy.array): Values of radius for grid search
        debug (bool, optional): Return debug info. Defaults to False.

    Returns:
        parameters (tuple): Best gaze parameters (phi, theta, radius)
        min_ind (tuple): Index of minimal error in grid for (phi, theta, radius)
        error (numpy array): len(grid_phi) x len(grid_theta) x len(grid_radius) array
            of reprojection errors
    """
    params = (0, 0, 0)
    error = np.inf
    errors = np.zeros((len(grid_phi), len(grid_theta), len(grid_radius)))
    for ip, phi in enumerate(grid_phi):
        for it, theta in enumerate(grid_theta):
            for ir, r in enumerate(grid_radius):
                el = reproj_ellipse(phi, theta, r, eye_centre=eye_centre, f_z0=f_z0)
                dst = ellipse_distance(source_ellipse, el)
                if debug:
                    errors[ip, it, ir] = dst

                if dst < error:
                    error = dst
                    params = (phi, theta, r)
    return params, error, errors


@njit(parallel=True, nogil=True)
def grid_search_best_eye(
    source_ellipses,
    ellipse_fits,
    grid_eye_x,
    grid_eye_y,
    grid_f_z0,
    progress_proxy,
    p_range=(np.deg2rad(30), np.deg2rad(30), 0.2),
    niter=3,
    grid_size=5,
    refit_from_p0=True,
):
    """Find best eye parameters on the given grid

    Grid search on eye parameters (centre x, y and f/z0 scale). For each combination,
    optimise phi/theta/radius for all source_ellipses and sum reprojection errors

    Args:
        source_ellipses (list): List of ellipses or ellipse parameter, input data
        ellipse_fits (list): List of phi/theta/radius parameters to initial search for
            each source_ellipse
        grid_eye_x (numpy.array): List of x values to test
        grid_eye_y (numpy.array): List of y values to test
        grid_f_z0 (numpy.array): List of f_z0 values to test
        progress_proxy (tqdm.tqdm, optional): Progress bar.
        p_range (tuple, optional): Range of phi/theta/radius to search for each
            source_ellipse. Defaults to (np.deg2rad(30), np.deg2rad(30), 0.2).
        niter (int, optional): Number of iteration for each source_ellipse.
            Defaults to 3.
        grid_size (int, optional): Number of values for each level of the grid.
            Defaults to 5.
        refit_from_p0 (bool, optional): Start inner search from p0. Defaults to False.


    Returns:
        params (tuple): Best (x, y, f_z0) eye parameters
        index (tuple): Index of best parameter in grid
        errors (numpy.array): Matrix of error for each position in the grid
    """

    best_eye = (0, 0, 0)
    best_error = np.inf
    eye_params = []
    for x in grid_eye_x:
        for y in grid_eye_y:
            for fz in grid_f_z0:
                eye_params.append((x, y, fz))
    errors = np.zeros(len(eye_params))

    for ip in prange(len(eye_params)):
        progress_proxy.update(1)
        x, y, fz = eye_params[ip]
        for ellipse, fit_params in zip(source_ellipses, ellipse_fits):
            if refit_from_p0:
                p0 = (0, 0, 1)
            else:
                p0 = (fit_params[0], fit_params[1], fit_params[2])
            p, single_error, all_errors = minimise_reprojection_error(
                ellipse,
                p0=p0,
                eye_centre=np.array([x, y]),
                f_z0=fz,
                p_range=p_range,
                niter=niter,
                grid_size=grid_size,
            )
            errors[ip] += single_error

    best_error = errors.min()
    best_eye = eye_params[errors.argmin()]
    return best_eye, best_error, errors


@njit
def predict_xy(t, xc, yc, a, b, angle):
    """Predict x- and y-coordinates using the estimated model.

    This is extracted from EllipseModel to avoid unessesary checks

    Parameters
    ----------
    t : array
        Angles in circle in radians. Angles start to count from positive
        x-axis to positive y-axis in a right-handed system.
    params : (5, ) array, optional
        Optional custom parameter set.

    Returns
    -------
    xy : (..., 2) array
        Predicted x- and y-coordinates.

    """

    ct = np.cos(t)
    st = np.sin(t)
    cangle = math.cos(angle)
    sangle = math.sin(angle)

    x = xc + a * cangle * ct - b * sangle * st
    y = yc + a * sangle * ct + b * cangle * st

    return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)


@njit(parallel=True, nogil=True)
def minimise_all(
    parameters,
    p0,
    eye_centre,
    f_z0,
    progress_proxy,
    p_range=(1.0, 1.0, 0.5),
    grid_size=10,
    niter=3,
    reduction_factor=3,
):
    """Run minimisation of reprojection error on all frames in parallel

    Args:

        parameters (numpy.array): N x 5 array of ellipse parameters
        is_valid (numpy.array): N array of bool indicating if ellipse is valid
        p0 (tuple): Starting estimates of parameter (phi, theta, radius), centre of grid
        eye_centre (numpy.array): x,y of eye centre in camera coordinate
        f_z0 (float): scale factor
        progress_proxy (tqdm.tqdm): Progress bar
        p_range (tuple, optional): range of grid for the 3 parameters. Defaults to
            (1, 1, 0.5)
        grid_size (int, optional): number of values for each level of the grid.
            Defaults to 10.
        niter (int, optional): number of iteration. Defaults to 3
        reduction_factor (int, optional): reduction of p_range at each iteration.
            Defaults to 3
    """
    nframes = len(parameters)
    eye_rotation = np.zeros((nframes, 3))
    eye_rotation += np.nan
    for ipos in prange(nframes):
        progress_proxy.update(1)
        ellipse_params = parameters[ipos]
        if np.sum(np.isnan(ellipse_params)) > 0:
            continue
        pa, e, all_errors = minimise_reprojection_error(
            ellipse_params,
            p0,
            eye_centre,
            f_z0,
            p_range,
            grid_size,
            niter,
            reduction_factor,
        )
        eye_rotation[ipos] = pa
    return eye_rotation


def optimise_eye_parameters(
    ellipses,
    gazes,
    p0,
    p_range=(50.0, 50.0, 30.0),
    grid_size=10,
    niter=5,
    reduction_factor=3,
    verbose=True,
    prange_inner=(np.pi / 3, np.pi / 3, 0.5),
    niter_inner=3,
    grid_size_inner=5,
    refit_from_p0=True,
    debug=False,
):
    """Optimise eye parameters by grid search"""

    params = tuple(p0)
    if verbose:
        p_display = np.round(params, 2)
        print(f"Initial eye parameters: {p_display}.", flush=True)

    errors = []

    for i_iter in range(niter):
        if verbose:
            print(f"Iteration {i_iter + 1}", flush=True)

        grids = [np.linspace(-r, r, grid_size) + p for r, p in zip(p_range, params)]
        # ensure last parameter is positive
        grids[-1] = grids[-1][grids[-1] > 0]
        n_p = np.prod([len(g) for g in grids])
        with ProgressBar(total=n_p) as progress:
            params, error, errs = grid_search_best_eye(
                ellipses,
                gazes,
                *grids,
                p_range=prange_inner,
                niter=niter_inner,
                grid_size=grid_size_inner,
                progress_proxy=progress,
                refit_from_p0=refit_from_p0,
            )
            if debug:
                errors.append((grids, errs))

        if verbose:
            p_display = np.round(params, 2)
            print(
                f"    Best eye parameters: {p_display}. Error: {error:.0f}",
                flush=True,
            )
        p_range = [p / reduction_factor for p in p_range]
    if debug:
        return params, error, errors
    return params, error


def get_gaze_vector(phi, theta):
    """Get the gaze vector from phi and theta

    Args:
        phi (float): Angle in radians
        theta (float): Angle in radians

    Returns:
        numpy.array: N x 3, vector(s) of gaze direction in camera coordinates
    """
    gaze_vec = np.array(
        [np.sin(theta), np.sin(phi) * np.cos(theta), -np.cos(phi) * np.cos(theta)]
    )
    if gaze_vec.ndim == 2:
        gaze_vec = gaze_vec.T
    return gaze_vec


def fit_circle(x, y):
    """
    Fit a circle to the points (x, y)

    Args:
        x (array): x coordinates of the points
        y (array): y coordinates of the points

    Returns:
        xc (float): x coordinate of the center of the circle
        yc (float): y coordinate of the center of the circle
        r (float): radius of the circle
        residual (float): sum of residuals of the fit
    """

    def calc_R(xc, yc):
        """calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """calculate the algebraic distance between the data points and the mean
        circle centered at c=(xc, yc)"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.nanmean(x), np.nanmean(y)
    center, ier = leastsq(f_2, center_estimate)

    xc, yc = center
    Ri = calc_R(*center)
    r = Ri.mean()
    residual = np.sum((Ri - r) ** 2)

    return xc, yc, r, residual


def segment_eye_border(img, local_threshold=65, block_size=3):
    """Segment the eye border from the image

    Args:
        img (numpy.array): Image to segment
        local_threshold (int, optional): Threshold for local thresholding.
            Defaults to 65.
        block_size (int, optional): Block size for local thresholding.
            Defaults to 3.

    Returns:
        numpy.array: Segmented image or None if no contour is found
    """
    th = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        local_threshold,
        block_size,
    )
    inverted_img = cv2.bitwise_not(th)
    # Define the kernel size and shape
    kernel_size = (7, 7)
    kernel_shape = cv2.MORPH_RECT
    # Create the kernel for binary erosion
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    eroded_img = cv2.erode(inverted_img, kernel, iterations=1)
    contours, _ = cv2.findContours(
        eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask for the largest object
    mask = np.zeros_like(eroded_img)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    # Apply the mask to the binary image
    segmented_img = cv2.bitwise_and(eroded_img, mask)
    return segmented_img
