import pickle
from pathlib import Path
from warnings import warn

import flexiznam as flz
import numpy as np
import pandas as pd


def add_behaviour(
    camera, dlc_res, ellipse, speed_threshold=0.01, log_speeds=False, verbose=True
):
    """Add running speed, optic flow and depth to ellipse dataframe

    This assumes that there can be a few triggers after the end of the scanimage session
    and cuts them (up to 5)
    Args:
        camera (flexiznam.CameraDataset): Camera dataset, used for finding data and
        flexilims interaction
        dlc_res (pandas.DataFrame): DLC results
        ellipse (pandas.DataFrame): Ellipse fits
        speed_threshold (float, optional): Threshold to cut running speeds. Defaults to
            0.01.
        log_speeds (bool, optional): If True, speeds at log10, otherwise raw. Defaults
            to False.
        verbose (bool, optional): If True tell how many frames are cut. Defaults to True

    Returns:
        pandas.DataFrame: Combined dataframe, copy of ellipse with speeds
    """
    # get data
    flm_sess = camera.flexilims_session
    assert flm_sess is not None
    recording = flz.get_entity(id=camera.origin_id, flexilims_session=flm_sess)

    sess_ds = flz.get_children(
        parent_id=recording.origin_id,
        flexilims_session=flm_sess,
        children_datatype="dataset",
    )
    suite_2p = sess_ds[sess_ds.dataset_type == "suite2p_rois"]
    if len(suite_2p) > 1:
        warn(f"Found {len(suite_2p)} suite2p_rois datasets for {recording.path}")
    assert len(suite_2p) != 0, f"No suite2p_rois dataset for {recording.path}"
    suite_2p = suite_2p.iloc[0]
    suite_2p = flz.Dataset.from_dataseries(suite_2p, flexilims_session=flm_sess)

    ops = np.load(
        suite_2p.path_full / "suite2p" / "plane0" / "ops.npy", allow_pickle=True
    ).item()
    processed = Path(flz.PARAMETERS["data_root"]["processed"])

    with open(processed / recording.path / "img_VS.pickle", "rb") as handle:
        param_logger = pickle.load(handle)

    sampling = ops["fs"]

    vrs = np.array(param_logger.EyeZ.diff() / param_logger.HarpTime.diff(), dtype=float)
    vrs = np.clip(vrs, speed_threshold, None)
    if "MouseZ" in param_logger.columns:
        rs = np.array(
            param_logger.MouseZ.diff() / param_logger.HarpTime.diff(), dtype=float
        )
        rs = np.clip(rs, speed_threshold, None)
    else:
        warn(f"No MouseZ for {recording.path}")
        assert not recording.protocol.lower().endswith("playback")
        rs = np.array(vrs)
    depth = np.array(param_logger.Depth, copy=True, dtype=float)
    depth[depth < 0] = np.nan
    of = np.degrees(vrs / depth)
    # convert to cm
    depth *= 100
    rs *= 100
    vrs *= 100
    if log_speeds:
        rs = np.log10(rs)
        vrs = np.log10(vrs)
        of = np.log10(of)
    if verbose:
        print(f"Running speed with {len(rs)}, vs {len(ellipse)} frames")
    ntocut = len(ellipse) - len(rs)
    if ntocut > 5:
        raise ValueError("{ntocut} more frames in video than SI triggers")
    elif ntocut > 0:
        if verbose:
            print(f"Cutting the last {ntocut} frames")
        data = pd.DataFrame(ellipse.iloc[:-ntocut, :], copy=True)
    else:
        raise NotImplementedError

    data["running_speed"] = rs
    data["optic_flow"] = of
    data["virtual_running_speed"] = vrs
    data["depth"] = np.round(depth, 0)
    return data, sampling
