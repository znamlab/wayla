import flexiznam as flz
import numpy as np
import pandas as pd


def get_data(
    camera,
    flexilims_session=None,
    likelihood_threshold=0.88,
    rsquare_threshold=0.99,
    maximum_reflection_distance=50,
    error_threshold=None,
    ds_is_cropped=True,
):
    """Get eye tracking data from camera dataset

    Args:
        camera (flexiznam.schema.camera_data.CameraData): Camera dataset
        flexilims_session (flexilims.Session): Flexilims session to interact with
            database. Must have the proper project. If None, use
            `camera.flexilims_sessions`. Default to None.
        likelihood_threshold (float, optional): Threshold on average DLC likelihood.
            Defaults to 0.88.
        rsquare_threshold (float, optional): Threshold on rsquare of ellipse fit.
            Defaults to 0.99.
        maximum_reflection_distance (int, optional): Maximum distance between
            of reflection to its median position, in px. Defaults to 50.
        error_threshold (float, optional): Threshold on error of ellipse fit, in px.
            If None, use 5 sd. Defaults to None.
        ds_is_cropped (bool, optional): Whether the dataset is cropped. Defaults to
            True.

    Returns:
        panda.DataFrame: DLC results
        panda.DataFrame: Ellipse fits
        flexiznam.schema.dataset.Dataset: DLC dataset
    """

    rec_ds = flz.get_children(
        parent_id=camera.origin_id,
        flexilims_session=flexilims_session,
        children_datatype="dataset",
    )
    cam_analysis = rec_ds[rec_ds.name.map(lambda x: camera.dataset_name in x)]
    dlc_ds = cam_analysis[cam_analysis.dataset_type == "dlc_tracking"]
    if ds_is_cropped:
        dlc_ds = dlc_ds[[(c is not None) for c in dlc_ds.cropping]]
    else:
        dlc_ds = dlc_ds[[(c is None) for c in dlc_ds.cropping]]
    assert len(dlc_ds) == 1
    dlc_ds = flz.Dataset.from_dataseries(
        dlc_ds.iloc[0], flexilims_session=flexilims_session
    )
    dlc_res = pd.read_hdf(dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"])
    # Get ellipse fits
    ellipse_csv = list(dlc_ds.path_full.glob("*ellipse_fits.csv"))
    assert len(ellipse_csv) == 1
    ellipse = pd.read_csv(ellipse_csv[0])
    # add dlc likelihood
    dlc_like = dlc_res.xs("likelihood", axis="columns", level=2)
    dlc_like.columns = dlc_like.columns.droplevel("scorer")
    reflection_like = dlc_like["reflection"]
    to_drop = [c for c in dlc_like.columns if c[-1].isalpha()]
    dlc_like = dlc_like.drop(axis="columns", labels=to_drop).mean(axis="columns")
    ellipse["dlc_avg_likelihood"] = dlc_like

    if error_threshold is None:
        error_threshold = np.nanmean(ellipse.error) + 5 * np.nanstd(ellipse.error)

    # add reflection distance
    reflection = dlc_res.xs("reflection", axis="columns", level=1)
    reflection.columns = reflection.columns.droplevel("scorer")
    reflection_dist = np.sqrt(
        (reflection.x - reflection.x.median()) ** 2
        + (reflection.y - reflection.y.median()) ** 2
    )
    ellipse["reflection_dist"] = reflection_dist

    valid = (
        (ellipse.dlc_avg_likelihood > likelihood_threshold)
        & (ellipse.rsquare > rsquare_threshold)
        & (ellipse.error < error_threshold)
        & (reflection_like > likelihood_threshold)
        & (reflection_dist < maximum_reflection_distance)
    )
    ellipse["valid"] = valid

    reflection = dlc_res.xs(axis="columns", level=1, key="reflection")
    reflection.columns = reflection.columns.droplevel("scorer")
    ellipse["reflection_x"] = reflection.x.values
    ellipse["reflection_y"] = reflection.y.values
    ellipse["reflection_likelihood"] = reflection_like.values
    ellipse["pupil_x"] = ellipse.centre_x - ellipse.reflection_x
    ellipse["pupil_y"] = ellipse.centre_y - ellipse.reflection_y
    ellipse.loc[~ellipse.valid, "pupil_x"] = np.nan
    ellipse.loc[~ellipse.valid, "pupil_y"] = np.nan
    
    # look for reprojection 
    repro_ds = rec_ds[rec_ds.dataset_type == "eye_reprojection"]
    if len(repro_ds) > 0:
        if len(repro_ds) > 1:
            raise IOError("More than one eye_reprojection dataset")
        repro_ds = flz.Dataset.from_flexilims(
            name=repro_ds.iloc[0].name, flexilims_session=flexilims_session
        )
        reprojection = np.load(repro_ds.path_full)
        for i, l in enumerate(['phi', 'theta', 'radius']):
            ellipse[l] = reprojection[:, i]
    return dlc_res, ellipse, dlc_ds


def get_tracking_datasets(camera_ds, flexilims_session):
    """Get the dlc tracking datasets corresponding to a camera dataset

    This will raise an error if more than one dataset is found for a given type

    Args:
        camera_ds (flexilims.Dataset): Camera dataset
        flexilims_session (flexilims.Session): Flexilims session

    Returns:
        dict: Dictionary with keys "cropped" and "uncropped", containing the
            corresponding datasets. If no dataset is found, the corresponding value is
            None
    """
    dlc_datasets = flz.get_children(
        parent_id=camera_ds.origin_id,
        children_datatype="dataset",
        flexilims_session=flexilims_session,
    )
    dlc_datasets = dlc_datasets[dlc_datasets["dataset_type"] == "dlc_tracking"]
    ds_dict = dict(cropped=None, uncropped=None)
    for ds_name, series in dlc_datasets.iterrows():
        ds = flz.Dataset.from_dataseries(series, flexilims_session=flexilims_session)
        vid = ds.extra_attributes["videos"]
        assert (
            len(vid) == 1
        ), f"{ds_name} tracking with more than one video, is that normal?"
        # exclude tracking for other videos
        if not vid[0].endswith(camera_ds.extra_attributes["video_file"]):
            continue
        if isinstance(ds.extra_attributes["cropping"], list):
            if ds_dict["cropped"] is not None:
                raise IOError("More than one cropped dataset")
            ds_dict["cropped"] = ds
        else:
            if ds_dict["uncropped"] is not None:
                raise IOError("More than one uncropped dataset")
            ds_dict["uncropped"] = ds
    return ds_dict
