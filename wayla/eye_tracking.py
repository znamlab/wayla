"""Main functions for eye tracking analysis

This module contains the main functions for eye tracking analysis. The main function
is `run_all`, which runs all the steps of the analysis.
"""

import os
import shutil
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd
import yaml
from . import diagnostics, eye_io
from . import eye_model_fitting as emf
from znamutils import slurm_it

envs = flz.PARAMETERS["conda_envs"]


def run_all(
    flexilims_session,
    dlc_model_detect,
    dlc_model_tracking,
    camera_ds_name,
    origin_id=None,
    conflicts="abort",
    use_slurm=True,
    dependency=None,
    run_detect=True,
    run_tracking=True,
    run_ellipse=True,
    run_reprojection=True,
    repro_kwargs=None,
):
    """Run all preprocessing steps for a session

    Args:
        flexilims_session (flexilims.Session): Flexilims session
        dlc_model_detect (str): Name of the dlc model to use for eye detection and
            initial cropping, must be in the `DLC_MODELS` project
        dlc_model_tracking (str): Name of the dlc model to use for eye tracking, must
            be in the `DLC_MODELS` project
        camera_ds_name (str): Name of the camera dataset on flexilims.
        origin_id (str, optional): ID of the origin dataset on flexilims. If None,
            origin is read from the camera dataset. Defaults to None.
        conflicts (str, optional): How to handle conflicts when creating the datasets
            on flexilims. Defaults to "abort".
        use_slurm (bool, optional): Start slurm jobs. Defaults to True.
        dependency (str, optional): Dependency for slurm. Defaults to None.
        run_detect (bool, optional): Whether to run the eye detection. Defaults to True.
        run_tracking (bool, optional): Whether to run the tracking. Defaults to True.
        run_ellipse (bool, optional): Whether to run the ellipse fitting. Defaults to
            True.
        run_reprojection (bool, optional): Whether to run the eye reprojection.
            Defaults to True.

    Returns:
        pandas.DataFrame: Log of job id of each step
    """
    project = flz.lookup_project(flexilims_session.project_id)
    if origin_id is None:
        camera_ds = flz.Dataset.from_flexilims(
            name=camera_ds_name, flexilims_session=flexilims_session
        )
        origin_id = camera_ds.origin_id
        cam_ds_short_name = camera_ds.dataset_name
    else:
        origin = flz.get_entity(id=origin_id, flexilims_session=flexilims_session)
        basename = "_".join(origin.genealogy)
        assert camera_ds_name.startswith(basename), (
            f"camera_ds_name {camera_ds_name} does not start with origin basename "
            f"{basename}"
        )
        cam_ds_short_name = camera_ds_name[len(basename) + 1 :]
    log = dict(dataset_name=cam_ds_short_name)

    # Run uncropped DLC
    ds = flz.Dataset.from_origin(
        origin_id=origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flexilims_session,
        base_name=f"{cam_ds_short_name}_dlc_tracking_uncropped",
        conflicts=conflicts,
    )
    ds.path_full.mkdir(parents=True, exist_ok=True)
    if run_detect:
        job_id = dlc_pupil(
            camera_ds_name,
            model_name=dlc_model_detect,
            project=project,
            crop=False,
            conflicts=conflicts,
            use_slurm=use_slurm,
            job_dependency=dependency,
            slurm_folder=ds.path_full,
        )
    if not use_slurm or not run_detect:
        job_id = None
    log["dlc_uncropped"] = job_id if job_id is not None else "Done"

    # look for cropped dataset
    ds = flz.Dataset.from_origin(
        origin_id=origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flexilims_session,
        base_name=f"{cam_ds_short_name}_dlc_tracking_cropped",
        conflicts=conflicts,
    )
    if run_tracking:
        # Run cropped DLC
        ds.path_full.mkdir(parents=True, exist_ok=True)
        job_id = dlc_pupil(
            camera_ds_name,
            model_name=dlc_model_tracking,
            project=project,
            crop=True,
            conflicts=conflicts,
            use_slurm=use_slurm,
            job_dependency=job_id,
            slurm_folder=ds.path_full,
        )
        if not use_slurm:
            job_id = None

        log["dlc_cropped"] = job_id if job_id is not None else "Done"

    if run_ellipse:
        # Run ellipse fit
        job_id = fit_ellipse(
            camera_ds_name,
            project=project,
            likelihood_threshold=None,
            job_dependency=job_id,
            use_slurm=use_slurm,
            slurm_folder=ds.path_full,
            conflicts=conflicts,
        )
        log["ellipse"] = job_id if job_id is not None else "Done"
        if not use_slurm:
            job_id = None

    if run_reprojection:
        # Run reprojection
        ds = flz.Dataset.from_origin(
            origin_id=origin_id,
            dataset_type="eye_reprojection",
            flexilims_session=flexilims_session,
            base_name=f"{cam_ds_short_name}_eye_reprojection",
            conflicts=conflicts,
        )
        if ds.path.suffix != ".npy":
            # When we create the dataset, the path is set to the folder, not the file
            ds.path_full.mkdir(parents=True, exist_ok=True)
        else:
            # if we already ran the analysis once, the path has been set to the results
            ds.path = ds.path.parent
        repro_kwargs = dict(
            theta0=0,
            phi0=0,
            likelihood_threshold=0.88,
            rsquare_threshold=0.99,
            error_threshold=None,
        )
        if repro_kwargs is not None:
            repro_kwargs.update(repro_kwargs)
        job_id = run_reproject_eye(
            project=project,
            camera_ds_name=camera_ds_name,
            conflicts=conflicts,
            use_slurm=use_slurm,
            slurm_folder=ds.path_full,
            job_dependency=job_id,
            **repro_kwargs,
        )
        if not use_slurm:
            job_id = None

        log["reprojection"] = job_id if job_id is not None else "Done"
    return pd.Series(log)


def clear_tracking_info(camera_ds, flexilims_session):
    """Clear tracking information for a camera dataset

    This will delete all tracking datasets associated with the camera dataset.
    and the reprojection dataset.
    Args:
        camera_ds (flexiznam.Dataset): Camera dataset
        flexilims_session (flexilims.Session): Flexilims session
    """
    ds_dict = eye_io.get_tracking_datasets(camera_ds, flexilims_session)
    for ds in ds_dict.values():
        if ds is not None:
            if ds.path_full.is_dir():
                print(f"        deleting {ds.path_full}")
                shutil.rmtree(ds.path_full)
            flexilims_session.delete(ds.id)
    cam_ds_short_name = camera_ds.dataset_name
    repro_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type="eye_reprojection",
        flexilims_session=flexilims_session,
        base_name=f"{cam_ds_short_name}_eye_reprojection",
        conflicts="skip",
    )
    if repro_ds.path_full.exists():
        if not repro_ds.path_full.is_dir():
            assert repro_ds.path_full.suffix == ".npy"
            p = repro_ds.path_full.parent
        else:
            p = repro_ds.path_full
        print(f"        deleting {p}")
        shutil.rmtree(p)
    if repro_ds.flexilims_status() != "not online":
        flexilims_session.delete(repro_ds.id)


def delete_tracking_dataset(ds, flexilims_session):
    """Delete a dlc_tracking dataset

    Args:
        ds (flexiznam.Dataset): dlc_tracking dataset to delete
        flexilims_session (flexilims.Session): Flexilims session
    """
    filenames = []
    for suffix in ["", "_filtered"]:
        p = ds.path_full / ds.extra_attributes["dlc_file"]
        basename = p.with_name(p.stem + suffix)
        for ext in [".h5", ".csv"]:
            filenames.append(basename.with_suffix(ext))
        filenames.append(basename.with_name(basename.stem + "_labeled.mp4"))
    filenames.append(p.with_name(p.stem + "_meta.pickle"))
    # also delete slurm files
    for ext in ["sh", "py", "err", "out"]:
        filenames.append(ds.path_full / f"dlc_track.{ext}")
    for fname in filenames:
        if fname.exists():
            print(f"        deleting {fname}")
            os.remove(fname)
    # also remove the flexilims entry
    flexilims_session.delete(ds.id)


@slurm_it(
    conda_env=envs["dlc"],
    module_list=["cuDNN/8.4.1.50-CUDA-11.7.0"],
    slurm_options=dict(
        ntasks=1,
        time="12:00:00",
        mem="32G",
        gres="gpu:1",
        partition="gpu",
    ),
)
def dlc_pupil(
    camera_ds_name,
    model_name,
    project,
    crop=False,
    conflicts="abort",
):
    """Run dlc tracking on a video

    This is the function that actually runs the tracking. It is called by run_dlc
    directly when not using slurm or by slurm_job.slurm_dlc_pupil when using slurm.

    Args:
        camera_ds_name (str): Name of the camera dataset on flexilims
        model_name (str): Name of the dlc model to use. Must be in the `DLC_models`
            project
        project (str): Name of the project on flexilims
        crop (bool, optional): Whether to crop the video. Defaults to False.
        conflicts (str, optional): How to handle conflicts when creating the dataset on
            flexilims. Defaults to "abort".
    """

    flexilims_session = flz.get_flexilims_session(project)
    camera_ds = flz.Dataset.from_flexilims(
        flexilims_session=flexilims_session, name=camera_ds_name
    )
    ds_dict = eye_io.get_tracking_datasets(camera_ds, flexilims_session)
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    video_path = Path(video_path)

    suffix = "cropped" if crop else "uncropped"
    basename = f"{camera_ds.dataset_name}_dlc_tracking_{suffix}"
    ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flexilims_session,
        base_name=basename,
        conflicts=conflicts,
    )

    if ds.flexilims_status() != "not online":
        if conflicts == "overwrite":
            delete_tracking_dataset(
                ds_dict["cropped" if crop else "uncropped"], flexilims_session
            )
        elif conflicts == "skip":
            print(f"  DLC {suffix} already done. Skip")
            return ds, ds.path_full

    processed_path = flz.get_data_root(which="processed", project="DLC_models")
    dlc_model_config = processed_path / "DLC_models" / model_name / "config.yaml"

    if crop:
        uncropped_ds = ds_dict["uncropped"]
        assert uncropped_ds is not None, "No uncropped dataset found"
        crop_info = create_crop_file(camera_ds, uncropped_ds, conflicts=conflicts)
        crop_info = [
            crop_info["xmin"],
            crop_info["xmax"],
            crop_info["ymin"],
            crop_info["ymax"],
        ]
        suffix = "cropped"
    else:
        crop_info = None
        suffix = "uncropped"

    target_folder = Path(ds.path_full)
    target_folder.mkdir(exist_ok=True, parents=True)

    print("Doing %s" % video_path, flush=True)
    analyse_kwargs = dict(
        config=dlc_model_config,
        videos=[str(video_path)],
        videotype="",
        shuffle=1,
        trainingsetindex=0,
        gputouse=None,
        save_as_csv=False,
        in_random_order=True,
        destfolder=str(target_folder),
        batchsize=None,
        cropping=crop_info,
        TFGPUinference=True,
        dynamic=(False, 0.5, 10),
        modelprefix="",
        robust_nframes=False,
        allow_growth=False,
        use_shelve=False,
        auto_track=True,
        n_tracks=None,
        calibrate=False,
        identity_only=False,
    )

    print("Analyzing", flush=True)
    # import dlc only in functions that need it as it takes a long time to load
    import deeplabcut

    # check that it will use the GPU
    import tensorflow

    print(f"Using tensorflow {tensorflow.__version__}", flush=True)
    from tensorflow.python.client import device_lib

    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    print("Available devices:")
    print(get_available_devices(), flush=True)

    out = deeplabcut.analyze_videos(**analyse_kwargs)

    dlc_output = target_folder / f"{video_path.stem}{out}.h5"
    if not dlc_output.exists():
        raise IOError(
            f"DLC ran but I cannot find the output. {dlc_output} does not exist."
        )

    # Adding to flexilims
    print("Updating flexilims", flush=True)
    ds.extra_attributes = dict(
        analyse_kwargs,
        dlc_prefix=out,
        dlc_file=f"{video_path.stem}{out}.h5",
    )
    ds.update_flexilims(mode="overwrite")

    # Save diagnostic plot
    print("Saving diagnostic plot", flush=True)
    if not crop:
        diagnostics.check_cropping(dlc_ds=ds, camera_ds=camera_ds, conflicts=conflicts)
    else:
        print("Labelling video")
        deeplabcut.create_labeled_video(
            config=dlc_model_config,
            videos=[str(video_path)],
            color_by="individual",
            keypoints_only=False,
            trailpoints=0,
            draw_skeleton=True,
        )
    return ds, ds.path_full


def create_crop_file(camera_ds, dlc_ds, conflicts="skip"):
    """Create a crop file for DLC tracking

    Uses the results of the uncropped tracking to find the crop area and save it in a
    crop file. This crop file can then be used to crop the video before tracking.

    Args:
        camera_ds (flexilims.Dataset): Camera dataset, must contain project information
        dlc_ds (flexilims.Dataset): dlc_tracking dataset, containing uncropped tracking
            results
        conflicts (str, optional): How to handle conflicts when creating the crop file.
            Defaults to "skip". Behaviour is "skip" or "overwrite", won't append.

    Returns:
        dict: Crop information

    """
    if dlc_ds.project is None:
        raise IOError("dlc_tracking dataset has no project information")

    metadata_path = camera_ds.path_full / camera_ds.extra_attributes["metadata_file"]
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if crop_file.exists() and (conflicts == "skip" or (not conflicts)):
        print("Crop file already exists. Delete manually to redo")
        with open(crop_file, "r") as fhandle:
            crop_info = yaml.safe_load(fhandle)
        return crop_info

    with open(metadata_path, "r") as fhandle:
        metadata = yaml.safe_load(fhandle)
    metadata = {k.lower(): v for k, v in metadata.items()}
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    print("Creating crop file")
    dlc_res = pd.read_hdf(dlc_file)
    # Find DLC crop area
    borders = np.zeros((4, 2))
    for iw, w in enumerate(
        (
            "temporal_eye_corner",
            "nasal_eye_corner",
            "top_eye_lid",
            "bottom_eye_lid",
        )
    ):
        vals = dlc_res.xs(w, level=1, axis=1)
        vals.columns = vals.columns.droplevel("scorer")
        v = np.nanmedian(vals[["x", "y"]].values, axis=0)
        borders[iw, :] = v

    borders = np.vstack([np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)])
    borders += ((np.diff(borders, axis=0) * 0.3).T @ np.array([[-1, 1]])).T
    for i, w in enumerate(["width", "height"]):
        borders[:, i] = np.clip(borders[:, i], 0, metadata[w])
    borders = borders.astype(int)
    crop_info = dict(
        xmin=int(borders[0, 0]),
        xmax=int(borders[1, 0]),
        ymin=int(borders[0, 1]),
        ymax=int(borders[1, 1]),
        dlc_source=str(dlc_ds.path),
        dlc_ds_id=dlc_ds.id,
    )
    with open(crop_file, "w") as fhandle:
        yaml.dump(crop_info, fhandle)
    print("Crop file created")
    return crop_info


@slurm_it(conda_env="cottage_analysis")
def fit_ellipse(
    camera_ds_name,
    project,
    likelihood_threshold=None,
    conflicts="skip",
    plot=True,
):
    flexilims_session = flz.get_flexilims_session(project)
    camera_ds = flz.Dataset.from_flexilims(
        name=camera_ds_name, flexilims_session=flexilims_session
    )
    ds_dict = eye_io.get_tracking_datasets(camera_ds, flexilims_session)
    if ds_dict["cropped"] is None:
        raise IOError("No cropped dataset found")
    dlc_ds = ds_dict["cropped"]
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    assert dlc_file.exists()

    target = dlc_ds.path_full / f"{dlc_file.stem}_ellipse_fits.csv"
    if target.exists():
        if conflicts == "overwrite":
            os.remove(target)
        else:
            print("  Ellipse fit already done. Skip")
            return target

    print("Doing %s" % dlc_file)
    ellipse_fits = emf.fit_ellipses(
        dlc_file,
        likelihood_threshold=likelihood_threshold,
    )
    print(f"Fitted, save to {target}")
    ellipse_fits.to_csv(target, index=False)

    if plot:
        print("Diagnostic plot")
        diagnostics.plot_ellipse_fit(
            camera_ds_name=camera_ds_name,
            project=project,
            likelihood_threshold=likelihood_threshold,
            duration=60,
            plot_reprojection=False,
        )
    print("Done")


@slurm_it(
    conda_env=envs["cottage_analysis"],
    slurm_options={
        "time": "48:00:00",
        "mem": "64G",
        "partition": "ncpu",
        "cpus-per-task": 32,
    },
)
def run_reproject_eye(
    camera_ds_name,
    project,
    phi0=0,
    theta0=np.deg2rad(20),
    likelihood_threshold=0.88,
    rsquare_threshold=0.99,
    error_threshold=None,
    conflicts="skip",
):
    """Run the reproject_eye function on a camera dataset

    DLC and ellipse fitting must have been done first

    Args:
        camera_ds_name (str): Name of the camera dataset on flexilims
        project (str): Name of the project on flexilims
        phi0 (int, optional): Initial guess for the phi angle. Defaults to 0.
        theta0 (float, optional): Initial guess for the theta angle. Defaults to
            np.deg2rad(20).
        likelihood_threshold (float, optional): Likelihood threshold for ellipse
            fitting. Defaults to 0.88.
        rsquare_threshold (float, optional): R^2 threshold for ellipse fitting.
            Defaults to 0.99.
        error_threshold (int, optional): Error threshold for ellipse fitting. If None,
            use mean + 5 sd. Defaults to None.
        conflicts (str, optional): How to handle conflicts when creating the datasets
            on flexilims. Defaults to "skip".
    """
    flexilims_session = flz.get_flexilims_session(project)
    camera_ds = flz.Dataset.from_flexilims(
        flexilims_session=flexilims_session, name=camera_ds_name
    )

    target_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type="eye_reprojection",
        flexilims_session=flexilims_session,
        base_name=f"{camera_ds.dataset_name}_eye_reprojection",
        conflicts=conflicts,
    )
    if target_ds.path.suffix != ".npy":
        assert (
            target_ds.path_full.is_dir()
        ), f"target_ds.path_full {target_ds.path_full} is not a directory"
        target_ds.path = target_ds.path / "eye_rotation_by_frame.npy"

    if target_ds.path_full.exists():
        if conflicts == "skip":
            print("  Reprojection already done. Skip")
            return target_ds
        elif conflicts == "abort":
            raise IOError("Reprojection already done")
        elif conflicts == "overwrite":
            print("  Reprojection already done. Overwrite")
            os.remove(target_ds.path_full)
        else:
            raise ValueError(f"Unknown conflict mode {conflicts}")

    kwargs = dict(
        theta0=theta0,
        phi0=phi0,
        likelihood_threshold=likelihood_threshold,
        rsquare_threshold=rsquare_threshold,
        error_threshold=error_threshold,
    )
    emf.reproject_ellipses(
        camera_ds=camera_ds,
        target_ds=target_ds,
        **kwargs,
    )
    target_ds.extra_attributes.update(**kwargs)
    target_ds.update_flexilims(mode=conflicts)
    return target_ds
