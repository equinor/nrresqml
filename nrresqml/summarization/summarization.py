import numpy as np
import h5py
import pathlib
import tqdm
import json

from nrresqml.summarization._utils import ResQmlData
from nrresqml.summarization.thumbnail import make_thumbnail_image


def summarize_resqml(fn: pathlib.Path, outdir: pathlib.Path) -> None:
    resqml_data = _read_resqml(fn)
    archel_stats = _compute_archel_stats(resqml_data)
    legend, bbox = make_thumbnail_image(resqml_data, outdir)
    archel_stats["thumbnail_legend"] = legend
    archel_stats["thumbnail_bbox"] = bbox
    _dump_model_description(archel_stats, outdir)


def _read_resqml(fn: pathlib.Path) -> ResQmlData:
    fn_str = str(fn)
    data = h5py.File(fn_str.replace(".epc", ".h5"), mode="r")
    cps_key = [c for c in data.keys() if c.startswith("control_points")][0]
    cpp_key = [c for c in data.keys() if c.startswith("control_point_parameters")][0]
    cps = data[cps_key]
    cpp = data[cpp_key]

    if cpp.ndim == 4 and cpp.shape[0] == 4:
        cpp = np.mean(cpp, axis=0)
        cpp = cpp.transpose((2, 0, 1))
        cps = cps[0, :, :, :]
    cpp_full = cpp

    xy_buffer = 1
    xy_step = slice(xy_buffer, -xy_buffer, 1)
    z_step = slice(None, None, 1)
    cps = cps[xy_step, xy_step, :]
    cpp = cpp[z_step, xy_step, xy_step]

    nz, nx, ny = cpp.shape
    n_pillars = (nx + 1, ny + 1)

    x0 = cps[0, 0, 0]
    y0 = cps[0, 0, 1]
    dx = cps[1, 0, 0] - cps[0, 0, 0]
    dy = cps[0, 1, 1] - cps[0, 0, 1]

    archel_name = "archel"
    archel = data[archel_name]

    cell_volumes = _compute_cell_volumes(cpp_full, dx, dy)

    return ResQmlData(x0, y0, dx, dy, nx, ny, nz, archel, cell_volumes, fn.stem)


def _compute_archel_stats(resqml_data: ResQmlData) -> dict:
    archel = resqml_data.archel
    unq_archel, unq_inverse, unq_count = np.unique(np.array(archel).flat, return_inverse=True, return_counts=True)
    cell_volumes = resqml_data.cell_volumes
    archel_volumes = [
        np.sum(np.array(cell_volumes).flat[unq_inverse == i])
        for i in tqdm.tqdm(
            range(unq_archel.size),
            total=unq_archel.size,
            desc="Calculating archel statistics",
            unit="archel value",
        )
    ]

    sum_of_volumes = np.sum(archel_volumes)
    assert np.isclose(sum_of_volumes, np.sum(cell_volumes))

    counts = {int(v): f"{int(c)}" for v, c in zip(unq_archel, unq_count)}
    proportions = {
        int(v): f"{c / archel.size:.2%}" for v, c in zip(unq_archel, unq_count)
    }
    volume_fractions = {
        int(v): f"{vol / sum_of_volumes:.2%}"
        for v, vol in zip(unq_archel, archel_volumes)
    }

    return {
        "model_name": resqml_data.model_name,
        "archel_cell_counts": counts,
        "archel_cell_count_proportions": proportions,
        "archel_volume_fractions": volume_fractions,
        "model_boundingbox": {
            "x0": resqml_data.x0,
            "x1": resqml_data.x0 + resqml_data.dx * resqml_data.nx,
            "y0": resqml_data.y0,
            "y1": resqml_data.y0 + resqml_data.dy * resqml_data.ny,
        }
    }


def _dump_model_description(archel_stats: dict, outdir: pathlib.Path) -> None:
    model_name = archel_stats["model_name"]
    fn_out = outdir / f"{model_name}_description.json"

    outdir.mkdir(parents=True, exist_ok=True)
    with open(fn_out, "w") as f:
        json.dump(archel_stats, f, indent=2)


def _compute_cell_volumes(cpp: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dz = np.diff(cpp, axis=0)
    dz = np.concatenate((dz, np.zeros_like(dz[0:1, :, :])), axis=0)
    cell_volumes = dx * dy * dz
    return cell_volumes
