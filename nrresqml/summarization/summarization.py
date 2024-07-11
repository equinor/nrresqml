import numpy as np
import h5py
import pathlib
import tqdm
import json
from nrresqml.summarization.thumbnail import make_thumbnail_image


def summarize_resqml(fn: pathlib.Path, outdir: pathlib.Path) -> None:
    resqml_data = _read_resqml(fn)
    archel_stats = _compute_archel_stats(resqml_data)
    _dump_archel_stats(archel_stats, outdir)
    make_thumbnail_image(resqml_data, outdir)


def _read_resqml(fn: pathlib.Path) -> dict:
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

    return {
        "x0": x0,
        "y0": y0,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "archel": archel,
        "model_name": fn.stem,
    }


def _compute_archel_stats(resqml_data: dict) -> dict:
    archel = resqml_data["archel"]
    archel_values = np.unique(archel)
    archel_counts = np.zeros_like(archel_values)
    for i, v in tqdm.tqdm(
        enumerate(archel_values),
        total=archel_values.size,
        desc="Calculating archel statistics",
        unit="archel value",
    ):
        archel_counts[i] = np.sum(archel == v)

    sum_of_counts = np.sum(archel_counts)
    assert sum_of_counts == archel.size

    proportions = {
        int(v): c / sum_of_counts for v, c in zip(archel_values, archel_counts)
    }
    counts = {int(v): int(c) for v, c in zip(archel_values, archel_counts)}

    return {
        "model_name": resqml_data["model_name"],
        "archel_proportions": proportions,
        "archel_counts": counts,
    }


def _dump_archel_stats(archel_stats: dict, outdir: pathlib.Path) -> None:
    model_name = archel_stats["model_name"]
    fn_out = outdir / f"{model_name}_archel_stats.json"

    outdir.mkdir(parents=True, exist_ok=True)
    with open(fn_out, "w") as f:
        json.dump(archel_stats, f, indent=2)
