import pathlib
import matplotlib.pyplot as plt
import numpy as np


def make_thumbnail_image(
    archel_data: dict, outdir: pathlib.Path, background_archels: list[int] = [0, 6]
) -> None:
    archel = archel_data["archel"]
    x0 = archel_data["x0"]
    y0 = archel_data["y0"]
    dx = archel_data["dx"]
    dy = archel_data["dy"]
    nx = archel_data["nx"]
    ny = archel_data["ny"]

    view_box = _find_cropbox(
        archel_data, background_archels, fraction=0.95, tolerance=0.01
    )

    print("Creating thumbnail image...")
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        np.where(
            _is_foreground(archel[-1, :, :], background_archels),
            archel[-1, :, :],
            np.nan,
        ),
        extent=(y0, y0 + dy * ny, x0, x0 + dx * nx),
        interpolation="none",
        origin="lower",
        cmap="tab10",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    x_min, x_max, y_min, y_max = view_box.values()
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)

    ax.set_aspect("equal", adjustable="box")

    model_name = archel_data["model_name"]
    output_path = outdir / f"{model_name}.png"

    plt.savefig(output_path, dpi=300)
    print(f"Saved thumbnail image to {output_path}")


def _is_foreground(array, background_values):
    return np.logical_and.reduce([array != bgv for bgv in background_values])


def _find_cropbox_full(archel_data: dict, background_archels) -> dict:
    archel = archel_data["archel"]
    x0 = archel_data["x0"]
    y0 = archel_data["y0"]
    dx = archel_data["dx"]
    dy = archel_data["dy"]
    nx = archel_data["nx"]
    ny = archel_data["ny"]

    xmesh, ymesh = np.meshgrid(
        np.linspace(x0, x0 + dx * nx, archel.shape[1]),
        np.linspace(y0, y0 + dy * ny, archel.shape[2]),
        indexing="ij",
    )
    x_foreground = np.where(
        _is_foreground(archel[-1, :, :], background_archels), xmesh, np.nan
    )
    y_foreground = np.where(
        np.all(archel[-1, :, :] != bgv for bgv in background_archels),
        ymesh,
        np.nan,
    )

    x_min = np.nanmin(x_foreground)
    x_max = np.nanmax(x_foreground)
    y_min = np.nanmin(y_foreground)
    y_max = np.nanmax(y_foreground)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def _find_cropbox(
    archel_data: dict, background_archels, fraction=1.0, tolerance=0.1
) -> dict:
    if fraction == 1.0:
        return _find_cropbox_full(archel_data, background_archels)
    elif fraction <= 0.0 or fraction > 1.0:
        raise ValueError("Fraction must be in the range (0, 1]")
    else:
        archel = archel_data["archel"]
        x0 = archel_data["x0"]
        y0 = archel_data["y0"]
        dx = archel_data["dx"]
        dy = archel_data["dy"]
        nx = archel_data["nx"]
        ny = archel_data["ny"]
        x1 = x0 + dx * nx
        y1 = y0 + dy * ny

        full_box = (x0, x1, y0, y1)

        xmesh, ymesh = np.meshgrid(
            np.linspace(x0, x0 + dx * nx, archel.shape[1]),
            np.linspace(y0, y0 + dy * ny, archel.shape[2]),
            indexing="ij",
        )
        x_foreground = np.where(
            _is_foreground(archel[-1, :, :], background_archels), xmesh, np.nan
        )
        y_foreground = np.where(
            np.all(archel[-1, :, :] != bgv for bgv in background_archels),
            ymesh,
            np.nan,
        )

        centroid = (
            np.nanmean(x_foreground),
            np.nanmean(y_foreground),
        )

        box_found = False

        scale_lower = 0.0
        scale_upper = 1.0
        fraction_lower = 0.0
        fraction_upper = 1.0

        scale = 0.5

        while not box_found:
            candidate_box = _inner_box(full_box, centroid, scale)
            actual_fraction = _fraction_inside_box(
                x_foreground, y_foreground, candidate_box
            )
            if abs(actual_fraction - fraction) < tolerance:
                box_found = True
            elif actual_fraction < fraction:
                scale_lower = scale
                scale = (scale_lower + scale_upper) / 2.0
            else:
                scale_upper = scale
                scale = (scale_lower + scale_upper) / 2.0

        x_min, x_max, y_min, y_max = candidate_box
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }


def _inner_box(
    box: tuple[float, float, float, float], point: tuple[float, float], scale: float
) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = box

    # distance from point to each edge
    dx_left = point[0] - x_min
    dx_right = x_max - point[0]
    dy_bottom = point[1] - y_min
    dy_top = y_max - point[1]

    if dx_left <= dx_right:
        x_min_inner = max(x_min, point[0] - scale * dx_left)
        x_max_inner = x_min_inner + scale * (x_max - x_min)
    else:
        x_max_inner = min(x_max, point[0] + scale * dx_right)
        x_min_inner = x_max_inner - scale * (x_max - x_min)

    if dy_bottom <= dy_top:
        y_min_inner = max(y_min, point[1] - scale * dy_bottom)
        y_max_inner = y_min_inner + scale * (y_max - y_min)
    else:
        y_max_inner = min(y_max, point[1] + scale * dy_top)
        y_min_inner = y_max_inner - scale * (y_max - y_min)

    return (x_min_inner, x_max_inner, y_min_inner, y_max_inner)


def _fraction_inside_box(
    x_foreground: np.ndarray,
    y_foreground: np.ndarray,
    box: tuple[float, float, float, float],
) -> float:
    x_min, x_max, y_min, y_max = box
    n_inside = np.sum(
        np.logical_and(
            np.logical_and(x_foreground >= x_min, x_foreground <= x_max),
            np.logical_and(y_foreground >= y_min, y_foreground <= y_max),
        )
    )
    n_total = np.sum(np.isfinite(x_foreground))
    return n_inside / n_total
