import pathlib
import matplotlib.pyplot as plt
import numpy as np


def make_thumbnail_image(archel_data: dict, outdir: pathlib.Path) -> None:
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
    background_archel_values = [0, 6]
    x_foreground = np.where(
        _is_foreground(archel[-1, :, :], background_archel_values), xmesh, np.nan
    )
    y_foreground = np.where(
        np.all(archel[-1, :, :] != bgv for bgv in background_archel_values),
        ymesh,
        np.nan,
    )

    x_min = np.nanmin(x_foreground)
    x_max = np.nanmax(x_foreground)
    y_min = np.nanmin(y_foreground)
    y_max = np.nanmax(y_foreground)

    print("Creating thumbnail image...")
    if True:
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_subplot(111)
        ax.imshow(
            np.where(
                _is_foreground(archel[-1, :, :], background_archel_values),
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

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)

        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        model_name = archel_data["model_name"]
        output_path = outdir / f"{model_name}.png"

        plt.savefig(output_path, dpi=300)
        print(f"Saved thumbnail image to {output_path}")


def _is_foreground(array, background_values):
    return np.logical_and.reduce([array != bgv for bgv in background_values])
