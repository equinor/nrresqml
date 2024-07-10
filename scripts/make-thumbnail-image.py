import numpy as np
import h5py
import matplotlib.pyplot as plt
import pathlib
import tqdm


def is_foreground(array, background_values):
    return np.logical_and.reduce([array != bgv for bgv in background_values])


resqml_file = r"C:\\Projects\\D3D\\roda-resqml\\roda.epc"
# resqml_file = r"C:\\Projects\\D3D\\sobrarbe-resqml\\sobrarbe.epc"


print(f"Reading {resqml_file}...")
data = h5py.File(resqml_file.replace(".epc", ".h5"), mode="r")
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

# Count occurrences of archel values
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

print("Archel value frequencies:")
for v, c in zip(archel_values, archel_counts):
    print(f"  {v}: {c} ({c/sum_of_counts:.1%})")

xmesh, ymesh = np.meshgrid(
    np.linspace(x0, x0 + dx * nx, archel.shape[1]),
    np.linspace(y0, y0 + dy * ny, archel.shape[2]),
    indexing="ij",
)
background_archel_values = [0, 6]
x_foreground = np.where(
    is_foreground(archel[-1, :, :], background_archel_values), xmesh, np.nan
)
y_foreground = np.where(
    np.all(archel[-1, :, :] != bgv for bgv in background_archel_values), ymesh, np.nan
)
x_centroid = np.nanmean(x_foreground.flatten())
y_centroid = np.nanmean(y_foreground.flatten())

x_min = np.nanmin(x_foreground)
x_max = np.nanmax(x_foreground)
y_min = np.nanmin(y_foreground)
y_max = np.nanmax(y_foreground)


print("Creating thumbnail image...")
if True:
    plt.figure()
    plt.imshow(
        np.where(
            is_foreground(archel[-1, :, :], background_archel_values),
            archel[-1, :, :],
            np.nan,
        ),
        extent=(y0, y0 + dy * ny, x0, x0 + dx * nx),
        interpolation="none",
        origin="lower",
        cmap="tab10",
    )

    if False:
        plt.scatter(y_centroid, x_centroid, color="white", s=100, marker="o")
        plt.scatter(y_centroid, x_centroid, color="black", s=100, marker="+")

    plt.axis("off")

    # Set axis limits
    plt.xlim = (y_min, y_max)
    plt.ylim = (x_min, x_max)

    # Set aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if False:
        plt.show()
    else:
        input_path = pathlib.Path(resqml_file)
        input_filename = input_path.stem
        output_filename = f"{input_filename}-thumbnail.png"
        output_path = input_path.parent / output_filename

        fig = plt.gcf()
        fig.set_size_inches(1.5, 1.5)

        plt.savefig(output_path, dpi=300)
        print(f"Saved thumbnail image to {output_path}")
