import os
import numpy as np
import sys
import rmsapi
import rmsapi.grids
import platform
import h5py

# USER PARAMETERS
resqml_file = "full/path/to/resqml/model.epc"
grid_name = "GridName"
# ---------------

# ADVANCED PARAMETERS
# Crops the outer part of the model with 'xy_buffer' cells. This reduces the lateral extent of the model, but seems to
# avoid RMS crashing for some grids.
xy_buffer = 1

# Instead of showing cell tops as flat (which is consistent with the Delft3D definition), cell tops can be shown as
# continuous. However, this is an approximation since it effectively shifts the grid half a grid cell in the I and J
# directions. The visual impression of the geometry will be representative, but properties of the grid can look odd when
# shifted half a cell. Turning smooth_approximation on may solve RMS crashing for some grids.
smooth_approximation = False
# ---------------

# Read h5 file and prepare data
data = h5py.File(resqml_file.replace(".epc", ".h5"), mode="r")
cps_key = [c for c in data.keys() if c.startswith("control_points")][0]
cpp_key = [c for c in data.keys() if c.startswith("control_point_parameters")][0]
cps = data[cps_key]
cpp = data[cpp_key]
# cpp data may be stored in a shared or split pillar format. In case of split pillar format, convert to shared, as this
# unifies the conversion to the RMS grid format
if cpp.ndim == 4 and cpp.shape[0] == 4:
    cpp = np.mean(cpp, axis=0)
    cpp = cpp.transpose((2, 0, 1))
    cps = cps[0, :, :, :]
cpp_full = cpp

# Adjust these parameters to extract only parts of the grid
if xy_buffer == 0:
    xy_buffer = None
xy_step = slice(xy_buffer, -xy_buffer, 1)
z_step = slice(None, None, 1)
cps = cps[xy_step, xy_step, :]
cpp = cpp[z_step, xy_step, xy_step]

# Get relevant grid parameters
nz, nx, ny = cpp.shape
n_pillars = (nx + 1, ny + 1)

x0 = cps[0, 0, 0]
y0 = cps[0, 0, 1]
dx = cps[1, 0, 0] - cps[0, 0, 0]
dy = cps[0, 1, 1] - cps[0, 0, 1]

cpp_t = cpp.transpose((1, 2, 0))
z_pillars = np.full(
    (4, n_pillars[0], n_pillars[1], nz), fill_value=np.nan, dtype=np.float64
)
if smooth_approximation is False:
    # Flat cell tops
    z_pillars[0, 1:, 1:, :] = cpp_t
    z_pillars[1, :-1, 1:, :] = cpp_t
    z_pillars[2, 1:, :-1, :] = cpp_t
    z_pillars[3, :-1, :-1, :] = cpp_t
else:
    # Smooth cell tops. This effectively shifts the grid half a cell in I and J direction
    z_pillars[0, 1:-1, 1:-1, :] = cpp_t[1:, 1:, :]
    z_pillars[1, :-1, 1:-1, :] = cpp_t[:, 1:, :]
    z_pillars[2, 1:-1, :-1, :] = cpp_t[1:, :, :]
    z_pillars[3, :-1, :-1, :] = cpp_t

    z_pillars[0, -1, 1:, :] = cpp_t[-1, :, :]
    z_pillars[0, 1:-1, -1, :] = cpp_t[:-1, -1, :]
    z_pillars[1, :-1, -1, :] = cpp_t[:, -1, :]
    z_pillars[2, -1, :-1, :] = cpp_t[-1, :, :]

z_pillars_ma = np.ma.array(z_pillars, mask=np.isnan(z_pillars))


# Determine what to name the grid (always create a new one)
gn = grid_name
i = 1
while gn in project.grid_models:
    gn = grid_name + f"_{i}"
    i += 1

# Create the grid definition
grid_model = project.grid_models.create(gn)
cpg = rmsapi.grids.CornerPointGridGeometry.create((nx, ny, nz - 1))

# Define grid geometry
for i in range(z_pillars_ma.shape[1]):
    # Create mask to mask cells on the boundary to conform to RMS definition
    for j in range(z_pillars_ma.shape[2]):
        current = z_pillars_ma[:, i, j, :]
        # Find top and bottom point
        x = x0 + i * dx
        y = y0 + j * dy
        bp = (x, y, np.min(current))
        tp = (x, y, np.max(current))
        # Set pillar data
        cpg.set_pillar_data(i, j, bp, tp, current)

# Initialize the geometry
grid_model.get_grid().set_geometry(cpg)

# Copy parameters
for p_name in data:
    if data[p_name].shape != cpp_full.shape:
        continue
    if data[p_name].dtype in (np.uint8, np.uint16):
        prop = grid_model.properties.create(
            p_name, rmsapi.GridPropertyType.discrete, data[p_name].dtype
        )
    elif data[p_name].dtype == np.int32:
        prop = grid_model.properties.create(
            p_name, rmsapi.GridPropertyType.discrete, np.uint16
        )
    else:
        prop = grid_model.properties.create(
            p_name, rmsapi.GridPropertyType.continuous, np.float32
        )
    z_itr = np.arange(
        z_step.start or 0, (((z_step.stop or nz) - 1) % nz), z_step.step or 1
    )
    _vals = data[p_name][z_itr, xy_step, xy_step].transpose((1, 2, 0))
    prop.set_values(_vals.flatten().astype(prop.data_type))
