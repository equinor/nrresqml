import numpy as np
import h5py
import matplotlib.pyplot as plt

resqml_file = r"C:\\Projects\\d3dmod\\check-resqml\\resqml-files\\sobrarbe.epc"

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


plt.figure()
plt.pcolormesh(archel[-1, :, :])
plt.show()
