import xarray as xr
import matplotlib.pyplot as plt
plt.rc('font', size = 16)

# calculate vertical meltwater distributions, averaged over all other dimensions

datadir = "data/"

ds_point = xr.open_dataset(datadir + "T3_128x128x32_fields.nc")
ds_line = xr.open_dataset(datadir + "T3_128x128x32_line_fields.nc")

# after 6 hours
mw_point = ds_point.meltwater.isel(time=8).sum(dim=("xC","yC"))
mw_line = ds_line.meltwater.isel(time=8).sum(dim=("xC","yC"))

plt.figure()
plt.plot(mw_point,mw_point.zC,linewidth=2)
plt.plot(mw_line,mw_line.zC,linewidth=2)
plt.xlabel('Normalized meltwater distribution')
plt.ylabel('z (m)')

plt.show()
