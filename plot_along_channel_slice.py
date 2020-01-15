import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import ffmpeg

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ds = xr.open_dataset("ice_shelf_meltwater_outflow_along_channel_yz_slice.nc")

Nt = ds.Time.size

for n in range(0, Nt, 100):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=300)

    t = ds.Time.values[n] / 86400
    fig.suptitle(f"Point source, Linear EOS, t = {t:.2f} days", fontsize=16)

    u = ds.u.isel(Time=n).squeeze()
    u.plot.contourf(ax=axes[0, 0], vmin=-0.5, vmax=0.5, levels=21, cmap=cmocean.cm.balance, extend="both")
    axes[0, 0].set_title("")

    w = ds.w.isel(Time=n).squeeze()
    w.plot.contourf(ax=axes[0, 1], vmin=-0.2, vmax=0.2, levels=21, cmap=cmocean.cm.balance, extend="both")
    axes[0, 1].set_title("")

    T = ds.T.isel(Time=n).squeeze()
    T.plot.contourf(ax=axes[1, 0], vmin=-2, vmax=1, levels=21, cmap=cmocean.cm.thermal, extend="both")
    axes[1, 0].set_title("")

    M = ds.meltwater.isel(Time=n).squeeze()
    M.plot.contourf(ax=axes[1, 1], vmin=0, vmax=1, levels=21, cmap=cmocean.cm.haline)
    axes[1, 1].set_title("")

    print(f"Saving frame {n}...") 
    plt.savefig(f"yz_slice_{n:05d}.png")

    plt.close("all")

#  (
#      ffmpeg
#      .input("yz_slice_%05d.png", framerate=10)
#      .output("yz_slice.mp4", crf=15, pix_fmt='yuv420p')
#      .overwrite_output()
#      .run()
#  )
