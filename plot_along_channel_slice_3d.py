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

ds = xr.open_dataset("ice_shelf_meltwater_outflow_3d_RoquetEOS_along_channel_yz_slice.nc")
ds_front = xr.open_dataset("ice_shelf_meltwater_outflow_3d_RoquetEOS_along_front_xz_slice.nc")
 
Nt = ds.time.size

for n in range(0, Nt, 1):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 9), dpi=150)

    t = ds.time.values[n] / 86400
    fig.suptitle(f"32x32x32, 3 C meltwater, t = {t:.2f} days", fontsize=16)

    u = ds.u.isel(time=n).squeeze()
    u.plot.contourf(ax=axes[0, 0], levels=21, cmap=cmocean.cm.balance, extend="both")
    axes[0, 0].set_title("")
    axes[0, 0].set_xlim(2500,0)

    v = ds.v.isel(time=n).squeeze()
    v.plot.contourf(ax=axes[0, 1], levels=21, cmap=cmocean.cm.balance, extend="both")
    axes[0, 1].set_title("")
    axes[0, 1].set_xlim(2500,0)

    T = ds.T.isel(time=n).squeeze()
    T.plot.contourf(ax=axes[1, 0], vmin=1, vmax = 3, levels=21, cmap=cmocean.cm.thermal, extend="both")
    axes[1, 0].set_title("")
    axes[1, 0].set_xlim(2500,0)

    M = ds.meltwater.isel(time=n).squeeze()
    M.plot.contourf(ax=axes[1, 1], vmin=0, vmax=1, levels=21, cmap=cmocean.cm.haline)
    axes[1, 1].set_title("")
    axes[1, 1].set_xlim(2500,0)

    T_front = ds_front.T.isel(time=n).squeeze()
    T_front.plot.contourf(ax=axes[2, 0], vmin=1, vmax = 3, levels=21, cmap=cmocean.cm.thermal, extend="both")
    axes[2, 0].set_title("")

    M_front = ds_front.meltwater.isel(time=n).squeeze()
    M_front.plot.contourf(ax=axes[2, 1], vmin=0, vmax=1, levels=21, cmap=cmocean.cm.haline)
    axes[2, 1].set_title("")

    plt.subplots_adjust(hspace=0.4)

    print(f"Saving frame {n}...") 
    plt.savefig(f"yz_slice_{n:05d}.png")

    plt.close("all")

(
    ffmpeg
    .input("yz_slice_%05d.png", framerate=10)
    .output("yz_slice.mp4", crf=15, pix_fmt='yuv420p')
    .overwrite_output()
    .run()
)
