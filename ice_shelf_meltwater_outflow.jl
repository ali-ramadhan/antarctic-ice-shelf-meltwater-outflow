using DelimitedFiles, Printf
using Interpolations, Plots
using Oceananigans

using Oceananigans.Diagnostics: AdvectiveCFL, DiffusiveCFL
using Oceananigans.OutputWriters: NetCDFOutputWriter

# Workaround for plotting many frames.
# See: https://github.com/JuliaPlots/Plots.jl/issues/1723
import GR
GR.inline("png")

#####
##### Some useful constants
#####

const km = 1000
const Ω_Earth = 7.292115e-5  # [s⁻¹]
const φ = -75  # degrees latitude

#####
##### Model grid and domain size
#####

arch = CPU()
FT = Float64

Nx = 32
Ny = 32
Nz = 32

Lx = 10km
Ly = 10km
Lz = 1km

end_time = 7day

#####
##### Set up source of meltwater: We will implement a source of meltwater as
##### a relaxation term towards a reference T and S value at a single point.
##### This is in effect weakly imposing a Value/Dirchlet boundary condition.
#####

λ = 1/(1minute)  # Relaxation timescale [s⁻¹].

# Temperature and salinity of the meltwater outflow.
T_source = -1
S_source = 33.95

# Index of the point source at the middle of the southern wall.
source_index = (Int(Nx/2), 1, Int(Nz/2))

# Point source
@inline T_point_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((i, j, k) == p.source_index, -p.λ * (C.T[i, j, k] - p.T_source), 0)

@inline S_point_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((i, j, k) == p.source_index, -p.λ * (C.S[i, j, k] - p.S_source), 0)

# Line source
@inline T_line_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((j, k) == (p.source_index[2], p.source_index[3]), -p.λ * (C.T[i, j, k] - p.T_source), 0)

@inline S_line_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((j, k) == (p.source_index[2], p.source_index[3]), -p.λ * (C.S[i, j, k] - p.S_source), 0)

params = (source_index=source_index, T_source=T_source, S_source=S_source, λ=λ)

# forcing = ModelForcing(T = T_point_source, S = S_point_source)
forcing = ModelForcing(T = T_line_source, S = S_line_source)

#####
##### Set up model
#####

model = Model(
           architecture = arch,
             float_type = FT,
                   grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(0, Ly), z=(-Lz, 0)),
                tracers = (:T, :S, :meltwater),
               coriolis = FPlane(rotation_rate=Ω_Earth, latitude=φ),
               buoyancy = SeawaterBuoyancy(),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = ChannelSolutionBCs(),
                forcing = forcing,
             parameters = params
)

#####
##### Read reference profiles from disk
##### As these profiles are derived from observations, we will have to do some
##### post-processing to be able to use them as initial conditions.
#####
##### We will get rid of all NaN values and use the remaining data to linearly
##### interpolate the T and S profiles to the model's grid.
#####

# The pressure is given in dbar so we will convert to depth (meters) assuming
# 1 dbar = 1 meter (this is approximately true).
z = readdlm("reference_pressure.txt")[:]

# We also flatten the arrays by indexing with a Colon [:] to convert the arrays
# from N×1 arrays to 1D arrays of length N.
T = readdlm("reference_temperature.txt")[:]
S = readdlm("reference_salinity.txt")[:]

# Get the indices of all the non-NaN values.
T_good_inds = findall(!isnan, T)
S_good_inds = findall(!isnan, S)

# Create T and S arrays that do not contain NaNs, along with corresponding
# z values.
T_good = T[T_good_inds]
S_good = S[S_good_inds]

z_T = z[T_good_inds]
z_S = z[S_good_inds]

# Linearly interpolate T and S profiles to model grid.
Ti = LinearInterpolation(z_T, T_good, extrapolation_bc=Flat())
Si = LinearInterpolation(z_S, S_good, extrapolation_bc=Flat())

zC = model.grid.zC
T₀ = Ti.(-zC)
S₀ = Si.(-zC)

# Plot and save figures of reference and interpolated profiles.
T_fpath = "temperature_profiles.png"
S_fpath = "salinity_profiles.png"

T_plot = plot(T_good, -z_T, label="Reference", xlabel="Temperature (C)", ylabel="Depth (m)", grid=false, dpi=300)
plot!(T_plot, T₀, zC, label="Interpolation")

@info "Saving temperature profiles to $T_fpath..."
savefig(T_plot, T_fpath)

S_plot = plot(S_good, -z_S, label="Reference", xlabel="Salinity (ppt)", ylabel="Depth (m)", grid=false, dpi=300)
plot!(S_plot, S₀, zC, label="Interpolation")

@info "Saving temperature profiles to $S_fpath..."
savefig(S_plot, S_fpath)

#####
##### Setting up initial conditions
#####

T₀_3D = repeat(reshape(T₀, 1, 1, Nz), Nx, Ny, 1)
S₀_3D = repeat(reshape(S₀, 1, 1, Nz), Nx, Ny, 1)

set!(model.tracers.T, T₀_3D)
set!(model.tracers.S, S₀_3D)

# Set meltwater concentration to 1 at the source.
# model.tracers.meltwater.data[source_index...] = 1  # Point source
model.tracers.meltwater.data[:, source_index[2], source_index[3]] .= 1  # Line source

#####
##### Write out 3D fields and slices to NetCDF
#####

fields = Dict(
        "u" => model.velocities.u,
        "v" => model.velocities.v,
        "w" => model.velocities.w,
        "T" => model.tracers.T,
        "S" => model.tracers.S,
"meltwater" => model.tracers.meltwater,
       "nu" => model.diffusivities.νₑ,
   "kappaT" => model.diffusivities.κₑ.T,
   "kappaS" => model.diffusivities.κₑ.S
)

output_attributes = Dict(
        "u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
        "v" => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
        "w" => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
        "T" => Dict("longname" => "Temperature", "units" => "C"),
        "S" => Dict("longname" => "Salinity", "units" => "g/kg"),
"meltwater" => Dict("longname" => "Meltwater concentration"),
       "nu" => Dict("longname" => "Nonlinear LES viscosity", "units" => "m^2/s"),
   "kappaT" => Dict("longname" => "Nonlinear LES diffusivity for temperature", "units" => "m^2/s"),
   "kappaS" => Dict("longname" => "Nonlinear LES diffusivity for salinity", "units" => "m^2/s")
)


ow = model.output_writers
ow[:field_writer] = NetCDFOutputWriter(model, fields; filename="ice_shelf_meltwater_outflow_fields.nc",
                                       interval=6hour, output_attributes=output_attributes)

ow[:depth_slice_writer] = NetCDFOutputWriter(model, fields; filename="ice_shelf_meltwater_outflow_source_xy_slice.nc",
                                             interval=5minute, output_attributes=output_attributes, zC=source_index[3], zF=source_index[3])

ow[:surface_slice_writer] = NetCDFOutputWriter(model, fields; filename="ice_shelf_meltwater_outflow_surface_xy_slice.nc",
                                               interval=5minute, output_attributes=output_attributes, zC=Nz, zF=Nz)

ow[:calving_front_slice_writer] = NetCDFOutputWriter(model, fields; filename="ice_shelf_meltwater_outflow_calving_front_xz_slice.nc",
                                                     interval=5minute, output_attributes=output_attributes, yC=1, yF=2)

ow[:along_channel_slice_writer] = NetCDFOutputWriter(model, fields; filename="ice_shelf_meltwater_outflow_along_channel_yz_slice.nc",
                                                     interval=5minute, output_attributes=output_attributes, xC=source_index[1], xF=source_index[1])

#####
##### Print banner
#####

#####
##### Time step!
#####

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.3, Δt=1second, max_change=1.2, max_Δt=30second)

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 50

# Convinient alias
C_mw = model.tracers.meltwater

while model.clock.time < end_time
    walltime = @elapsed begin
        time_step!(model; Nt=Ni, Δt=wizard.Δt)

        # C_mw.data[source_index...] = 1  # Point source
        C_mw.data[:, source_index[2], source_index[3]] .= 1  # Line source
    end

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / end_time)

    # Find maximum velocities.
    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = cfl(model)

    # Find maximum ν and κ.
    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)
    dCFL = dcfl(model)

    # Calculate a new adaptive time step.
    update_Δt!(wizard, model)

    # Print progress statement.
    @printf("[%06.2f%%] i: %d, t: %5.2f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: %6.4g, next Δt: %8.5g s, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time / day, umax, vmax, wmax, CFL, νmax, κmax, dCFL, wizard.Δt, prettytime(walltime / Ni))
end
