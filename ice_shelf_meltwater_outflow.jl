using DelimitedFiles, Printf
using Interpolations, Plots
using Oceananigans

using Oceananigans.Diagnostics: cell_advection_timescale

####
#### Some useful constants
####

const km = 1000
const Ω_Earth = 7.292115e-5  # [s⁻¹]
const φ = -75  # degrees latitude

####
#### Model grid and domain size
####

arch = CPU()
FT = Float64

Nx = 32
Ny = 32
Nz = 32

Lx = 10km
Ly = 10km
Lz = 1km

end_time = 7day

####
#### Set up source of meltwater: We will implement a source of meltwater as
#### a relaxation term towards a reference T and S value at a single point.
#### This is in effect weakly imposing a Value/Dirchlet boundary condition.
####

λ = 1/(1minute)  # Relaxation timescale [s⁻¹].

# Temperature and salinity of the meltwater outflow.
T_source = -1
S_source = 33.95

# Index of the point source at the middle of the southern wall.
source_index = (1, Int(Ny/2), Int(Nz/2))

@inline T_point_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((i, j, k) == p.source_index, -p.λ * (C.T[i, j, k] - p.T_source), 0)

@inline S_point_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((i, j, k) == p.source_index, -p.λ * (C.S[i, j, k] - p.S_source), 0)

forcing = ModelForcing(T = T_point_source, S = S_point_source)
params = (source_index=source_index, T_source=T_source, S_source=S_source, λ=λ)

####
#### Set up model
####

model = Model(
           architecture = arch,
             float_type = FT,
                   grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(0, Ly), z=(-Lz, 0)),
               coriolis = FPlane(rotation_rate=Ω_Earth, latitude=φ),
               buoyancy = SeawaterBuoyancy(),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = ChannelSolutionBCs(),
                forcing = forcing,
             parameters = params
)

####
#### Read reference profiles from disk
#### As these profiles are derived from observations, we will have to do some
#### post-processing to be able to use them as initial conditions.
####
#### We will get rid of all NaN values and use the remaining data to linearly
#### interpolate the T and S profiles to the model's grid.
####

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

####
#### Setting up initial conditions
####

T₀_3D = repeat(reshape(T₀, 1, 1, Nz), Nx, Ny, 1)
S₀_3D = repeat(reshape(S₀, 1, 1, Nz), Nx, Ny, 1)

set!(model.tracers.T, T₀_3D)
set!(model.tracers.S, S₀_3D)

####
#### Write out 3D fields to JLD2
####

fields = Dict(
     :u => model -> Array(model.velocities.u.data.parent),
     :v => model -> Array(model.velocities.v.data.parent),
     :w => model -> Array(model.velocities.w.data.parent),
     :T => model -> Array(model.tracers.T.data.parent),
     :S => model -> Array(model.tracers.S.data.parent),
    :nu => model -> Array(model.diffusivities.νₑ.data.parent),
:kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
:kappaS => model -> Array(model.diffusivities.κₑ.S.data.parent)
)

# field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix="meltwater_outflow_point_source_fields",
#                                 init=init_save_parameters_and_bcs,
#                                 max_filesize=100GiB, interval=6hour, force=true, verbose=true)
# push!(model.output_writers, field_writer)

####
#### Write out slices to JLD2
####

####
#### Time step!
####

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.3, Δt=1second, max_change=1.2, max_Δt=30second)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 50

while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)

    k = Int(Nz/2)
    u_slice = rotr90(model.velocities.u.data[1:Nx+1, 1:Ny, k])
    w_slice = rotr90(model.velocities.w.data[1:Nx, 1:Ny, k])
    T_slice = rotr90(model.tracers.T.data[1:Nx, 1:Ny, k])
    S_slice = rotr90(model.tracers.S.data[1:Nx, 1:Ny, k])

    xC, xF, yC = model.grid.xC, model.grid.xF, model.grid.yC
    pu = contour(xF, yC, u_slice; xlabel="x", ylabel="y", fill=true, levels=10, color=:balance, clims=(-0.2, 0.2))
    pw = contour(xC, yC, w_slice; xlabel="x", ylabel="y", fill=true, levels=10, color=:balance, clims=(-0.2, 0.2))
    pT = contour(xC, yC, T_slice; xlabel="x", ylabel="y", fill=true, levels=10, color=:thermal)
    pS = contour(xC, yC, S_slice; xlabel="x", ylabel="y", fill=true, levels=10, color=:haline) 

    t = @sprintf("%.2f hours", model.clock.time / hour)
    display(plot(pu, pw, pT, pS, title=["u (m/s), t=$t" "w (m/s)" "T (C)" "S (ppt)"], show=true))

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / end_time)

    # Calculate advective CFL number.
    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = wizard.Δt / cell_advection_timescale(model)

    # Calculate diffusive CFL number.
    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)

    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    νCFL = wizard.Δt / (Δ^2 / νmax)
    κCFL = wizard.Δt / (Δ^2 / κmax)

    # Calculate a new adaptive time step.
    update_Δt!(wizard, model)

    # Print progress statement.
    @printf("[%06.2f%%] i: %d, t: %5.2f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: (%6.4g, %6.4g), next Δt: %8.5g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time / day, umax, vmax, wmax, CFL, νmax, κmax, νCFL, κCFL, wizard.Δt, prettytime(walltime / Ni))
end
