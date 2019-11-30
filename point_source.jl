using DelimitedFiles
using Oceananigans
using Interpolations
using Plots

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

Nx = 128
Ny = 128
Nz = 100

Lx = 10km
Ly = 10km
Lz = 1km

end_time = 7day

####
#### Set up boundary conditions
####

# Point source of freshwater with S = 30 ppt on the middle of the Southern wall.
Sᶠ = 35 * ones(Ny, Nz)
Sᶠ[Int(Ny/2), Int(Nz/2)] += -5

S_bcs = ChannelBCs(south = BoundaryCondition(Value, Sᶠ))
bcs = ChannelSolutionBCs(S = S_bcs)

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
    boundary_conditions = bcs
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

exit()

####
#### Setting up initial conditions
####

∂T∂z = 0.01
T₀(x, y, z) = 10 + ∂T∂z * z
S₀(x, y, z) = 35

set!(model.tracers.T, T₀)
set!(model.tracers.S, S₀)

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

field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix="meltwater_outflow_point_source_fields",
                                init=init_save_parameters_and_bcs,
                                max_filesize=100GiB, interval=6hour, force=true, verbose=true)
push!(model.output_writers, field_writer)

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
