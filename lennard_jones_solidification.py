"""
==================================================================
lennard_jones_solidification.py — Controlled cooling & phase transition
==================================================================

Purpose
-------
Study the liquid -> solid transition of a 3D Lennard-Jones system
through progressive cooling.

The program computes:
    - the Lennard-Jones forces
    - the Velocity Verlet time integration
    - the thermal evolution (T_ini -> T_fin)
    - the number of neighbours (local order indicator)
    - the energies and temperature

Special feature:
-> variable target temperature (linear schedule)
-> soft periodic thermostat
-> visualization of the local structural order

Main discrete variables:
    DIM            : spatial dimension (3)
    Natom          : number of particles (64)
    itmax          : total number of iterations
    fastSteps      : sub-steps per animation frame
    rescale_every  : thermostat application frequency
    latticeSide    : discrete size of the initial lattice

Lennard-Jones reduced units:
    m = 1, sigma = 1, epsilon = 1
-> dimensionless system.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


# ───────────────────────── General parameters ─────────────────────────

DIM     = 3          # Spatial dimension (3D)
Natom   = 64          # Total number of particles
d0      = 1.0         # Initial lattice spacing
Ratom   = 0.3         # Particle radius used for display only
vini    = 0.2         # Initial velocity magnitude
h       = 0.01        # Integration time step
itmax   = 2001         # Total number of iterations
fastSteps = 5          # Number of sub-steps per frame

VELOCITY_MODE = "random"   # "random" or "fixed"

latticeSide = int(Natom**(1 / DIM) + 0.99)   # Particles per side
L = latticeSide * d0                          # Size of the simulation box

T_ini   = 0.75         # Initial temperature
T_fin   = 0.0005       # Final temperature
rescale_every = 2      # Thermostat frequency


# ───────────────────────── Periodic boundary conditions ─────────────────────────

def distance_Periodic(dr, L):
    """Minimum image convention."""
    return dr - L * np.rint(dr / L)

def coordonee_Periodic(posi, L):
    """Wrap particle coordinates back into the main simulation box."""
    return (posi + 0.5 * L) % L - 0.5 * L


# ───────────────────────── Lennard-Jones forces ─────────────────────────

def forces(fr, posi):
    """
    Compute the total Lennard-Jones force acting on every particle,
    using the minimum image convention and Newton's third law.
    """

    fr[:, :] = 0.0   # Reset the force array

    for i in range(Natom - 1):
        for j in range(i + 1, Natom):

            dr = distance_Periodic(posi[i] - posi[j], L)
            r2 = dr.dot(dr)
            rm6 = r2 ** -3

            # Analytical expression of the LJ force
            fij = (48.0 / r2) * (rm6 - 0.5) * rm6 * dr

            fr[i] += fij
            fr[j] -= fij

    return fr


# ───────────────────────── System energies ─────────────────────────

def Ekinetic(vel):
    """Total kinetic energy."""
    return 0.5 * np.sum(vel * vel)

def Epotential(posi):
    """Total Lennard-Jones potential energy."""
    Ep = 0.0
    for i in range(Natom):
        for j in range(i):
            dr = distance_Periodic(posi[i] - posi[j], L)
            r = np.linalg.norm(dr)
            Ep += 4.0 * (r**-12 - r**-6)
    return Ep


# ───────────────────────── Structural analysis ─────────────────────────

def compute_neighbors(posi, rc=1.35):
    """
    Count, for every particle, the number of neighbours located
    within a cutoff radius rc. Used as a simple local order
    parameter to detect crystallization.
    """
    neigh = np.zeros(Natom)

    for i in range(Natom):
        for j in range(Natom):

            if i == j:
                continue

            dr = distance_Periodic(posi[i] - posi[j], L)
            r = np.linalg.norm(dr)

            if r < rc:
                neigh[i] += 1

    return neigh


# ───────────────────────── Velocity Verlet integrator ─────────────────────────

def veloverlet(h, nsteps, T_cible):
    """
    Advance the system by `nsteps` Velocity Verlet steps of size `h`,
    applying a soft velocity-rescaling thermostat every
    `rescale_every` sub-steps to drive the system towards the
    target temperature `T_cible`.
    """

    global vel, posi, fr

    for nt in range(nsteps):

        vel += 0.5 * h * fr       # Half-step velocity update
        posi += h * vel           # Position update
        posi[:] = coordonee_Periodic(posi, L)

        fr = forces(fr, posi)     # Recompute forces
        vel += 0.5 * h * fr       # Second half-step velocity update

        # Soft thermostat
        if nt % rescale_every == 0:
            Ek = Ekinetic(vel)
            alpha = math.sqrt(DIM / 2.0 * Natom * T_cible / Ek)
            vel *= alpha


# ───────────────────────── Cooling schedule ─────────────────────────

def T_schedule(i_frame, total_frames):
    """Linear interpolation of the target temperature, T_ini -> T_fin."""
    frac = i_frame / max(total_frames - 1, 1)
    return T_ini * (1 - frac) + T_fin * frac


# ───────────────────────── Main animation loop ─────────────────────────

def animate(i):
    """
    Animation callback: advances the dynamics under the current
    target temperature, updates the structural order colour map
    and the energy/temperature curves.
    """

    global Ekpm

    T_cible = T_schedule(i, itmax // fastSteps + 1)

    if i:
        veloverlet(h, fastSteps, T_cible)

    # Compute the number of neighbours
    neigh = compute_neighbors(posi)

    # Normalized order parameter
    order = np.clip((neigh - 6) / 6, 0, 1)

    atoms.set_array(order)
    atoms.set_offsets(posi[:, :2])

    currtime = i * fastSteps * h
    ttime.append(currtime)

    ek = Ekinetic(vel) / Natom
    ep = Epotential(posi) / Natom
    T_inst = 2.0 * ek / DIM

    Ekpm = np.append(Ekpm, [[ek, T_inst, ep, ek + ep]], axis=0)

    for k in range(4):
        Ecurve[k].set_data(ttime, Ekpm[:, k])

    if i % 20 == 0:
        print(
            f"\n t={currtime:.2f}"
            f" | mean neighbours={np.mean(neigh):.2f}"
            f" | T={T_inst:.3f}"
        )

    if len(ttime) > 1:
        axE.set_xlim(0, ttime[-1] + 0.05)

        ymin = np.min(Ekpm[:, :3]) - 0.2
        ymax = np.max(Ekpm[:, :3]) + 0.2
        axE.set_ylim(ymin, ymax)

    return [atoms] + Ecurve


# ───────────────────────── Initial positions ─────────────────────────

fr   = np.zeros((Natom, DIM))
posi = np.zeros((Natom, DIM))

for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = (
            (i // latticeSide**k) % latticeSide
            - (latticeSide - 1) * 0.5
        ) * d0


# ───────────────────────── Initial velocities ─────────────────────────

if VELOCITY_MODE == "random":

    vel = np.random.standard_normal((Natom, DIM))

elif VELOCITY_MODE == "fixed":

    directions = np.random.standard_normal((Natom, DIM))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    vel = vini * directions

else:
    raise ValueError("Choose 'random' or 'fixed'")

# Rescale the initial velocities to match T_ini
Ek0 = Ekinetic(vel)
alpha0 = math.sqrt(DIM / 2.0 * Natom * T_ini / Ek0)
vel *= alpha0


# ───────────────────────── Plotting setup ─────────────────────────

ttime = []
Ekpm  = np.empty((0, 4))

xmax = 0.7 * L
atomArea = (Ratom * 250 / xmax)**2

plt.style.use('dark_background')

fig = plt.figure('Lennard-Jones — Phase transition', figsize=(7, 9))

ax = fig.add_subplot(3, 1, (1, 2),
                     xlim=(-xmax, xmax),
                     ylim=(-xmax, xmax),
                     aspect="equal")

ax.set_title("Lennard-Jones")

atoms = ax.scatter(
    posi[:, 0],
    posi[:, 1],
    c=np.zeros(Natom),
    cmap='coolwarm',
    vmin=0,
    vmax=1,
    s=atomArea
)

axE = fig.add_subplot(3, 1, 3,
                      xlim=(0, itmax * h),
                      ylim=(-4.0, 2.5))

Ecurve = axE.plot(
    ttime, np.empty((0,)),
    ttime, np.empty((0,)),
    ttime, np.empty((0,)),
    ttime, np.empty((0,))
)

axE.set(
    xlabel=r"t / $\tau_0$",
    ylabel=r"energy / $\varepsilon_0$ per atom"
)

axE.legend(["$E_k$", "$T$", "$E_p$", "$E_m$"])
axE.axhline(0, color='white', lw=0.4, ls='--')

plt.tight_layout()

fr = forces(fr, posi)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=itmax // fastSteps + 1,
    blit=True,
    interval=1,
    repeat=False
)

plt.show()

print("\n=== Done ===")
print(f"t_final = {ttime[-1]:.2f}  |  final Ekpm = {Ekpm[-1]}")
