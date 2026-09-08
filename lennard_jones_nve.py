"""
==============================================================
 Lennard-Jones Molecular Dynamics Simulation (NVE ensemble)
==============================================================

Purpose
-------
Simulate the time evolution of a set of particles interacting
through a Lennard-Jones potential, in the microcanonical (NVE)
ensemble: the number of particles N, the volume V and the total
energy E are conserved (no thermostat is applied).

The program computes:
    - the interatomic forces
    - the trajectories, via the Velocity Verlet integrator
    - the energies (kinetic, potential, total)
    - the instantaneous temperature

The simulation uses Lennard-Jones reduced units:
    mass     m       = 1
    length   sigma   = 1
    energy   epsilon = 1
    k_B              = 1
-> all quantities are dimensionless.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ───────────────────────── Physical and numerical parameters ─────────────────────────

DIM = 2              # Spatial dimension (here 2D)
Natom = 64            # Total number of particles
d0 = 1.2              # Initial spacing between particles on the lattice

# Number of particles per side of the square lattice
latticeSide = int(Natom**(1 / DIM) + .99)

# Size of the periodic simulation box
L = (latticeSide + 1) * d0

Ratom = 0.3           # Particle radius used for display only
vini = 0.5            # Amplitude of the initial random velocities
h = 0.01              # Integration time step
itmax = 1001           # Total number of iterations
fastSteps = 10          # Number of integration steps between two animation frames

kb = 1.0              # Boltzmann constant (reduced units)


# ───────────────────────── Periodic boundary conditions ─────────────────────────

def distance_Periodic(dr, L):
    """
    Apply the minimum image convention.
    Returns the shortest vector between two particles
    in a periodic system.
    """
    return dr - L * np.rint(dr / L)


def coordonee_Periodic(posi, L):
    """
    Wrap particle coordinates back into the main simulation
    box whenever they cross a periodic boundary.
    """
    return (posi + 0.5 * L) % L - 0.5 * L


# ───────────────────────── Lennard-Jones force calculation ─────────────────────────

def forces(fr, posi):
    """
    Compute the total Lennard-Jones force acting on every particle.

    Loops over all unique pairs (i < j), applies the minimum image
    convention, and accumulates the pairwise force using Newton's
    third law (action / reaction).
    """

    fr[:, :] = 0.0   # Reset the force array

    # Double loop over all unique pairs (i < j)
    for i in range(Natom - 1):
        for j in range(i + 1, Natom):

            # Vector distance between particles i and j
            dr = posi[i] - posi[j]

            # Apply periodic boundary conditions
            dr = distance_Periodic(dr, L)

            # Squared distance
            r2 = dr.dot(dr)

            # (1/r^6) term
            rm6 = r2 ** -3

            # Force derived from the LJ potential:
            # F = 48/r^2 * (1/r^6 - 1/2) * (1/r^6) * r_vector
            fij = (48.0 / r2) * (rm6 - 0.5) * rm6 * dr

            # Newton's third law
            fr[i] += fij
            fr[j] -= fij

    return fr


# ───────────────────────── System energies ─────────────────────────

def Ekinetic(vel):
    """
    Total kinetic energy:
    Ek = 1/2 * sum(v^2)   (with m = 1)
    """
    return 0.5 * np.sum(vel * vel)


def Epotential(posi):
    """
    Total Lennard-Jones potential energy, summed over all
    unique pairs (i < j).
    """
    Ep = 0.0

    for i in range(Natom):
        for j in range(i):

            dr = posi[i] - posi[j]
            dr = distance_Periodic(dr, L)

            r = np.linalg.norm(dr)

            # LJ potential
            Ep += 4.0 * (r**-12 - r**-6)

    return Ep


# ───────────────────────── Instantaneous temperature ─────────────────────────

def Temperature(vel):
    """
    Temperature from the equipartition theorem:
    kB * T = 2 * Ek / (3N)

    (Note: DIM = 2 here, but this original formula keeps the
    3N normalization instead of DIM * N.)
    """
    Ek = Ekinetic(vel)
    T = 2.0 * Ek / (3.0 * Natom * kb)
    return T


# ───────────────────────── Velocity Verlet integrator ─────────────────────────

def veloverlet(h, nsteps):
    """
    Advance the system by `nsteps` integration steps of size `h`
    using the Velocity Verlet scheme.
    """

    global vel, posi, fr

    for nt in range(nsteps):

        # 1) Half-step velocity update
        vel += 0.5 * h * fr

        # 2) Position update
        posi += h * vel

        # 3) Recompute forces at the new positions
        fr = forces(fr, posi)

        # 4) Second half-step velocity update
        vel += 0.5 * h * fr


# ───────────────────────── Animation loop ─────────────────────────

def animate(i):
    """
    Animation callback: advances the dynamics, updates the
    scatter plot of particle positions, and updates the
    energy/temperature curves.
    """

    global Ekpm

    # Advance the dynamics, except on the very first frame
    if i:
        veloverlet(h, fastSteps)

    # Update particle positions on the plot
    atoms.set_offsets(posi[:, :2])

    # Current simulation time
    currtime = i * fastSteps * h
    ttime.append(currtime)

    # Compute thermodynamic observables
    ep = Epotential(posi) / Natom
    ek = Ekinetic(vel) / Natom
    T = Temperature(vel)

    # Store: [Ek, Ep, Etot, T]
    Ekpm = np.append(Ekpm, [[ek, ep, ek + ep, T]], axis=0)

    # Update the energy curves
    for k in range(4):
        Ecurve[k].set_data(ttime, Ekpm[:, k])

    return [atoms] + Ecurve


# ───────────────────────── System initialization ─────────────────────────

fr = np.zeros((Natom, DIM))    # Force array
posi = np.zeros((Natom, DIM))  # Position array

# Build a centered square lattice
for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = (i // latticeSide**k) % latticeSide
        posi[i, k] = (posi[i, k] - (latticeSide - 1) * 0.5) * d0

# Initial velocities drawn from a Gaussian distribution
vel = vini * np.random.standard_normal((Natom, DIM))

ttime = []                     # List of recorded times
Ekpm = np.empty((0, 4))        # Storage for the energy/temperature history


# ───────────────────────── Plotting setup ─────────────────────────

xmax = 0.7 * L
atomCol = np.random.uniform(0, 1, Natom)
atomArea = (Ratom * 250 / xmax)**2

plt.style.use('dark_background')
fig = plt.figure('Lennard-Jones', figsize=(7, 8))

# Particle positions subplot
ax = fig.add_subplot(3, 1, (1, 2),
                     xlim=(-xmax, xmax),
                     ylim=(-xmax, xmax),
                     aspect="equal")

atoms = ax.scatter(posi[:, 0], posi[:, 1],
                   c=atomCol, s=atomArea)

# Energy subplot
axE = fig.add_subplot(3, 1, 3,
                      xlim=(0, itmax * h),
                      ylim=(-3., 5.))

Ecurve = axE.plot(ttime, Ekpm)
plt.legend(["Ek", "Ep", "Etot", "T"])

plt.tight_layout()

# Initial force calculation
fr = forces(fr, posi)

# Run the animation
ani = animation.FuncAnimation(
    fig, animate,
    frames=itmax // fastSteps + 1,
    blit=True,
    interval=1,
    repeat=False
)

plt.show()
