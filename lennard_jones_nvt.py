"""
==============================================================
 lennard_jones_nvt.py — Lennard-Jones MD with a thermostat (NVT)
==============================================================

Purpose
-------
Simulate the dynamics of a system of Lennard-Jones particles
while keeping the temperature constant using a simple velocity
rescaling thermostat (canonical / NVT ensemble).

The program computes:
    - the interatomic forces (Lennard-Jones)
    - the time integration (Velocity Verlet)
    - the energies Ek, Ep, Etot
    - the instantaneous temperature

Main discrete variables:
    DIM         : spatial dimension (2)
    Natom       : number of atoms (64)
    itmax       : total number of iterations
    fastSteps   : internal steps between two animation frames
    latticeSide : number of particles per lattice side

Lennard-Jones reduced units:
    m = 1, sigma = 1, epsilon = 1, k_B = 1
-> all quantities are dimensionless.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ───────────────────────── System parameters ─────────────────────────

DIM = 2                # Spatial dimension (2D)
Natom = 64              # Number of particles
d0 = 1.2                # Initial lattice spacing

latticeSide = int(Natom**(1 / DIM) + .99)   # Number of particles per side
L = (latticeSide + 1) * d0                  # Size of the periodic box

Ratom = 0.3             # Particle radius used for display only
vini = 0.5              # Amplitude of the initial velocities
h = 0.01                # Integration time step
itmax = 1001             # Total number of iterations
fastSteps = 5             # Internal steps between two display updates

kb = 1.0                # Boltzmann constant (reduced units)


# ───────────────────────── Periodic boundary conditions ─────────────────────────

def distance_Periodic(dr, L):
    """Minimum image convention: shortest vector between two particles."""
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

            dr  = posi[i] - posi[j]        # Raw distance vector
            dr  = distance_Periodic(dr, L) # Periodic distance vector

            r2  = dr.dot(dr)               # Squared distance
            rm6 = r2 ** -3                 # 1/r^6

            # Force derived from the LJ potential
            fij = (48.0 / r2) * (rm6 - 0.5) * rm6 * dr

            # Action-reaction principle
            fr[i] += fij
            fr[j] -= fij

    return fr


# ───────────────────────── Energies and temperature ─────────────────────────

def Ekinetic(vel):
    """Total kinetic energy."""
    return 0.5 * np.sum(vel * vel)

def Epotential(posi):
    """Total Lennard-Jones potential energy."""
    Ep = 0.0
    for i in range(Natom):
        for j in range(i):
            dr = posi[i] - posi[j]
            dr = distance_Periodic(dr, L)
            r  = np.linalg.norm(dr)
            Ep += 4.0 * (r**-12 - r**-6)
    return Ep

def Temperature(vel):
    """Instantaneous temperature from the equipartition theorem."""
    Ek = Ekinetic(vel)
    T = 2.0 * Ek / (DIM * Natom * kb)
    return T


# ───────────────────────── Integrator + thermostat ─────────────────────────

def veloverlet(h, nsteps):
    """
    Advance the system by `nsteps` Velocity Verlet steps, applying
    a simple velocity-rescaling thermostat after every step so that
    the kinetic energy is pinned to the target value (Natom * kb * T_target,
    with T_target = 1 implied here).
    """

    global vel, posi, fr

    for _ in range(nsteps):

        vel += 0.5 * h * fr         # Half-step velocity update
        posi += h * vel             # Position update
        posi[:] = coordonee_Periodic(posi, L)

        fr = forces(fr, posi)       # Recompute forces
        vel += 0.5 * h * fr         # Second half-step velocity update

        # Velocity-rescaling thermostat
        alpha = math.sqrt(Natom * 1 / Ekinetic(vel))
        vel = alpha * vel           # Rescale velocities


# ───────────────────────── Animation loop ─────────────────────────

def animate(i):
    """Animation callback: advances the dynamics and updates all plots."""

    global Ekpm

    if i:
        veloverlet(h, fastSteps)

    atoms.set_offsets(posi[:, :2])   # Update particle positions on the plot

    currtime = i * fastSteps * h
    ttime.append(currtime)

    ep = Epotential(posi) / Natom
    ek = Ekinetic(vel) / Natom
    T  = Temperature(vel)

    Ekpm = np.append(Ekpm, [[ek, ep, ek + ep, T]], axis=0)

    for k in range(4):
        Ecurve[k].set_data(ttime, Ekpm[:, k])

    return [atoms] + Ecurve


# ───────────────────────── Initialization ─────────────────────────

fr = np.zeros((Natom, DIM))
posi = np.zeros((Natom, DIM))

# Centered square lattice
for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = (i // latticeSide**k) % latticeSide
        posi[i, k] = (posi[i, k] - (latticeSide - 1) * 0.5) * d0

# Gaussian initial velocities
vel = vini * np.random.standard_normal((Natom, DIM))

ttime = []
Ekpm = np.empty((0, 4))


# ───────────────────────── Visualization setup ─────────────────────────

xmax = 0.7 * L
atomCol = np.random.uniform(0, 1, Natom)
atomArea = (Ratom * 250 / xmax)**2

plt.style.use('dark_background')
fig = plt.figure('Lennard-Jones', figsize=(7, 8))

ax = fig.add_subplot(3, 1, (1, 2),
                     xlim=(-xmax, xmax),
                     ylim=(-xmax, xmax),
                     aspect="equal")

atoms = ax.scatter(posi[:, 0], posi[:, 1],
                  c=atomCol, s=atomArea)

axE = fig.add_subplot(3, 1, 3,
                      xlim=(0, itmax * h),
                      ylim=(-3., 5.))

Ecurve = axE.plot(ttime, Ekpm)

plt.legend(["Ek", "Ep", "Etot", "T"])
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
