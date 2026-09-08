"""
==================================================================
lennard_jones_nvt_trajectories.py — 3D LJ dynamics with trajectory tracking
==================================================================

Purpose
-------
Simulate a 3D Lennard-Jones system and analyze the individual
dynamical behaviour of a few selected particles.

The program computes:
    - the Lennard-Jones forces with a radial cutoff
    - the time integration (Verlet)
    - the kinetic, potential and total energies
    - the 3D trajectories of 3 selected particles

Special feature:
-> explicit recording of the (x, y, z) positions
-> visualization of the spatial trajectories

Main discrete variables:
    DIM      : dimension (3)
    Natom    : total number of particles
    itmax    : number of time iterations
    tracked  : indices of the tracked particles
    n_side   : number of particles per side of the cubic lattice

Lennard-Jones reduced units:
    m = 1, sigma = 1, epsilon = 1
-> dynamics expressed in dimensionless units.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # required for 3D plotting


# ───────────────────────── System parameters ─────────────────────────

DIM = 3              # 3D simulation
Natom = 64            # Total number of particles
d0 = 1.0              # Initial cubic lattice spacing
dt = 0.01             # Time step
itmax = 1500           # Total number of time iterations

# Number of particles per side of the cube (cube root of N)
n_side = int(round(Natom ** (1 / 3)))

# Size of the cubic periodic box
Lbox = n_side * d0


# ───────────────────────── Periodic boundary conditions ─────────────────────────

def PBC(r):
    """Minimum image convention, used to simulate an infinite system."""
    return r - Lbox * np.round(r / Lbox)

def wrap(r):
    """Wrap particle coordinates back into the main simulation box."""
    return r % Lbox


# ───────────────────────── Initial positions ─────────────────────────

R = []  # Temporary list used to build the initial positions

# Build a simple cubic lattice
for x in range(n_side):
    for y in range(n_side):
        for z in range(n_side):
            R.append([x * d0, y * d0, z * d0])

R = np.array(R)   # Convert to a numpy array


# ───────────────────────── Initial velocities ─────────────────────────

# Random Gaussian velocities
V = np.random.normal(0, 1, (Natom, DIM))

# Remove the mean velocity (center of mass stays at rest)
V -= np.mean(V, axis=0)


# ───────────────────────── Lennard-Jones forces with cutoff ─────────────────────────

def forces(R):
    """
    Compute the Lennard-Jones forces and potential energy for all
    particle pairs within a radial cutoff (r_c = 3 in these units).
    """

    F = np.zeros_like(R)   # Force array, initialized to zero
    Ep = 0.0                # Total potential energy

    # Double loop over all unique pairs
    for i in range(Natom - 1):
        for j in range(i + 1, Natom):

            rij = PBC(R[i] - R[j])        # Periodic distance vector
            r2 = np.dot(rij, rij)         # Squared distance

            # Radial cutoff: pairs farther than r_c = 3 are ignored
            if r2 < 9.0:

                inv_r6 = 1 / r2**3
                inv_r12 = inv_r6**2

                # Force derived from the LJ potential
                fij = (48 * inv_r12 - 24 * inv_r6) * rij / r2

                # Action / reaction
                F[i] += fij
                F[j] -= fij

                # Accumulate potential energy
                Ep += 4 * (inv_r12 - inv_r6)

    return F, Ep


# ───────────────────────── Kinetic energy ─────────────────────────

def kinetic(V):
    """Kinetic energy: 1/2 * m * v^2 (m = 1 in reduced units)."""
    return 0.5 * np.sum(V * V)


# ───────────────────────── Tracked particles ─────────────────────────

tracked = [0, 1, 2]   # Indices of the particles whose trajectory is recorded

# Dictionary storing the x, y, z coordinates of each tracked particle
traj = {i: {"x": [], "y": [], "z": []} for i in tracked}


# ───────────────────────── Lists used for the energy plots ─────────────────────────

Ek_list = []   # Kinetic energy
Ep_list = []   # Potential energy
Et_list = []   # Total energy
time = []      # Time


# ───────────────────────── Main loop (Verlet integration) ─────────────────────────

F, Ep = forces(R)   # Initial force calculation

for it in range(itmax):

    # Position update (Verlet formula)
    R += V * dt + 0.5 * F * dt * dt

    # Apply periodic boundary conditions
    R = wrap(R)

    # Recompute forces at the new positions
    F_new, Ep = forces(R)

    # Velocity update
    V += 0.5 * (F + F_new) * dt

    F = F_new   # Update the stored forces

    # Kinetic energy
    Ek = kinetic(V)

    # Store energies per particle
    Ek_list.append(Ek / Natom)
    Ep_list.append(Ep / Natom)
    Et_list.append((Ek + Ep) / Natom)

    time.append(it * dt)

    # Record the trajectory of each tracked particle
    for i in tracked:
        traj[i]["x"].append(R[i, 0])
        traj[i]["y"].append(R[i, 1])
        traj[i]["z"].append(R[i, 2])


# ───────────────────────── Energy plot ─────────────────────────

plt.figure(figsize=(7, 4))

plt.plot(time, Ek_list, label="Kinetic energy")
plt.plot(time, Ep_list, label="Potential energy")
plt.plot(time, Et_list, label="Total energy")

plt.xlabel("Time")
plt.ylabel("Energy per particle")
plt.legend()
plt.title("Energy evolution")

plt.show()


# ───────────────────────── 3D trajectories ─────────────────────────

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory of each tracked particle
for i in tracked:
    ax.plot(
        traj[i]["x"],
        traj[i]["y"],
        traj[i]["z"],
        label=f"Particle {i}"
    )

ax.set_title("Trajectories of 3 particles")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

plt.show()
