# Lennard-Jones Molecular Dynamics Simulation

A collection of Python simulations of a 2D/3D particle system interacting
through the **Lennard-Jones potential**, implemented from scratch with
`numpy` and animated with `matplotlib`. The project covers energy-conserving
dynamics (NVE), temperature control (NVT), individual particle trajectory
tracking, and a liquid-to-solid phase transition obtained by progressive
cooling.

<p align="center">
  <img width="818" height="355" alt="image" src="https://github.com/user-attachments/assets/6c71bae6-0aeb-46f1-bb8a-5f956c5ecff0" />

</p>

## Overview

Molecular dynamics (MD) is a numerical method used to study the time
evolution of a system of interacting particles by integrating Newton's
equations of motion. This repository implements a minimal, educational MD
engine for a classic pairwise interaction — the Lennard-Jones (LJ)
potential:

```
V(r) = 4ε [ (σ/r)^12 − (σ/r)^6 ]
```

All simulations use **LJ reduced units** (`m = σ = ε = k_B = 1`), periodic
boundary conditions with the minimum image convention, and the
**Velocity Verlet** integration scheme.

## Contents

| Script | Ensemble | Dimension | Description |
|---|---|---|---|
| [`src/lennard_jones_nve.py`](src/lennard_jones_nve.py) | NVE (microcanonical) | 2D | Energy-conserving dynamics with live animation of positions, kinetic/potential/total energy and temperature. |
| [`src/lennard_jones_nvt.py`](src/lennard_jones_nvt.py) | NVT (canonical) | 2D | Same as above, with a velocity-rescaling thermostat that keeps the temperature constant. |
| [`src/lennard_jones_nvt_trajectories.py`](src/lennard_jones_nvt_trajectories.py) | NVE, 3D | 3D | 3D simulation with a radial force cutoff; records and plots the individual trajectories of three selected particles. |
| [`src/lennard_jones_solidification.py`](src/lennard_jones_solidification.py) | Controlled cooling | 3D | Progressive cooling from a liquid-like state (`T_ini = 0.75`) to a near-zero temperature (`T_fin = 0.0005`), with a local order parameter (neighbour count) used to visualize crystallization. |

## Physics background

- **Force field**: standard 12-6 Lennard-Jones potential, with the force
  computed analytically as the gradient of the potential.
- **Boundary conditions**: periodic boundary conditions (PBC) with the
  minimum image convention, so the simulated particles behave as if
  embedded in an infinite, repeating system.
- **Integrator**: Velocity Verlet, a symplectic, time-reversible scheme
  well suited to Hamiltonian systems such as this one.
- **Thermostat**: simple isokinetic velocity rescaling
  (`v -> α·v` with `α = sqrt(T_target / T_instantaneous)`), applied either
  every step (NVT) or on a fixed schedule with a linearly decreasing
  target temperature (solidification).
- **Order parameter**: in the solidification script, the local
  coordination number (number of neighbours within a cutoff radius) is
  used as a simple proxy for crystalline order.

## Installation

```bash
git clone https://github.com/<your-username>/lennard-jones-md-simulation.git
cd lennard-jones-md-simulation
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, `numpy`, `matplotlib`.

## Usage

Each script is self-contained and can be run directly. A `matplotlib`
animation window will open showing the particles and, below it, the
evolution of the energies and temperature over time.

```bash
python src/lennard_jones_nve.py
python src/lennard_jones_nvt.py
python src/lennard_jones_nvt_trajectories.py
python src/lennard_jones_solidification.py
```

Simulation parameters (number of particles, time step, number of
iterations, initial/final temperature, etc.) are defined as plain
constants near the top of each file and can be edited directly to explore
different regimes.

## Performance note

The current implementation uses explicit Python `for` loops over all
particle pairs (`O(N²)` per force evaluation), which keeps the code easy
to read but limits practical system sizes to a few hundred particles.
Possible improvements — listed here as ideas for future work rather than
implemented features — include vectorizing the force calculation with
`numpy` broadcasting, using cell lists / neighbour lists to reduce the
complexity to roughly `O(N)`, or moving performance-critical sections to
`numba`.

## Adding a demo

GitHub renders animated GIFs and images directly in the README, which
makes the project much easier to present. To add one:

1. Run a simulation and save the animation, e.g. with
   `ani.save("demo.gif", writer="pillow", fps=30)` instead of
   (or in addition to) `plt.show()`.
2. Place the file in a `media/` or `assets/` folder in the repository.
3. Reference it at the top of this README:
   `![demo](media/demo.gif)`.

## Results 

<img width="750" height="606" alt="image" src="https://github.com/user-attachments/assets/b0dbb721-d1a2-4751-acc5-c497d38420be" />
<img width="508" height="495" alt="image" src="https://github.com/user-attachments/assets/d8e28357-36a1-434d-aa3c-78227bdc8995" />
<img width="626" height="265" alt="image" src="https://github.com/user-attachments/assets/9bd31197-406f-4340-93a2-84cd728d7909" />




## Author

AWayToCreate — feel free to open an issue or a pull request for questions,
bug reports, or suggestions.
