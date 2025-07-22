# QSimJy — Quantum Dot Device Simulator

QSimJy is a modular toolkit for **approximate simulation and design of semiconductor quantum‑dot devices**.  
It bundles electrostatics, Schrödinger‑equation solvers, micromagnetics and genetic optimisation into a single Python package, with optional C++/OpenMP back‑ends for heavy kernels.

---

## Key Features

- **DXF / GDS import** – load and visualise gate layouts with minimal boiler‑plate.  
- **High‑performance kernels** – 100× speed‑ups via C++/OpenMP for potential and stray‑field calculations.  
- **Extensible workflow** – mix‑and‑match modules or script full pipelines in notebooks.  
- **Fabrication awareness** – lithography‑blur models & penalty terms keep designs realistic.

---

## Package Modules

| Module | Purpose | Highlights |
| ------ | ------- | ---------- |
| `qsimjy.quantumdot` | Electrostatic potential solver | DXF import, bias assignment, fast Green‑function kernel |
| `qsimjy.solver` | Schrödinger equation | Finite‑difference grid, sparse eigensolvers |
| `qsimjy.magnet` | Micromagnetic simulation | CAD‑defined magnets, MuMax3 control, C++ stray‑field kernel |
| `qsimjy.optimization.genetic` | Genetic optimisation | Polygon‑blending & grid‑mesh GAs for magnet design |

---

## Installation

```bash
git clone https://github.com/your‑username/qsimjy.git
cd qsimjy
pip install .
```

> **Requirements**: Python ≥ 3.8, `numpy`, `scipy`, `ezdxf`, `shapely`, `matplotlib`, and the `mumax3` CLI (optional).

### Optional native extensions

To build the optional C++/OpenMP kernels:

```bash
pip install pybind11 meson ninja
meson setup build && meson compile -C build
```

---

## Quick Start

```python
import qsimjy as qsim

# 1. Load and bias a DXF gate layout
qd = qsim.quantumdot.QuantumDotDevice()
qd.add_gate_set(qsim.quantumdot.load_dxf('device.dxf'))
qd.set_boundary([(-0.3, -0.3), (0.3, 0.3)])
qd.calc_potential()

# 2. Solve bound states
solver = qsim.solver.Schrodinger2D(qd.potential_xlist,
                                   qd.potential_ylist)
solver.set_potential(qd.potential_value)
energies, wavefuncs = solver.solve()
```

---

## Tutorials & Documentation

Interactive Jupyter‑Book with step‑by‑step notebooks:

1. **Approximate Potential Calculation**  
2. **Micromagnet Simulation with MuMax3**  
3. **Genetic Optimisation for Micromagnet Design**

See the `docs/` folder or https://your‑username.github.io/qsimjy for rendered pages.

---

## Contributing

1. Fork the repo and create a feature branch.  
2. Follow the coding style (PEP 8 for Python, `clang‑format` for C++).  
3. Add or update unit tests in `tests/`.  
4. Open a pull request; CI must pass before review.

---

## License

This project is licensed under the **MIT License**.  
© 2025 Integrated Quantum Systems Lab.

---

## Citation

```
J. Yun et al., "QSimJy: A Modular Quantum‑Dot Device Simulator",
Integrated Quantum Systems Lab, SNU (2025).
```

---

## Contact

Questions, bug reports and feature requests are welcome via the  
[GitHub issues page](https://github.com/your‑username/qsimjy/issues).

Happy simulating!
