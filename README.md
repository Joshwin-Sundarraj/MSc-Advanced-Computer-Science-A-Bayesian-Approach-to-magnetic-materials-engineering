# MSc-Advanced-Computer-Science-A-Bayesian-Approach-to-magnetic-materials-engineering

# Magnetic Bayesian Optimization (2D: ku, damp)

A Bayesian Optimization implementation for magnetic parameter extraction of anisotropy and effective damping `(ku, damp)`.  
This supports Gaussian Process (GP) and Random Forest (RF) surrogate models with a **Sequential Domain Reduction (SDR)** implemented in PyTorch
based on https://github.com/bayesian-optimization/BayesianOptimization . This implementation allows to assign different zooming factors for each parameter.
Additionally a Neighbour Aware Search (NAS) mechanism has been implemented.

- For reproducibility deterministic seeding for Sobol init, acquisition optimization, and RF (unless overridden) is included.
- Optional per iteration logs and total runtime print are available.
- Common BoTorch warnings are suppressed by default, this can be disabled.

> **No pip installation required.** Keep your **notebook**vand your **data folders** in the same local project directory.  
> The **big dataset (~9 GB)** is **not** in the repo (GitHub size limit). It’s **available on request**. See “Data layout”.

---

## Project layout (recommended)

```
your_project/
│─ your_experiment.ipynb              # <- your notebook (same repo)
│
├─ ensemble/                          # <- your ensemble/reference outputs (in repo)
│   └─ ...                            #     e.g., Wc reference files
└─ big_dataset/                       # <- ~9 GB, NOT on GitHub (available on request)
│   └─ single_spin_reference_nofield_30_new/
│      └─ Anis_<ku>e-22Damp_<damp>/output
└─ README.md
```
---

## Data layout & variables you provide

Your **objective function** will typically load simulated outputs from disk and compare to a reference trace:

- `source`: path to your **big dataset** root (local, not in Git).  
  Example: `data/big_dataset/single_spin_reference_nofield_30_new`
- `Wc`: your **reference** tensor (e.g., from `data/ensemble/...`).
- `Ku`, `Damp`: your **grids** (numpy arrays) of allowed values.

Example prep:

```python
import os, numpy as np, torch

# Grids (real units)
Ku   = np.arange(0.0, 2.7, 0.015)
Damp = np.arange(0.01, 0.2001, 0.001)

# Bounds (real units)
bounds_real = torch.tensor([[Ku.min(), Damp.min()],
                            [Ku.max(), Damp.max()]], dtype=torch.double)

# Local paths
source = "data/big_dataset/single_spin_reference_nofield_30_new"  # large dataset (local only)
ref_file = "data/ensemble/Anis_2.625e-22Damp_0.051/output"       # example reference

# Reference signal
Wc = torch.tensor(np.loadtxt(ref_file), dtype=torch.double)
```

> **Big dataset (~9 GB)**: not included in the repo due to GitHub’s 2 GB limit.  
> **Availability**: the dataset is **available on request**. Place it in the root as shown above.

---

## Define your objective (examples)

Your function receives a **2-D torch tensor** `(ku, damp)` in **real units** and returns a **scalar torch tensor** (larger is better). Use `snap` to map to your discrete grids.

### Value-space loss
```python
import os, numpy as np, torch

def objective_func(candidate: torch.Tensor) -> torch.Tensor:
    ku_val   = float(snap(candidate[0], Ku))
    damp_val = float(snap(candidate[1], Damp))
    ku_str   = format(ku_val, '.3f')
    damp_str = format(damp_val, '.3f')

    filename = os.path.join(source, f'Anis_{ku_str}e-22Damp_{damp_str}', 'output')
    W0 = torch.tensor(np.loadtxt(filename), dtype=torch.double)

    chi = torch.sum((W0[6:, 2:5] - Wc[6:, 2:5]) ** 2)
    return -chi
```

### Value + low-frequency FFT loss
```python
import os, numpy as np, torch

def objective_func_fft(candidate: torch.Tensor) -> torch.Tensor:
    ku_val   = float(snap(candidate[0], Ku))
    damp_val = float(snap(candidate[1], Damp))
    ku_str   = format(ku_val, '.3f'); damp_str = format(damp_val, '.3f')

    filename = os.path.join(source, f'Anis_{ku_str}e-22Damp_{damp_str}', 'output')
    W0 = torch.tensor(np.loadtxt(filename), dtype=torch.double)

    W0_sig = W0[6:, 2:5]
    Wc_sig = Wc[6:, 2:5]

    alpha = 1.0
    beta  = 6.073e-3
    N     = 9

    value_loss = torch.sum((W0_sig - Wc_sig) ** 2)
    W0_fft = torch.fft.rfft(W0_sig, dim=0)[:N]
    Wc_fft = torch.fft.rfft(Wc_sig, dim=0)[:N]
    fft_loss = torch.sum(torch.view_as_real(W0_fft - Wc_fft) ** 2)

    return -(alpha * value_loss + beta * fft_loss)
```

---

## Running BO

```python

cfg = MagneticBayesianOptimizationConfig(
    objective=objective_func,   # or objective_func_fft
    ku=Ku, damp=Damp, bounds_real=bounds_real,

    # Model
    model_type="gp",
    kernel="matern",

    # BO loop
    n_initial=500,
    n_runs=150,

    # Heuristics
    use_nas=True,
    use_sdr=True,
    filter_outside_window=True,
    shrink_factor=torch.tensor([0.9, 0.945], dtype=torch.double),
    min_window=0.0,

    # Acquisition (EI)
    xi=0.1,                     # EI exploration bias (main proposals)
    neighbor_xi=0.01,           # EI bias for NAS

    # Acquisition optimizer
    raw_samples=500,
    num_restarts=10,

    # Reproducibility / UX
    seed=123,                   # global seed; if None, one is generated
    suppress_warnings=True,
    verbose=True,
    print_total_time=True,
)

runner = MagneticBayesianOptimization(cfg)
ku_opt, damp_opt = runner.run()
print("Optimal (ku, damp):", ku_opt, damp_opt)
```

---

## Configuration reference

**Class:** `MagneticBayesianOptimizationConfig`

### Required
- `objective: Callable[[Tensor], Tensor]` — your objective function `(ku, damp)` → scalar torch tensor (maximize).
- `ku: np.ndarray` — grid of allowed ku values.
- `damp: np.ndarray` — grid of allowed damping values.
- `bounds_real: torch.Tensor[[2,2], double]` — search box in real units: `[[ku_min, damp_min], [ku_max, damp_max]]`.

### Model selection
- `model_type: "gp"|"rf"` — default `"gp"`.
- `kernel: "matern"|"rbf"` — GP only; default `"matern"`.

### Random Forest (when `model_type="rf"`)
- `rf_n_estimators: int = 100` - Number of trees
- `rf_random_state: Optional[int] = None` (defaults to global `seed` if `None`)

### BO loop
- `n_initial: int = 500` — Sobol QMC initial points (real units).
- `n_runs: int = 150` — iterations.

### Acquisition (EI)
- `xi: float = 0.1` — exploration bias for EI on main proposals.
- `neighbor_xi: float = 0.01` — EI’s `xi` for NAS

### NAS / SDR
- `use_nas: bool = True` — evaluate snapped proposal & 4 neighbors; add the best.
- `use_sdr: bool = True` — contract domain around current best per dimension.
- `shrink_factor: torch.Tensor[2] = [0.9, 0.945]` — SDR η per dimension (ku, damp).
- `min_window: float = 0.0` — per dim minimum window (real units).
- `filter_outside_window: bool = True` — drop training points outside SDR window.  
  - **Note (RF):** filtering changes the RF training set; even with fixed seed this can slightly shift results. A warning is issued when points are filtered.

### Acquisition optimizer
- `raw_samples: int = 500` — random initial samples for acq optimization.
- `num_restarts: int = 10` — local restarts.

### Logs and reproducibility
- `seed: Optional[int] = None` — global seed (Sobol, acq optimizer, RF unless overridden).
- `suppress_warnings: bool = True` — hide common BoTorch/linear_operator warnings.
- `verbose: bool = False` — per iteration logs (best Y, snapped best point, current domain).
- `print_total_time: bool = True` — print wall clock total.

---

## Reproducibility

- A single **global seed** drives:
  - Sobol initial sampling (`SobolEngine(..., seed=seed)`),
  - BoTorch acquisition optimizer (`options={"seed": seed}`),
  - RF `random_state` (unless `rf_random_state` is set).
- Ensure your **objective** is deterministic (file I/O, math).
- With **RF + filtering**, expect minor differences as the SDR window changes the training pool.

---

## Troubleshooting
**Data not found**  
- Verify `source` and file naming: `Anis_{ku:.3f}e-22Damp_{damp:.3f}/output`.
- ku and damp in this should be passed as strings to ensure they are 3 decimal places each.

---

## License

MIT License

Copyright (c) 2025 Joshwin-Sundarraj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgements

Built with PyTorch, BoTorch, GPyTorch, scikit-learn, and linear-operator.
