# torchfilter

![build](https://github.com/stanford-iprl-lab/torchfilter/workflows/build/badge.svg)
![mypy](https://github.com/stanford-iprl-lab/torchfilter/workflows/mypy/badge.svg)
![lint](https://github.com/stanford-iprl-lab/torchfilter/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/stanford-iprl-lab/torchfilter/branch/master/graph/badge.svg)](https://codecov.io/gh/stanford-iprl-lab/torchfilter)

**`torchfilter`** is a library for discrete-time Bayesian filtering in PyTorch.
By writing filters as differentiable PyTorch modules, we can optimize for system
models/parameters that directly minimize end-to-end state estimation error.

The package is broken down into six submodules:

<table>
  <tbody valign="top">
    <tr>
      <td><code>torchfilter.<strong>base</strong></code></td>
      <td>
        Base classes that define standard interfaces for implementing filter,
        dynamics, measurement, and virtual sensor models as standard PyTorch
        modules.
      </td>
    </tr>
    <tr>
      <td><code>torchfilter.<strong>filters</strong></code></td>
      <td>
        Differentiable filters implemented as standard PyTorch modules, which
        can either be used directly or subclassed. Currently implemented:
        <ul>
            <li>Particle Filter (PF)</li>
            <li>Extended Kalman Filter (EKF)</li>
            <li>Unscented Kalman Filter (UKF)</li>
            <li>Square Root Unscented Kalman Filter (SR-UKF)</li>
        </ul>
        For our EKF, UKF, and SR-UKF, we also provide <em>virtual sensor</em>
        implementations, which use a (raw observation => predicted state)
        mapping + an identity as the measurement model.
      </td>
    </tr>
    <tr>
      <td><code>torchfilter.<strong>train</strong></code></td>
      <td>
        Training loop helpers. These are currently coupled tightly with a custom
        experiment management framework, but may be useful as a reference.
      </td>
    </tr>
    <tr>
      <td><code>torchfilter.<strong>data</strong></code></td>
      <td>
        Dataset interfaces used for our training loop helpers.
      </td>
    </tr>
    <tr>
      <td><code>torchfilter.<strong>utils</strong></code></td>
      <td>
        General utilities; currently only contains helpers for performing
        unscented transforms.
      </td>
    </tr>
    <tr>
      <td><code>torchfilter.<strong>types</strong></code></td>
      <td>Data structures and semantic type aliases.</td>
    </tr>
  </tbody>
</table>

For more details, see the
[API reference](https://stanford-iprl-lab.github.io/torchfilter).

For a linear system example, see `tests/_linear_system_models.py`.

---

### Installation

From source:

```bash
$ git clone https://github.com/stanford-iprl-lab/torchfilter.git
$ cd torchfilter
$ pip install -e .
```

---

### Development

Tests can be run with `pytest`, and documentation can be built by running
`make dirhtml` in the `docs/` directory.

Tooling: [black](https://github.com/psf/black) and
[isort](https://github.com/timothycrosley/isort) for formatting,
[flake8](https://flake8.pycqa.org/en/latest/) for linting, and
[mypy](https://github.com/python/mypy) for static type-checking.

Until `numpy 1.20.0` [is released](https://github.com/numpy/numpy/pull/16515),
type-checking works best with NumPy stubs installed manually:

```
pip install https://github.com/numpy/numpy-stubs/tarball/master
```
