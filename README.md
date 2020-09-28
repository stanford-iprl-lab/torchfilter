# torchfilter

![build](https://github.com/stanford-iprl-lab/torchfilter/workflows/build/badge.svg)
![mypy](https://github.com/stanford-iprl-lab/torchfilter/workflows/mypy/badge.svg)
![lint](https://github.com/stanford-iprl-lab/torchfilter/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/stanford-iprl-lab/torchfilter/branch/master/graph/badge.svg)](https://codecov.io/gh/stanford-iprl-lab/torchfilter)

**`torchfilter`** is a library for discrete-time Bayesian filtering in PyTorch.
By writing filters as standard PyTorch modules, we get:

- The ability to optimize for system models/parameters that directly minimize
  end-to-end state estimation error
- Automatic Jacobians with autograd
- GPU acceleration (particularly useful for particle filters)

The package is broken down into six submodules:

<table>
  <tbody valign="top">
    <tr>
      <td>
        <a
          href="https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/base/"
        >
          <code>torchfilter.<strong>base</strong></code>
        </a>
      </td>
      <td>
        Base classes that define standard interfaces for implementing filter,
        dynamics, measurement, and virtual sensor models as PyTorch
        modules.
      </td>
    </tr>
    <tr>
      <td>
        <a
          href="https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/filters/"
        >
          <code>torchfilter.<strong>filters</strong></code>
        </a>
      </td>
      <td>
        Differentiable filters implemented as PyTorch modules, which
        can either be used directly or subclassed. Currently implemented:
        <ul>
          <li>Particle Filter (PF)</li>
          <li>Extended Kalman Filter (EKF)</li>
          <li>Unscented Kalman Filter (UKF)</li>
          <li>Square Root Unscented Kalman Filter (SR-UKF)</li>
        </ul>
        <p>
          For our PF, we include both standard resampling and the
          soft/differentiable approach from <a href="#references">[1]</a>.
        </p>
        <p>
          UKFs and SR-UKFs are implemented using equations from
          <a href="#references">[2]</a>; approach for handling heteroscedastic
          noises in the former is based on <a href="#references">[3]</a>.
        </p>
        <p>
          For our EKF, UKF, and SR-UKF, we also provide &ldquo;virtual
          sensor&rdquo; implementations, which use a (raw observation =>
          state prediction/uncertainty) mapping and an identity as the measurement model. This
          is similar to the discriminative strategy described in <a href="#references">[4]</a>.
        </p>
      </td>
    </tr>
    <tr>
      <td>
        <a
          href="https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/train/"
        >
          <code>torchfilter.<strong>train</strong></code>
        </a>
      </td>
      <td>
        Training loop helpers. These are currently coupled tightly with a custom
        experiment management framework, but may be useful as a reference.
      </td>
    </tr>
    <tr>
      <td>
        <a
          href="https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/data/"
        >
          <code>torchfilter.<strong>data</strong></code>
        </a>
      </td>
      <td>Dataset interfaces used for our training loop helpers.</td>
    </tr>
    <tr>
      <td>
        <a
          href="https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/utils/"
        >
          <code>torchfilter.<strong>utils</strong></code>
        </a>
      </td>
      <td>
        General utilities; currently only contains helpers for performing
        unscented transforms.
      </td>
    </tr>
    <tr>
      <td>
        <a
          href="https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/types/"
        >
          <code>torchfilter.<strong>types</strong></code>
        </a>
      </td>
      <td>Data structures and semantic type aliases.</td>
    </tr>
  </tbody>
</table>

For more details, see the
[API reference](https://stanford-iprl-lab.github.io/torchfilter).

For a linear system example, see `tests/_linear_system_models.py`. A more complex
application can be found in code for our IROS 2020 work <a
href="#references">[5]</a>: [GitHub repository](https://github.com/brentyi/multimodalfilter).

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
[mypy](https://github.com/python/mypy) for static type checking.

Until `numpy 1.20.0` [is released](https://github.com/numpy/numpy/pull/16515),
static analysis works best with NumPy stubs installed manually:

```
pip install https://github.com/numpy/numpy-stubs/tarball/master
```

---

### References

This library is based on code written for our IROS 2020 work [5].

[1] P. Karkus, D. Hsu, and W. S. Lee,
"[Particle filter networks with application to visual localization](https://arxiv.org/abs/1805.08975)",
in _Conference on Robot Learning_, 2018, pp. 169–178.

[2] R. Van der Merwe and E. A. Wan,
"[The square-root unscented Kalman filter for state and parameter-estimation](https://ieeexplore.ieee.org/document/940586)",
in _IEEE International Conference on Acoustics, Speech, and Signal Processing_,
2001, pp. 3461-3464 vol.6.

[3] A. Kloss, G. Martius, and J. Bohg,
"[How to Train Your Differentiable Filter](https://al.is.tuebingen.mpg.de/publications/kloss_rss_ws_2020)",
in _Robotics: Science and Systems (RSS) Workshop on Structured Approaches to
Robot Learning for Improved Generalization_, 2020.

[4] T. Haarnoja, A. Ajay, S. Levine, and P. Abbeel,
"[Backprop KF: Learning discriminative deterministic state estimators](https://arxiv.org/abs/1605.07148)",
in _Advances in Neural Information Processing Systems_, 2016, pp. 4376–4384.

[5] M. Lee*, B. Yi*, R. Martín-Martín, S. Savarese, J. Bohg,
"[Multimodal Sensor Fusion with Differentiable Filters](https://sites.google.com/view/multimodalfilter)",
in _IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS)_, 2020.

---

### Contact

brentyi (at) stanford (dot) edu
