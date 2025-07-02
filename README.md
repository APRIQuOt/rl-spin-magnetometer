# rl-spin-magnetometer

Repository for results presented in ["Reinforcement Learning for Optimal Control of Spin Magnetometers."](https://arxiv.org/abs/2506.21475) We implement the soft actor-critic (SAC) reinforcement learning (RL) algorithm to optimize the control of a spin-magnetometer, for optimal measurement sensitivity.

# Installation

Simulations of the spin system, and calculations of the quantum Fisher information are implemented in Julia, while the RL is in Python through PyTorch. Here, we include instructions for properly setting up both the Python and Julia environments.

## Python

A suitable `python` environment may be installed from the `requirements.txt` file included here. Primary dependencies are:

- `python` version 3.12
- [`matplotlib`](https://matplotlib.org/) for plotting results
- [`jupyter`](https://jupyter.org/) for running the example notebook
- [`numpy`](https://numpy.org/)
- [`torch`](https://pytorch.org/) PyTorch ML library for RL implementation
- [`juliacall`](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) JuliaCall for interfacing AI in Python and simulations in Julia
- [`juliapkg`](https://github.com/JuliaPy/pyjuliapkg) a helper package for handling Julia environment in JuliaCall.

## Julia

While the RL implementation is done in Python, through PyTorch, simulations are performed in [Julia](https://julialang.org/). Simulations are called directly from Python through the aforementioned `juliacall` package. Julia may either be installed seperately, or JuliaCall will handle the installation automatically. See the [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) documentation for details.

Once installed, the Julia environment must be constructed with the correct dependencies; since the code for simulations are custom, this must be done through the `juliapkg` package manager in Python. Simply run the included `juliacall-pkg-manager.py` script, which adds the `SensorUtils` Julia package to the environment in development mode. After adding this package, the script will download necessary Julia packages and precompile them.
