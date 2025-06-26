"""
Script to setup Julia environment, for `juliacall`.

- The `uuid` for `SensorUtils` comes from its `Project.toml` file.
- The `path` must be an absolute path to the location of SensorUtils location. Here we
  use the current working directory.
"""

import juliapkg as jlp
from juliacall import Main as jl
import os

cwd = os.path.abspath(os.getcwd()) # Current working directory
uuid = "be1392fb-b7fc-4b79-8e52-3b9da734796b"

# Add SensorUtils in dev mode
jlp.add("SensorUtils", uuid, dev=True, path=cwd + "/SensorUtils")

# Instantiate and precompile packages
jl.seval("using Pkg")
jl.seval("Pkg.instantiate()")
jl.seval("Pkg.precompile()")

# Resolve pre-compilation
jlp.resolve()
jlp.project()
jlp.status()
