# Calling from Python

Using `ModeCouplingTheory` from Python can be useful for solving mode-coupling theory repeatedly or for large systems of equations, making the overhead of initializing Julia negligible. (Or perhaps if one prefers to work in Python.)

## Installing

First, install `juliacall` through the `pip` package manager, with

```bash
pip install juliacall
```
This package allows one to call julia from python 

In `Python` (only versions $\geq$ 3 are supported), run:

```python
from juliacall import Main as jl
```

which will install the latest stable version of Julia the first time it is called. Now install `ModeCouplingTheory.jl`, with

```python

jl.Pkg.add("ModeCouplingTheory.jl")
```

To import this package in order to use it we need to run:

```python
jl.seval("using ModeCouplingTheory")
```

## Usage

And now we can use ModeCouplingTheory.jl in Python:

```python
# since Python doesn't like all unicode symbols (α, β, ∂, ...), we use standard letters:
k = 3.99999
a = 1.0
b = 0.0
c = 1.0
d = 0.0
F0 = 1.0
dF0 = 0.0
kernel = jl.SchematicF2Kernel(k)
problem = jl.MemoryEquation(a, b, c, d, F0, dF0, kernel)
sol = jl.solve(problem)

import matplotlib.pyplot as plt
import numpy as np
t = get_t(sol)
F = get_t(sol)
plt.plot(np.log10(t), F)
plt.show()
```

et voilà!

![image](images/Figure 2022-08-26 115331.png)

See the documentation of [pyjulia](https://pyjulia.readthedocs.io/en/latest/usage.html) for more information on how to call julia from python.
