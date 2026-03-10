# QOCO.jl

Julia wrapper for the [QOCO](https://github.com/qoco-org/qoco) solver — a
Quadratic Objective Conic Optimizer.

QOCO solves second-order cone programs (SOCPs) with quadratic objectives of the
form:

```
min   (1/2)x'Px + c'x
s.t.  Ax = b
      Gx <=_C h
```

where `C = R_+^l × SOC(q₁) × … × SOC(qₙ)` is a product of the nonnegative
orthant and second-order cones.

## Installation

```julia
using Pkg
Pkg.add("QOCO")
```

## Usage with JuMP

```julia
using JuMP, QOCO

model = Model(QOCO.Optimizer)
set_silent(model)

@variable(model, x[1:2] >= 0)
@constraint(model, x[1] + x[2] == 1)
@objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

optimize!(model)

value.(x)  # [0.5, 0.5]
```

## Usage with MathOptInterface

```julia
import MathOptInterface as MOI

optimizer = QOCO.Optimizer()
MOI.set(optimizer, MOI.Silent(), true)
```

## Solver Settings

Settings can be passed as keyword arguments to the optimizer or set via
`MOI.RawOptimizerAttribute`:

```julia
# Via constructor
optimizer = QOCO.Optimizer(; max_iters = 500, abstol = 1e-8)

# Via MOI
MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iters"), 500)
```

| Parameter         | Type    | Default  | Description                         |
|-------------------|---------|----------|-------------------------------------|
| `max_iters`       | Int     | 200      | Maximum number of iterations        |
| `bisect_iters`    | Int     | 5        | Bisection iterations                |
| `ruiz_iters`      | Int     | 0        | Ruiz equilibration iterations       |
| `iter_ref_iters`  | Int     | 1        | Iterative refinement iterations     |
| `kkt_static_reg`  | Float64 | 1e-8     | KKT static regularization           |
| `kkt_dynamic_reg` | Float64 | 1e-8     | KKT dynamic regularization          |
| `abstol`          | Float64 | 1e-7     | Absolute tolerance                  |
| `reltol`          | Float64 | 1e-7     | Relative tolerance                  |
| `abstol_inacc`    | Float64 | 1e-5     | Absolute tolerance (inaccurate)     |
| `reltol_inacc`    | Float64 | 1e-5     | Relative tolerance (inaccurate)     |
| `verbose`         | Bool    | false    | Print solver output                 |
