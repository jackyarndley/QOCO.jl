# Thin Julia wrapper over the QOCO C API (QOCO_jll v0.1.6)

# Type aliases matching definitions.h
const QOCOInt = Cint
const QOCOFloat = Cdouble

# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

# qoco_solve_status
const QOCO_UNSOLVED = QOCOInt(0)
const QOCO_SOLVED = QOCOInt(1)
const QOCO_SOLVED_INACCURATE = QOCOInt(2)
const QOCO_NUMERICAL_ERROR = QOCOInt(3)
const QOCO_MAX_ITER = QOCOInt(4)

# qoco_error_code
const QOCO_NO_ERROR = QOCOInt(0)
const QOCO_DATA_VALIDATION_ERROR = QOCOInt(1)
const QOCO_SETTINGS_VALIDATION_ERROR = QOCOInt(2)
const QOCO_SETUP_ERROR = QOCOInt(3)
const QOCO_AMD_ERROR = QOCOInt(4)
const QOCO_MALLOC_ERROR = QOCOInt(5)

# --------------------------------------------------------------------------
# Structs (must match C memory layout exactly)
# --------------------------------------------------------------------------

"""
    QOCOCscMatrix

CSC sparse matrix, matches `QOCOCscMatrix` in `qoco_linalg.h`.
Field order: m, n, nnz, i, p, x.
"""
mutable struct QOCOCscMatrix
    m::QOCOInt
    n::QOCOInt
    nnz::QOCOInt
    i::Ptr{QOCOInt}    # row indices, length nnz
    p::Ptr{QOCOInt}    # column pointers, length n+1
    x::Ptr{QOCOFloat}  # values, length nnz
end

"""
    QOCOSettings

Solver settings struct matching `QOCOSettings` in `structs.h`.
Note: `verbose` is `unsigned char` in C (mapped to `UInt8`).
"""
mutable struct QOCOSettings
    max_iters::QOCOInt
    bisect_iters::QOCOInt
    ruiz_iters::QOCOInt
    iter_ref_iters::QOCOInt
    kkt_static_reg::QOCOFloat
    kkt_dynamic_reg::QOCOFloat
    abstol::QOCOFloat
    reltol::QOCOFloat
    abstol_inacc::QOCOFloat
    reltol_inacc::QOCOFloat
    verbose::UInt8
end

"""
    QOCOSolution

Solution struct matching `QOCOSolution` in `structs.h`.
Field order: x, s, y, z, iters, setup_time_sec, solve_time_sec, obj, pres, dres, gap, status.
"""
struct QOCOSolution
    x::Ptr{QOCOFloat}
    s::Ptr{QOCOFloat}
    y::Ptr{QOCOFloat}
    z::Ptr{QOCOFloat}
    iters::QOCOInt
    setup_time_sec::QOCOFloat
    solve_time_sec::QOCOFloat
    obj::QOCOFloat
    pres::QOCOFloat
    dres::QOCOFloat
    gap::QOCOFloat
    status::QOCOInt
end

"""
    QOCOSolver

Top-level solver handle matching `QOCOSolver` in `structs.h` (v0.1.6).

Because `qoco_cleanup` calls `qoco_free(solver)` (freeing the solver struct
itself), we must allocate the solver with C's malloc so C can free it safely.
Use `qoco_solver_alloc()` to create a solver and pass the returned
`Ptr{QOCOSolver}` to all API functions.
"""
struct QOCOSolver
    settings::Ptr{QOCOSettings}
    work::Ptr{Cvoid}         # QOCOWorkspace*, opaque
    sol::Ptr{QOCOSolution}
end

"""
    qoco_solver_alloc() -> Ptr{QOCOSolver}

Allocate a zero-initialised `QOCOSolver` on the C heap.
The pointer is freed by `qoco_cleanup!` — do **not** call `Libc.free` on it.
"""
function qoco_solver_alloc()
    ptr = Ptr{QOCOSolver}(Libc.calloc(1, sizeof(QOCOSolver)))
    ptr == C_NULL && error("Failed to allocate QOCOSolver")
    return ptr
end

# --------------------------------------------------------------------------
# C API wrappers
# --------------------------------------------------------------------------

"""
    set_default_settings!(settings::QOCOSettings)

Populate `settings` with QOCO's compiled-in defaults.
"""
function set_default_settings!(settings::QOCOSettings)
    ccall(
        (:set_default_settings, qoco[]),
        Cvoid,
        (Ref{QOCOSettings},),
        settings,
    )
    return settings
end

function default_settings()
    s = QOCOSettings(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0x00)
    set_default_settings!(s)
    return s
end

"""
    qoco_set_csc!(A, m, n, nnz, Ax, Ap, Ai)

Fill a `QOCOCscMatrix` in-place. Wraps `qoco_set_csc`.
"""
function qoco_set_csc!(
    A::QOCOCscMatrix, m::Integer, n::Integer, nnz::Integer,
    Ax::Vector{QOCOFloat}, Ap::Vector{QOCOInt}, Ai::Vector{QOCOInt},
)
    ccall(
        (:qoco_set_csc, qoco[]),
        Cvoid,
        (Ref{QOCOCscMatrix}, QOCOInt, QOCOInt, QOCOInt,
         Ptr{QOCOFloat}, Ptr{QOCOInt}, Ptr{QOCOInt}),
        A, QOCOInt(m), QOCOInt(n), QOCOInt(nnz), Ax, Ap, Ai,
    )
    return A
end

"""
    qoco_setup!(solver_ptr, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings)

Initialise the solver with problem data. Returns `QOCOInt` error code
(0 = success).  All array/matrix arguments must stay alive until
`qoco_cleanup!` is called.
"""
function qoco_setup!(
    solver_ptr::Ptr{QOCOSolver},
    n::Integer, m::Integer, p::Integer,
    P::QOCOCscMatrix, c::Vector{QOCOFloat},
    A::QOCOCscMatrix, b::Vector{QOCOFloat},
    G::QOCOCscMatrix, h::Vector{QOCOFloat},
    l::Integer, nsoc::Integer, q::Vector{QOCOInt},
    settings::QOCOSettings,
)
    return ccall(
        (:qoco_setup, qoco[]),
        QOCOInt,
        (Ptr{QOCOSolver}, QOCOInt, QOCOInt, QOCOInt,
         Ref{QOCOCscMatrix}, Ptr{QOCOFloat},
         Ref{QOCOCscMatrix}, Ptr{QOCOFloat},
         Ref{QOCOCscMatrix}, Ptr{QOCOFloat},
         QOCOInt, QOCOInt, Ptr{QOCOInt},
         Ref{QOCOSettings}),
        solver_ptr, QOCOInt(n), QOCOInt(m), QOCOInt(p),
        P, c, A, b, G, h,
        QOCOInt(l), QOCOInt(nsoc), q, settings,
    )
end

"""
    qoco_solve!(solver_ptr)

Run the solver. Returns `QOCOInt` solve status.
"""
function qoco_solve!(solver_ptr::Ptr{QOCOSolver})
    return ccall(
        (:qoco_solve, qoco[]),
        QOCOInt,
        (Ptr{QOCOSolver},),
        solver_ptr,
    )
end

"""
    qoco_cleanup!(solver_ptr)

Free all memory allocated by `qoco_setup!`, including the solver struct
itself.  Do not use `solver_ptr` after this call.
"""
function qoco_cleanup!(solver_ptr::Ptr{QOCOSolver})
    return ccall(
        (:qoco_cleanup, qoco[]),
        QOCOInt,
        (Ptr{QOCOSolver},),
        solver_ptr,
    )
end

"""
    qoco_update_settings!(solver_ptr, settings)

Update solver settings after setup.
"""
function qoco_update_settings!(solver_ptr::Ptr{QOCOSolver}, settings::QOCOSettings)
    return ccall(
        (:qoco_update_settings, qoco[]),
        QOCOInt,
        (Ptr{QOCOSolver}, Ref{QOCOSettings}),
        solver_ptr, settings,
    )
end

"""
    qoco_update_vector_data!(solver_ptr, c, b, h)

Update vector problem data in-place. Pass `C_NULL` for any vector
that should not be updated.
"""
function qoco_update_vector_data!(
    solver_ptr::Ptr{QOCOSolver},
    c::Union{Vector{QOCOFloat}, Ptr{Nothing}},
    b::Union{Vector{QOCOFloat}, Ptr{Nothing}},
    h::Union{Vector{QOCOFloat}, Ptr{Nothing}},
)
    ccall(
        (:qoco_update_vector_data, qoco[]),
        Cvoid,
        (Ptr{QOCOSolver}, Ptr{QOCOFloat}, Ptr{QOCOFloat}, Ptr{QOCOFloat}),
        solver_ptr, c, b, h,
    )
    return
end

"""
    qoco_update_matrix_data!(solver_ptr, Px, Ax, Gx)

Update matrix nonzero values in-place (same sparsity pattern).
Pass `C_NULL` for any matrix that should not be updated.
"""
function qoco_update_matrix_data!(
    solver_ptr::Ptr{QOCOSolver},
    Px::Union{Vector{QOCOFloat}, Ptr{Nothing}},
    Ax::Union{Vector{QOCOFloat}, Ptr{Nothing}},
    Gx::Union{Vector{QOCOFloat}, Ptr{Nothing}},
)
    ccall(
        (:qoco_update_matrix_data, qoco[]),
        Cvoid,
        (Ptr{QOCOSolver}, Ptr{QOCOFloat}, Ptr{QOCOFloat}, Ptr{QOCOFloat}),
        solver_ptr, Px, Ax, Gx,
    )
    return
end

"""
    get_solution(solver_ptr) -> QOCOSolution

Read the `QOCOSolution` from the solver after solve.
Must be called **before** `qoco_cleanup!`.
"""
function get_solution(solver_ptr::Ptr{QOCOSolver})
    solver = unsafe_load(solver_ptr)
    solver.sol == C_NULL && error("Solver has not been solved yet")
    return unsafe_load(solver.sol)
end
