using QOCO
using Test

@testset "C Wrapper" begin
    @testset "Default settings" begin
        settings = QOCO.default_settings()
        @test settings.max_iters == 200
        @test settings.bisect_iters == 5
        @test settings.ruiz_iters == 0
        @test settings.iter_ref_iters == 1
        @test settings.abstol ≈ 1e-7
        @test settings.reltol ≈ 1e-7
        @test settings.verbose == 0x00
    end

    @testset "Small QP via C API" begin
        # min (1/2) x'Px + c'x
        # s.t. Gx <=_C h (nonneg cone: x >= 0)
        #      Ax = b
        #
        # P = [2 0; 0 2], c = [-1; -1]
        # G = [-1 0; 0 -1], h = [0; 0] (x >= 0 ⟺ -x ≤ 0)
        # A = [1 1], b = [1] (x1 + x2 = 1)
        #
        # Solution: x = [0.5, 0.5], obj = -0.75

        n = 2   # variables
        m = 2   # conic constraints (nonneg)
        p = 1   # equality constraints
        l = 2   # nonneg cone dimension
        nsoc = 0
        q = QOCO.QOCOInt[0]

        # P (upper triangular, CSC, 0-indexed)
        P = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Px = QOCO.QOCOFloat[2.0, 2.0]
        Pp = QOCO.QOCOInt[0, 1, 2]
        Pi = QOCO.QOCOInt[0, 1]
        QOCO.qoco_set_csc!(P, n, n, 2, Px, Pp, Pi)

        c_vec = QOCO.QOCOFloat[-1.0, -1.0]

        # A (CSC, 0-indexed)
        A = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Ax = QOCO.QOCOFloat[1.0, 1.0]
        Ap = QOCO.QOCOInt[0, 1, 2]
        Ai = QOCO.QOCOInt[0, 0]
        QOCO.qoco_set_csc!(A, p, n, 2, Ax, Ap, Ai)

        b_vec = QOCO.QOCOFloat[1.0]

        # G (CSC, 0-indexed)
        G = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Gx = QOCO.QOCOFloat[-1.0, -1.0]
        Gp = QOCO.QOCOInt[0, 1, 2]
        Gi = QOCO.QOCOInt[0, 1]
        QOCO.qoco_set_csc!(G, m, n, 2, Gx, Gp, Gi)

        h_vec = QOCO.QOCOFloat[0.0, 0.0]

        settings = QOCO.default_settings()
        settings.verbose = 0x00

        solver_ptr = QOCO.qoco_solver_alloc()
        err = QOCO.qoco_setup!(solver_ptr, n, m, p, P, c_vec, A, b_vec, G, h_vec, l, nsoc, q, settings)
        @test err == QOCO.QOCO_NO_ERROR

        QOCO.qoco_solve!(solver_ptr)
        sol = QOCO.get_solution(solver_ptr)

        @test sol.status == QOCO.QOCO_SOLVED
        x = unsafe_wrap(Array, sol.x, n)
        @test x[1] ≈ 0.5 atol = 1e-5
        @test x[2] ≈ 0.5 atol = 1e-5
        @test sol.obj ≈ -0.5 atol = 1e-5

        QOCO.qoco_cleanup!(solver_ptr)
    end

    @testset "SOC constraint via C API" begin
        # min x1
        # s.t. ||[x2; x3]|| <= x1 (SOC)
        #      x2 = 1, x3 = 0
        # Solution: x = [1, 1, 0]

        n = 3
        p = 2   # x2=1, x3=0
        m = 3   # SOC of dim 3
        l = 0
        nsoc = 1
        q = QOCO.QOCOInt[3]

        # P = 0 (linear objective)
        P = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Px = QOCO.QOCOFloat[0.0]
        Pp = QOCO.QOCOInt[0, 0, 0, 0]
        Pi = QOCO.QOCOInt[0]
        QOCO.qoco_set_csc!(P, n, n, 0, Px, Pp, Pi)

        c_vec = QOCO.QOCOFloat[1.0, 0.0, 0.0]

        # A: x2 = 1, x3 = 0
        A = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        A_x = QOCO.QOCOFloat[1.0, 1.0]
        A_p = QOCO.QOCOInt[0, 0, 1, 2]
        A_i = QOCO.QOCOInt[0, 1]
        QOCO.qoco_set_csc!(A, p, n, 2, A_x, A_p, A_i)

        b_vec = QOCO.QOCOFloat[1.0, 0.0]

        # G: SOC ||[x2;x3]|| <= x1
        # Written as: (x1, x2, x3) ∈ SOC(3)
        # QOCO: h - Gx ∈ SOC → G = -I (rows for SOC variables), h = 0
        G = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        G_x = QOCO.QOCOFloat[-1.0, -1.0, -1.0]
        G_p = QOCO.QOCOInt[0, 1, 2, 3]
        G_i = QOCO.QOCOInt[0, 1, 2]
        QOCO.qoco_set_csc!(G, m, n, 3, G_x, G_p, G_i)

        h_vec = QOCO.QOCOFloat[0.0, 0.0, 0.0]

        settings = QOCO.default_settings()
        settings.verbose = 0x00

        solver_ptr = QOCO.qoco_solver_alloc()
        err = QOCO.qoco_setup!(solver_ptr, n, m, p, P, c_vec, A, b_vec, G, h_vec, l, nsoc, q, settings)
        @test err == QOCO.QOCO_NO_ERROR

        QOCO.qoco_solve!(solver_ptr)
        sol = QOCO.get_solution(solver_ptr)

        @test sol.status == QOCO.QOCO_SOLVED
        x = unsafe_wrap(Array, sol.x, n)
        @test x[1] ≈ 1.0 atol = 1e-5
        @test x[2] ≈ 1.0 atol = 1e-5
        @test abs(x[3]) < 1e-5

        QOCO.qoco_cleanup!(solver_ptr)
    end
end
