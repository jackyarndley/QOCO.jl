using QOCO
using Test
import MathOptInterface as MOI
using JuMP

@testset "MOI Wrapper" begin
    @testset "Solver attributes" begin
        opt = QOCO.Optimizer()
        @test MOI.get(opt, MOI.SolverName()) == "QOCO"
        @test MOI.get(opt, MOI.SolverVersion()) == "0.1.6"
        @test MOI.is_empty(opt)
    end

    @testset "Settings" begin
        opt = QOCO.Optimizer(; max_iters = 100, abstol = 1e-6)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iters")) == 100
        @test MOI.get(opt, MOI.RawOptimizerAttribute("abstol")) == 1e-6
        MOI.set(opt, MOI.RawOptimizerAttribute("reltol"), 1e-8)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("reltol")) == 1e-8
    end

    @testset "Silent" begin
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        @test MOI.get(opt, MOI.Silent()) == true
    end

    @testset "Simple QP via MOI" begin
        # min (1/2)(2x1^2 + 2x2^2) - x1 - x2
        # s.t. x1 + x2 = 1
        #      x1, x2 >= 0
        # Solution: x = [0.5, 0.5], obj = -0.5

        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)

        # Objective: (1/2)(2x1^2 + 2x2^2) - x1 - x2
        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        # x1 + x2 = 1
        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        # x1, x2 >= 0
        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(opt, MOI.ResultCount()) == 1

        x1_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]])
        x2_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]])
        @test x1_val ≈ 0.5 atol = 1e-5
        @test x2_val ≈ 0.5 atol = 1e-5
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -0.5 atol = 1e-5
    end

    @testset "Linear objective with SOC via MOI" begin
        # min x1
        # s.t. (x1, x2, x3) ∈ SOC(3)
        #      x2 = 1, x3 = 0
        # Solution: x = [1, 1, 0]

        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)

        # Objective: min x1
        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1])],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        # (x1, x2, x3) ∈ SOC(3)
        soc_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [0.0, 0.0, 0.0],
        )
        MOI.add_constraint(model, soc_func, MOI.SecondOrderCone(3))

        # x2 = 1, x3 = 0
        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [-1.0, 0.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        x1_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]])
        x2_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]])
        x3_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[3]])
        @test x1_val ≈ 1.0 atol = 1e-4
        @test x2_val ≈ 1.0 atol = 1e-4
        @test abs(x3_val) < 1e-4
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol = 1e-4
    end

    @testset "Maximization" begin
        # max -x1^2 - x2^2 + x1 + x2
        # s.t. x1 + x2 = 1, x1,x2 >= 0
        # Equivalent to min x1^2 + x2^2 - x1 - x2 s.t. same
        # Solution: x = [0.5, 0.5], obj = 0.5

        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(-2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(-2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(1.0, x[1]),
            MOI.ScalarAffineTerm(1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.5 atol = 1e-4
    end

    @testset "Empty model" begin
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        model = MOI.Utilities.Model{Float64}()
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0
    end

    @testset "Iteration limit" begin
        # Use very few iterations so the solver cannot converge
        opt = QOCO.Optimizer(; max_iters = 1)
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        status = MOI.get(opt, MOI.TerminationStatus())
        @test status in (MOI.ITERATION_LIMIT, MOI.ALMOST_OPTIMAL)
    end

    @testset "Infeasible problem" begin
        # x1 = 1, x1 = 2 (contradictory equality constraints)
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 1)

        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1])],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        # P = 0
        # x1 = 1 AND x1 = 2  (infeasible)
        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[1])),
            ],
            [-1.0, -2.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        # QOCO doesn't have a dedicated infeasible status; it will report
        # either iteration limit, numerical error, or inaccurate solution
        status = MOI.get(opt, MOI.TerminationStatus())
        @test status in (
            MOI.ITERATION_LIMIT,
            MOI.NUMERICAL_ERROR,
            MOI.ALMOST_OPTIMAL,
            MOI.OTHER_ERROR,
        )
    end
end

@testset "MOI.Test" begin
    optimizer = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            QOCO.Optimizer(),
        ),
        Float64,
    )
    MOI.set(optimizer, MOI.Silent(), true)

    MOI.Test.runtests(
        optimizer,
        MOI.Test.Config(
            atol = 1e-3,
            rtol = 1e-3,
            optimal_status = MOI.OPTIMAL,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
            ],
        );
        exclude = String[
            # QOCO does not support these cone types
            "test_conic_ExponentialCone",
            "test_conic_DualExponentialCone",
            "test_conic_PowerCone",
            "test_conic_DualPowerCone",
            "test_conic_GeometricMeanCone",
            "test_conic_RootDetConeTriangle",
            "test_conic_RootDetConeSquare",
            "test_conic_LogDetConeTriangle",
            "test_conic_LogDetConeSquare",
            "test_conic_PositiveSemidefiniteConeTriangle",
            "test_conic_PositiveSemidefiniteConeSquare",
            "test_conic_RelativeEntropyCone",
            "test_conic_NormSpectralCone",
            "test_conic_NormNuclearCone",
            "test_conic_NormInfinityCone",
            "test_conic_NormOneCone",
            "test_conic_HermitianPositiveSemidefiniteConeTriangle",
            "test_conic_NormCone",
            # No integer support
            "test_basic_VectorOfVariables_",
            "test_solve_SOS",
            "test_constraint_ZeroOne",
            "test_constraint_Integer",
            # No incremental interface
            "test_model_",
            # QOCO doesn't detect infeasibility or unboundedness
            "test_conic_linear_INFEASIBLE",
            "test_conic_NonnegToNonworking",
            "test_conic_SecondOrderCone_INFEASIBLE",
            "test_conic_SecondOrderCone_negative_post_bound_2",
            "test_conic_SecondOrderCone_negative_post_bound_3",
            "test_conic_SecondOrderCone_no_initial_bound",
            "test_conic_RotatedSecondOrderCone_INFEASIBLE",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            "test_infeasible",
            "test_linear_INFEASIBLE",
            "test_linear_DUAL_INFEASIBLE",
            "test_unbounded",
            # No time limit support
            "test_attribute_TimeLimitSec",
            # Tests requiring features beyond copy_to
            "test_modification_",
            "test_objective_ObjectiveFunction_",
            "test_variable_",
            # Constraint query not fully supported
            "test_constraint_",
            # Deletion not supported
            "test_solve_result_index",
            # Quadratic constraints not supported (only quadratic objectives)
            "test_quadratic_constraint_",
            # Integration tests that require incremental or delete
            "test_linear_integration_delete_variables",
            "test_solve_optimize_twice",
        ],
    )
end

@testset "JuMP smoke test" begin
    model = Model(QOCO.Optimizer)
    set_silent(model)

    @variable(model, x[1:2] >= 0)
    @constraint(model, x[1] + x[2] == 1)
    @objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value(x[1]) ≈ 0.5 atol = 1e-4
    @test value(x[2]) ≈ 0.5 atol = 1e-4
    @test objective_value(model) ≈ -0.5 atol = 1e-4
end
