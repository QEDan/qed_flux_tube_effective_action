
VALIDATION_PY = scripts/test_zero_field.py scripts/compare_analytics.py \
                scripts/visualize_benchmark.py \
                scripts/test_delta_shell_validation.py scripts/test_sech2_shell_validation.py \
                scripts/validate_wkb.py scripts/plot_greens_function_comparison.py \
		scripts/plot_sech2_greens_function.py \
                scripts/validate_step_ea.py scripts/validate_density_log.py

VALIDATION_SAGEMATH_PY = symbolic_validations/numerical_spectrum.py symbolic_validations/verify_analytic_spectrum.py \
                         symbolic_validations/verify_isomorphism.py symbolic_validations/verify_ode.py \
                         symbolic_validations/verify_spectrum.py symbolic_validations/verify_step_interior.py \
                         symbolic_validations/verify_uv_subtraction.py \
                         symbolic_validations/confirm_geometry_mismatch.py
VALIDATION_SAGE = symbolic_validations/verify_ode.sage \
                         symbolic_validations/derive_delta_shell.sage \
                         symbolic_validations/derive_sech2_shell.sage \
                         symbolic_validations/derive_wkb_limit.sage \
                         symbolic_validations/derive_flux_quantization.sage

validate:
	@if ! command -v sage > /dev/null; then \
		echo "Error: 'sage' command not found. Please activate your SageMath environment (e.g., 'conda activate sage') before running validations."; \
		exit 1; \
	fi
	@echo "--- Running Numerical Validations ---"
	@failed=0; \
	for script in $(VALIDATION_PY); do \
		echo ""; \
		echo "Executing $$script..."; \
		PYTHONPATH=. python3 $$script || failed=1; \
	done; \
	echo "--- Running SageMath/Symbolic Validations ---"; \
	for script in $(VALIDATION_SAGEMATH_PY); do \
		echo ""; \
		echo "Executing $$script..."; \
		sage -python $$script || failed=1; \
	done; \
	for script in $(VALIDATION_SAGE); do \
		echo ""; \
		echo "Executing $$script..."; \
		sage $$script || failed=1; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo ""; \
		echo "------------------------------------"; \
		echo "✅ All automated validations PASSED."; \
	else \
		echo ""; \
		echo "------------------------------------"; \
		echo "❌ Some validations FAILED. Check output above."; \
		exit 1; \
	fi
	@echo ""
	@echo "--- Visual Inspection Required ---"
	@echo "1. scripts/compare_analytics.py: Check results/analytic_vs_numerical.png. Verify high-degree overlap between numerical and analytic solutions in the top panels, and that residuals in the bottom panel are noise-level."
	@echo "2. scripts/visualize_benchmark.py: Check results/test_benchmark_absolute_visualization.png. Confirm that the numerical integration tracks the analytic benchmark for absolute amplitude."
	@echo "3. scripts/test_delta_shell_validation.py: Check results/delta_shell_greens_function_comparison.png. Verify agreement between numerical and analytic Green's functions. Small boundary-induced peaks are acceptable (see TECHNICAL_VALIDATION.md)."
	@echo "4. scripts/test_sech2_shell_validation.py: Check results/sech2_shell_greens_function_validation.png. Verify agreement between numerical Sech2 and analytic delta-shell equivalent."
	@echo "5. scripts/plot_greens_function_comparison.py: Check results/greens_function_comparison.png. Verify the 2-panel comparison and residual behavior according to TECHNICAL_VALIDATION.md criteria."
	@echo "6. scripts/plot_sech2_greens_function.py: Check results/sech2_shell_greens_function_comparison.png. Verify comparison plot; divergences should be localized to shell boundaries."
	@echo "7. scripts/validate_wkb.py: Check results/wkb_validation_visual.png. Verify oscillatory match between numerical solver and WKB benchmark, observing amplitude and offset residuals."
	@echo "8. scripts/validate_step_ea.py: Check results/step_ea_density_comparison.png. Verify that the analytic and numerical densities overlap (median ratio approx 1.0)."
	@echo "9. scripts/validate_density_log.py: Check results/step_ea_density_log_rho.png. Verify the asymptotic power-law decay of the negative density as rho -> infinity."

.PHONY: all clean validate

test: all
	PYTHONPATH=. pytest tests
