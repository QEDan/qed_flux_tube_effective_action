CC = gcc
CFLAGS = -O3 -fPIC -fopenmp -Wall
LDFLAGS = -shared -fopenmp

TARGET = libsolver.so
SRCS = src/c/solver.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

VALIDATION_PY = scripts/test_zero_field.py scripts/compare_analytics.py \
                scripts/compare_full_regime.py scripts/visualize_benchmark.py

VALIDATION_SAGEMATH_PY = sagemath/numerical_spectrum.py sagemath/verify_analytic_spectrum.py \
                         sagemath/verify_isomorphism.py sagemath/verify_ode.py \
                         sagemath/verify_spectrum.py sagemath/verify_step_interior.py \
                         sagemath/verify_uv_subtraction.py
VALIDATION_SAGE = sagemath/verify_ode.sage

validate: all
	@if ! command -v sage > /dev/null; then \
		echo "Error: 'sage' command not found. Please activate your SageMath environment (e.g., 'conda activate sage') before running validations."; \
		exit 1; \
	fi
	@echo "--- Running Numerical Validations ---"
	@failed=0; \
	for script in $(VALIDATION_PY); do \
		echo "Executing $$script..."; \
		python3 $$script || failed=1; \
	done; \
	echo "--- Running SageMath/Symbolic Validations ---"; \
	for script in $(VALIDATION_SAGEMATH_PY); do \
		echo "Executing $$script..."; \
		sage -python $$script || failed=1; \
	done; \
	for script in $(VALIDATION_SAGE); do \
		echo "Executing $$script..."; \
		sage $$script || failed=1; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "------------------------------------"; \
		echo "All automated validations PASSED."; \
	else \
		echo "------------------------------------"; \
		echo "Some validations FAILED. Check output above."; \
		exit 1; \
	fi
	@echo ""
	@echo "--- Visual Inspection Required ---"
	@echo "1. scripts/compare_analytics.py: Check results/analytic_vs_numerical_smoothed.png. Look for overlap between analytic and numerical curves."
	@echo "2. scripts/compare_full_regime.py: Check results/analytic_vs_numerical_full_residuals_v2.png. Look for small residuals and overlap."
	@echo "3. scripts/visualize_benchmark.py: Check results/test_benchmark_absolute_visualization.png. Look for overlap of numerical/analytic absolute solutions."

.PHONY: all clean validate
