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
                scripts/visualize_benchmark.py

VALIDATION_SAGEMATH_PY = symbolic_validations/numerical_spectrum.py symbolic_validations/verify_analytic_spectrum.py \
                         symbolic_validations/verify_isomorphism.py symbolic_validations/verify_ode.py \
                         symbolic_validations/verify_spectrum.py symbolic_validations/verify_step_interior.py \
                         symbolic_validations/verify_uv_subtraction.py
VALIDATION_SAGE = symbolic_validations/verify_ode.sage

validate: all
	@if ! command -v sage > /dev/null; then \
		echo "Error: 'sage' command not found. Please activate your SageMath environment (e.g., 'conda activate sage') before running validations."; \
		exit 1; \
	fi
	@echo "--- Running Numerical Validations ---"
	@failed=0; \
	for script in $(VALIDATION_PY); do \
		echo ""; \
		echo "Executing $$script..."; \
		python3 $$script || failed=1; \
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
		echo "All automated validations PASSED."; \
	else \
		echo ""; \
		echo "------------------------------------"; \
		echo "Some validations FAILED. Check output above."; \
		exit 1; \
	fi
	@echo ""
	@echo "--- Visual Inspection Required ---"
	@echo "1. scripts/compare_analytics.py: Check results/analytic_vs_numerical.png. Verify high-degree overlap in the top two panels (Real/Imag) and near-zero residuals in the bottom panel."
	@echo "2. scripts/visualize_benchmark.py: Check results/test_benchmark_absolute_visualization.png. Confirm that the dashed numerical integration curve perfectly tracks the solid analytic benchmark for the absolute amplitude."


.PHONY: all clean validate

test: all
	pytest tests
