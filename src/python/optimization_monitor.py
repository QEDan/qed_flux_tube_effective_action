import torch
import logging

class OptimizationMonitor:
    """
    Lightweight health monitor inspired by DLCheck for standalone PyTorch optimization.
    Monitors weight distributions, gradient stability, and NaN propagation.
    """
    def __init__(self, model, exploding_grad_threshold=1e3, vanishing_grad_threshold=1e-9):
        self.model = model
        self.exploding_grad_threshold = exploding_grad_threshold
        self.vanishing_grad_threshold = vanishing_grad_threshold
        self.logger = logging.getLogger("OptimizationMonitor")
        logging.basicConfig(level=logging.INFO)

    def check_health(self):
        """Perform health checks on parameters and gradients."""
        self._check_nan_inf()
        self._check_gradient_stability()
        self._check_weight_stats()

    def _check_nan_inf(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    self.logger.error(f"NaN/Inf gradient detected in {name}")
            if not torch.isfinite(param).all():
                self.logger.error(f"NaN/Inf weight detected in {name}")

    def _check_gradient_stability(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if grad_norm > self.exploding_grad_threshold:
                    self.logger.warning(f"Exploding gradient in {name}: norm={grad_norm:.2e}")
                elif grad_norm < self.vanishing_grad_threshold and grad_norm > 0:
                    self.logger.warning(f"Vanishing gradient in {name}: norm={grad_norm:.2e}")

    def _check_weight_stats(self):
        for name, param in self.model.named_parameters():
            # Check for weight saturation / divergence
            percentile_25, percentile_75 = torch.quantile(param, torch.tensor([0.25, 0.75], device=param.device))
            if abs(percentile_25) > 1e3 or abs(percentile_75) > 1e3:
                self.logger.warning(f"Weights in {name} appear to be diverging.")
