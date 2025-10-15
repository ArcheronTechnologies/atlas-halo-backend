"""
Scaling module for automatic scaling and load management.
"""

from .auto_scaler import auto_scaler, run_scaling_evaluation, execute_auto_scaling

__all__ = ['auto_scaler', 'run_scaling_evaluation', 'execute_auto_scaling']