"""
GreenCurve: Renewable Energy Prediction

This package provides functionality to predict the 24-hour renewable energy production
curve for the upcoming day based on historical production data.
"""

__version__ = "0.1.0"

from .predictor import predict_energy_curve

__all__ = ["predict_energy_curve"]
