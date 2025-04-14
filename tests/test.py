#!/usr/bin/env python3
"""
tests/test.py

Este script importa las funciones de pronóstico del módulo y ejecuta la comparación de pronósticos.
Se usa pytest para ejecutar el test.
"""

import pytest
from greencurve.predictor import compare_forecasts

def test_forecast_comparison():
    country = "NL"
    forecast_steps = 48  # 48 horas
    # Ejecuta la comparación de pronósticos.
    compare_forecasts(country, forecast_steps)
    # Puedes agregar aserciones adicionales según tus necesidades.

if __name__ == "__main__":
    pytest.main(["-s", __file__])
