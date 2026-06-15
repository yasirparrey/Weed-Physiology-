"""Dicamba spray window predictor for Kansas based on EPA label requirements."""

from .predictor import DicambaSprayPredictor, predict_spray_windows
from .models import SprayStatus, HourlyAssessment, DailySummary

__all__ = [
    "DicambaSprayPredictor",
    "predict_spray_windows",
    "SprayStatus",
    "HourlyAssessment",
    "DailySummary",
]

__version__ = "1.0.0"
