"""
Algorithm Strategy Implementations

This module contains all concrete strategy implementations for the trading system.
Each strategy inherits from the base Strategy class and implements Optuna-based
hyperparameter optimization.

Strategy Categories:
- Boosting: lgbm, xgboost, catboost, histgb, ngboost, rf
- Classical ML: svc, knn, logistic, nb, adaboost
- Neural Networks: pytorch_nn, mlp
- Specialized: gp (Gaussian Process)
- Meta: ensemble (combines multiple strategies)
"""

# Import all strategy classes for convenience
from .lgbm_strategy import LGBMOptunaStrategy
from .xgboost_strategy import XGBoostOptunaStrategy
from .catboost_strategy import CatBoostOptunaStrategy
from .histgb_strategy import HistGBOptunaStrategy
from .ngboost_strategy import NGBoostOptunaStrategy
from .rf_strategy import RandomForestOptunaStrategy
from .svc_strategy import SVCOptunaStrategy
from .knn_strategy import KNNOptunaStrategy
from .logistic_strategy import LogisticRegressionOptunaStrategy
from .nb_strategy import GaussianNBOptunaStrategy
from .adaboost_strategy import AdaBoostOptunaStrategy
from .pytorch_nn_strategy import PyTorchNeuralNetOptunaStrategy
from .mlp_strategy import MLPOptunaStrategy
from .gp_strategy import GaussianProcessOptunaStrategy
from .ensemble_strategy import EnsembleOptunaStrategy

__all__ = [
    'LGBMOptunaStrategy',
    'XGBoostOptunaStrategy',
    'CatBoostOptunaStrategy',
    'HistGBOptunaStrategy',
    'NGBoostOptunaStrategy',
    'RandomForestOptunaStrategy',
    'SVCOptunaStrategy',
    'KNNOptunaStrategy',
    'LogisticRegressionOptunaStrategy',
    'GaussianNBOptunaStrategy',
    'AdaBoostOptunaStrategy',
    'PyTorchNeuralNetOptunaStrategy',
    'MLPOptunaStrategy',
    'GaussianProcessOptunaStrategy',
    'EnsembleOptunaStrategy',
]
