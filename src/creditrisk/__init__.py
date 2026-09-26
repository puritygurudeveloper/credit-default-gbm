"""Cost-sensitive consumer credit default modelling.

The package is deliberately small and functional: every stage of the pipeline
is a pure function over ``pandas`` objects, so a stage can be unit tested in
isolation and swapped without touching the rest of the chain.

Pipeline
--------
``data``        synthetic loan book with a latent creditworthiness factor
``features``    weight-of-evidence binning and ratio construction
``models``      scorecard (logistic on WOE) and gradient boosting
``calibration`` probability calibration and reliability diagnostics
``evaluate``    discrimination metrics (AUC, Gini, KS, lift)
``economics``   turning a calibrated PD into an approve/decline decision
``explain``     SHAP attributions for model governance
"""

from creditrisk.__about__ import __version__

__all__ = ["__version__"]
