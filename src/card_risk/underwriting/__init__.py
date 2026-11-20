'''
Underwriting (application) model on a simulated applicant population.

The transaction data has no application or credit bureau attributes and no default
outcomes, so this package simulates them from a known ground-truth process. Only
booked applicants get observed outcomes, as in real life.
'''

from card_risk.underwriting.features import (
    UW_CAT_FEATURES,
    UW_NUM_FEATURES,
    build_underwriting_features,
)
from card_risk.underwriting.model import UnderwritingModel, convert_pd_to_underwriting_score
from card_risk.underwriting.simulate import simulate_applicant_population

__all__ = [
    "UW_CAT_FEATURES",
    "UW_NUM_FEATURES",
    "UnderwritingModel",
    "build_underwriting_features",
    "convert_pd_to_underwriting_score",
    "simulate_applicant_population",
]
