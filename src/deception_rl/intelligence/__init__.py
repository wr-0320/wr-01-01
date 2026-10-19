"""RIDe components for adversarial intelligence decision experiments."""

from deception_rl.intelligence.adapter import (
    INTELLIGENCE_FEATURE_SIZE,
    encode_intelligence_sequence,
)
from deception_rl.intelligence.ride import (
    HonestIntelligenceReasoner,
    IntelligenceRIDeModel,
    RIDeIntelligenceController,
    TemperatureCalibrator,
)

__all__ = [
    "INTELLIGENCE_FEATURE_SIZE",
    "encode_intelligence_sequence",
    "HonestIntelligenceReasoner",
    "IntelligenceRIDeModel",
    "RIDeIntelligenceController",
    "TemperatureCalibrator",
]

