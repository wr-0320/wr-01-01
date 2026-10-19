"""Privileged headroom diagnostics and budget-matched active baselines."""

from deception_rl.intelligence.ride_v2.active import (
    MultiSourceEntropyVerify,
    MultiSourceRandomVerify,
)
from deception_rl.intelligence.ride_v2.oracle import (
    FullOracleDefender,
    OracleChangePoint,
    OracleDiagnosticStep,
    OracleSourceReliability,
    OracleVerifyPolicy,
)
from deception_rl.intelligence.ride_v2.reliability import (
    DynamicReliabilityController,
    DynamicSourceReliabilityFilter,
    extract_reliability_evidence,
)
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    HierarchicalSourceRegionReliabilityFilter,
    SourceRegionReliabilityController,
    extract_source_region_evidence,
)

__all__ = [
    "FullOracleDefender",
    "DynamicReliabilityController",
    "DynamicSourceReliabilityFilter",
    "HierarchicalSourceRegionReliabilityFilter",
    "MultiSourceEntropyVerify",
    "MultiSourceRandomVerify",
    "OracleChangePoint",
    "OracleDiagnosticStep",
    "OracleSourceReliability",
    "OracleVerifyPolicy",
    "SourceRegionReliabilityController",
    "extract_reliability_evidence",
    "extract_source_region_evidence",
]
