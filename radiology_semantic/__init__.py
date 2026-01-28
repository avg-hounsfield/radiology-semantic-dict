"""
Radiology Semantic Dictionary
=============================
A lightweight Python package for mapping radiology findings to diagnoses.

Features:
- 8,300+ imaging findings with pathology mappings
- 2,900+ medical concepts with synonyms
- 360+ differential diagnosis groups
- 450+ radiology synonym mappings
- 69 named imaging signs with differentials
- 7 ACR classification systems (BI-RADS, LI-RADS, etc.)
- 25 radiologic measurement thresholds (Fleischner, TI-RADS, organ sizes)
- 36 board-tested mnemonics (VITAMIN D, CHICAGO, FLAMES, etc.)
- 59 syndrome-imaging associations with screening recommendations

Example:
    >>> from radiology_semantic import SemanticRadDict
    >>> srd = SemanticRadDict()
    >>> result = srd.findings_to_diagnosis(["fat stranding", "RLQ"], modality="CT")
    >>> print(result.primary_diagnosis)
    'Appendicitis'

No API calls, no LLM required - pure dictionary lookups.
"""

from .dictionary import (
    SemanticRadDict,
    DiagnosisResult,
    Finding,
    DifferentialGroup,
    MeasurementThreshold,
    Mnemonic,
    SyndromeAssociation
)

__version__ = "1.1.1"
__author__ = "Radiology AI Assistant Team"
__all__ = [
    "SemanticRadDict",
    "DiagnosisResult",
    "Finding",
    "DifferentialGroup",
    "MeasurementThreshold",
    "Mnemonic",
    "SyndromeAssociation"
]
