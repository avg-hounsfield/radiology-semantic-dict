"""
Radiology Semantic Dictionary
=============================
A Python package for interpreting radiology report findings.

v1.2.0 adds NLP preprocessing for real-world report interpretation:
- Negation detection ("no pneumothorax" -> negated=True)
- Uncertainty quantification ("possible", "cannot exclude" -> certainty levels)
- Temporal context ("new", "stable", "worsening" -> temporality)
- Measurement extraction ("6mm nodule" -> value=6, unit=mm)
- Laterality detection (left/right/bilateral)

Core Features:
- 8,300+ imaging findings with pathology mappings
- 2,900+ medical concepts with synonyms
- 360+ differential diagnosis groups
- 450+ radiology synonym mappings
- 69 named imaging signs with differentials
- 7 ACR classification systems (BI-RADS, LI-RADS, etc.)
- 25 radiologic measurement thresholds (Fleischner, TI-RADS, organ sizes)
- 36 board-tested mnemonics
- 59 syndrome-imaging associations

Example (new interpret_finding API):
    >>> from radiology_semantic import SemanticRadDict
    >>> srd = SemanticRadDict()
    >>> result = srd.interpret_finding("No evidence of pulmonary embolism")
    >>> print(result['negated'], result['certainty'])
    True negated

Example (classic findings_to_diagnosis API):
    >>> result = srd.findings_to_diagnosis(["fat stranding", "RLQ"], modality="CT")
    >>> print(result.primary_diagnosis)
    'Appendicitis'

No API calls, no LLM required - pure regex-based NLP and dictionary lookups.
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

from .nlp import (
    RadiologyNLP,
    ExtractedFinding,
    ExtractedMeasurement,
    Certainty,
    Temporality,
    is_negated,
    get_certainty,
    get_temporality,
    extract_measurements,
)

__version__ = "1.2.0"
__author__ = "Radiology AI Assistant Team"
__all__ = [
    # Core dictionary
    "SemanticRadDict",
    "DiagnosisResult",
    "Finding",
    "DifferentialGroup",
    "MeasurementThreshold",
    "Mnemonic",
    "SyndromeAssociation",
    # NLP preprocessing
    "RadiologyNLP",
    "ExtractedFinding",
    "ExtractedMeasurement",
    "Certainty",
    "Temporality",
    "is_negated",
    "get_certainty",
    "get_temporality",
    "extract_measurements",
]
