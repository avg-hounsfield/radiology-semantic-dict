#!/usr/bin/env python3
"""Test the new NLP preprocessing module."""

import sys
sys.path.insert(0, '.')

from radiology_semantic.nlp import RadiologyNLP, Certainty, Temporality

nlp = RadiologyNLP()

print("=" * 70)
print("TESTING NLP PREPROCESSING MODULE")
print("=" * 70)

# =============================================================================
# NEGATION TESTS
# =============================================================================
print("\n### NEGATION DETECTION ###")
print("-" * 50)

negation_tests = [
    ("No pneumothorax", True),
    ("No evidence of pulmonary embolism", True),
    ("Lungs are clear without consolidation", True),
    ("No acute intracranial abnormality", True),
    ("Appendix is normal", True),
    ("Gallbladder is unremarkable", True),
    ("Large right pleural effusion", False),
    ("3cm mass in the liver", False),
    ("Acute appendicitis", False),
    ("There is a pneumothorax", False),
    ("Cannot exclude pulmonary embolism", False),  # This is UNCERTAINTY, not negation
    ("Ruled out for PE", True),
]

passed = 0
for text, expected in negation_tests:
    result = nlp.detect_negation(text)
    is_correct = result[0] == expected
    if is_correct:
        passed += 1
    status = "[PASS]" if is_correct else "[FAIL]"
    print(f"{status} '{text[:50]}...' -> negated={result[0]} (expected {expected})")

print(f"\nNegation: {passed}/{len(negation_tests)} passed")

# =============================================================================
# CERTAINTY TESTS
# =============================================================================
print("\n### CERTAINTY DETECTION ###")
print("-" * 50)

certainty_tests = [
    ("Definite pneumonia in the right lower lobe", Certainty.DEFINITE),
    ("Probable pneumonia", Certainty.PROBABLE),
    ("Likely represents pneumonia", Certainty.PROBABLE),
    ("Possible pneumonia", Certainty.POSSIBLE),
    ("Cannot exclude pulmonary embolism", Certainty.POSSIBLE),
    ("May represent early appendicitis", Certainty.POSSIBLE),
    ("Suspicious for malignancy", Certainty.POSSIBLE),
    ("Unlikely to represent malignancy", Certainty.UNLIKELY),
    ("Low probability for PE", Certainty.UNLIKELY),
    ("No evidence of pneumonia", Certainty.NEGATED),
    ("Consolidation in the right lower lobe", Certainty.DEFINITE),
]

passed = 0
for text, expected in certainty_tests:
    result = nlp.detect_certainty(text)
    is_correct = result == expected
    if is_correct:
        passed += 1
    status = "[PASS]" if is_correct else "[FAIL]"
    print(f"{status} '{text[:45]}...' -> {result.value} (expected {expected.value})")

print(f"\nCertainty: {passed}/{len(certainty_tests)} passed")

# =============================================================================
# TEMPORALITY TESTS
# =============================================================================
print("\n### TEMPORALITY DETECTION ###")
print("-" * 50)

temporal_tests = [
    ("New consolidation in the right lower lobe", Temporality.NEW),
    ("Unchanged pulmonary nodule", Temporality.STABLE),
    ("Stable appearance of the liver lesion", Temporality.STABLE),
    ("Interval increase in pleural effusion", Temporality.WORSENED),
    ("Worsening pneumonia", Temporality.WORSENED),
    ("Improving consolidation", Temporality.IMPROVED),
    ("Resolving pneumonia", Temporality.IMPROVED),
    ("Chronic appearing changes", Temporality.CHRONIC),
    ("Old infarct", Temporality.CHRONIC),
    ("Acute stroke", Temporality.ACUTE),
    ("There is a pulmonary nodule", Temporality.UNKNOWN),
]

passed = 0
for text, expected in temporal_tests:
    result = nlp.detect_temporality(text)
    is_correct = result == expected
    if is_correct:
        passed += 1
    status = "[PASS]" if is_correct else "[FAIL]"
    print(f"{status} '{text[:45]}...' -> {result.value} (expected {expected.value})")

print(f"\nTemporality: {passed}/{len(temporal_tests)} passed")

# =============================================================================
# MEASUREMENT TESTS
# =============================================================================
print("\n### MEASUREMENT EXTRACTION ###")
print("-" * 50)

measurement_tests = [
    ("6mm pulmonary nodule", [(6.0, "mm")]),
    ("3.5 cm liver lesion", [(3.5, "cm")]),
    ("2.3 x 1.5 x 1.2 cm mass", [(2.3, "cm")]),  # Max dimension
    ("15 cm spleen", [(15.0, "cm")]),
    ("Measures 4 mm in diameter", [(4.0, "mm")]),
    ("No measurements", []),
]

passed = 0
for text, expected in measurement_tests:
    result = nlp.extract_measurements(text)
    if len(result) == len(expected):
        if len(expected) == 0:
            is_correct = True
        else:
            is_correct = abs(result[0].value - expected[0][0]) < 0.01 and result[0].unit == expected[0][1]
    else:
        is_correct = False

    if is_correct:
        passed += 1
    status = "[PASS]" if is_correct else "[FAIL]"
    extracted = [(m.value, m.unit) for m in result] if result else []
    print(f"{status} '{text}' -> {extracted}")

print(f"\nMeasurements: {passed}/{len(measurement_tests)} passed")

# =============================================================================
# FULL PROCESSING TEST
# =============================================================================
print("\n### FULL FINDING PROCESSING ###")
print("-" * 50)

full_tests = [
    "No evidence of acute pulmonary embolism",
    "New 6mm right lower lobe pulmonary nodule, recommend follow-up CT in 6 months",
    "Stable 3 cm left adrenal mass, likely adenoma",
    "Possible early appendicitis with fat stranding in the RLQ",
    "Interval worsening of bilateral pleural effusions",
]

for text in full_tests:
    result = nlp.process(text)
    print(f"\nInput: {text}")
    print(f"  Negated: {result.negated}")
    print(f"  Certainty: {result.certainty.value}")
    print(f"  Temporality: {result.temporality.value}")
    print(f"  Measurements: {[(m.value, m.unit) for m in result.measurements]}")
    print(f"  Laterality: {result.laterality or 'none'}")

print("\n" + "=" * 70)
print("NLP MODULE TEST COMPLETE")
print("=" * 70)
