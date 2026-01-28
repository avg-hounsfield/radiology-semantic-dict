#!/usr/bin/env python3
"""
Test the integrated NLP + Dictionary for real-world report interpretation.
This tests the new interpret_finding() method.
"""

import sys
sys.path.insert(0, '.')

from radiology_semantic import SemanticRadDict

srd = SemanticRadDict()

print("=" * 70)
print("REAL-WORLD INTERPRETATION TEST (v2 - NLP Integrated)")
print("=" * 70)

# =============================================================================
# TEST 1: NEGATION HANDLING
# =============================================================================
print("\n### TEST 1: NEGATION HANDLING ###")
print("-" * 50)

negation_tests = [
    ("No pneumothorax", True, "Should detect as negated, no diagnoses"),
    ("No evidence of pulmonary embolism", True, "Should detect as negated"),
    ("Lungs are clear without consolidation", True, "Should detect as negated"),
    ("No acute intracranial abnormality", True, "Should detect as negated"),
    ("Large right pleural effusion", False, "Should NOT be negated"),
    ("Acute appendicitis with fat stranding", False, "Should NOT be negated"),
]

passed = 0
for text, expected_negated, description in negation_tests:
    result = srd.interpret_finding(text)

    is_correct = result['negated'] == expected_negated
    has_no_dx_if_negated = len(result['diagnoses']) == 0 if expected_negated else True

    if is_correct and has_no_dx_if_negated:
        passed += 1
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"{status} '{text[:50]}...'")
    print(f"       Negated: {result['negated']} (expected {expected_negated})")
    print(f"       Diagnoses: {len(result['diagnoses'])} returned")
    print()

print(f"Negation: {passed}/{len(negation_tests)} passed")

# =============================================================================
# TEST 2: CERTAINTY HANDLING
# =============================================================================
print("\n### TEST 2: CERTAINTY HANDLING ###")
print("-" * 50)

certainty_tests = [
    ("Definite pneumonia", "definite", "Full confidence"),
    ("Probable pneumonia", "probable", "Reduced confidence"),
    ("Possible appendicitis", "possible", "Low confidence"),
    ("Cannot exclude pulmonary embolism", "possible", "Uncertainty phrase"),
    ("Unlikely to represent malignancy", "unlikely", "Very low confidence"),
]

passed = 0
for text, expected_certainty, description in certainty_tests:
    result = srd.interpret_finding(text)

    is_correct = result['certainty'] == expected_certainty
    if is_correct:
        passed += 1
        status = "[PASS]"
    else:
        status = "[FAIL]"

    # Show confidence adjustment
    dx_conf = result['diagnoses'][0]['confidence'] if result['diagnoses'] else 0

    print(f"{status} '{text}'")
    print(f"       Certainty: {result['certainty']} (expected {expected_certainty})")
    print(f"       Top diagnosis confidence: {dx_conf:.2f}")
    print()

print(f"Certainty: {passed}/{len(certainty_tests)} passed")

# =============================================================================
# TEST 3: TEMPORAL HANDLING
# =============================================================================
print("\n### TEST 3: TEMPORAL HANDLING ###")
print("-" * 50)

temporal_tests = [
    ("New consolidation in the right lower lobe", "new"),
    ("Stable pulmonary nodule", "stable"),
    ("Interval worsening of pleural effusion", "worsened"),
    ("Resolving pneumonia", "improved"),
    ("Chronic appearing white matter changes", "chronic"),
    ("Acute stroke", "acute"),
]

passed = 0
for text, expected_temp in temporal_tests:
    result = srd.interpret_finding(text)

    is_correct = result['temporality'] == expected_temp
    if is_correct:
        passed += 1
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"{status} '{text}'")
    print(f"       Temporality: {result['temporality']} (expected {expected_temp})")
    print()

print(f"Temporality: {passed}/{len(temporal_tests)} passed")

# =============================================================================
# TEST 4: MEASUREMENT EXTRACTION
# =============================================================================
print("\n### TEST 4: MEASUREMENT EXTRACTION + RECOMMENDATIONS ###")
print("-" * 50)

measurement_tests = [
    ("6mm pulmonary nodule", 6.0, "mm", True),
    ("3.5 cm liver lesion", 3.5, "cm", True),
    ("15 cm spleen suggesting splenomegaly", 15.0, "cm", True),
    ("2 cm thyroid nodule with calcifications", 2.0, "cm", True),
]

passed = 0
for text, expected_val, expected_unit, has_recommendation in measurement_tests:
    result = srd.interpret_finding(text)

    has_measurement = len(result['measurements']) > 0
    if has_measurement:
        m = result['measurements'][0]
        val_correct = abs(m['value'] - expected_val) < 0.01
        unit_correct = m['unit'] == expected_unit
        is_correct = val_correct and unit_correct
    else:
        is_correct = False

    if is_correct:
        passed += 1
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"{status} '{text}'")
    print(f"       Measurements: {result['measurements']}")
    print(f"       Recommendations: {len(result['recommendations'])} available")
    print()

print(f"Measurements: {passed}/{len(measurement_tests)} passed")

# =============================================================================
# TEST 5: FULL INTEGRATION - COMPLEX FINDINGS
# =============================================================================
print("\n### TEST 5: FULL INTEGRATION - COMPLEX FINDINGS ###")
print("-" * 50)

complex_tests = [
    # (text, expected_negated, expected_certainty, expected_to_have_diagnoses)
    ("No evidence of acute pulmonary embolism", True, "negated", False),
    ("New 6mm right lower lobe pulmonary nodule", False, "definite", True),
    ("Stable 3 cm left adrenal mass, likely adenoma", False, "probable", True),
    ("Possible early appendicitis with fat stranding in the RLQ", False, "possible", True),
    ("Cannot exclude small bowel obstruction given dilated loops", False, "possible", True),
    ("Unchanged bilateral pleural effusions", False, "definite", True),
]

passed = 0
for text, exp_neg, exp_cert, exp_has_dx in complex_tests:
    result = srd.interpret_finding(text)

    neg_ok = result['negated'] == exp_neg
    cert_ok = result['certainty'] == exp_cert
    dx_ok = (len(result['diagnoses']) > 0) == exp_has_dx

    if neg_ok and cert_ok and dx_ok:
        passed += 1
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"{status} '{text[:55]}...'")
    print(f"       Negated: {result['negated']}, Certainty: {result['certainty']}, Temporality: {result['temporality']}")
    print(f"       Laterality: {result['laterality'] or 'none'}, Measurements: {result['measurements']}")
    if result['diagnoses']:
        top_dx = result['diagnoses'][0]
        print(f"       Top Dx: {top_dx['name']} (conf: {top_dx['confidence']:.2f})")
    else:
        print(f"       No diagnoses (negated finding)")
    print()

print(f"Complex findings: {passed}/{len(complex_tests)} passed")

# =============================================================================
# TEST 6: REPORT INTERPRETATION
# =============================================================================
print("\n### TEST 6: FULL REPORT INTERPRETATION ###")
print("-" * 50)

sample_report = """
FINDINGS:
The heart is normal in size. No pericardial effusion.
The lungs are clear without focal consolidation, pleural effusion, or pneumothorax.
New 8mm right lower lobe pulmonary nodule. Recommend follow-up CT in 6 months per Fleischner guidelines.
Stable bilateral hilar lymphadenopathy.
No acute intracranial abnormality.
"""

print("Input Report:")
print(sample_report)
print("-" * 50)
print("Interpreted Findings:")

results = srd.interpret_report(sample_report, modality="CT")

for i, r in enumerate(results):
    print(f"\n[{i+1}] {r['text'][:60]}...")
    print(f"    Negated: {r['negated']}, Certainty: {r['certainty']}, Temporality: {r['temporality']}")
    if r['diagnoses']:
        print(f"    Diagnoses: {[d['name'] for d in r['diagnoses'][:2]]}")
    if r['measurements']:
        print(f"    Measurements: {r['measurements']}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("v2 SUMMARY: NLP-INTEGRATED INTERPRETATION")
print("=" * 70)
print("""
The new interpret_finding() method now provides:

1. NEGATION DETECTION - Negated findings return empty diagnosis list
2. CERTAINTY QUANTIFICATION - Confidence scores adjusted by hedge words
3. TEMPORAL CONTEXT - Finding status (new/stable/worsened/etc.) extracted
4. MEASUREMENT EXTRACTION - Values and units parsed from text
5. LATERALITY - Left/right/bilateral detected
6. INTEGRATED OUTPUT - Structured dict with all attributes

This is a significant improvement for real-world report interpretation.

REMAINING GAPS:
- Common incidental finding vocabulary still limited
- Guideline recommendations not yet fully connected to measurements
- Complex multi-finding reasoning not implemented
- No body region extraction (beyond laterality)
""")
