#!/usr/bin/env python3
"""Test the new clinical reasoning module."""

import sys
sys.path.insert(0, '.')

from radiology_semantic import SemanticRadDict, ClinicalReasoner, combine_findings

srd = SemanticRadDict()

print("=" * 70)
print("CLINICAL REASONING TEST")
print("=" * 70)

# =============================================================================
# TEST 1: MULTI-FINDING COMBINATION
# =============================================================================
print("\n### TEST 1: MULTI-FINDING COMBINATION ###")
print("-" * 50)

multi_finding_tests = [
    # (findings, expected_diagnosis, expected_min_confidence)
    (
        ["Fat stranding in the RLQ", "Appendicolith", "Periappendiceal fluid"],
        "Appendicitis",
        0.85
    ),
    (
        ["Gallbladder wall thickening", "Pericholecystic fluid", "Cholelithiasis"],
        "Acute cholecystitis",
        0.85
    ),
    (
        ["Small bowel dilation", "Decompressed colon", "Transition point"],
        "Small bowel obstruction",
        0.85
    ),
    (
        ["Filling defect in pulmonary artery", "DVT in lower extremity"],
        "Pulmonary embolism",
        0.90
    ),
    (
        ["Restricted diffusion in MCA territory", "Loss of gray white differentiation"],
        "stroke",  # Accept any stroke pattern
        0.80
    ),
    (
        ["Periventricular white matter lesions", "Dawson fingers", "Corpus callosum lesions"],
        "Multiple sclerosis",
        0.85
    ),
]

passed = 0
for findings, expected_dx, min_confidence in multi_finding_tests:
    result = srd.analyze_findings(findings)

    top_dx = result.get('top_diagnosis')
    if top_dx and expected_dx.lower() in top_dx['name'].lower():
        if top_dx['confidence'] >= min_confidence:
            passed += 1
            print(f"[PASS] {expected_dx}")
            print(f"       Confidence: {top_dx['confidence']:.2f} (>= {min_confidence})")
            print(f"       Findings: {findings[:2]}...")
        else:
            print(f"[FAIL] {expected_dx} - confidence too low")
            print(f"       Got: {top_dx['confidence']:.2f}, expected >= {min_confidence}")
    else:
        print(f"[FAIL] {expected_dx}")
        print(f"       Got: {top_dx}")
        print(f"       Findings: {findings}")
    print()

print(f"Multi-finding: {passed}/{len(multi_finding_tests)} passed")

# =============================================================================
# TEST 2: MEASUREMENT THRESHOLD EVALUATION
# =============================================================================
print("\n### TEST 2: MEASUREMENT THRESHOLD EVALUATION ###")
print("-" * 50)

measurement_tests = [
    # (value, unit, context, expected_threshold_met, expected_significance_contains)
    (15, "cm", "spleen", True, "splenomegaly"),
    (10, "cm", "spleen", False, ""),  # Normal, threshold not met
    (6, "cm", "aorta", True, "repair"),  # 6cm hits surgical threshold
    (8, "mm", "pulmonary nodule", True, "PET"),
    (4, "mm", "pulmonary nodule", True, "No routine follow-up"),
    (7, "mm", "CBD", True, "dilated"),
]

passed = 0
for value, unit, context, expected_met, expected_contains in measurement_tests:
    recommendations = srd.get_clinical_recommendation(value, unit, context)

    if not recommendations and not expected_met:
        passed += 1
        print(f"[PASS] {value}{unit} {context} - no threshold triggered (as expected)")
    elif recommendations:
        relevant = [r for r in recommendations if r['met'] == expected_met]
        if relevant:
            r = relevant[0]
            if expected_contains.lower() in r['significance'].lower() or not expected_contains:
                passed += 1
                print(f"[PASS] {value}{unit} {context}")
                print(f"       Threshold: {r['threshold_name']}")
                print(f"       Action: {r['action'][:50]}...")
            else:
                print(f"[FAIL] {value}{unit} {context} - wrong significance")
                print(f"       Expected to contain: {expected_contains}")
                print(f"       Got: {r['significance']}")
        else:
            print(f"[FAIL] {value}{unit} {context} - threshold met mismatch")
    else:
        print(f"[FAIL] {value}{unit} {context} - no recommendations but expected some")
    print()

print(f"Measurements: {passed}/{len(measurement_tests)} passed")

# =============================================================================
# TEST 3: URGENCY DETECTION
# =============================================================================
print("\n### TEST 3: URGENCY DETECTION ###")
print("-" * 50)

urgency_tests = [
    (["Saddle embolus", "RV strain"], "urgent"),
    (["Aortic dissection", "Intimal flap"], "urgent"),
    (["Restricted diffusion", "MCA territory", "Dense MCA sign"], "urgent"),
    (["Hepatic steatosis", "Cholelithiasis"], "routine"),
    (["Pulmonary nodule 6mm", "Stable"], "routine"),
]

passed = 0
for findings, expected_urgency in urgency_tests:
    result = srd.analyze_findings(findings)
    actual_urgency = result.get('urgency', 'routine')

    if actual_urgency == expected_urgency:
        passed += 1
        print(f"[PASS] {findings[0][:30]}... -> {actual_urgency}")
    else:
        print(f"[FAIL] {findings[0][:30]}...")
        print(f"       Expected: {expected_urgency}, Got: {actual_urgency}")
    print()

print(f"Urgency: {passed}/{len(urgency_tests)} passed")

# =============================================================================
# TEST 4: FULL ANALYSIS INTEGRATION
# =============================================================================
print("\n### TEST 4: FULL ANALYSIS INTEGRATION ###")
print("-" * 50)

# Simulate a complex case
findings = [
    "7mm appendicolith in the RLQ",
    "Fat stranding surrounding the appendix",
    "Periappendiceal fluid collection",
    "Appendix measures 12mm in diameter",
    "No free intraperitoneal air"
]

print("Case: Suspected appendicitis")
print(f"Findings: {findings}")
print()

result = srd.analyze_findings(findings, modality="CT")

print("Analysis Results:")
print(f"  Top Diagnosis: {result['top_diagnosis']}")
print(f"  Urgency: {result['urgency']}")
print(f"  Body Regions: {result['body_regions']}")
print(f"  Reasoning: {result['reasoning']}")

if result['combined_diagnoses']:
    print(f"\n  Combined Diagnoses:")
    for dx in result['combined_diagnoses'][:3]:
        print(f"    - {dx['name']} (conf: {dx['confidence']:.2f})")
        print(f"      Supporting: {dx['supporting_findings']}")

if result['measurement_evaluations']:
    print(f"\n  Measurement Evaluations:")
    for m in result['measurement_evaluations'][:3]:
        print(f"    - {m['value']}{m['unit']}: {m['significance'][:50]}...")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("CLINICAL REASONING SUMMARY")
print("=" * 70)
print("""
New capabilities in v1.3.0:

1. MULTI-FINDING COMBINATION
   - 25+ clinical patterns (appendicitis, cholecystitis, SBO, PE, stroke, etc.)
   - Confidence boosting based on supporting findings
   - Exclusion criteria (e.g., acalculous cholecystitis excludes gallstones)

2. MEASUREMENT THRESHOLD EVALUATION
   - 25 thresholds connected to clinical recommendations
   - Fleischner, TI-RADS, organ size criteria
   - Actionable recommendations with urgency levels

3. URGENCY DETECTION
   - Automatic urgency classification (routine/soon/urgent)
   - Pattern-based urgent diagnoses (dissection, saddle PE, stroke)
   - Keyword-based urgent finding detection

4. INTEGRATED ANALYSIS
   - analyze_findings() combines all reasoning
   - Returns structured output with confidence and reasoning
""")
