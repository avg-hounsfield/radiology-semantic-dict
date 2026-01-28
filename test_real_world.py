#!/usr/bin/env python3
"""
Test the semantic dictionary against realistic radiology report language.
This exposes gaps that would prevent real-world usage.
"""

import sys
sys.path.insert(0, '.')

from radiology_semantic import SemanticRadDict

srd = SemanticRadDict()

print("=" * 70)
print("REAL-WORLD RADIOLOGY REPORT LANGUAGE TEST")
print("=" * 70)

# ============================================================================
# TEST 1: NEGATION HANDLING
# ============================================================================
print("\n### TEST 1: NEGATION HANDLING ###")
print("Real reports frequently negate findings. Does the system understand?")
print("-" * 50)

negation_tests = [
    # (input, should_NOT_diagnose)
    (["no pneumothorax"], "pneumothorax"),
    (["no evidence of pulmonary embolism"], "pulmonary embolism"),
    (["lungs are clear without consolidation"], "pneumonia"),
    (["no acute intracranial abnormality"], "stroke"),
    (["appendix is normal without inflammation"], "appendicitis"),
    (["gallbladder is unremarkable"], "cholecystitis"),
]

for findings, should_not_have in negation_tests:
    result = srd.findings_to_diagnosis(findings)
    primary = result.primary_diagnosis.lower() if result.primary_diagnosis else ""
    top_3 = [d[0].lower() if isinstance(d, tuple) else d.lower() for d in result.differentials[:3]]

    # Check if we incorrectly diagnosed something that was negated
    false_positive = should_not_have.lower() in primary or any(should_not_have.lower() in d for d in top_3)

    status = "[FAIL - FALSE POSITIVE]" if false_positive else "[PASS]" if not result.primary_diagnosis else "[WARN - unexpected result]"
    print(f"{status} Input: {findings}")
    if result.primary_diagnosis:
        print(f"         Got: {result.primary_diagnosis}")
    print()

# ============================================================================
# TEST 2: HEDGING/UNCERTAINTY LANGUAGE
# ============================================================================
print("\n### TEST 2: HEDGING/UNCERTAINTY LANGUAGE ###")
print("Reports often hedge. Does the system recognize uncertainty?")
print("-" * 50)

hedge_tests = [
    (["possible pneumonia"], "pneumonia", "uncertain"),
    (["cannot exclude pulmonary embolism"], "pulmonary embolism", "uncertain"),
    (["findings may represent early appendicitis"], "appendicitis", "uncertain"),
    (["correlate clinically for cholecystitis"], "cholecystitis", "uncertain"),
    (["low probability for malignancy"], "malignancy", "low"),
    (["suspicious for renal cell carcinoma"], "renal cell carcinoma", "high"),
]

for findings, expected_dx, expected_certainty in hedge_tests:
    result = srd.findings_to_diagnosis(findings)
    primary = result.primary_diagnosis.lower() if result.primary_diagnosis else ""

    found_dx = expected_dx.lower() in primary
    # Check if confidence reflects uncertainty (it probably doesn't)

    print(f"Input: {findings}")
    print(f"  Expected: {expected_dx} with {expected_certainty} certainty")
    print(f"  Got: {result.primary_diagnosis} (conf: {result.confidence:.2f})")
    print(f"  [{'OK' if found_dx else 'MISS'}] - Note: No uncertainty handling in output")
    print()

# ============================================================================
# TEST 3: MEASUREMENT-BASED FINDINGS
# ============================================================================
print("\n### TEST 3: MEASUREMENT-BASED FINDINGS ###")
print("Reports include measurements. Can we interpret them?")
print("-" * 50)

measurement_tests = [
    (["6mm pulmonary nodule"], "Should trigger Fleischner - what follow-up?"),
    (["15cm spleen"], "Should recognize splenomegaly (>12cm)"),
    (["4cm adrenal mass"], "Should flag for workup (>4cm concerning)"),
    (["8mm gallstone"], "Should note cholelithiasis"),
    (["3.5cm aorta"], "Should recognize aneurysm if >3cm abdominal"),
    (["2cm thyroid nodule with microcalcifications"], "TI-RADS scoring?"),
]

for findings, expected_interpretation in measurement_tests:
    result = srd.findings_to_diagnosis(findings)

    print(f"Input: {findings}")
    print(f"  Question: {expected_interpretation}")
    print(f"  Got: {result.primary_diagnosis or 'No diagnosis'}")
    # Check if we have measurement threshold data
    thresholds = srd.get_measurement_threshold(findings[0].split()[0] if findings else "")
    print(f"  Thresholds available: {len(thresholds) if thresholds else 0}")
    print()

# ============================================================================
# TEST 4: COMPLEX REPORT PHRASES
# ============================================================================
print("\n### TEST 4: COMPLEX REPORT PHRASES ###")
print("Real report language is verbose. Can we parse it?")
print("-" * 50)

complex_tests = [
    # Real report-style phrasing
    (["ill-defined hypoattenuating lesion in the right hepatic lobe with arterial enhancement and portal venous washout"], "hepatocellular carcinoma"),
    (["segmental pulmonary artery filling defects bilaterally consistent with acute PE"], "pulmonary embolism"),
    (["diffuse periventricular and subcortical white matter T2/FLAIR hyperintensities"], "multiple sclerosis"),
    (["fat-containing right adnexal lesion with Rokitansky nodule"], "dermoid cyst"),
    (["dilated loops of small bowel with air-fluid levels and transition point in the RLQ"], "small bowel obstruction"),
    (["subcentimeter mesenteric lymph nodes likely reactive"], "reactive lymphadenopathy"),
]

for findings, expected in complex_tests:
    result = srd.findings_to_diagnosis(findings)
    primary = result.primary_diagnosis.lower() if result.primary_diagnosis else ""
    top_5 = [d[0].lower() if isinstance(d, tuple) else d.lower() for d in result.differentials[:5]]

    found = expected.lower() in primary or any(expected.lower() in d for d in top_5)
    status = "[PASS]" if found else "[FAIL]"

    print(f"{status} Input: {findings[0][:60]}...")
    print(f"         Expected: {expected}")
    print(f"         Got: {result.primary_diagnosis or 'None'}")
    print()

# ============================================================================
# TEST 5: TEMPORAL/COMPARISON LANGUAGE
# ============================================================================
print("\n### TEST 5: TEMPORAL/COMPARISON LANGUAGE ###")
print("Reports compare to priors. Is this handled?")
print("-" * 50)

temporal_tests = [
    (["unchanged pulmonary nodule since 2020"], "stable - no action needed"),
    (["new consolidation in the right lower lobe"], "acute - needs attention"),
    (["interval increase in pleural effusion"], "worsening"),
    (["resolving pneumonia"], "improving"),
    (["stable cirrhotic liver morphology"], "chronic, stable"),
]

for findings, expected_interpretation in temporal_tests:
    result = srd.findings_to_diagnosis(findings)
    print(f"Input: {findings}")
    print(f"  Should interpret as: {expected_interpretation}")
    print(f"  Got: {result.primary_diagnosis or 'No diagnosis'}")
    print(f"  [MISSING] Temporal/comparison logic not implemented")
    print()

# ============================================================================
# TEST 6: COVERAGE CHECK - COMMON FINDINGS
# ============================================================================
print("\n### TEST 6: COVERAGE - COMMON CT ABDOMEN FINDINGS ###")
print("Testing vocabulary coverage for routine findings")
print("-" * 50)

common_findings = [
    "hepatic steatosis",
    "cholelithiasis",
    "renal cyst",
    "diverticulosis",
    "hiatal hernia",
    "atherosclerotic disease",
    "degenerative changes",
    "incidental adrenal nodule",
    "pancreatic atrophy",
    "prostatomegaly",
]

covered = 0
for finding in common_findings:
    result = srd.findings_to_diagnosis([finding])
    has_result = bool(result.primary_diagnosis)
    if has_result:
        covered += 1
        print(f"[OK] {finding} -> {result.primary_diagnosis}")
    else:
        print(f"[MISS] {finding} -> No mapping")

print(f"\nCoverage: {covered}/{len(common_findings)} ({100*covered/len(common_findings):.0f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF CRITICAL GAPS FOR REAL-WORLD USAGE")
print("=" * 70)
print("""
1. NO NEGATION HANDLING
   - "No pneumothorax" will still match pneumothorax
   - This is a CRITICAL flaw for report interpretation

2. NO UNCERTAINTY QUANTIFICATION
   - "Possible pneumonia" vs "definite pneumonia" treated identically
   - No way to express diagnostic confidence from hedge words

3. NO MEASUREMENT PARSING
   - "6mm nodule" doesn't trigger Fleischner guidelines
   - Measurements in findings not extracted or interpreted

4. LIMITED COMPLEX PHRASE PARSING
   - Relies on keyword matching, not semantic understanding
   - Multi-clause findings often fail

5. NO TEMPORAL REASONING
   - "Unchanged", "new", "resolving" not interpreted
   - Critical for clinical decision-making

6. VOCABULARY GAPS
   - Many common incidental findings not mapped
   - Informal/colloquial terms may fail

VERDICT: Not ready for production report interpretation without:
- NLP preprocessing for negation detection
- Uncertainty/hedge word extraction
- Measurement extraction and threshold lookup
- Much larger finding vocabulary
""")
