#!/usr/bin/env python3
"""Test unseen cases to verify generalization."""

import sys
sys.path.insert(0, '.')

from radiology_semantic import SemanticRadDict

srd = SemanticRadDict()

# Completely unseen test cases (never used to tune patterns)
UNSEEN_CASES = [
    # GI variations
    (['liver mass', 'arterial enhancement', 'washout'], 'hepatocellular carcinoma'),
    (['pancreatic head mass', 'dilated bile duct'], 'pancreatic cancer'),
    (['double duct', 'pancreatic mass'], 'pancreatic adenocarcinoma'),
    (['small bowel', 'dilation', 'decompressed colon'], 'small bowel obstruction'),

    # Thoracic variations
    (['PE', 'DVT'], 'pulmonary embolism'),
    (['filling defect', 'pulmonary artery'], 'pulmonary embolism'),
    (['GGO', 'bilateral', 'peripheral'], 'covid-19'),
    (['halo sign', 'neutropenic'], 'invasive aspergillosis'),

    # Neuro variations
    (['restricted diffusion', 'MCA territory'], 'stroke'),
    (['periventricular lesions', 'corpus callosum'], 'multiple sclerosis'),
    (['dawson fingers', 'white matter lesions'], 'multiple sclerosis'),
    (['DWI hyperintensity', 'acute onset'], 'stroke'),

    # GU variations
    (['kidney mass', 'enhancing', 'heterogeneous'], 'renal cell carcinoma'),
    (['hydronephrosis', 'ureteral stone'], 'ureterolithiasis'),

    # Trauma variations
    (['splenic laceration', 'hemoperitoneum'], 'splenic injury'),

    # Gallbladder
    (['gallbladder wall necrosis', 'rim sign'], 'gangrenous cholecystitis'),
]

def test_unseen_cases():
    passed = 0
    failed = 0

    print("Testing unseen cases for generalization...\n")

    for findings, expected in UNSEEN_CASES:
        result = srd.findings_to_diagnosis(findings)

        # Check if expected is in top 3 (differentials are tuples: (name, score))
        top_3 = [d[0].lower() if isinstance(d, tuple) else d.lower() for d in result.differentials[:3]]
        expected_lower = expected.lower()

        # Also check primary
        primary = result.primary_diagnosis.lower() if result.primary_diagnosis else ''

        success = expected_lower in top_3 or expected_lower in primary or any(expected_lower in d for d in top_3)

        if success:
            passed += 1
            print(f"[PASS] {findings}")
            print(f"       Got: {result.primary_diagnosis}")
        else:
            failed += 1
            top_names = [d[0] if isinstance(d, tuple) else d for d in result.differentials[:3]]
            print(f"[FAIL] {findings}")
            print(f"       Expected: {expected}")
            print(f"       Got: {top_names}")
        print()

    total = passed + failed
    accuracy = passed / total * 100 if total > 0 else 0

    print("=" * 60)
    print(f"UNSEEN TEST RESULTS: {passed}/{total} passed ({accuracy:.1f}%)")
    print("=" * 60)

    return accuracy

if __name__ == '__main__':
    test_unseen_cases()
