"""
Benchmark: Radiology Semantic Dictionary Performance
====================================================
Measures accuracy of finding-to-diagnosis mapping.

Metrics:
- Accuracy@k: Is correct diagnosis in top k results?
- MRR: Mean Reciprocal Rank (1/position of correct answer)
- Exact Match: Is correct diagnosis the #1 result?
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))
from radiology_semantic import SemanticRadDict

# =============================================================================
# TEST CASES - Clinically realistic finding combinations
# =============================================================================

TEST_CASES = [
    # =========================================================================
    # ABDOMINAL - Classic presentations
    # =========================================================================
    {
        "id": "abd_001",
        "findings": ["fat stranding", "RLQ", "dilated appendix"],
        "modality": "CT",
        "expected": "appendicitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_002",
        "findings": ["fat stranding", "right lower quadrant", "lymphadenopathy"],
        "modality": "CT",
        "expected": "appendicitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_003",
        "findings": ["periappendiceal fat stranding", "appendicolith"],
        "modality": "CT",
        "expected": "appendicitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_004",
        "findings": ["fat stranding", "LLQ", "diverticula", "wall thickening"],
        "modality": "CT",
        "expected": "diverticulitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_005",
        "findings": ["fat stranding", "left lower quadrant", "colon wall thickening"],
        "modality": "CT",
        "expected": "diverticulitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_006",
        "findings": ["gallbladder wall thickening", "pericholecystic fluid", "RUQ"],
        "modality": "CT",
        "expected": "cholecystitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_007",
        "findings": ["gallstones", "murphy sign", "gallbladder distension"],
        "modality": "US",
        "expected": "cholecystitis",
        "difficulty": "easy",
        "category": "GI"
    },
    {
        "id": "abd_008",
        "findings": ["double duct sign", "pancreatic mass"],
        "modality": "CT",
        "expected": "pancreatic adenocarcinoma",
        "difficulty": "medium",
        "category": "GI"
    },
    {
        "id": "abd_009",
        "findings": ["arterial enhancement", "washout", "capsule", "liver mass"],
        "modality": "CT",
        "expected": "hepatocellular carcinoma",
        "difficulty": "medium",
        "category": "GI"
    },
    {
        "id": "abd_010",
        "findings": ["coffee bean sign", "dilated colon"],
        "modality": "CT",
        "expected": "sigmoid volvulus",
        "difficulty": "medium",
        "category": "GI"
    },
    {
        "id": "abd_011",
        "findings": ["whirlpool sign", "mesenteric vessels"],
        "modality": "CT",
        "expected": "volvulus",
        "difficulty": "medium",
        "category": "GI"
    },
    {
        "id": "abd_012",
        "findings": ["target sign", "small bowel", "intussusception"],
        "modality": "CT",
        "expected": "intussusception",
        "difficulty": "medium",
        "category": "GI"
    },

    # =========================================================================
    # THORACIC
    # =========================================================================
    {
        "id": "tho_001",
        "findings": ["ground glass opacity", "bilateral", "peripheral"],
        "modality": "CT",
        "expected": "covid-19",
        "difficulty": "medium",
        "category": "Thoracic"
    },
    {
        "id": "tho_002",
        "findings": ["tree in bud", "centrilobular nodules"],
        "modality": "CT",
        "expected": "infection",
        "difficulty": "medium",
        "category": "Thoracic"
    },
    {
        "id": "tho_003",
        "findings": ["honeycombing", "basal predominant", "traction bronchiectasis"],
        "modality": "CT",
        "expected": "usual interstitial pneumonia",
        "difficulty": "medium",
        "category": "Thoracic"
    },
    {
        "id": "tho_004",
        "findings": ["halo sign", "nodule", "immunocompromised"],
        "modality": "CT",
        "expected": "aspergillosis",
        "difficulty": "medium",
        "category": "Thoracic"
    },
    {
        "id": "tho_005",
        "findings": ["air crescent sign", "mycetoma"],
        "modality": "CT",
        "expected": "aspergilloma",
        "difficulty": "medium",
        "category": "Thoracic"
    },

    # =========================================================================
    # NEURO
    # =========================================================================
    {
        "id": "neuro_001",
        "findings": ["ring enhancing lesion", "periventricular", "white matter"],
        "modality": "MRI",
        "expected": "multiple sclerosis",
        "difficulty": "medium",
        "category": "Neuro"
    },
    {
        "id": "neuro_002",
        "findings": ["ring enhancing lesion", "edema", "mass effect"],
        "modality": "MRI",
        "expected": "glioblastoma",
        "difficulty": "medium",
        "category": "Neuro"
    },
    {
        "id": "neuro_003",
        "findings": ["restricted diffusion", "vascular territory", "acute"],
        "modality": "MRI",
        "expected": "stroke",
        "difficulty": "easy",
        "category": "Neuro"
    },
    {
        "id": "neuro_004",
        "findings": ["cerebellopontine angle mass", "ice cream cone"],
        "modality": "MRI",
        "expected": "vestibular schwannoma",
        "difficulty": "medium",
        "category": "Neuro"
    },
    {
        "id": "neuro_005",
        "findings": ["dural tail", "extra-axial mass", "homogeneous enhancement"],
        "modality": "MRI",
        "expected": "meningioma",
        "difficulty": "medium",
        "category": "Neuro"
    },

    # =========================================================================
    # GU
    # =========================================================================
    {
        "id": "gu_001",
        "findings": ["hydronephrosis", "ureteral stone", "perinephric stranding"],
        "modality": "CT",
        "expected": "ureterolithiasis",
        "difficulty": "easy",
        "category": "GU"
    },
    {
        "id": "gu_002",
        "findings": ["renal mass", "enhancement", "clear cell"],
        "modality": "CT",
        "expected": "renal cell carcinoma",
        "difficulty": "medium",
        "category": "GU"
    },
    {
        "id": "gu_003",
        "findings": ["adrenal mass", "low attenuation", "lipid rich"],
        "modality": "CT",
        "expected": "adrenal adenoma",
        "difficulty": "medium",
        "category": "GU"
    },

    # =========================================================================
    # MSK
    # =========================================================================
    {
        "id": "msk_001",
        "findings": ["soap bubble appearance", "epiphyseal", "subarticular"],
        "modality": "XR",
        "expected": "giant cell tumor",
        "difficulty": "medium",
        "category": "MSK"
    },
    {
        "id": "msk_002",
        "findings": ["sunburst periosteal reaction", "Codman triangle"],
        "modality": "XR",
        "expected": "osteosarcoma",
        "difficulty": "medium",
        "category": "MSK"
    },
    {
        "id": "msk_003",
        "findings": ["onion skin periosteal reaction", "permeative"],
        "modality": "XR",
        "expected": "ewing sarcoma",
        "difficulty": "medium",
        "category": "MSK"
    },

    # =========================================================================
    # CHALLENGING - Natural language / fuzzy input
    # =========================================================================
    {
        "id": "fuzzy_001",
        "findings": ["inflammation around appendix", "RLQ pain"],
        "modality": "CT",
        "expected": "appendicitis",
        "difficulty": "hard",
        "category": "Fuzzy"
    },
    {
        "id": "fuzzy_002",
        "findings": ["liver lesion lights up early", "washes out"],
        "modality": "CT",
        "expected": "hepatocellular carcinoma",
        "difficulty": "hard",
        "category": "Fuzzy"
    },
    {
        "id": "fuzzy_003",
        "findings": ["twisted bowel", "swirl sign"],
        "modality": "CT",
        "expected": "volvulus",
        "difficulty": "hard",
        "category": "Fuzzy"
    },
    {
        "id": "fuzzy_004",
        "findings": ["brain tumor", "ring enhancement", "necrosis"],
        "modality": "MRI",
        "expected": "glioblastoma",
        "difficulty": "hard",
        "category": "Fuzzy"
    },
    {
        "id": "fuzzy_005",
        "findings": ["bright on T1", "melanin", "hemorrhage"],
        "modality": "MRI",
        "expected": "melanoma metastasis",
        "difficulty": "hard",
        "category": "Fuzzy"
    },
]


@dataclass
class BenchmarkResult:
    """Results from a single test case."""
    test_id: str
    expected: str
    predicted: str
    rank: int  # Position of correct answer (0 if not found)
    in_top_1: bool
    in_top_3: bool
    in_top_5: bool
    confidence: float
    difficulty: str
    category: str
    all_predictions: List[Tuple[str, float]]


def normalize_diagnosis(diagnosis: str) -> str:
    """Normalize diagnosis name for comparison."""
    if not diagnosis:
        return ""
    d = diagnosis.lower().strip()
    # Remove common suffixes/prefixes
    d = d.replace("acute ", "").replace("chronic ", "")
    d = d.replace(" disease", "").replace(" syndrome", "")
    return d


def diagnoses_match(expected: str, predicted: str) -> bool:
    """Check if two diagnoses match (fuzzy)."""
    exp = normalize_diagnosis(expected)
    pred = normalize_diagnosis(predicted)

    # Exact match
    if exp == pred:
        return True

    # One contains the other
    if exp in pred or pred in exp:
        return True

    # Key word overlap
    exp_words = set(exp.split())
    pred_words = set(pred.split())
    overlap = exp_words & pred_words
    if len(overlap) >= 1 and len(overlap) / len(exp_words) >= 0.5:
        return True

    return False


def find_rank(expected: str, differentials: List[Tuple[str, float]]) -> int:
    """Find the rank of the expected diagnosis in the differentials list."""
    for i, (diag, _) in enumerate(differentials, 1):
        if diagnoses_match(expected, diag):
            return i
    return 0  # Not found


def run_benchmark(srd: SemanticRadDict, test_cases: List[Dict]) -> List[BenchmarkResult]:
    """Run benchmark on all test cases."""
    results = []

    for tc in test_cases:
        result = srd.findings_to_diagnosis(
            findings=tc["findings"],
            modality=tc.get("modality"),
            top_k=10
        )

        predicted = result.primary_diagnosis or ""
        rank = find_rank(tc["expected"], result.differentials)

        results.append(BenchmarkResult(
            test_id=tc["id"],
            expected=tc["expected"],
            predicted=predicted,
            rank=rank,
            in_top_1=rank == 1,
            in_top_3=1 <= rank <= 3,
            in_top_5=1 <= rank <= 5,
            confidence=result.confidence,
            difficulty=tc["difficulty"],
            category=tc["category"],
            all_predictions=result.differentials[:5]
        ))

    return results


def compute_metrics(results: List[BenchmarkResult]) -> Dict:
    """Compute aggregate metrics."""
    n = len(results)
    if n == 0:
        return {}

    # Overall metrics
    accuracy_1 = sum(1 for r in results if r.in_top_1) / n
    accuracy_3 = sum(1 for r in results if r.in_top_3) / n
    accuracy_5 = sum(1 for r in results if r.in_top_5) / n

    # MRR (Mean Reciprocal Rank)
    mrr = sum(1/r.rank if r.rank > 0 else 0 for r in results) / n

    # By difficulty
    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.difficulty == diff]
        if subset:
            by_difficulty[diff] = {
                "n": len(subset),
                "accuracy@1": sum(1 for r in subset if r.in_top_1) / len(subset),
                "accuracy@3": sum(1 for r in subset if r.in_top_3) / len(subset),
                "mrr": sum(1/r.rank if r.rank > 0 else 0 for r in subset) / len(subset)
            }

    # By category
    by_category = {}
    categories = set(r.category for r in results)
    for cat in categories:
        subset = [r for r in results if r.category == cat]
        if subset:
            by_category[cat] = {
                "n": len(subset),
                "accuracy@1": sum(1 for r in subset if r.in_top_1) / len(subset),
                "accuracy@3": sum(1 for r in subset if r.in_top_3) / len(subset),
                "mrr": sum(1/r.rank if r.rank > 0 else 0 for r in subset) / len(subset)
            }

    return {
        "total_cases": n,
        "accuracy@1": accuracy_1,
        "accuracy@3": accuracy_3,
        "accuracy@5": accuracy_5,
        "mrr": mrr,
        "by_difficulty": by_difficulty,
        "by_category": by_category
    }


def print_results(results: List[BenchmarkResult], metrics: Dict):
    """Print formatted results."""
    print("=" * 70)
    print("RADIOLOGY SEMANTIC DICTIONARY BENCHMARK")
    print("=" * 70)

    print(f"\nTotal test cases: {metrics['total_cases']}")
    print(f"\n{'OVERALL METRICS':=^70}")
    print(f"  Accuracy@1 (Exact Match):  {metrics['accuracy@1']:.1%}")
    print(f"  Accuracy@3 (Top 3):        {metrics['accuracy@3']:.1%}")
    print(f"  Accuracy@5 (Top 5):        {metrics['accuracy@5']:.1%}")
    print(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.3f}")

    print(f"\n{'BY DIFFICULTY':=^70}")
    for diff, m in metrics['by_difficulty'].items():
        print(f"  {diff.upper():8} (n={m['n']:2}): Acc@1={m['accuracy@1']:.1%}, Acc@3={m['accuracy@3']:.1%}, MRR={m['mrr']:.3f}")

    print(f"\n{'BY CATEGORY':=^70}")
    for cat, m in sorted(metrics['by_category'].items()):
        print(f"  {cat:12} (n={m['n']:2}): Acc@1={m['accuracy@1']:.1%}, Acc@3={m['accuracy@3']:.1%}, MRR={m['mrr']:.3f}")

    # Show failures
    failures = [r for r in results if not r.in_top_5]
    if failures:
        print(f"\n{'FAILURES (not in top 5)':=^70}")
        for r in failures:
            print(f"  [{r.test_id}] Expected: {r.expected}")
            print(f"           Got: {r.predicted} (conf: {r.confidence:.2f})")
            print(f"           Top 5: {[d[0] for d in r.all_predictions]}")

    # Show partial matches (in top 5 but not top 1)
    partial = [r for r in results if r.in_top_5 and not r.in_top_1]
    if partial:
        print(f"\n{'PARTIAL MATCHES (rank 2-5)':=^70}")
        for r in partial[:10]:  # Show first 10
            print(f"  [{r.test_id}] Expected: {r.expected} (rank {r.rank})")
            print(f"           Got #1: {r.predicted}")


def main():
    print("Loading SemanticRadDict...")
    srd = SemanticRadDict()
    print(f"Stats: {srd.stats()}")

    print(f"\nRunning benchmark with {len(TEST_CASES)} test cases...")
    results = run_benchmark(srd, TEST_CASES)

    metrics = compute_metrics(results)
    print_results(results, metrics)

    # Save results to JSON
    output = {
        "metrics": metrics,
        "results": [
            {
                "id": r.test_id,
                "expected": r.expected,
                "predicted": r.predicted,
                "rank": r.rank,
                "in_top_1": r.in_top_1,
                "in_top_3": r.in_top_3,
                "in_top_5": r.in_top_5,
                "confidence": r.confidence,
                "difficulty": r.difficulty,
                "category": r.category,
                "predictions": r.all_predictions
            }
            for r in results
        ]
    }

    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
