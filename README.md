# Radiology Semantic Dictionary

A lightweight Python package for mapping radiology imaging findings to diagnoses. No API calls, no LLM required - pure dictionary lookups with sub-millisecond response times.

## Features

- **8,300+ imaging findings** with pathology mappings
- **2,900+ medical concepts** with synonyms and definitions
- **360+ differential diagnosis groups** by finding pattern
- **340+ radiology synonym mappings** (CT -> computed tomography, etc.)
- **69 named imaging signs** with differentials (whirlpool sign, halo sign, etc.)
- **7 ACR classification systems** (BI-RADS, LI-RADS, Lung-RADS, PI-RADS, TI-RADS, O-RADS, CAD-RADS)
- **25 measurement thresholds** (Fleischner criteria, TI-RADS FNA, organ sizes)
- **36 board-tested mnemonics** (VITAMIN D, CHICAGO, FLAMES, CRASH, etc.)
- **59 syndrome associations** with screening recommendations (VHL, TSC, NF1, etc.)

## Installation

```bash
pip install radiology-semantic-dict
```

## Quick Start

```python
from radiology_semantic import SemanticRadDict

# Initialize the dictionary
srd = SemanticRadDict()

# Map findings to diagnosis
result = srd.findings_to_diagnosis(
    findings=["fat stranding", "RLQ", "lymphadenopathy"],
    modality="CT"
)

print(result.primary_diagnosis)  # "Appendicitis"
print(result.confidence)         # 0.85
print(result.differentials)      # [("Appendicitis", 0.85), ("Crohn disease", 0.45), ...]
print(result.pathognomonic_findings)  # ["periappendiceal fat stranding"]
print(result.suggested_lookfor)  # ["appendicolith", "target sign"]
```

## API Reference

### SemanticRadDict

The main class providing all lookup functionality.

#### `findings_to_diagnosis(findings, modality=None, body_region=None, top_k=5)`

Map a list of findings to likely diagnoses.

```python
result = srd.findings_to_diagnosis(
    findings=["ring-enhancing lesion", "periventricular"],
    modality="MRI",
    body_region="brain"
)
```

**Returns:** `DiagnosisResult` with:
- `primary_diagnosis`: Most likely diagnosis
- `confidence`: Confidence score (0-1)
- `differentials`: List of (diagnosis, confidence) tuples
- `matching_findings`: List of Finding objects that matched
- `pathognomonic_findings`: List of pathognomonic finding names
- `suggested_lookfor`: Additional findings to look for

#### `get_findings_for_pathology(pathology, modality=None)`

Get all known imaging findings for a disease/condition.

```python
findings = srd.get_findings_for_pathology("appendicitis", modality="CT")
for f in findings:
    print(f"{f.name} - Pathognomonic: {f.is_pathognomonic}")
```

#### `get_differential(finding_pattern, modality=None, body_region=None)`

Get differential diagnoses for a finding pattern.

```python
ddx = srd.get_differential("ground glass opacity", modality="CT")
for d in ddx:
    print(f"{d.presentation}: {d.differentials}")
```

#### `get_imaging_sign(sign_name)`

Look up a named imaging sign.

```python
sign = srd.get_imaging_sign("whirlpool sign")
print(sign['indicates'])      # "Volvulus (sigmoid, cecal) or midgut malrotation"
print(sign['differential'])   # ["Sigmoid volvulus", "Cecal volvulus", ...]
print(sign['board_relevance']) # "HIGH"
```

#### `expand_synonyms(term)`

Get synonyms for a radiology term.

```python
synonyms = srd.expand_synonyms("CT")
# ["computed tomography", "cat scan", "ct scan"]

synonyms = srd.expand_synonyms("hemorrhage")
# ["bleeding", "hematoma", "blood", "haemorrhage"]
```

#### `get_classification(system, category=None)`

Get ACR classification system information.

```python
# Get all BI-RADS categories
birads = srd.get_classification("BI-RADS")

# Get specific category
birads_4 = srd.get_classification("BI-RADS", category="4")
print(birads_4[0]['malignancy_risk'])  # "2-95%"
print(birads_4[0]['management'])       # "Tissue diagnosis recommended"
```

#### `get_measurement_threshold(query, modality=None, body_region=None)`

Look up radiologic measurement thresholds.

```python
thresholds = srd.get_measurement_threshold("spleen")
for t in thresholds:
    print(f"{t.name}: {t.threshold_operator}{t.threshold_value}{t.unit}")
# Spleen length (splenomegaly threshold): >12cm
# Spleen length (massive splenomegaly): >20cm

# Filter by modality
thyroid = srd.get_measurement_threshold("thyroid", modality="US")
for t in thyroid:
    print(f"{t.name}: {t.clinical_significance}")
```

#### `get_mnemonic(mnemonic_name)`

Look up a specific mnemonic.

```python
m = srd.get_mnemonic("CHICAGO")
print(m.expansion)
# 'Crohn, Hernia, Intussusception, Cancer, Adhesions, Gallstone ileus, Obturation'
print(m.topic)
# 'Small bowel obstruction causes'
```

#### `search_mnemonics(query, category=None, body_region=None)`

Search mnemonics by topic or keyword.

```python
# Find all neuro mnemonics
neuro = srd.search_mnemonics("", body_region="Neuro")
for m in neuro:
    print(f"{m.mnemonic}: {m.topic}")

# Find mnemonics about T1 signal
t1_mnemonics = srd.search_mnemonics("T1")
```

#### `get_syndrome_associations(syndrome)`

Get imaging findings associated with a syndrome.

```python
vhl = srd.get_syndrome_associations("VHL")
for a in vhl:
    print(f"{a.associated_finding}: {a.frequency}")
# Hemangioblastoma (CNS): 60-80%
# Renal cell carcinoma (clear cell): 25-45%
# Pheochromocytoma: 10-20%
```

#### `search_syndromes_by_finding(finding)`

Find syndromes associated with a specific imaging finding.

```python
syndromes = srd.search_syndromes_by_finding("cardiac myxoma")
for s in syndromes:
    print(f"Consider: {s.syndrome_name}")
# Consider: Carney Complex
```

#### `get_screening_recommendations(syndrome)`

Get screening recommendations for a genetic syndrome.

```python
recs = srd.get_screening_recommendations("TSC")
for r in recs:
    print(f"{r['finding']}: {r['recommendation']}")
# SEGA: MRI every 1-3 years until age 25
# Renal angiomyolipoma: MRI abdomen every 1-3 years
```

#### `stats()`

Get statistics about loaded data.

```python
print(srd.stats())
# {
#     'imaging_findings': 8308,
#     'medical_concepts': 2946,
#     'differential_groups': 361,
#     'classification_systems': 41,
#     'synonym_mappings': 342,
#     'imaging_signs': 69,
#     'measurement_thresholds': 25,
#     'mnemonics': 36,
#     'syndrome_associations': 59
# }
```

## Use Cases

### Clinical Decision Support

```python
# Real-time finding interpretation
result = srd.findings_to_diagnosis(
    ["hepatic lesion", "arterial enhancement", "washout"],
    modality="CT"
)
if "Hepatocellular carcinoma" in result.primary_diagnosis:
    print("Consider LI-RADS classification")
    lirads = srd.get_classification("LI-RADS")
```

### Educational Applications

```python
# Board exam preparation
sign = srd.get_imaging_sign("Hampton hump")
print(f"Sign: {sign['name']}")
print(f"Indicates: {sign['indicates']}")
print(f"Board relevance: {sign['board_relevance']}")
```

### NLP Pipeline Enhancement

```python
# Expand radiology terms for better text matching
terms = ["CT", "MRI", "tumor"]
expanded = []
for term in terms:
    expanded.extend(srd.expand_synonyms(term))
# Now use expanded terms for text search/matching
```

### Structured Reporting

```python
# Auto-suggest differentials based on findings
findings = extract_findings_from_report(report_text)  # Your NLP
ddx = srd.findings_to_diagnosis(findings, modality="CT")
print(f"Consider: {', '.join([d[0] for d in ddx.differentials[:3]])}")
```

### Measurement Decision Support

```python
# Check if a measurement triggers action
spleen_length = 14.5  # cm
thresholds = srd.get_measurement_threshold("spleen")
for t in thresholds:
    if t.threshold_operator == ">" and spleen_length > t.threshold_value:
        print(f"ALERT: {t.clinical_significance}")
        print(f"Action: {t.action_if_met}")
```

### Syndrome Workup Assistant

```python
# Patient has a hemangioblastoma - check for syndromes
finding = "hemangioblastoma"
syndromes = srd.search_syndromes_by_finding(finding)
for s in syndromes:
    print(f"Consider {s.syndrome_name}")
    recs = srd.get_screening_recommendations(s.syndrome_name.split()[0])
    for r in recs:
        print(f"  - Screen for: {r['finding']}")
```

### Board Exam Study Tool

```python
# Quiz yourself on mnemonics
import random
all_mnemonics = srd.search_mnemonics("")
m = random.choice(all_mnemonics)
print(f"What does {m.mnemonic} stand for?")
# User answers...
print(f"Answer: {m.expansion}")
print(f"Topic: {m.topic}")
```

## Data Sources

This dictionary was curated from:
- RadLex (NIH Radiology Lexicon) - 12,000+ terms
- ACR Appropriateness Criteria
- Standard radiology textbooks and references
- Board examination resources

All data has been sanitized to remove citations, page numbers, and source artifacts.

## Performance

- **Initialization:** ~100ms (one-time load)
- **Lookups:** <1ms per query
- **Memory:** ~50MB for full dictionary

## License

MIT License - Free for commercial and non-commercial use.

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
