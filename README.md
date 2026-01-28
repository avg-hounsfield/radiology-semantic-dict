# Radiology Semantic Dictionary

A lightweight Python package for mapping radiology imaging findings to diagnoses. No API calls, no LLM required - pure dictionary lookups with sub-millisecond response times.

## Features

- **8,300+ imaging findings** with pathology mappings
- **2,900+ medical concepts** with synonyms and definitions
- **360+ differential diagnosis groups** by finding pattern
- **340+ radiology synonym mappings** (CT -> computed tomography, etc.)
- **50+ named imaging signs** with differentials (whirlpool sign, halo sign, etc.)
- **7 ACR classification systems** (BI-RADS, LI-RADS, Lung-RADS, PI-RADS, TI-RADS, O-RADS, CAD-RADS)

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
#     'imaging_signs': 26
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
