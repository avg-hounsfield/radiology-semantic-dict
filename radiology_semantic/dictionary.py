"""
Radiology Semantic Dictionary - Core API
=========================================
Provides semantic mapping between radiology findings and diagnoses.

Enhanced with:
- Anatomical location weighting (RLQ -> appendix, RUQ -> gallbladder)
- Concept co-occurrence boosting
- Modality-specific matching
- 100+ named imaging signs
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict


# =============================================================================
# ANATOMICAL LOCATION MAPPINGS
# =============================================================================
ANATOMICAL_LOCATIONS = {
    # Abdominal quadrants -> associated organs/pathologies
    'rlq': {
        'full_name': 'right lower quadrant',
        'organs': ['appendix', 'cecum', 'terminal ileum', 'right ovary', 'right ureter'],
        'pathologies': ['appendicitis', 'cecal volvulus', 'crohn disease', 'ovarian torsion', 'ureteral stone'],
        'boost': 3.0
    },
    'right lower quadrant': {
        'aliases': ['rlq'],
        'organs': ['appendix', 'cecum', 'terminal ileum', 'right ovary', 'right ureter'],
        'pathologies': ['appendicitis', 'cecal volvulus', 'crohn disease', 'ovarian torsion', 'ureteral stone'],
        'boost': 3.0
    },
    'ruq': {
        'full_name': 'right upper quadrant',
        'organs': ['liver', 'gallbladder', 'right kidney', 'hepatic flexure', 'duodenum'],
        'pathologies': ['cholecystitis', 'cholelithiasis', 'hepatitis', 'liver abscess', 'hepatocellular carcinoma'],
        'boost': 3.0
    },
    'right upper quadrant': {
        'aliases': ['ruq'],
        'organs': ['liver', 'gallbladder', 'right kidney', 'hepatic flexure', 'duodenum'],
        'pathologies': ['cholecystitis', 'cholelithiasis', 'hepatitis', 'liver abscess', 'hepatocellular carcinoma'],
        'boost': 3.0
    },
    'llq': {
        'full_name': 'left lower quadrant',
        'organs': ['sigmoid colon', 'left ovary', 'left ureter', 'descending colon'],
        'pathologies': ['diverticulitis', 'sigmoid volvulus', 'ovarian torsion', 'ureteral stone'],
        'boost': 3.0
    },
    'left lower quadrant': {
        'aliases': ['llq'],
        'organs': ['sigmoid colon', 'left ovary', 'left ureter', 'descending colon'],
        'pathologies': ['diverticulitis', 'sigmoid volvulus', 'ovarian torsion', 'ureteral stone'],
        'boost': 3.0
    },
    'luq': {
        'full_name': 'left upper quadrant',
        'organs': ['spleen', 'stomach', 'left kidney', 'splenic flexure', 'pancreatic tail'],
        'pathologies': ['splenic laceration', 'gastric ulcer', 'pancreatitis'],
        'boost': 3.0
    },
    'left upper quadrant': {
        'aliases': ['luq'],
        'organs': ['spleen', 'stomach', 'left kidney', 'splenic flexure', 'pancreatic tail'],
        'pathologies': ['splenic laceration', 'gastric ulcer', 'pancreatitis'],
        'boost': 3.0
    },
    'epigastric': {
        'organs': ['stomach', 'duodenum', 'pancreas', 'aorta'],
        'pathologies': ['pancreatitis', 'peptic ulcer', 'aortic aneurysm', 'gastric cancer'],
        'boost': 2.5
    },
    'periumbilical': {
        'organs': ['small bowel', 'aorta', 'mesenteric vessels'],
        'pathologies': ['small bowel obstruction', 'mesenteric ischemia', 'aortic aneurysm'],
        'boost': 2.0
    },
    'suprapubic': {
        'organs': ['bladder', 'uterus', 'prostate'],
        'pathologies': ['cystitis', 'bladder cancer', 'uterine fibroid'],
        'boost': 2.0
    },
    'flank': {
        'organs': ['kidney', 'ureter', 'retroperitoneum'],
        'pathologies': ['pyelonephritis', 'renal stone', 'renal cell carcinoma'],
        'boost': 2.5
    },
    # Neuroanatomical locations
    'periventricular': {
        'organs': ['lateral ventricles', 'white matter'],
        'pathologies': ['multiple sclerosis', 'pvl', 'lymphoma', 'metastasis'],
        'boost': 2.5
    },
    'basal ganglia': {
        'organs': ['caudate', 'putamen', 'globus pallidus'],
        'pathologies': ['hypertensive hemorrhage', 'carbon monoxide poisoning', 'wilson disease'],
        'boost': 2.5
    },
    'cerebellopontine angle': {
        'aliases': ['cpa'],
        'organs': ['cerebellum', 'pons', 'cn vii', 'cn viii'],
        'pathologies': ['vestibular schwannoma', 'meningioma', 'epidermoid'],
        'boost': 3.0
    },
}

# =============================================================================
# FINDING CO-OCCURRENCE PATTERNS
# =============================================================================
FINDING_COOCCURRENCE = {
    # If you see these findings together, boost this diagnosis
    ('fat stranding', 'rlq'): {'appendicitis': 5.0, 'crohn disease': 2.0},
    ('fat stranding', 'right lower quadrant'): {'appendicitis': 5.0, 'crohn disease': 2.0},
    ('fat stranding', 'llq'): {'diverticulitis': 5.0, 'colitis': 2.0},
    ('fat stranding', 'left lower quadrant'): {'diverticulitis': 5.0, 'colitis': 2.0},
    ('fat stranding', 'ruq'): {'cholecystitis': 4.0, 'pancreatitis': 2.0},
    ('fat stranding', 'appendix'): {'appendicitis': 6.0},
    ('fat stranding', 'periappendiceal'): {'appendicitis': 6.0},
    ('appendicolith',): {'appendicitis': 5.0},
    ('target sign', 'rlq'): {'appendicitis': 4.0, 'crohn disease': 3.0},
    ('wall thickening', 'gallbladder'): {'cholecystitis': 4.0},
    ('pericholecystic fluid',): {'cholecystitis': 5.0},
    ('murphy sign',): {'cholecystitis': 4.0},
    ('double duct sign',): {'pancreatic adenocarcinoma': 5.0, 'ampullary carcinoma': 3.0},
    ('ring enhancing', 'brain'): {'abscess': 3.0, 'glioblastoma': 3.0, 'metastasis': 3.0},
    ('ring enhancing', 'periventricular'): {'lymphoma': 4.0, 'toxoplasmosis': 3.0},
    ('arterial enhancement', 'washout'): {'hepatocellular carcinoma': 5.0},
    ('arterial enhancement', 'washout', 'capsule'): {'hepatocellular carcinoma': 6.0},
    ('ground glass', 'bilateral', 'peripheral'): {'covid-19': 4.0, 'organizing pneumonia': 3.0},
    ('tree in bud',): {'infection': 4.0, 'tuberculosis': 3.0, 'aspiration': 3.0},
    ('honeycomb', 'basal'): {'usual interstitial pneumonia': 5.0, 'ipf': 5.0},
    ('whirlpool sign',): {'volvulus': 5.0, 'sigmoid volvulus': 4.0, 'ovarian torsion': 4.0},
    ('coffee bean sign',): {'sigmoid volvulus': 5.0},
    ('birds beak',): {'sigmoid volvulus': 4.0, 'achalasia': 4.0},
}


@dataclass
class Finding:
    """Represents an imaging finding."""
    name: str
    description: Optional[str] = None
    modality: Optional[str] = None
    pathology_name: Optional[str] = None
    is_pathognomonic: bool = False
    high_yield: bool = False
    acr_topic: Optional[str] = None


@dataclass
class MeasurementThreshold:
    """Represents a radiologic measurement threshold."""
    name: str
    threshold_value: float
    threshold_operator: str
    unit: str
    clinical_significance: str
    finding_type: Optional[str] = None
    measurement_type: Optional[str] = None
    action_if_met: Optional[str] = None
    action_if_not_met: Optional[str] = None
    modality: Optional[str] = None
    body_region: Optional[str] = None
    high_yield: bool = False


@dataclass
class Mnemonic:
    """Represents a radiology mnemonic."""
    mnemonic: str
    expansion: str
    topic: str
    category: Optional[str] = None
    body_region: Optional[str] = None
    high_yield: bool = False
    explanation: Optional[str] = None


@dataclass
class SyndromeAssociation:
    """Represents a syndrome-imaging association."""
    syndrome_name: str
    associated_finding: str
    finding_type: Optional[str] = None
    body_region: Optional[str] = None
    frequency: Optional[str] = None
    clinical_notes: Optional[str] = None
    screening_recommendation: Optional[str] = None
    high_yield: bool = False


@dataclass
class DifferentialGroup:
    """Represents a differential diagnosis group for a finding pattern."""
    presentation: str
    differentials: List[str]
    modality: Optional[str] = None
    body_region: Optional[str] = None
    clinical_context: Optional[str] = None


@dataclass
class DiagnosisResult:
    """Result from findings-to-diagnosis lookup."""
    primary_diagnosis: Optional[str] = None
    confidence: float = 0.0
    differentials: List[Tuple[str, float]] = field(default_factory=list)
    matching_findings: List[Finding] = field(default_factory=list)
    pathognomonic_findings: List[str] = field(default_factory=list)
    suggested_lookfor: List[str] = field(default_factory=list)
    anatomical_match: Optional[str] = None
    reasoning: List[str] = field(default_factory=list)


class SemanticRadDict:
    """
    Radiology Semantic Dictionary for findings-to-diagnosis mapping.

    Enhanced with anatomical location weighting and co-occurrence boosting.

    Example:
        >>> srd = SemanticRadDict()
        >>> result = srd.findings_to_diagnosis(["fat stranding", "RLQ"], modality="CT")
        >>> print(result.primary_diagnosis)
        'Appendicitis'
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the semantic dictionary."""
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)

        # Load all data files
        self._imaging_findings: List[Dict] = []
        self._medical_concepts: List[Dict] = []
        self._differential_groups: List[Dict] = []
        self._classification_systems: List[Dict] = []
        self._synonyms: Dict[str, List[str]] = {}
        self._imaging_signs: Dict[str, Dict] = {}
        self._measurement_thresholds: List[Dict] = []
        self._mnemonics: List[Dict] = []
        self._syndrome_associations: List[Dict] = []

        self._load_data()
        self._build_indices()

    def _load_json(self, filename: str) -> List[Dict]:
        """Load a JSON data file."""
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _load_data(self):
        """Load all data files."""
        self._imaging_findings = self._load_json("imaging_findings.json")
        self._medical_concepts = self._load_json("medical_concepts.json")
        self._differential_groups = self._load_json("differential_groups.json")
        self._classification_systems = self._load_json("classification_systems.json")
        self._synonyms = self._load_json("radiology_synonyms.json")

        # Load imaging signs
        signs_data = self._load_json("imaging_signs.json")
        if isinstance(signs_data, dict):
            self._imaging_signs = signs_data
        elif isinstance(signs_data, list):
            self._imaging_signs = {s.get('key', s.get('name', '')): s for s in signs_data}

        # Load expanded data (Phase 1 & 2)
        self._measurement_thresholds = self._load_json("measurement_thresholds.json")
        self._mnemonics = self._load_json("mnemonics.json")
        self._syndrome_associations = self._load_json("syndrome_associations.json")

    def _build_indices(self):
        """Build lookup indices for fast querying."""
        # Index findings by pathology name (normalized)
        self._findings_by_pathology: Dict[str, List[Dict]] = defaultdict(list)
        for f in self._imaging_findings:
            pathology = self._normalize(f.get('pathology_name', ''))
            if pathology:
                self._findings_by_pathology[pathology].append(f)
                # Also index by key words
                for word in pathology.split():
                    if len(word) > 3:
                        self._findings_by_pathology[word].append(f)

        # Index findings by name (normalized)
        self._findings_by_name: Dict[str, List[Dict]] = defaultdict(list)
        for f in self._imaging_findings:
            name = self._normalize(f.get('name', ''))
            if name:
                self._findings_by_name[name].append(f)
                # Also index individual words
                for word in name.split():
                    if len(word) > 3:
                        self._findings_by_name[word].append(f)

        # Index differentials by presentation
        self._ddx_by_presentation: Dict[str, List[Dict]] = defaultdict(list)
        for d in self._differential_groups:
            pres = self._normalize(d.get('presentation', ''))
            if pres:
                self._ddx_by_presentation[pres].append(d)
                for word in pres.split():
                    if len(word) > 3:
                        self._ddx_by_presentation[word].append(d)

        # Build reverse synonym lookup
        self._reverse_synonyms: Dict[str, str] = {}
        for canonical, syns in self._synonyms.items():
            for syn in syns:
                self._reverse_synonyms[syn.lower()] = canonical

        # Index concepts by name
        self._concepts_by_name: Dict[str, Dict] = {}
        for c in self._medical_concepts:
            name = c.get('name', '').lower()
            if name:
                self._concepts_by_name[name] = c
                for syn in c.get('synonyms', []):
                    self._concepts_by_name[syn.lower()] = c

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _expand_terms(self, terms: List[str]) -> Set[str]:
        """Expand terms using synonym dictionary."""
        expanded = set()
        for term in terms:
            term_lower = term.lower()
            expanded.add(term_lower)

            # Add synonyms if term is canonical
            if term_lower in self._synonyms:
                expanded.update(s.lower() for s in self._synonyms[term_lower])

            # Add canonical if term is a synonym
            if term_lower in self._reverse_synonyms:
                canonical = self._reverse_synonyms[term_lower]
                expanded.add(canonical.lower())
                if canonical.lower() in self._synonyms:
                    expanded.update(s.lower() for s in self._synonyms[canonical.lower()])

        return expanded

    def _detect_anatomical_location(self, findings: List[str]) -> Optional[Dict]:
        """Detect anatomical location from findings and return location info."""
        findings_lower = ' '.join(f.lower() for f in findings)

        for loc_key, loc_info in ANATOMICAL_LOCATIONS.items():
            if loc_key in findings_lower:
                return {'key': loc_key, **loc_info}

            # Check aliases
            aliases = loc_info.get('aliases', [])
            for alias in aliases:
                if alias in findings_lower:
                    return {'key': loc_key, **loc_info}

        return None

    def _get_cooccurrence_boost(self, findings: List[str]) -> Dict[str, float]:
        """Get diagnosis boosts based on finding co-occurrence patterns."""
        boosts: Dict[str, float] = defaultdict(float)
        findings_lower = [f.lower() for f in findings]
        findings_text = ' '.join(findings_lower)

        for pattern, diagnoses in FINDING_COOCCURRENCE.items():
            # Check if all parts of the pattern are present
            match = True
            for part in pattern:
                if part.lower() not in findings_text:
                    match = False
                    break

            if match:
                for diagnosis, boost in diagnoses.items():
                    boosts[diagnosis.lower()] += boost

        return dict(boosts)

    def findings_to_diagnosis(
        self,
        findings: List[str],
        modality: Optional[str] = None,
        body_region: Optional[str] = None,
        top_k: int = 5
    ) -> DiagnosisResult:
        """
        Map a list of findings to likely diagnoses.

        Uses anatomical location weighting and co-occurrence boosting.

        Args:
            findings: List of finding descriptions
            modality: Optional imaging modality filter
            body_region: Optional body region filter
            top_k: Number of top differentials to return

        Returns:
            DiagnosisResult with primary diagnosis, confidence, and reasoning
        """
        result = DiagnosisResult()
        if not findings:
            return result

        # Normalize and expand input terms
        normalized_findings = [self._normalize(f) for f in findings]
        expanded_terms = self._expand_terms(findings)

        # Detect anatomical location
        location = self._detect_anatomical_location(findings)
        if location:
            result.anatomical_match = location.get('key', '')
            result.reasoning.append(f"Anatomical location: {location.get('key', '').upper()}")

        # Get co-occurrence boosts
        cooccurrence_boosts = self._get_cooccurrence_boost(findings)
        if cooccurrence_boosts:
            top_boost = max(cooccurrence_boosts.items(), key=lambda x: x[1])
            result.reasoning.append(f"Pattern match: {top_boost[0]} (boost: +{top_boost[1]:.1f})")

        # Score pathologies
        pathology_scores: Dict[str, float] = defaultdict(float)
        pathology_matches: Dict[str, List[Dict]] = defaultdict(list)
        pathology_pathognomonic: Dict[str, List[str]] = defaultdict(list)

        # Apply co-occurrence boosts first
        for diagnosis, boost in cooccurrence_boosts.items():
            pathology_scores[diagnosis] += boost

        # Apply anatomical location boosts
        if location:
            for pathology in location.get('pathologies', []):
                pathology_scores[pathology.lower()] += location.get('boost', 2.0)

        # Search findings index
        for term in expanded_terms:
            for finding in self._findings_by_name.get(term, []):
                pathology = finding.get('pathology_name', '')
                if not pathology:
                    continue

                # Check modality filter
                if modality and finding.get('modality'):
                    if modality.upper() not in finding['modality'].upper():
                        continue

                # Score based on finding properties
                score = 1.0
                if finding.get('is_pathognomonic'):
                    score = 5.0
                    pathology_pathognomonic[pathology].append(finding['name'])
                elif finding.get('high_yield'):
                    score = 2.0

                # Boost if anatomical location matches
                if location:
                    pathology_lower = pathology.lower()
                    if pathology_lower in [p.lower() for p in location.get('pathologies', [])]:
                        score *= 1.5

                pathology_scores[pathology] += score
                if finding not in pathology_matches[pathology]:
                    pathology_matches[pathology].append(finding)

        # Check differential groups
        for term in expanded_terms:
            for ddx in self._ddx_by_presentation.get(term, []):
                if modality and ddx.get('modality'):
                    if modality.upper() not in ddx['modality'].upper():
                        continue

                if body_region and ddx.get('body_region'):
                    if body_region.lower() not in ddx['body_region'].lower():
                        continue

                differentials = ddx.get('differentials', [])
                for i, diff in enumerate(differentials):
                    if isinstance(diff, dict):
                        diff_name = diff.get('name', '')
                        likelihood = diff.get('likelihood', 'uncommon')
                        score = {'common': 2.0, 'uncommon': 1.0, 'rare': 0.5}.get(likelihood, 1.0)
                    else:
                        diff_name = str(diff)
                        score = max(0.5, 2.0 - (i * 0.3))

                    if diff_name:
                        pathology_scores[diff_name] += score

        # Check imaging signs for additional matches
        for term in expanded_terms:
            sign = self.get_imaging_sign(term)
            if sign:
                differentials = sign.get('differential', [])
                for i, diff in enumerate(differentials):
                    score = max(1.0, 4.0 - (i * 0.5))
                    pathology_scores[diff] += score
                result.reasoning.append(f"Imaging sign matched: {sign.get('name', term)}")

        # Sort by score
        sorted_pathologies = sorted(
            pathology_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        if sorted_pathologies:
            max_score = sorted_pathologies[0][1]
            result.primary_diagnosis = sorted_pathologies[0][0]
            result.confidence = min(0.95, sorted_pathologies[0][1] / (max_score + 3))

            result.differentials = [
                (name, min(0.95, score / (max_score + 3)))
                for name, score in sorted_pathologies
            ]

            # Add matching findings
            if result.primary_diagnosis:
                primary_key = result.primary_diagnosis
                result.matching_findings = [
                    Finding(
                        name=f.get('name', ''),
                        description=f.get('description'),
                        modality=f.get('modality'),
                        pathology_name=f.get('pathology_name'),
                        is_pathognomonic=f.get('is_pathognomonic', False),
                        high_yield=f.get('high_yield', False)
                    )
                    for f in pathology_matches.get(primary_key, [])
                ]

                result.pathognomonic_findings = pathology_pathognomonic.get(primary_key, [])

                # Get "look for" suggestions
                primary_lower = result.primary_diagnosis.lower()
                all_findings = self._findings_by_pathology.get(primary_lower, [])
                matched_names = {f.name.lower() for f in result.matching_findings}
                result.suggested_lookfor = [
                    f.get('name', '')
                    for f in all_findings
                    if f.get('name', '').lower() not in matched_names
                    and (f.get('is_pathognomonic') or f.get('high_yield'))
                ][:5]

        return result

    def get_findings_for_pathology(
        self,
        pathology: str,
        modality: Optional[str] = None
    ) -> List[Finding]:
        """Get all known imaging findings for a pathology."""
        pathology_lower = pathology.lower()
        findings = self._findings_by_pathology.get(pathology_lower, [])

        if not findings:
            expanded = self._expand_terms([pathology])
            for term in expanded:
                findings.extend(self._findings_by_pathology.get(term, []))

        if modality:
            findings = [f for f in findings
                       if not f.get('modality') or modality.upper() in f['modality'].upper()]

        return [
            Finding(
                name=f.get('name', ''),
                description=f.get('description'),
                modality=f.get('modality'),
                pathology_name=f.get('pathology_name'),
                is_pathognomonic=f.get('is_pathognomonic', False),
                high_yield=f.get('high_yield', False),
                acr_topic=f.get('acr_topic')
            )
            for f in findings
        ]

    def get_differential(
        self,
        finding_pattern: str,
        modality: Optional[str] = None,
        body_region: Optional[str] = None
    ) -> List[DifferentialGroup]:
        """Get differential diagnoses for a finding pattern."""
        normalized = self._normalize(finding_pattern)
        results = []

        for ddx in self._ddx_by_presentation.get(normalized, []):
            if modality and ddx.get('modality'):
                if modality.upper() not in ddx['modality'].upper():
                    continue
            if body_region and ddx.get('body_region'):
                if body_region.lower() not in ddx['body_region'].lower():
                    continue

            differentials = []
            for d in ddx.get('differentials', []):
                if isinstance(d, dict):
                    differentials.append(d.get('name', ''))
                else:
                    differentials.append(str(d))

            results.append(DifferentialGroup(
                presentation=ddx.get('presentation', ''),
                differentials=differentials,
                modality=ddx.get('modality'),
                body_region=ddx.get('body_region'),
                clinical_context=ddx.get('clinical_context')
            ))

        if not results:
            for word in normalized.split():
                if len(word) > 3:
                    for ddx in self._ddx_by_presentation.get(word, []):
                        differentials = []
                        for d in ddx.get('differentials', []):
                            if isinstance(d, dict):
                                differentials.append(d.get('name', ''))
                            else:
                                differentials.append(str(d))

                        results.append(DifferentialGroup(
                            presentation=ddx.get('presentation', ''),
                            differentials=differentials,
                            modality=ddx.get('modality'),
                            body_region=ddx.get('body_region'),
                            clinical_context=ddx.get('clinical_context')
                        ))

        return results

    def get_imaging_sign(self, sign_name: str) -> Optional[Dict]:
        """Look up a named imaging sign."""
        normalized = self._normalize(sign_name)
        normalized_key = normalized.replace(' ', '_')

        if normalized_key in self._imaging_signs:
            return self._imaging_signs[normalized_key]

        for key, sign in self._imaging_signs.items():
            if normalized in self._normalize(sign.get('name', '')):
                return sign
            if normalized in self._normalize(key):
                return sign

        return None

    def search_imaging_signs(
        self,
        query: str,
        modality: Optional[str] = None,
        anatomy: Optional[str] = None
    ) -> List[Dict]:
        """Search imaging signs by keyword, modality, or anatomy."""
        results = []
        query_lower = query.lower()

        for key, sign in self._imaging_signs.items():
            score = 0

            # Check name match
            if query_lower in sign.get('name', '').lower():
                score += 3
            if query_lower in key.lower():
                score += 2

            # Check indicates/description
            if query_lower in sign.get('indicates', '').lower():
                score += 2
            if query_lower in sign.get('description', '').lower():
                score += 1

            # Check differential
            for diff in sign.get('differential', []):
                if query_lower in diff.lower():
                    score += 1

            # Modality filter
            if modality:
                sign_modalities = sign.get('modality', [])
                if isinstance(sign_modalities, str):
                    sign_modalities = [sign_modalities]
                if not any(modality.upper() in m.upper() for m in sign_modalities):
                    continue

            # Anatomy filter
            if anatomy:
                if anatomy.lower() not in sign.get('anatomy', '').lower():
                    continue

            if score > 0:
                results.append({'sign': sign, 'score': score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['sign'] for r in results]

    def expand_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a radiology term."""
        term_lower = term.lower()

        if term_lower in self._synonyms:
            return self._synonyms[term_lower]

        if term_lower in self._reverse_synonyms:
            canonical = self._reverse_synonyms[term_lower]
            return [canonical] + self._synonyms.get(canonical.lower(), [])

        return []

    def get_concept(self, term: str) -> Optional[Dict]:
        """Look up a medical concept."""
        return self._concepts_by_name.get(term.lower())

    def get_classification(self, system: str, category: Optional[str] = None) -> List[Dict]:
        """Get classification system information."""
        results = []
        system_upper = system.upper().replace('-', '')

        for entry in self._classification_systems:
            entry_name = entry.get('name', '').upper().replace('-', '')
            if system_upper in entry_name:
                if category is not None:
                    entry_cat = str(entry.get('category', ''))
                    if entry_cat == str(category) or entry_cat.startswith(str(category)):
                        results.append(entry)
                else:
                    results.append(entry)

        return sorted(results, key=lambda x: str(x.get('category', '')))

    # =========================================================================
    # MEASUREMENT THRESHOLDS (Phase 1)
    # =========================================================================

    def get_measurement_threshold(
        self,
        query: str,
        modality: Optional[str] = None,
        body_region: Optional[str] = None
    ) -> List[MeasurementThreshold]:
        """
        Look up radiologic measurement thresholds.

        Args:
            query: Search term (e.g., "spleen", "thyroid nodule", "AAA")
            modality: Optional modality filter (CT, US, MRI)
            body_region: Optional body region filter

        Returns:
            List of matching MeasurementThreshold objects

        Example:
            >>> thresholds = srd.get_measurement_threshold("spleen")
            >>> for t in thresholds:
            ...     print(f"{t.name}: {t.threshold_operator}{t.threshold_value}{t.unit}")
            Spleen length (splenomegaly threshold): >12cm
        """
        query_lower = query.lower()
        results = []

        for m in self._measurement_thresholds:
            score = 0
            name = m.get('name', '').lower()
            significance = m.get('clinical_significance', '').lower()
            finding_type = m.get('finding_type', '').lower()

            # Name matching
            if query_lower in name:
                score += 3
            for word in query_lower.split():
                if word in name:
                    score += 1
                if word in significance:
                    score += 0.5

            # Modality filter
            if modality:
                m_modality = m.get('modality', '')
                if m_modality and modality.upper() not in m_modality.upper():
                    continue

            # Body region filter
            if body_region:
                m_region = m.get('body_region', '')
                if m_region and body_region.lower() not in m_region.lower():
                    continue

            if score > 0:
                results.append((score, m))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        return [
            MeasurementThreshold(
                name=m.get('name', ''),
                threshold_value=m.get('threshold_value', 0),
                threshold_operator=m.get('threshold_operator', ''),
                unit=m.get('unit', ''),
                clinical_significance=m.get('clinical_significance', ''),
                finding_type=m.get('finding_type'),
                measurement_type=m.get('measurement_type'),
                action_if_met=m.get('action_if_met'),
                action_if_not_met=m.get('action_if_not_met'),
                modality=m.get('modality'),
                body_region=m.get('body_region'),
                high_yield=m.get('high_yield', False)
            )
            for _, m in results
        ]

    def list_measurement_categories(self) -> Dict[str, List[str]]:
        """
        List all measurement categories with their thresholds.

        Returns:
            Dict mapping body_region to list of measurement names
        """
        categories: Dict[str, List[str]] = defaultdict(list)
        for m in self._measurement_thresholds:
            region = m.get('body_region', 'other')
            categories[region].append(m.get('name', ''))
        return dict(categories)

    # =========================================================================
    # MNEMONICS DATABASE (Phase 2)
    # =========================================================================

    def get_mnemonic(self, mnemonic: str) -> Optional[Mnemonic]:
        """
        Look up a specific mnemonic by name.

        Args:
            mnemonic: The mnemonic abbreviation (e.g., "VITAMIN D", "CHICAGO")

        Returns:
            Mnemonic object if found, None otherwise

        Example:
            >>> m = srd.get_mnemonic("CHICAGO")
            >>> print(m.expansion)
            'Crohn, Hernia, Intussusception, Cancer, Adhesions, Gallstone ileus, Obturation'
        """
        mnemonic_upper = mnemonic.upper()

        for m in self._mnemonics:
            if m.get('mnemonic', '').upper() == mnemonic_upper:
                return Mnemonic(
                    mnemonic=m.get('mnemonic', ''),
                    expansion=m.get('expansion', ''),
                    topic=m.get('topic', ''),
                    category=m.get('category'),
                    body_region=m.get('body_region'),
                    high_yield=m.get('high_yield', False),
                    explanation=m.get('explanation')
                )
        return None

    def search_mnemonics(
        self,
        query: str,
        category: Optional[str] = None,
        body_region: Optional[str] = None
    ) -> List[Mnemonic]:
        """
        Search mnemonics by topic, expansion, or body region.

        Args:
            query: Search term
            category: Filter by category (differential, syndrome, physics, etc.)
            body_region: Filter by body region (Neuro, GI, Thoracic, etc.)

        Returns:
            List of matching Mnemonic objects

        Example:
            >>> mnemonics = srd.search_mnemonics("small bowel")
            >>> for m in mnemonics:
            ...     print(f"{m.mnemonic}: {m.topic}")
        """
        query_lower = query.lower()
        results = []

        for m in self._mnemonics:
            score = 0
            mnemonic_name = m.get('mnemonic', '').lower()
            topic = m.get('topic', '').lower()
            expansion = m.get('expansion', '').lower()
            explanation = m.get('explanation', '').lower()

            # Direct mnemonic match (highest)
            if query_lower in mnemonic_name or mnemonic_name in query_lower:
                score += 4

            # Topic match
            if query_lower in topic:
                score += 3
            for word in query_lower.split():
                if len(word) > 2 and word in topic:
                    score += 1

            # Expansion match
            if query_lower in expansion:
                score += 2

            # Explanation match
            if query_lower in explanation:
                score += 1

            # Category filter
            if category:
                if category.lower() != m.get('category', '').lower():
                    continue

            # Body region filter
            if body_region:
                m_region = m.get('body_region', '').lower()
                if body_region.lower() not in m_region:
                    continue

            if score > 0:
                results.append((score, m))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            Mnemonic(
                mnemonic=m.get('mnemonic', ''),
                expansion=m.get('expansion', ''),
                topic=m.get('topic', ''),
                category=m.get('category'),
                body_region=m.get('body_region'),
                high_yield=m.get('high_yield', False),
                explanation=m.get('explanation')
            )
            for _, m in results
        ]

    def list_mnemonics_by_category(self) -> Dict[str, List[str]]:
        """
        List all mnemonics organized by category.

        Returns:
            Dict mapping category to list of mnemonic names
        """
        categories: Dict[str, List[str]] = defaultdict(list)
        for m in self._mnemonics:
            cat = m.get('category', 'other')
            categories[cat].append(m.get('mnemonic', ''))
        return dict(categories)

    def list_mnemonics_by_region(self) -> Dict[str, List[str]]:
        """
        List all mnemonics organized by body region.

        Returns:
            Dict mapping body_region to list of mnemonic names
        """
        regions: Dict[str, List[str]] = defaultdict(list)
        for m in self._mnemonics:
            region = m.get('body_region', 'General')
            regions[region].append(m.get('mnemonic', ''))
        return dict(regions)

    # =========================================================================
    # SYNDROME ASSOCIATIONS (Phase 2)
    # =========================================================================

    def get_syndrome_associations(self, syndrome: str) -> List[SyndromeAssociation]:
        """
        Get all imaging findings associated with a syndrome.

        Args:
            syndrome: Syndrome name (e.g., "VHL", "Tuberous Sclerosis", "NF1")

        Returns:
            List of SyndromeAssociation objects

        Example:
            >>> assocs = srd.get_syndrome_associations("VHL")
            >>> for a in assocs:
            ...     print(f"{a.associated_finding} ({a.frequency})")
        """
        syndrome_lower = syndrome.lower()
        results = []

        for s in self._syndrome_associations:
            syn_name = s.get('syndrome_name', '').lower()

            # Match by full name or abbreviation in parentheses
            if syndrome_lower in syn_name:
                results.append(SyndromeAssociation(
                    syndrome_name=s.get('syndrome_name', ''),
                    associated_finding=s.get('associated_finding', ''),
                    finding_type=s.get('finding_type'),
                    body_region=s.get('body_region'),
                    frequency=s.get('frequency'),
                    clinical_notes=s.get('clinical_notes'),
                    screening_recommendation=s.get('screening_recommendation'),
                    high_yield=s.get('high_yield', False)
                ))

        return results

    def search_syndromes_by_finding(self, finding: str) -> List[SyndromeAssociation]:
        """
        Find syndromes that are associated with a specific imaging finding.

        Args:
            finding: Imaging finding (e.g., "hemangioblastoma", "cardiac myxoma")

        Returns:
            List of SyndromeAssociation objects

        Example:
            >>> assocs = srd.search_syndromes_by_finding("cardiac myxoma")
            >>> for a in assocs:
            ...     print(f"Consider {a.syndrome_name}")
        """
        finding_lower = finding.lower()
        results = []

        for s in self._syndrome_associations:
            assoc_finding = s.get('associated_finding', '').lower()

            if finding_lower in assoc_finding:
                results.append(SyndromeAssociation(
                    syndrome_name=s.get('syndrome_name', ''),
                    associated_finding=s.get('associated_finding', ''),
                    finding_type=s.get('finding_type'),
                    body_region=s.get('body_region'),
                    frequency=s.get('frequency'),
                    clinical_notes=s.get('clinical_notes'),
                    screening_recommendation=s.get('screening_recommendation'),
                    high_yield=s.get('high_yield', False)
                ))

        return results

    def get_screening_recommendations(self, syndrome: str) -> List[Dict]:
        """
        Get screening recommendations for a syndrome.

        Args:
            syndrome: Syndrome name

        Returns:
            List of dicts with finding and recommendation

        Example:
            >>> recs = srd.get_screening_recommendations("TSC")
            >>> for r in recs:
            ...     print(f"{r['finding']}: {r['recommendation']}")
        """
        associations = self.get_syndrome_associations(syndrome)
        recommendations = []

        for a in associations:
            if a.screening_recommendation:
                recommendations.append({
                    'finding': a.associated_finding,
                    'recommendation': a.screening_recommendation,
                    'frequency': a.frequency
                })

        return recommendations

    def list_syndromes(self) -> List[str]:
        """
        List all syndromes in the database.

        Returns:
            List of unique syndrome names
        """
        syndromes = set()
        for s in self._syndrome_associations:
            syndromes.add(s.get('syndrome_name', ''))
        return sorted(list(syndromes))

    def stats(self) -> Dict[str, int]:
        """Get statistics about loaded data."""
        return {
            'imaging_findings': len(self._imaging_findings),
            'medical_concepts': len(self._medical_concepts),
            'differential_groups': len(self._differential_groups),
            'classification_systems': len(self._classification_systems),
            'synonym_mappings': len(self._synonyms),
            'imaging_signs': len(self._imaging_signs),
            'measurement_thresholds': len(self._measurement_thresholds),
            'mnemonics': len(self._mnemonics),
            'syndrome_associations': len(self._syndrome_associations),
        }
