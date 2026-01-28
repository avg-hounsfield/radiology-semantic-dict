"""
Clinical Reasoning Module
=========================
Provides intelligent reasoning for radiology interpretation:
- Multi-finding combination for improved differential diagnosis
- Measurement threshold evaluation with actionable recommendations
- Finding pattern recognition

This module elevates the package from keyword matching to clinical reasoning.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict


@dataclass
class MeasurementEvaluation:
    """Result of evaluating a measurement against thresholds."""
    measurement_value: float
    measurement_unit: str
    threshold_name: str
    threshold_value: float
    threshold_operator: str
    threshold_met: bool
    clinical_significance: str
    recommended_action: str
    urgency: str = "routine"  # routine, soon, urgent


@dataclass
class CombinedDiagnosis:
    """Result of combining multiple findings for diagnosis."""
    diagnosis: str
    confidence: float
    supporting_findings: List[str]
    reasoning: str
    pattern_name: Optional[str] = None


# =============================================================================
# MULTI-FINDING PATTERNS
# =============================================================================
# These patterns combine multiple findings for higher-confidence diagnoses.
# Each pattern specifies required findings, optional findings, and the diagnosis.

MULTI_FINDING_PATTERNS = {
    # =========================================================================
    # APPENDICITIS PATTERNS
    # =========================================================================
    'appendicitis_classic': {
        'required': ['fat stranding', 'rlq'],
        'optional': ['appendicolith', 'appendiceal thickening', 'periappendiceal fluid'],
        'diagnosis': 'Appendicitis',
        'base_confidence': 0.75,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'appendicitis_with_appendicolith': {
        'required': ['appendicolith'],
        'optional': ['fat stranding', 'rlq', 'periappendiceal fluid'],
        'diagnosis': 'Appendicitis',
        'base_confidence': 0.80,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'appendicitis_perforated': {
        'required': ['appendicitis', 'free fluid'],
        'optional': ['abscess', 'extraluminal air', 'phlegmon'],
        'diagnosis': 'Perforated appendicitis',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # CHOLECYSTITIS PATTERNS
    # =========================================================================
    'cholecystitis_classic': {
        'required': ['gallbladder wall thickening', 'pericholecystic fluid'],
        'optional': ['cholelithiasis', 'murphy sign', 'gallbladder distension'],
        'diagnosis': 'Acute cholecystitis',
        'base_confidence': 0.80,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'cholecystitis_acalculous': {
        'required': ['gallbladder wall thickening', 'pericholecystic fluid'],
        'excluded': ['cholelithiasis', 'gallstone'],
        'optional': ['gallbladder distension', 'sludge'],
        'diagnosis': 'Acalculous cholecystitis',
        'base_confidence': 0.70,
        'boost_per_optional': 0.05,
        'max_confidence': 0.90,
    },
    'gangrenous_cholecystitis': {
        'required': ['cholecystitis'],
        'optional': ['rim sign', 'intramural gas', 'irregular wall', 'mucosal sloughing', 'abscess'],
        'min_optional': 1,
        'diagnosis': 'Gangrenous cholecystitis',
        'base_confidence': 0.75,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # SMALL BOWEL OBSTRUCTION PATTERNS
    # =========================================================================
    'sbo_classic': {
        'required': ['small bowel dilation', 'decompressed colon'],
        'optional': ['transition point', 'air fluid levels', 'small bowel feces sign'],
        'diagnosis': 'Small bowel obstruction',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },
    'sbo_closed_loop': {
        'required': ['small bowel obstruction'],
        'optional': ['u-shaped loop', 'mesenteric swirl', 'beak sign', 'two transition points'],
        'min_optional': 1,
        'diagnosis': 'Closed loop small bowel obstruction',
        'base_confidence': 0.75,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
        'urgency': 'urgent',
    },
    'sbo_strangulation': {
        'required': ['small bowel obstruction'],
        'optional': ['bowel wall thickening', 'mesenteric haziness', 'reduced enhancement', 'ascites', 'pneumatosis'],
        'min_optional': 2,
        'diagnosis': 'Strangulated small bowel obstruction',
        'base_confidence': 0.80,
        'boost_per_optional': 0.04,
        'max_confidence': 0.95,
        'urgency': 'urgent',
    },

    # =========================================================================
    # PULMONARY EMBOLISM PATTERNS
    # =========================================================================
    'pe_classic': {
        'required': ['filling defect', 'pulmonary artery'],
        'optional': ['dvt', 'right heart strain', 'wedge infarct', 'pleural effusion'],
        'diagnosis': 'Pulmonary embolism',
        'base_confidence': 0.90,
        'boost_per_optional': 0.02,
        'max_confidence': 0.98,
    },
    'pe_saddle': {
        'required': ['saddle embolus'],
        'optional': ['right heart strain', 'rv dilation', 'septal bowing'],
        'diagnosis': 'Saddle pulmonary embolism',
        'base_confidence': 0.95,
        'boost_per_optional': 0.01,
        'max_confidence': 0.99,
        'urgency': 'urgent',
    },
    'pe_with_infarct': {
        'required': ['pulmonary embolism', 'wedge-shaped opacity'],
        'optional': ['pleural effusion', 'pleural-based'],
        'diagnosis': 'Pulmonary embolism with infarction',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # STROKE PATTERNS
    # =========================================================================
    'stroke_acute': {
        'required': ['restricted diffusion', 'vascular territory'],
        'optional': ['t2 hyperintensity', 'loss of gray white', 'swelling', 'vessel occlusion'],
        'diagnosis': 'Acute ischemic stroke',
        'base_confidence': 0.90,
        'boost_per_optional': 0.02,
        'max_confidence': 0.98,
        'urgency': 'urgent',
    },
    'stroke_hemorrhagic_conversion': {
        'required': ['stroke', 'hemorrhage'],
        'optional': ['petechial', 'hematoma', 'blood products'],
        'diagnosis': 'Hemorrhagic conversion of stroke',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },
    'stroke_large_vessel': {
        'required': ['restricted diffusion'],
        'optional': ['mca', 'ica', 'hyperdense vessel', 'dot sign', 'insular ribbon'],
        'min_optional': 1,
        'diagnosis': 'Large vessel ischemic stroke',
        'base_confidence': 0.80,
        'boost_per_optional': 0.04,
        'max_confidence': 0.95,
        'urgency': 'urgent',
    },

    # =========================================================================
    # PANCREATITIS PATTERNS
    # =========================================================================
    'pancreatitis_acute': {
        'required': ['pancreatic enlargement', 'peripancreatic fat stranding'],
        'optional': ['peripancreatic fluid', 'gallstones', 'pancreatic necrosis'],
        'diagnosis': 'Acute pancreatitis',
        'base_confidence': 0.80,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'pancreatitis_necrotizing': {
        'required': ['pancreatitis', 'necrosis'],
        'optional': ['gas', 'non-enhancement', 'walled-off necrosis'],
        'diagnosis': 'Necrotizing pancreatitis',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # HCC PATTERNS (LI-RADS)
    # =========================================================================
    'hcc_classic': {
        'required': ['washout'],  # Washout is pathognomonic
        'required_any': ['arterial enhancement', 'arterially enhancing', 'hyperenhancing', 'aphe'],
        'optional': ['capsule', 'threshold growth', 'cirrhosis', 'cirrhotic'],
        'context': ['liver', 'hepatic', 'segment'],
        'diagnosis': 'Hepatocellular carcinoma (LI-RADS 5)',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },
    'hcc_probable': {
        'required_any': ['arterial enhancement', 'arterially enhancing', 'hyperenhancing', 'aphe'],
        'optional': ['washout', 'capsule', 'cirrhosis', 'cirrhotic'],
        'min_optional': 0,
        'context': ['liver', 'hepatic', 'segment'],
        'diagnosis': 'Probable HCC (LI-RADS 4)',
        'base_confidence': 0.70,
        'boost_per_optional': 0.05,
        'max_confidence': 0.85,
    },

    # =========================================================================
    # MULTIPLE SCLEROSIS PATTERNS
    # =========================================================================
    'ms_classic': {
        'required': ['periventricular', 'white matter lesions'],
        'optional': ['dawson fingers', 'corpus callosum', 'juxtacortical', 'infratentorial', 'spinal cord'],
        'diagnosis': 'Multiple sclerosis',
        'base_confidence': 0.75,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'ms_mcdonald': {
        'required': ['white matter lesions'],
        'optional': ['periventricular', 'juxtacortical', 'infratentorial', 'spinal cord'],
        'min_optional': 2,  # Need dissemination in space
        'diagnosis': 'Multiple sclerosis (McDonald criteria)',
        'base_confidence': 0.80,
        'boost_per_optional': 0.04,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # RENAL CELL CARCINOMA PATTERNS
    # =========================================================================
    'rcc_classic': {
        'required': ['renal mass', 'enhancement'],
        'optional': ['heterogeneous', 'necrosis', 'renal vein invasion', 'lymphadenopathy'],
        'diagnosis': 'Renal cell carcinoma',
        'base_confidence': 0.75,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'rcc_with_thrombus': {
        'required': ['renal mass', 'renal vein thrombus'],
        'optional': ['ivc thrombus', 'enhancement'],
        'diagnosis': 'Renal cell carcinoma with venous invasion',
        'base_confidence': 0.90,
        'boost_per_optional': 0.02,
        'max_confidence': 0.98,
    },

    # =========================================================================
    # TRAUMA PATTERNS
    # =========================================================================
    'splenic_injury': {
        'required': ['splenic laceration'],
        'optional': ['hemoperitoneum', 'active extravasation', 'subcapsular hematoma', 'perisplenic fluid'],
        'diagnosis': 'Splenic injury',
        'base_confidence': 0.90,
        'boost_per_optional': 0.02,
        'max_confidence': 0.98,
    },
    'liver_injury': {
        'required': ['hepatic laceration'],
        'optional': ['hemoperitoneum', 'active extravasation', 'perihepatic fluid', 'subcapsular hematoma'],
        'diagnosis': 'Hepatic injury',
        'base_confidence': 0.90,
        'boost_per_optional': 0.02,
        'max_confidence': 0.98,
    },
    'bowel_injury': {
        'required': ['bowel wall thickening'],
        'optional': ['mesenteric stranding', 'free fluid', 'extraluminal air', 'mesenteric hematoma'],
        'min_optional': 1,
        'context': ['trauma'],
        'diagnosis': 'Bowel injury',
        'base_confidence': 0.70,
        'boost_per_optional': 0.06,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # DIVERTICULITIS PATTERNS
    # =========================================================================
    'diverticulitis_uncomplicated': {
        'required': ['diverticula', 'fat stranding'],
        'optional': ['llq', 'sigmoid', 'bowel wall thickening'],
        'excluded': ['abscess', 'perforation', 'free air'],
        'diagnosis': 'Uncomplicated diverticulitis',
        'base_confidence': 0.80,
        'boost_per_optional': 0.05,
        'max_confidence': 0.95,
    },
    'diverticulitis_complicated': {
        'required': ['diverticulitis'],
        'optional': ['abscess', 'perforation', 'free air', 'fistula', 'obstruction'],
        'min_optional': 1,
        'diagnosis': 'Complicated diverticulitis',
        'base_confidence': 0.85,
        'boost_per_optional': 0.03,
        'max_confidence': 0.95,
    },

    # =========================================================================
    # AORTIC PATHOLOGY PATTERNS
    # =========================================================================
    'aortic_dissection': {
        'required': ['intimal flap'],
        'optional': ['true lumen', 'false lumen', 'entry tear', 'branch involvement'],
        'diagnosis': 'Aortic dissection',
        'base_confidence': 0.95,
        'boost_per_optional': 0.01,
        'max_confidence': 0.99,
        'urgency': 'urgent',
    },
    'ruptured_aaa': {
        'required': ['aortic aneurysm', 'retroperitoneal hematoma'],
        'optional': ['active extravasation', 'discontinuity', 'draped aorta'],
        'diagnosis': 'Ruptured abdominal aortic aneurysm',
        'base_confidence': 0.95,
        'boost_per_optional': 0.01,
        'max_confidence': 0.99,
        'urgency': 'urgent',
    },
}


# =============================================================================
# CLINICAL REASONER CLASS
# =============================================================================

class ClinicalReasoner:
    """
    Provides clinical reasoning for radiology interpretation.

    Combines multiple findings for improved diagnosis accuracy,
    evaluates measurements against thresholds, and provides
    actionable recommendations.
    """

    def __init__(self, measurement_thresholds: List[Dict] = None):
        """
        Initialize the clinical reasoner.

        Args:
            measurement_thresholds: List of threshold dicts from JSON data
        """
        self._thresholds = measurement_thresholds or []
        self._threshold_keywords = self._build_threshold_index()

    def _build_threshold_index(self) -> Dict[str, List[Dict]]:
        """Build keyword index for threshold lookup."""
        index = defaultdict(list)

        # Keyword aliases for common abbreviations
        ALIASES = {
            'cbd': ['common bile duct', 'bile duct', 'biliary', 'bile'],
            'bile': ['cbd', 'common bile duct', 'bile duct', 'biliary'],
            'aaa': ['abdominal aortic aneurysm', 'aortic aneurysm', 'aorta'],
            'aorta': ['aortic', 'aaa'],
            'nodule': ['pulmonary nodule', 'lung nodule', 'thyroid nodule'],
            'spleen': ['splenic', 'splenomegaly'],
            'liver': ['hepatic', 'hepatomegaly'],
            'kidney': ['renal', 'nephro'],
            'prostate': ['prostatic', 'bph'],
        }

        # Also add the aliases directly to index for context matching
        for alias, expansions in ALIASES.items():
            # Create empty list if not exists
            if alias not in index:
                index[alias] = []

        for t in self._thresholds:
            name = t.get('name', '').lower()
            # Extract keywords from name
            keywords = re.findall(r'\b\w+\b', name)
            for kw in keywords:
                if len(kw) > 2:  # Allow 3-letter abbreviations
                    index[kw].append(t)
                    # Add aliases
                    for alias, expansions in ALIASES.items():
                        if kw == alias or kw in expansions:
                            for exp in [alias] + expansions:
                                if t not in index[exp]:
                                    index[exp].append(t)

            # Also index by body region
            region = t.get('body_region', '').lower()
            if region:
                index[region].append(t)

            # Index by finding_type
            finding_type = t.get('finding_type', '').lower()
            if finding_type:
                index[finding_type].append(t)

        return dict(index)

    def evaluate_measurement(
        self,
        value: float,
        unit: str,
        context: str,
        body_region: str = ""
    ) -> List[MeasurementEvaluation]:
        """
        Evaluate a measurement against relevant thresholds.

        Args:
            value: Numeric measurement value
            unit: Unit of measurement (mm, cm, etc.)
            context: Text context describing what was measured
            body_region: Body region if known

        Returns:
            List of MeasurementEvaluation results
        """
        evaluations = []
        context_lower = context.lower()

        # Normalize units
        value_cm = value
        if unit.lower() == 'mm':
            value_cm = value / 10

        # Find relevant thresholds
        relevant_thresholds = []
        for keyword, thresholds in self._threshold_keywords.items():
            if keyword in context_lower:
                relevant_thresholds.extend(thresholds)

        if body_region:
            for t in self._thresholds:
                if t.get('body_region', '').lower() == body_region.lower():
                    if t not in relevant_thresholds:
                        relevant_thresholds.append(t)

        # Evaluate against each threshold
        seen = set()
        for t in relevant_thresholds:
            t_name = t.get('name', '')
            if t_name in seen:
                continue
            seen.add(t_name)

            t_value = t.get('threshold_value', 0)
            t_unit = t.get('unit', '').lower()
            t_op = t.get('threshold_operator', '')

            # Convert threshold to same unit
            t_value_cm = t_value
            if t_unit == 'mm':
                t_value_cm = t_value / 10
            elif t_unit != 'cm':
                continue  # Skip non-size thresholds for now

            # Check if threshold is met
            threshold_met = False
            if t_op == '>':
                threshold_met = value_cm > t_value_cm
            elif t_op == '>=':
                threshold_met = value_cm >= t_value_cm
            elif t_op == '<':
                threshold_met = value_cm < t_value_cm
            elif t_op == '<=':
                threshold_met = value_cm <= t_value_cm

            action = t.get('action_if_met', '') if threshold_met else t.get('action_if_not_met', '')

            # Determine urgency
            urgency = 'routine'
            significance = t.get('clinical_significance', '').lower()
            if 'urgent' in significance or 'emergent' in significance:
                urgency = 'urgent'
            elif 'surgical' in action.lower() or 'repair' in action.lower():
                urgency = 'soon'

            evaluations.append(MeasurementEvaluation(
                measurement_value=value,
                measurement_unit=unit,
                threshold_name=t_name,
                threshold_value=t_value,
                threshold_operator=t_op,
                threshold_met=threshold_met,
                clinical_significance=t.get('clinical_significance', ''),
                recommended_action=action,
                urgency=urgency,
            ))

        # Sort by: 1) threshold_met (True first), 2) threshold_value descending
        # This prioritizes actionable (met) thresholds and higher (more severe) values
        evaluations.sort(key=lambda e: (-e.threshold_met, -e.threshold_value))

        return evaluations

    def combine_findings(
        self,
        findings: List[str],
        existing_diagnoses: List[Tuple[str, float]] = None
    ) -> List[CombinedDiagnosis]:
        """
        Combine multiple findings to generate higher-confidence diagnoses.

        Args:
            findings: List of finding strings
            existing_diagnoses: Optional existing diagnoses with scores

        Returns:
            List of CombinedDiagnosis with reasoning
        """
        results = []
        findings_lower = [f.lower() for f in findings]
        findings_text = ' '.join(findings_lower)

        for pattern_name, pattern in MULTI_FINDING_PATTERNS.items():
            required = pattern.get('required', [])
            required_any = pattern.get('required_any', [])  # At least one must match
            optional = pattern.get('optional', [])
            excluded = pattern.get('excluded', [])
            context = pattern.get('context', [])
            min_optional = pattern.get('min_optional', 0)

            # Check if all required findings are present
            required_met = all(
                any(req in f for f in findings_lower) or req in findings_text
                for req in required
            )

            if not required_met:
                continue

            # Check if at least one of required_any is present (synonym matching)
            if required_any:
                required_any_met = any(
                    any(req in f for f in findings_lower) or req in findings_text
                    for req in required_any
                )
                if not required_any_met:
                    continue

            # Check if any excluded findings are present
            if excluded:
                exclusion_met = any(
                    any(exc in f for f in findings_lower) or exc in findings_text
                    for exc in excluded
                )
                if exclusion_met:
                    continue

            # Check context if specified
            if context:
                context_met = any(
                    any(ctx in f for f in findings_lower) or ctx in findings_text
                    for ctx in context
                )
                if not context_met:
                    continue

            # Count optional findings present
            optional_found = []
            for opt in optional:
                if any(opt in f for f in findings_lower) or opt in findings_text:
                    optional_found.append(opt)

            # Check minimum optional requirement
            if len(optional_found) < min_optional:
                continue

            # Calculate confidence
            base_conf = pattern.get('base_confidence', 0.7)
            boost = pattern.get('boost_per_optional', 0.05)
            max_conf = pattern.get('max_confidence', 0.95)

            confidence = min(max_conf, base_conf + len(optional_found) * boost)

            # Build reasoning
            reasoning_parts = [f"Required: {', '.join(required)}"]
            if optional_found:
                reasoning_parts.append(f"Supporting: {', '.join(optional_found)}")

            results.append(CombinedDiagnosis(
                diagnosis=pattern['diagnosis'],
                confidence=confidence,
                supporting_findings=required + optional_found,
                reasoning=' | '.join(reasoning_parts),
                pattern_name=pattern_name,
            ))

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)

        return results

    def get_urgency(self, findings: List[str]) -> str:
        """
        Determine overall urgency based on findings.

        Returns: 'routine', 'soon', or 'urgent'
        """
        combined = self.combine_findings(findings)

        for diagnosis in combined:
            pattern = MULTI_FINDING_PATTERNS.get(diagnosis.pattern_name, {})
            if pattern.get('urgency') == 'urgent':
                return 'urgent'

        # Check for urgent keywords
        urgent_keywords = [
            'dissection', 'rupture', 'ruptured', 'hemorrhage', 'bleeding',
            'tension', 'saddle', 'massive', 'acute stroke', 'strangulation',
            'ischemia', 'infarct', 'perforation', 'free air',
            # Hemodynamically significant PE
            'right heart strain', 'rv strain', 'rv dilation', 'septal bowing',
            'clot in transit', 'right ventricular dysfunction',
            # Critical findings
            'herniation', 'midline shift', 'uncal', 'tonsillar',
        ]
        findings_text = ' '.join(f.lower() for f in findings)
        if any(kw in findings_text for kw in urgent_keywords):
            return 'urgent'

        return 'routine'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def combine_findings(findings: List[str]) -> List[CombinedDiagnosis]:
    """Quick multi-finding combination."""
    reasoner = ClinicalReasoner()
    return reasoner.combine_findings(findings)


def evaluate_measurement(
    value: float,
    unit: str,
    context: str
) -> List[MeasurementEvaluation]:
    """Quick measurement evaluation."""
    reasoner = ClinicalReasoner()
    return reasoner.evaluate_measurement(value, unit, context)
