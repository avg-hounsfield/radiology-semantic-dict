"""
Radiology NLP Preprocessing Module
===================================
Handles text preprocessing for real-world radiology report interpretation:
- Negation detection
- Uncertainty/hedge word classification
- Measurement extraction
- Temporal marker detection
- Anatomical entity extraction

No external dependencies - pure Python with regex-based patterns.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class Certainty(Enum):
    """Certainty levels for findings."""
    DEFINITE = "definite"       # Stated as fact
    PROBABLE = "probable"       # Likely, probably, most consistent with
    POSSIBLE = "possible"       # Possible, may represent, cannot exclude
    UNLIKELY = "unlikely"       # Low probability, doubtful
    NEGATED = "negated"         # Explicitly ruled out


class Temporality(Enum):
    """Temporal status of findings."""
    NEW = "new"                 # New finding
    STABLE = "stable"           # Unchanged from prior
    IMPROVED = "improved"       # Better than prior
    WORSENED = "worsened"       # Worse than prior
    RESOLVED = "resolved"       # Previously present, now gone
    CHRONIC = "chronic"         # Long-standing
    ACUTE = "acute"             # Recent onset
    UNKNOWN = "unknown"         # No temporal context


@dataclass
class ExtractedMeasurement:
    """A measurement extracted from text."""
    value: float
    unit: str
    dimension: str = ""         # "diameter", "length", "volume", etc.
    raw_text: str = ""


@dataclass
class ExtractedFinding:
    """A finding extracted from report text with full context."""
    text: str                                   # Original finding text
    normalized: str = ""                        # Cleaned/normalized text
    negated: bool = False                       # Is this finding negated?
    certainty: Certainty = Certainty.DEFINITE   # How certain is this finding?
    temporality: Temporality = Temporality.UNKNOWN  # Temporal status
    measurements: List[ExtractedMeasurement] = field(default_factory=list)
    anatomical_location: str = ""               # Detected anatomy (deprecated, use body_regions)
    body_regions: List[str] = field(default_factory=list)  # Detected body regions
    laterality: str = ""                        # left/right/bilateral


# =============================================================================
# NEGATION PATTERNS
# =============================================================================
# Based on NegEx algorithm patterns, adapted for radiology

NEGATION_PREFIXES = [
    r'\bno\b',
    r'\bno evidence of\b',
    r'\bwithout\b',
    r'\bwithout evidence of\b',
    r'\bnegative for\b',
    r'\bdenies\b',
    r'\bdenied\b',
    r'\babsence of\b',
    r'\babsent\b',
    r'\bnot\b',
    r'\bnot seen\b',
    r'\bnot identified\b',
    r'\bnot demonstrated\b',
    r'\bnot visualized\b',
    r'\bno signs of\b',
    r'\bno findings of\b',
    r'\bno definite\b',
    r'\bno significant\b',
    r'\bno acute\b',
    r'\bno suspicious\b',
    r'\bfails to demonstrate\b',
    r'\bfailed to demonstrate\b',
    r'\bruled out\b',
    r'\brules out\b',
    # Note: "cannot exclude" is UNCERTAINTY, not negation - handled separately
    # r'\bexclude\b',  # Removed - too many false positives with "cannot exclude"
    # r'\bexcludes\b',
    # r'\bexcluded\b',
    r'\bfree of\b',
    r'\bresolve[ds]?\b',
    r'\bcleared\b',
    r'\bunremarkable\b',
    r'\bnormal\b',
    r'\bwithin normal limits\b',
]

NEGATION_SUFFIXES = [
    r'\bis absent\b',
    r'\bare absent\b',
    r'\bnot seen\b',
    r'\bnot present\b',
    r'\bnot identified\b',
    r'\bruled out\b',
    r'\bexcluded\b',
    r'\bhas resolved\b',
    r'\bhas cleared\b',
]

# Terms that END negation scope (conjunctions that start a new clause)
NEGATION_TERMINATORS = [
    r'\bbut\b',
    r'\bhowever\b',
    r'\balthough\b',
    r'\bthough\b',
    r'\bexcept\b',
    r'\bapart from\b',
    r'\bother than\b',
    r'\bwhich\b',
    r'\bthat\b',
]

# =============================================================================
# UNCERTAINTY PATTERNS
# =============================================================================

CERTAINTY_PATTERNS = {
    Certainty.PROBABLE: [
        r'\bprobably\b',
        r'\bprobable\b',
        r'\blikely\b',
        r'\bmost likely\b',
        r'\bhighly likely\b',
        r'\bmost consistent with\b',
        r'\bconsistent with\b',
        r'\bcompatible with\b',
        r'\bsuggestive of\b',
        r'\bsuggest[s]?\b',
        r'\bfavored\b',
        r'\bfavoring\b',
        r'\bpresumed\b',
        r'\bpresumably\b',
        r'\brepresent[s]?\b',
        r'\bappear[s]? to\b',
    ],
    Certainty.POSSIBLE: [
        r'\bpossible\b',
        r'\bpossibly\b',
        r'\bmay represent\b',
        r'\bmay be\b',
        r'\bmight be\b',
        r'\bcould be\b',
        r'\bcould represent\b',
        r'\bcannot exclude\b',
        r'\bcannot rule out\b',
        r'\bcannot be excluded\b',
        r'\bquestionable\b',
        r'\buncertain\b',
        r'\bindeterminate\b',
        r'\bequivocal\b',
        r'\bconsider\b',
        r'\bto consider\b',
        r'\bdifferential includes\b',
        r'\bdifferential consideration\b',
        r'\bworrisome for\b',
        r'\bconcerning for\b',
        r'\bsuspicious for\b',
        r'\bsuspicious\b',
        r'\braise[s]? concern\b',
        r'\braise[s]? the possibility\b',
    ],
    Certainty.UNLIKELY: [
        r'\bunlikely\b',
        r'\blow probability\b',
        r'\blow suspicion\b',
        r'\bdoubtful\b',
        r'\bimprobable\b',
        r'\bprobably not\b',
        r'\bprobably benign\b',
        r'\blikely benign\b',
        r'\bmost likely benign\b',
        r'\balmost certainly benign\b',
    ],
}

# =============================================================================
# TEMPORAL PATTERNS
# =============================================================================

TEMPORAL_PATTERNS = {
    Temporality.NEW: [
        r'\bnew\b',
        r'\bnewly\b',
        r'\binterval development\b',
        r'\bnew since\b',
        r'\bfirst seen\b',
        r'\bnot previously seen\b',
        r'\bnot seen on prior\b',
        r'\bnot present on prior\b',
        r'\bemerging\b',
        r'\bdeveloping\b',
        r'\bde novo\b',
    ],
    Temporality.STABLE: [
        r'\bstable\b',
        r'\bunchanged\b',
        r'\bno change\b',
        r'\bno significant change\b',
        r'\bno interval change\b',
        r'\bsimilar\b',
        r'\bsimilar to prior\b',
        r'\bcomparable\b',
        r'\bpersistent\b',
        r'\bgrossly unchanged\b',
        r'\bessentially unchanged\b',
        r'\bredemon\w*\b',  # redemonstrated
    ],
    Temporality.IMPROVED: [
        r'\bimproved\b',
        r'\bimproving\b',
        r'\bimprovement\b',
        r'\bdecreased\b',
        r'\bdecreasing\b',
        r'\bdiminished\b',
        r'\bdiminishing\b',
        r'\breduced\b',
        r'\breducing\b',
        r'\bresolved\b',
        r'\bresolving\b',
        r'\bsmaller\b',
        r'\bless\b',
        r'\binterval decrease\b',
        r'\binterval improvement\b',
        r'\bnearly resolved\b',
        r'\balmost resolved\b',
    ],
    Temporality.WORSENED: [
        r'\bworse\b',
        r'\bworsened\b',
        r'\bworsening\b',
        r'\bincreased\b',
        r'\bincreasing\b',
        r'\bprogressed\b',
        r'\bprogressing\b',
        r'\bprogression\b',
        r'\blarger\b',
        r'\bmore\b',
        r'\binterval increase\b',
        r'\binterval worsening\b',
        r'\benlarged\b',
        r'\benlarging\b',
        r'\bexpanded\b',
        r'\bexpanding\b',
    ],
    Temporality.CHRONIC: [
        r'\bchronic\b',
        r'\bold\b',
        r'\blong-?standing\b',
        r'\blong-?term\b',
        r'\bestablished\b',
        r'\bknown\b',
        r'\bpreexisting\b',
        r'\bpre-existing\b',
        r'\bremote\b',
        r'\bsequela\b',
        r'\bsequel\w*\b',
    ],
    Temporality.ACUTE: [
        r'\bacute\b',
        r'\bacutely\b',
        r'\brecent\b',
        r'\brecently\b',
        r'\bearly\b',
        r'\bfresh\b',
        r'\bacute-on-chronic\b',
        r'\bsuperimposed\b',
    ],
    Temporality.RESOLVED: [
        r'\bresolved\b',
        r'\bno longer\b',
        r'\bno longer seen\b',
        r'\bno longer present\b',
        r'\bcleared\b',
        r'\bgone\b',
        r'\babsent\b',  # when comparing to prior
    ],
}

# =============================================================================
# MEASUREMENT PATTERNS
# =============================================================================

MEASUREMENT_PATTERN = re.compile(
    r'(?P<value>\d+(?:\.\d+)?)\s*'
    r'(?P<unit>mm|cm|inches?|in|cc|ml|mL|liters?|hu|hounsfield)\b'  # Removed standalone 'm'
    r'(?:\s+(?P<dimension>diameter|length|width|height|size|volume|thickness))?',
    re.IGNORECASE
)

# Alternative: "X x Y x Z cm" format
MULTI_DIM_PATTERN = re.compile(
    r'(?P<d1>\d+(?:\.\d+)?)\s*x\s*(?P<d2>\d+(?:\.\d+)?)'
    r'(?:\s*x\s*(?P<d3>\d+(?:\.\d+)?))?'
    r'\s*(?P<unit>mm|cm)',
    re.IGNORECASE
)

# =============================================================================
# LATERALITY PATTERNS
# =============================================================================

LATERALITY_PATTERNS = {
    'left': [r'\bleft\b', r'\bl\s+(?=\w)', r'\blt\b'],
    'right': [r'\bright\b', r'\br\s+(?=\w)', r'\brt\b'],
    'bilateral': [r'\bbilateral\b', r'\bbilaterally\b', r'\bboth\b'],
}

# =============================================================================
# BODY REGION PATTERNS
# =============================================================================

BODY_REGION_PATTERNS = {
    # Brain/Head
    'brain': [r'\bbrain\b', r'\bcerebr\w+\b', r'\bintracranial\b', r'\bcranial\b'],
    'head': [r'\bhead\b', r'\bskull\b', r'\bcalvari\w+\b'],
    'orbit': [r'\borbit\w*\b', r'\borbital\b', r'\beye\b', r'\bocular\b'],
    'sinus': [r'\bsinus\b', r'\bparanasal\b', r'\bethmoid\b', r'\bmaxillary\b', r'\bsphenoid\b', r'\bfrontal sinus\b'],

    # Neck
    'neck': [r'\bneck\b', r'\bcervical\b(?!\s*spine)'],
    'thyroid': [r'\bthyroid\b'],
    'parotid': [r'\bparotid\b', r'\bsubmandibular\b'],
    'larynx': [r'\blaryn\w+\b', r'\bvocal\b', r'\bglott\w+\b'],

    # Chest
    'chest': [r'\bchest\b', r'\bthorac\w+\b', r'\bthorax\b'],
    'lung': [r'\blung\b', r'\blungs\b', r'\bpulmonary\b', r'\bpulmonic\b'],
    'right_upper_lobe': [r'\bright upper lobe\b', r'\brul\b', r'\brll\b'],
    'right_middle_lobe': [r'\bright middle lobe\b', r'\brml\b'],
    'right_lower_lobe': [r'\bright lower lobe\b', r'\brll\b'],
    'left_upper_lobe': [r'\bleft upper lobe\b', r'\blul\b'],
    'left_lower_lobe': [r'\bleft lower lobe\b', r'\blll\b'],
    'lingula': [r'\blingula\b'],
    'mediastinum': [r'\bmediastin\w+\b'],
    'pleura': [r'\bpleura\w*\b', r'\bpleural\b'],
    'heart': [r'\bheart\b', r'\bcardiac\b', r'\bmyocardi\w+\b', r'\bpericardi\w+\b'],
    'aorta': [r'\baorta\b', r'\baortic\b'],

    # Abdomen
    'abdomen': [r'\babdomen\b', r'\babdominal\b'],
    'liver': [r'\bliver\b', r'\bhepatic\b', r'\bhepato\w+\b'],
    'gallbladder': [r'\bgallbladder\b', r'\bgb\b', r'\bcholecyst\w+\b'],
    'bile_duct': [r'\bbile duct\b', r'\bbiliary\b', r'\bcbd\b', r'\bcommon bile duct\b', r'\bcholedoch\w+\b'],
    'pancreas': [r'\bpancrea\w+\b'],
    'spleen': [r'\bspleen\b', r'\bsplenic\b'],
    'stomach': [r'\bstomach\b', r'\bgastric\b'],
    'small_bowel': [r'\bsmall bowel\b', r'\bsmall intestine\b', r'\bjejun\w+\b', r'\bile\w+\b', r'\bduoden\w+\b'],
    'colon': [r'\bcolon\b', r'\bcolonic\b', r'\blarge bowel\b', r'\bcecum\b', r'\bsigmoid\b', r'\brectum\b', r'\brectal\b'],
    'appendix': [r'\bappendix\b', r'\bappendiceal\b', r'\bperiappendiceal\b'],
    'kidney': [r'\bkidney\b', r'\brenal\b', r'\bnephr\w+\b'],
    'ureter': [r'\bureter\w*\b', r'\bureteral\b'],
    'bladder': [r'\bbladder\b', r'\bvesic\w+\b'],
    'adrenal': [r'\badrenal\b', r'\bsuprarenal\b'],
    'retroperitoneum': [r'\bretroperiton\w+\b'],
    'mesentery': [r'\bmesentery\b', r'\bmesenteric\b'],
    'peritoneum': [r'\bperiton\w+\b'],

    # Pelvis
    'pelvis': [r'\bpelvi\w*\b'],
    'prostate': [r'\bprostat\w+\b'],
    'uterus': [r'\buter\w+\b', r'\bendometri\w+\b', r'\bmyometri\w+\b'],
    'ovary': [r'\bovar\w+\b', r'\badnexa\w*\b'],
    'rectum': [r'\brectum\b', r'\brectal\b', r'\bperirectal\b'],

    # Spine
    'cervical_spine': [r'\bcervical spine\b', r'\bc-spine\b', r'\bc\d\b'],
    'thoracic_spine': [r'\bthoracic spine\b', r'\bt-spine\b', r'\bt\d+\b'],
    'lumbar_spine': [r'\blumbar spine\b', r'\bl-spine\b', r'\bl\d\b'],
    'sacrum': [r'\bsacr\w+\b', r'\bsacroiliac\b'],
    'spine': [r'\bspine\b', r'\bspinal\b', r'\bvertebr\w+\b'],

    # Extremities
    'shoulder': [r'\bshoulder\b', r'\bglenohumeral\b', r'\brotator cuff\b'],
    'elbow': [r'\belbow\b'],
    'wrist': [r'\bwrist\b', r'\bcarpal\b'],
    'hand': [r'\bhand\b', r'\bmetacarpal\b', r'\bphalanx\b', r'\bphalanges\b'],
    'hip': [r'\bhip\b', r'\bfemoral head\b', r'\bacetabul\w+\b'],
    'knee': [r'\bknee\b', r'\bpatella\b', r'\bmeniscus\b', r'\bmeniscal\b'],
    'ankle': [r'\bankle\b', r'\bmalleol\w+\b', r'\btalus\b'],
    'foot': [r'\bfoot\b', r'\bmetatarsal\b', r'\bcalcane\w+\b'],

    # Vessels
    'carotid': [r'\bcarotid\b'],
    'pulmonary_artery': [r'\bpulmonary arter\w+\b', r'\bpa\b'],
    'iliac': [r'\biliac\b'],
    'femoral': [r'\bfemoral\b'],
    'popliteal': [r'\bpopliteal\b'],
}


# =============================================================================
# NLP PROCESSOR CLASS
# =============================================================================

class RadiologyNLP:
    """
    NLP processor for radiology report text.

    Extracts structured information from free-text findings including:
    - Negation status
    - Certainty level
    - Temporal context
    - Measurements
    - Laterality

    Example:
        >>> nlp = RadiologyNLP()
        >>> result = nlp.process("No evidence of pulmonary embolism")
        >>> result.negated
        True
        >>> result.certainty
        Certainty.NEGATED
    """

    def __init__(self):
        # Compile patterns for efficiency
        self._negation_prefix_re = re.compile(
            '|'.join(f'({p})' for p in NEGATION_PREFIXES),
            re.IGNORECASE
        )
        self._negation_suffix_re = re.compile(
            '|'.join(f'({p})' for p in NEGATION_SUFFIXES),
            re.IGNORECASE
        )
        self._negation_terminator_re = re.compile(
            '|'.join(f'({p})' for p in NEGATION_TERMINATORS),
            re.IGNORECASE
        )

        self._certainty_patterns = {
            level: re.compile('|'.join(f'({p})' for p in patterns), re.IGNORECASE)
            for level, patterns in CERTAINTY_PATTERNS.items()
        }

        self._temporal_patterns = {
            temp: re.compile('|'.join(f'({p})' for p in patterns), re.IGNORECASE)
            for temp, patterns in TEMPORAL_PATTERNS.items()
        }

        self._laterality_patterns = {
            side: re.compile('|'.join(f'({p})' for p in patterns), re.IGNORECASE)
            for side, patterns in LATERALITY_PATTERNS.items()
        }

    def detect_negation(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if the finding is negated.

        Returns:
            Tuple of (is_negated, negation_trigger)
        """
        text_lower = text.lower()

        # Check prefix negations
        prefix_match = self._negation_prefix_re.search(text_lower)
        if prefix_match:
            # Check if there's a terminator that limits scope
            terminator = self._negation_terminator_re.search(text_lower[prefix_match.end():])
            if terminator:
                # Negation scope ended - check if main content is after terminator
                # For simplicity, if terminator exists, be conservative
                pass
            return True, prefix_match.group()

        # Check suffix negations
        suffix_match = self._negation_suffix_re.search(text_lower)
        if suffix_match:
            return True, suffix_match.group()

        return False, None

    def detect_certainty(self, text: str) -> Certainty:
        """
        Detect the certainty level of the finding.

        Returns highest-priority certainty level found.
        """
        text_lower = text.lower()

        # Check negation first (takes priority)
        is_negated, _ = self.detect_negation(text)
        if is_negated:
            return Certainty.NEGATED

        # Check uncertainty levels in order of priority
        for level in [Certainty.UNLIKELY, Certainty.POSSIBLE, Certainty.PROBABLE]:
            if self._certainty_patterns[level].search(text_lower):
                return level

        return Certainty.DEFINITE

    def detect_temporality(self, text: str) -> Temporality:
        """
        Detect the temporal status of the finding.
        """
        text_lower = text.lower()

        # Check in priority order
        priority_order = [
            Temporality.RESOLVED,
            Temporality.NEW,
            Temporality.WORSENED,
            Temporality.IMPROVED,
            Temporality.ACUTE,
            Temporality.CHRONIC,
            Temporality.STABLE,
        ]

        for temp in priority_order:
            if self._temporal_patterns[temp].search(text_lower):
                return temp

        return Temporality.UNKNOWN

    def extract_measurements(self, text: str) -> List[ExtractedMeasurement]:
        """
        Extract all measurements from the text.
        """
        measurements = []
        matched_spans = set()  # Track already-matched positions

        # Multi-dimensional measurements first (X x Y x Z)
        for match in MULTI_DIM_PATTERN.finditer(text):
            dims = [match.group('d1'), match.group('d2')]
            if match.group('d3'):
                dims.append(match.group('d3'))

            # Report the largest dimension
            max_val = max(float(d) for d in dims)
            measurements.append(ExtractedMeasurement(
                value=max_val,
                unit=match.group('unit').lower(),
                dimension='max diameter',
                raw_text=match.group()
            ))
            # Mark this span as matched to avoid double-counting
            matched_spans.add((match.start(), match.end()))

        # Single measurements (skip if overlapping with multi-dim)
        for match in MEASUREMENT_PATTERN.finditer(text):
            # Check if this overlaps with any multi-dim match
            overlaps = any(
                not (match.end() <= start or match.start() >= end)
                for start, end in matched_spans
            )
            if overlaps:
                continue

            measurements.append(ExtractedMeasurement(
                value=float(match.group('value')),
                unit=match.group('unit').lower(),
                dimension=match.group('dimension') or '',
                raw_text=match.group()
            ))

        return measurements

    def detect_laterality(self, text: str) -> str:
        """
        Detect laterality (left/right/bilateral).
        """
        text_lower = text.lower()

        for side, pattern in self._laterality_patterns.items():
            if pattern.search(text_lower):
                return side

        return ""

    def detect_body_regions(self, text: str) -> List[str]:
        """
        Detect all body regions mentioned in the text.

        Returns:
            List of body region keys (e.g., ['lung', 'pleura'])
        """
        text_lower = text.lower()
        regions = []

        for region, patterns in BODY_REGION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    regions.append(region)
                    break  # Found this region, move to next

        return regions

    def process(self, text: str) -> ExtractedFinding:
        """
        Process a finding text and extract all structured information.

        Args:
            text: Raw finding text from a radiology report

        Returns:
            ExtractedFinding with all detected attributes
        """
        is_negated, _ = self.detect_negation(text)

        return ExtractedFinding(
            text=text,
            normalized=self._normalize(text),
            negated=is_negated,
            certainty=self.detect_certainty(text),
            temporality=self.detect_temporality(text),
            measurements=self.extract_measurements(text),
            body_regions=self.detect_body_regions(text),
            laterality=self.detect_laterality(text),
        )

    def process_report(self, report_text: str) -> List[ExtractedFinding]:
        """
        Process an entire report by splitting into sentences and processing each.

        This is a simple sentence splitter - production use would benefit
        from more sophisticated segmentation.
        """
        # Split on periods, but be careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', report_text)

        findings = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                findings.append(self.process(sentence))

        return findings

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_negated(text: str) -> bool:
    """Quick check if a finding is negated."""
    nlp = RadiologyNLP()
    return nlp.detect_negation(text)[0]


def get_certainty(text: str) -> Certainty:
    """Quick check for certainty level."""
    nlp = RadiologyNLP()
    return nlp.detect_certainty(text)


def get_temporality(text: str) -> Temporality:
    """Quick check for temporal status."""
    nlp = RadiologyNLP()
    return nlp.detect_temporality(text)


def extract_measurements(text: str) -> List[ExtractedMeasurement]:
    """Quick extraction of measurements."""
    nlp = RadiologyNLP()
    return nlp.extract_measurements(text)
