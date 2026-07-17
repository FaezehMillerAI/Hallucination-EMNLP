import re
from .sentence_splitter import split_sentences
from .uncertainty_parser import parse_uncertainty

# Lists of common medical findings, anatomies, and severities for rule-based decomposition
ANATOMIES = [
    "right lower lung", "left lower lung", "right upper lung", "left upper lung",
    "right mid lung", "left mid lung", "cardiomediastinal silhouette", "heart",
    "pleural space", "diaphragm", "mediastinum", "hilar region", "costophrenic angle",
    "lungs", "right lung", "left lung", "lung"
]

FINDINGS = [
    "infiltration", "opacity", "consolidation", "pleural effusion", "effusion",
    "pneumothorax", "cardiomegaly", "congestion", "edema", "atelectasis", "nodule",
    "mass", "lesion", "fracture", "abnormality", "normal"
]

SEVERITIES = ["mild", "moderate", "severe", "minimal", "massive", "small", "large", "increased", "decreased", "stable"]

NEGATIONS = [r"\bno\b", r"\bnot\b", r"\bclear of\b", r"\bfree of\b", r"\bwithout\b", r"\babsence of\b", r"\bnegative for\b"]

def find_span(substring: str, text: str) -> tuple:
    """Find start and end character indexes of a substring in a text."""
    match = re.search(re.escape(substring), text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return -1, -1

def decompose_claims(text: str) -> list:
    """
    Decomposes an answer or report string into a list of atomic claim dictionaries.
    
    Each claim contains:
      - sentence: str
      - subject: str
      - predicate: str
      - object: str
      - anatomy: str
      - severity: str
      - negation: bool
      - uncertainty: dict (from parse_uncertainty)
      - claim_type: str
      - token_span: tuple (start_char, end_char)
    """
    sentences = split_sentences(text)
    atomic_claims = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # 1. Detect anatomy
        detected_anatomy = "unknown"
        for anatomy in ANATOMIES:
            if anatomy in sentence_lower:
                detected_anatomy = anatomy
                break
                
        # 2. Detect finding/subject
        detected_finding = "abnormality"
        for finding in FINDINGS:
            if finding in sentence_lower:
                detected_finding = finding
                break
                
        # 3. Detect severity
        detected_severity = "normal"
        for severity in SEVERITIES:
            if severity in sentence_lower:
                detected_severity = severity
                break
                
        # 4. Detect negation
        negated = False
        for neg in NEGATIONS:
            if re.search(neg, sentence_lower):
                negated = True
                break
                
        # 5. Get uncertainty
        uncertainty = parse_uncertainty(sentence)
        
        # Determine subject, predicate, object based on structure
        # VQA format: "Does Right lower lung suffer from Infiltration?" -> Subject: Right lower lung, Predicate: suffer from, Object: Infiltration
        if "suffer from" in sentence_lower:
            subject = detected_anatomy
            predicate = "suffers from"
            obj = detected_finding
            claim_type = "finding_presence"
        # VQA answer format: "The location of Right lower lung is at <seg>. The answer is No"
        elif "location" in sentence_lower and "<seg>" in sentence:
            subject = detected_anatomy
            predicate = "located at"
            obj = "<seg>"
            claim_type = "anatomical_localization"
        else:
            subject = detected_finding if detected_finding != "abnormality" else detected_anatomy
            predicate = "is"
            obj = "absent" if negated else "present"
            claim_type = "normality_statement" if "normal" in sentence_lower else "finding_presence"
            
        start, end = find_span(sentence, text)
        
        claim = {
            "sentence": sentence,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "anatomy": detected_anatomy,
            "severity": detected_severity,
            "negation": negated,
            "uncertainty": uncertainty,
            "claim_type": claim_type,
            "token_span": (start, end)
        }
        atomic_claims.append(claim)
        
    return atomic_claims
