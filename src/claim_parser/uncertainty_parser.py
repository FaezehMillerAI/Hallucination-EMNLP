import re

# Dictionary mapping uncertainty phrases to uncertainty level (0.0 to 1.0)
UNCERTAINTY_lexicon = {
    r"\bpossible\b": 0.7,
    r"\bpossibly\b": 0.7,
    r"\bprobable\b": 0.5,
    r"\bprobably\b": 0.5,
    r"\bmay\b": 0.6,
    r"\bmight\b": 0.7,
    r"\bcould\b": 0.6,
    r"\bsuggests?\b": 0.4,
    r"\bsuggestive of\b": 0.5,
    r"\bsuspected\b": 0.6,
    r"\bconcerning for\b": 0.5,
    r"\brule out\b": 0.8,
    r"\bevaluate for\b": 0.8,
    r"\bequivocal\b": 0.8,
    r"\bcannot be excluded\b": 0.7,
    r"\bcannot exclude\b": 0.7,
    r"\bquestionable\b": 0.8,
    r"\bperhaps\b": 0.8,
    r"\bpotential\b": 0.5,
    r"\bnot excluded\b": 0.6,
}

def parse_uncertainty(text: str) -> dict:
    """
    Parses linguistic uncertainty cues in the text.
    Returns a dict with:
      - 'uncertain': bool (True if uncertainty detected)
      - 'uncertainty_score': float (0.0 to 1.0)
      - 'markers': list of str (matched phrases)
    """
    text_lower = text.lower()
    matched_markers = []
    max_score = 0.0
    
    for pattern, score in UNCERTAINTY_lexicon.items():
        if re.search(pattern, text_lower):
            matched_markers.append(pattern.replace(r"\b", ""))
            if score > max_score:
                max_score = score
                
    return {
        "uncertain": len(matched_markers) > 0,
        "uncertainty_score": max_score,
        "markers": matched_markers
    }
