import re

def split_sentences(text: str) -> list:
    """
    Splits a paragraph of medical text into individual sentences,
    handling common medical and general abbreviations.
    """
    if not text:
        return []
        
    # Split on standard sentence punctuation followed by whitespace
    raw_splits = re.split(r'(?<=[.!?])\s+', text.strip())
    
    abbreviations = ["e.g.", "i.e.", "dr.", "mr.", "mrs.", "ms.", "vs.", "fig.", "vol.", "no.", "pt."]
    
    sentences = []
    temp = ""
    for split in raw_splits:
        if temp:
            temp += " " + split
        else:
            temp = split
            
        # Check if last word ends with an abbreviation or if it's a decimal/initial
        words = temp.lower().split()
        if words:
            last_word = words[-1]
            is_abbr = any(last_word.endswith(abbr) for abbr in abbreviations)
            is_initial = len(last_word) <= 2 and last_word.endswith(".")
            is_decimal = last_word.endswith(".") and len(words) > 1 and words[-2].isdigit()
            
            if is_abbr or is_initial or is_decimal:
                continue  # Merge with next split
                
        sentences.append(temp)
        temp = ""
        
    if temp:
        sentences.append(temp)
        
    return [s for s in sentences if s]
