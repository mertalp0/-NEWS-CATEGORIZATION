import re

def clean_text(text):
    """
    Cleans the text: Converts to lowercase and removes punctuation.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
