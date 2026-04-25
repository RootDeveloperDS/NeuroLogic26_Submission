import re
import string

def clean_text(text: str) -> str:
    """
    Standardizes input text by removing noise, URLs, and punctuation.
    Designed for fast execution and memory efficiency.
    """
    if not isinstance(text, str):
        return ""
    
    # Converts to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # to Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    
    # toRemove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Removes extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
