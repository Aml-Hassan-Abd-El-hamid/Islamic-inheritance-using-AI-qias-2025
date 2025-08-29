import re
from typing import List

class ArabicTextProcessor:
    """Arabic text processing utilities"""
    
    def __init__(self):
        self.arabic_stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بين', 'تحت', 'فوق', 'أمام', 'خلف',
            'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'هو', 'هي', 'أنت', 'أنتم', 'نحن', 'هم',
            'كان', 'كانت', 'يكون', 'تكون', 'قد', 'لقد', 'إن', 'أن', 'لكن', 'غير',
            'ما', 'لا', 'لم', 'لن', 'ليس', 'ليست'
        }
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text"""
        if not text:
            return ""
        
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        
        # Normalize Arabic characters
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')
        text = text.replace('ى', 'ي')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """Extract Arabic keywords from text"""
        text = self.normalize_arabic_text(text)
        words = re.findall(r'[\u0600-\u06FF]+', text)
        keywords = [
            word for word in words 
            if len(word) >= min_length and word not in self.arabic_stopwords
        ]
        return keywords
