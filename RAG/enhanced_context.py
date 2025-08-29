import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import re
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
from arabic_text_processor import ArabicTextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedContext:
    def __init__(self, model_name: str = 'Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka'):
        """Initialize the RAG system"""
        logger.info(f"Initializing RAG System with model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.text_processor = ArabicTextProcessor()
        self.creation_time = datetime.now()
        
    def load_data(self, json_files: List[str] = None, txt_files: List[str] = None):
        """Load data from JSON and TXT files"""
        initial_count = len(self.documents)
        
        # Load JSON files
        if json_files:
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        valid_docs = self._validate_json_documents(data)
                        self.documents.extend(valid_docs)
                    logger.info(f"Loaded {len(valid_docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading JSON file {file_path}: {e}")
        
        # Load TXT files
        if txt_files:
            for file_path in txt_files:
                doc_count = self._load_txt_file(file_path)
                logger.info(f"Loaded {doc_count} documents from {file_path}")
        
        total_loaded = len(self.documents) - initial_count
        logger.info(f"Total new documents loaded: {total_loaded}")
        logger.info(f"Total documents in system: {len(self.documents)}")
    
    def _validate_json_documents(self, data: List[Dict]) -> List[Dict]:
        """Validate and clean JSON documents"""
        valid_docs = []
        for doc in data:
            if isinstance(doc, dict) and 'Question' in doc and 'Answer' in doc:
                doc['Question'] = self.text_processor.normalize_arabic_text(doc.get('Question', ''))
                doc['Answer'] = self.text_processor.normalize_arabic_text(doc.get('Answer', ''))
                
                if doc['Question'].strip() and doc['Answer'].strip():
                    valid_docs.append(doc)
        
        return valid_docs
    
    def _load_txt_file(self, file_path: str) -> int:
        """Load TXT file and split into documents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = self.text_processor.normalize_arabic_text(content)
            filename = os.path.basename(file_path).replace('.txt', '')
            
            # Check for Q&A patterns
            if self._has_qa_patterns(content):
                return self._split_by_qa_pairs(content, filename)
            else:
                return self._split_by_chunks(content, filename)
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return 0
    
    def _has_qa_patterns(self, content: str) -> bool:
        """Check if content has Q&A patterns"""
        qa_patterns = [r'سؤال:', r'السؤال:', r'س\d*:', r'ج\d*:', r'الجواب:', r'الإجابة:']
        return any(re.search(pattern, content) for pattern in qa_patterns)
    
    def _split_by_qa_pairs(self, content: str, filename: str) -> int:
        """Split content by Q&A pairs"""
        # Simple pattern matching for Q&A
        patterns = [
            r'(سؤال\s*\d*\s*:|السؤال\s*\d*\s*:)(.*?)(?=(سؤال\s*\d*\s*:|السؤال\s*\d*\s*:)|$)',
            r'(س\d*\s*:)(.*?)(?=(س\d*\s*:)|$)'
        ]
        
        doc_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                for i, match in enumerate(matches):
                    qa_text = match[1].strip() if len(match) > 1 else match[0].strip()
                    
                    if len(qa_text) < 50:
                        continue
                    
                    question, answer = self._extract_qa_from_text(qa_text)
                    
                    if question and answer:
                        doc_id = f"TXT_{filename}_QA_{doc_count+1:03d}"
                        self.documents.append({
                            'ID': doc_id,
                            'Question': question,
                            'Answer': answer,
                            'Source': 'TXT_QA',
                            'Filename': filename
                        })
                        doc_count += 1
                break
        
        return doc_count
    
    def _extract_qa_from_text(self, text: str) -> Tuple[str, str]:
        """Extract question and answer from text"""
        answer_patterns = [r'(ج\s*\d*\s*:|الجواب\s*:|الإجابة\s*:)']
        
        for pattern in answer_patterns:
            parts = re.split(pattern, text, maxsplit=1)
            if len(parts) >= 3:
                question = parts[0].strip()
                answer = parts[2].strip()
                return question, answer
        
        # Fallback: split in half
        words = text.split()
        if len(words) > 10:
            mid_point = len(words) // 2
            question = ' '.join(words[:mid_point])
            answer = ' '.join(words[mid_point:])
            return question, answer
        
        return text[:100] + "؟", text
    
    def _split_by_chunks(self, content: str, filename: str, chunk_size: int = 400) -> int:
        """Split content into chunks"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        doc_count = 0
        for i, chunk in enumerate(chunks):
            if len(chunk) > 100:
                question = self._generate_chunk_question(chunk, filename)
                doc_id = f"TXT_{filename}_CHUNK_{doc_count+1:03d}"
                self.documents.append({
                    'ID': doc_id,
                    'Question': question,
                    'Answer': chunk,
                    'Source': 'TXT_Chunk',
                    'Filename': filename
                })
                doc_count += 1
        
        return doc_count
    
    def _generate_chunk_question(self, chunk: str, filename: str) -> str:
        """Generate question for chunk"""
        keywords = self.text_processor.extract_keywords(chunk, min_length=3)
        
        if any(word in chunk for word in ['ميراث', 'الوارث', 'الورثة']):
            if keywords:
                return f"ما هي أحكام {' و'.join(keywords[:2])} في الميراث؟"
            return f"أحكام الميراث في {filename}"
        elif any(word in chunk for word in ['البنت', 'الابن', 'الأب', 'الأم']):
            return f"ما هي حقوق الورثة المذكورة؟"
        else:
            if keywords:
                return f"ما المقصود بـ {keywords[0]}؟"
            return f"محتوى من {filename}"
    
    def create_embeddings(self, batch_size: int = 32):
        """Create embeddings for all documents"""
        if not self.documents:
            logger.warning("No documents to create embeddings for")
            return
        
        logger.info("Creating embeddings...")
        
        # Prepare texts
        texts = []
        for doc in self.documents:
            question_text = f"السؤال: {doc['Question']}"
            answer_text = f"الجواب: {doc['Answer']}"
            combined_text = f"{question_text}\n{answer_text}"
            texts.append(combined_text)
        
        # Create embeddings
        self.embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        
        logger.info(f"Created embeddings for {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 1, similarity_threshold: float = 0.8) -> List[Dict]:
        """Search using cosine similarity"""
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Run create_embeddings() first.")
        
        logger.info(f"Searching for: {query}")
        
        # Normalize query
        normalized_query = self.text_processor.normalize_arabic_text(query)
        
        # Get query embedding
        query_embedding = self.model.encode([normalized_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top results above threshold
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            similarity_score = similarities[idx]
            
            if similarity_score >= similarity_threshold:
                result = {
                    'rank': rank + 1,
                    'similarity_score': float(similarity_score),
                    'document': self.documents[idx],
                    'question': self.documents[idx]['Question'],
                    'answer': self.documents[idx]['Answer']
                }
                results.append(result)
        
        return results
