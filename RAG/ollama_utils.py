import re
from typing import List, Dict
from enhanced_context import EnhancedContext
from ollama_utils import generate_with_ollama

class RAGSystem(EnhancedContext):
    """RAG System with MCQ answering capability"""
    
    def answer_mcq_with_generation(self, question: str, options: List[str], similarity_threshold: float = 0.8, top_k: int = 1) -> Dict:
        """Answer MCQ using RAG + Generation with Ollama"""
        # Get relevant documents
        results = self.search(question, top_k=top_k, similarity_threshold=similarity_threshold)
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"مرجع {i+1}: {r['document']['Answer']}"
            for i, r in enumerate(results)
        ])
        
        # Create prompt
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        
        prompt = f"""اعتماداً على المراجع الشرعية التالية المتعلقة بالميراث الإسلامي، أجب عن السؤال التالي باختيار الرقم الصحيح فقط للإجابة الصحيحة:
                

                السؤال: {question}

                الخيارات:
                {options_text}

                اختر الإجابة الصحيحة ورقمها فقط (1، 2، 3، إلخ):
                {context}
                """

        # Generate answer
        response = generate_with_ollama(prompt)
        
        # Extract number from response (simple parsing)
        numbers = re.findall(r'\b([1-6])\b', response)
        predicted_num = int(numbers[0]) if numbers else 1
        
        # Validate prediction is within range
        if predicted_num > len(options):
            predicted_num = 1
        
        return {
            'predicted_answer': f'option{predicted_num}',
            'raw_response': response,
            'retrieved_sources': [r['document']['ID'] for r in results],
            'context_used': len(results),
            'fallback_used': False
        }
