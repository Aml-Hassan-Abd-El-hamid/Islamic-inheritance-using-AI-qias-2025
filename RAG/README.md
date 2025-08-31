# Arabic Islamic Inheritance RAG System

A Retrieval-Augmented Generation (RAG) system for answering multiple-choice questions about Islamic inheritance law using Arabic text processing and semantic search.

## Overview

This project implements a comprehensive RAG system that combines dense vector embeddings with language model generation to answer complex questions about Islamic inheritance law. The system achieved **44.0% overall accuracy** on multiple-choice questions, representing a **12.5 percentage point improvement** over the base generation model alone.

The orginal work was done via a kaggle notebook that can be found here: https://www.kaggle.com/code/amlhassan/rag-using-qwen

## Key Features

- **Arabic Text Processing**: Specialized normalization and tokenization for Arabic text
- **Multi-format Data Loading**: Support for both JSON Q&A pairs and unstructured TXT files
- **Semantic Search**: Dense vector embeddings using state-of-the-art Arabic language models
- **Generation**: Integration with Qwen2.5:7B via Ollama for answer generation
- **MCQ Answering**: Structured prompting for multiple-choice question answering

## Performance Results

Our RAG system demonstrated significant improvements over standalone generation models:

| Generation Model | Embedding Model | Beginner | Advanced | Overall |
|------------------|-----------------|----------|----------|---------|
| Qwen3:4B | N/A | 7.3% | 14.1% | 10.8% |
| Qwen3:8B | N/A | 12.7% | 17.3% | 15.0% |
| Qwen2.5:7B | N/A | 33.0% | 30.0% | **31.5%** |
| Qwen2.5:7B | paraphrase-multilingual-MiniLM-L12-v2 | 44.0% | 32.0% | 38.0% |
| Qwen2.5:7B | intfloat/multilingual-e5-base | 45.6% | 39.6% | 42.6% |
| Qwen2.5:7B | **Arabic-all-nli-triplet-Matryoshka** | **51.4%** | 36.6% | **44.0%** |

The best configuration achieved a **39.7% relative improvement** over the base model.

## Installation

1. **Clone the repository**:
```bash
git clone "https://github.com/Aml-Hassan-Abd-El-hamid/Islamic-inheritance-using-AI-qias-2025.git"
```

2. **Run setup script**:
```bash
python setup.py
```

This will:
- Install required Python packages
- Create necessary directories
- Download data files (if configured)

3. **Manual installation** (alternative):
```bash
pip install -r requirements.txt
mkdir data results
```

## Project Structure

```
RAG/
├── arabic_text_processor.py  # Arabic text processing utilities
├── enhanced_context.py       # Core RAG system implementation
├── ollama_utils.py           # Ollama setup and API utilities  
├── rag_system.py            # MCQ answering functionality
├── main.py                  # Main execution script
├── setup.py                # Setup and installation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Data directory
│   ├── *.json             # Q&A JSON files
│   ├── *.txt              # Text corpus files
│   └── *.csv              # Test datasets
```

## Usage

### Basic Usage

1. **Prepare your data**: Place your data files in the `data/` directory:
   - JSON files with Q&A pairs
   - TXT files with Islamic inheritance content
   - CSV test file with MCQ questions

2. **Update file paths** in `main.py`:
```python
json_files = [
    'data/your_qa_file1.json',
    'data/your_qa_file2.json'
]

txt_files = [
    'data/your_text_file1.txt',
    'data/your_text_file2.txt'
]
```

3. **Run the system**:
```bash
python main.py
```

## Technical Details

### Embedding Models Tested

- `intfloat/multilingual-e5-base`
- `paraphrase-multilingual-MiniLM-L12-v2` 
- `Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka` ✓ (Best performance)

### Generation Model

- **Qwen2.5:7B** via Ollama inference framework
- Temperature: 0.1 (for consistent answers)
- Top-p: 0.9

### Retrieval Configuration

- **Top-K**: 3 documents
- **Similarity Threshold**: 0.7 (cosine similarity)
- **Document Types**: Q&A pairs, text chunks

### Arabic Text Processing

- Diacritic removal
- Character normalization (أ, إ, آ → ا)
- Stopword filtering
- Keyword extraction

## Data Format

### JSON Q&A Format
```json
[
    {
        "Question": "ما هي حصة الزوجة في الميراث؟",
        "Answer": "حصة الزوجة في الميراث هي الثمن إذا كان للزوج ولد..."
    }
]
```

### CSV Test Format
```csv
question,option1,option2,option3,option4,label
"ما هي حصة البنت؟","النصف","الثلث","الربع","السدس","A"
```

## Configuration

You can modify the following parameters in the code:

- **Embedding model**: Change in `EnhancedContext.__init__()`
- **Similarity threshold**: Adjust in `answer_mcq_with_generation()`
- **Top-K documents**: Modify `top_k` parameter
- **Generation model**: Update in `generate_with_ollama()`

## Troubleshooting

### Ollama Issues
- Ensure Ollama is properly installed: `ollama --version`
- Check if service is running: `curl http://localhost:11434/api/tags`
- Manually start service: `ollama serve`

### Memory Issues
- Reduce batch size in `create_embeddings()`
- Use CPU-only FAISS: `faiss-cpu` instead of `faiss-gpu`
- Consider smaller embedding models

### Performance Issues
- Use GPU acceleration if available
- Increase batch size for embedding creation
- Adjust similarity threshold for retrieval



