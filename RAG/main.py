import pandas as pd
import logging
from rag_system import RAGSystem
from ollama_utils import setup_ollama, check_ollama_running

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("="*50)
    print("Setting up Islamic Inheritance RAG System")
    print("="*50)
    
    # Setup Ollama first
    ollama_success = setup_ollama()
    if not ollama_success:
        print("WARNING: Ollama setup failed. Will use keyword matching fallback only.")
    
    # Verify Ollama is working
    if check_ollama_running():
        print("✓ Ollama service is running and ready!")
    else:
        print("✗ Ollama service is not accessible. Using fallback method only.")
    
    print("\nInitializing RAG system...")
    
    # Create RAG system
    rag = RAGSystem()
    
    # Update these paths to your actual data files
    json_files = [
        'data/فقه_الموارث_batch_1.json',
        'data/فقه_الموارث_batch_2.json', 
        'data/فقه_الموارث_batch_3.json',
        'data/فقه_الموارث_batch_4.json'
    ]
    
    txt_files = [
        'data/Ayaat.txt',
        'data/article and other rules.txt',
        'data/inheritance.txt',
        'data/the book.txt',
        'data/articles collection.txt',
        'data/el_wagez.txt'
    ]
    
    # Load both JSON and TXT data
    rag.load_data(json_files=json_files, txt_files=txt_files)
    rag.create_embeddings()
    
    # Load test data
    # Update this path to your test CSV file
    df = pd.read_csv('data/Task1_MCQ_Test_gold_labels.csv')
    
    print(f"\nTesting with {len(df)} questions...")
    print("Using RAG + Generation method with fallback to keyword matching")
    
    predictions = []
    correct_so_far = []
    fallback_count = 0
    
    label_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    
    for idx, row in df.iterrows():
        # Get options
        options = [row[f'option{i}'] for i in range(1, 7) if pd.notna(row[f'option{i}'])]
        
        # Get prediction
        result = rag.answer_mcq_with_generation(row['question'], options)
        
        # Track fallback usage
        if result.get('fallback_used', False):
            fallback_count += 1
        
        predicted_num = int(result['predicted_answer'].replace('option', ''))
        predictions.append(predicted_num)
        
        # Accuracy tracking
        true_label_num = label_to_num[row['label']]
        correct = predicted_num == true_label_num
        correct_so_far.append(correct)
        
        # Print progress every 25 rows
        if (idx + 1) % 25 == 0:
            acc = sum(correct_so_far) / len(correct_so_far)
            print(f"Progress: {idx + 1}/{len(df)} rows | Accuracy: {acc:.3f} | Fallbacks: {fallback_count}")
    
    # Add predictions and correctness to the DataFrame
    df['predicted'] = predictions
    df['label_num'] = df['label'].map(label_to_num)
    df['correct'] = df['predicted'] == df['label_num']
    
    # Final results
    overall_acc = df['correct'].mean()
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)      
    print(f"Overall Accuracy: {overall_acc:.3f}")
    print(f"Total Questions: {len(df)}")
    print(f"Correct Answers: {df['correct'].sum()}")
    print(f"Fallback Method Used: {fallback_count}/{len(df)} times ({fallback_count/len(df)*100:.1f}%)")
    
    # Accuracy by level if available
    if 'level' in df.columns:
        print("\nAccuracy by level:")
        level_acc = df.groupby('level')['correct'].mean()
        for level, acc in level_acc.items():
            print(f"Level {level}: {acc:.3f}")
    
    print("\nFirst few predictions:")
    print(df[['question', 'label', 'predicted', 'correct']].head())
    
    # Save results
    output_file = 'results/rag_mcq_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
