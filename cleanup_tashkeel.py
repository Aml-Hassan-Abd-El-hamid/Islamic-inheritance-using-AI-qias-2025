import pandas as pd
import re
import string

# Define path to input/output
input_file = "data/Task1_MCQ_Test.csv"
output_file = "data/Task1_MCQ_Test_cleaned_no_tashkeel.csv"

# Arabic diacritics (tashkeel)
arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')

# Custom punctuation removal (preserve ? ( ) )
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = arabic_diacritics.sub('', text)  # Remove diacritics
    text = re.sub(r'[^\w\s\u0600-\u06FF\(\)\؟\.]', '', text)  # Remove punctuation except () and ؟ and . 


    return text

# Load the CSV
df = pd.read_csv(input_file, encoding="utf-8-sig")

# Clean all text columns (except maybe 'level' or numeric fields)
for col in df.columns:
    if df[col].dtype == object:
        print(col)
        df[col] = df[col].apply(clean_text)

# Save cleaned version
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Cleaned file saved to {output_file}")
