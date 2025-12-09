import json
import os
import glob
import pandas as pd

DATA_DIR = "../open_ragbench_local"

def load_jsonl(filename):
    path = os.path.join(DATA_DIR, filename)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("Loading dataset from local files...")

corpus_files = glob.glob(os.path.join(DATA_DIR, "corpus", "*.json"))
print(f"Found {len(corpus_files)} corpus documents.")

corpus_data = []
for corpus_file in corpus_files:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        doc = json.load(f)
        corpus_data.append(doc)

# Change from dict to list
corpus_file_list = []  # Changed from dict to list
for corpus_file in corpus_files:
    print("Step 1: Reading corpus file...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        doc = json.load(f)
        print("Step 2: Reading corpus file...")

        title = doc.get("title")
        abstract = doc.get("abstract")
        
        sections = doc.get("sections", {})

        print(f"sections type:{type(sections)}")
        # print("Step 3: Reading corpus file...")
        for index, value in enumerate(sections):
            if type(value) == dict:
                print("================================")
                print(f"Section: {value}")
                print("================================")
                # Append each row as a new dict instead of updating
                corpus_file_list.append({
                    "title": title, 
                    "abstract": abstract, 
                    "section_id": str(value.get("section_id")), 
                    "section_text": value.get("text")
                })
                print("Step 4: Reading corpus file...")

# Convert to DataFrame
if corpus_file_list:  # Changed variable name
    df = pd.DataFrame(corpus_file_list)  # Changed variable name
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nDataFrame columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Save to CSV
    output_path = os.path.join(DATA_DIR, "corpus_data_sections.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nDataFrame saved to: {output_path}")
else:
    print("No corpus data found.")