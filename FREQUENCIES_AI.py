import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm

# large spacy model
try:
    nlp = spacy.load("ca_core_news_lg")
except OSError:
    print("Downloading spaCy Large Model for Catalan...")
    from spacy.cli import download
    download("ca_core_news_lg")
    nlp = spacy.load("ca_core_news_lg")

#reading the generated corpus
df = pd.read_csv('corpus_gemma4.csv')

all_lemmas = []

print("Extracting AI text and processing abstract per abstract...")

# Loop through rows 
for idx, row in tqdm(df.dropna(subset=['Generated abstract']).iterrows(), total=len(df)):
    complete_abstract = str(row['Generated abstract'])
    batch = str(row['Batch'])
    
    # batch 1, only getting text after the slash
    if 'continuation' in batch.lower():
        if '/' in complete_abstract:
            abstract_text = complete_abstract.split('/', 1)[1].strip()
        else:
            abstract_text = complete_abstract.strip()
            
    # batch 2 (Text generation): Take the entire abstract text
    else:
        abstract_text = complete_abstract.strip()
        
    # applying the same text processing
    doc = nlp(abstract_text.lower())
    
    for token in doc:
        # filters: alphabetical, not a stop word, lemma longer than 2 letters
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 2:
            all_lemmas.append(token.lemma_)

# counting frequencies of lemmas
counting = Counter(all_lemmas)

# counting PMW (per million words) 
total_words = len(all_lemmas)
print(total_words)

data_frequencies = []
for word, freq in counting.most_common(100): # top 100 most common words
    #  normalization: (freq / total) * 1,000,000
    pmw = (freq / total_words) * 1000000
    data_frequencies.append({
        'Word': word,
        'Absolute frequency': freq,
        'PMW': round(pmw, 2)
    })

#new dataframe and save
df_frequencies = pd.DataFrame(data_frequencies)

print("\nTop 50 most frequent words in the AI-generated abstracts:")
print(df_frequencies.head(50).to_string(index=False))

#save into a csv
df_frequencies.to_csv('frequencies_ai.csv', index=False, encoding='utf-8-sig')
print("\nFrequencies and PMW values have been saved to 'frequencies_ai.csv'!")