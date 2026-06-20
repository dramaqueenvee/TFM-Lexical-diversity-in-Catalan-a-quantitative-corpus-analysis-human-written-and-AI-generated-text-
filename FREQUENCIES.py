import pandas as pd
import spacy
from collections import Counter
import csv

# load spacy for catalan
try:
    nlp = spacy.load("ca_core_news_lg")
except OSError:
    print("Downloading spaCy...")
    from spacy.cli import download
    download("ca_core_news_lg")
    nlp = spacy.load("ca_core_news_lg")

custom_stopword = {"cat"}
for word in custom_stopword:
    nlp.vocab[word].is_stop = True

def analyze_corpus(route_csv, name_output):
    print(f"Analyzing {route_csv}...")
    df = pd.read_csv(route_csv)
    
    all_lemmas = []
    
    # processing abstract per abstract
    for abstract in df['Abstract'].dropna():
        doc = nlp(abstract.lower())
        
        for token in doc:
            # no punctuation and no stop words, more than 2 letters
            if token.is_alpha and not token.is_stop and len(token.lemma_) > 2:
                all_lemmas.append(token.lemma_)
                
    # count frequencies
    counting = Counter(all_lemmas)
    
    # count PMW (per million word)
    total_words = len(all_lemmas)
    print(total_words)
    
    data_frequencies = []
    for word, freq in counting.most_common(100): # 100 most common words
        pmw = (freq / total_words) * 1000000
        data_frequencies.append({
            'Word': word,
            'Absolute frequency': freq,
            'PMW': round(pmw, 2)
        })
        
    # save in csv with utf-8-sig and quoting
    df_freq = pd.DataFrame(data_frequencies)
    df_freq.to_csv(name_output, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"Frequencies saved to: {name_output}")

# executing the analysis for both corpora
if __name__ == "__main__":
 analyze_corpus('corpus_pre_llm.csv', 'frequencies_pre_llm.csv')
 analyze_corpus('corpus_post_llm.csv', 'frequencies_post_llm.csv')
 