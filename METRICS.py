import numpy as np
import pandas as pd
from collections import Counter
import re

#from raw text to tokens function
import re
import pandas as pd

# tokenization function
def text_to_tokens(text):
    if pd.isna(text):
        return []
    text_clean = text.lower()
    text_clean = re.sub(r'-\d+[\d\.]*', '', text_clean)
# using regex to get only words with letters (including accented) and ignoring punctuation, numbers, etc.
    tokens = re.findall(r'[a-zàèòíúéóç·ïüñ]+', text_clean)
    return tokens

raw_pre = pd.read_csv('corpus_pre_llm.csv')
raw_post = pd.read_csv('corpus_post_llm.csv')
raw_ai = pd.read_csv('corpus_gemma4.csv')

# each corpus as a list of tokens
tokens_pre = []
for txt in raw_pre['Abstract']: tokens_pre.extend(text_to_tokens(txt))

tokens_post = []
for txt in raw_post['Abstract']: tokens_post.extend(text_to_tokens(txt))

tokens_ai = []
for txt in raw_ai['Generated abstract']:
    if pd.isna(txt):
        continue
        
    if '/' in txt:
        # Batch 1: find slash and take text after it
        pure_ai_text = txt.split('/', 1)[1]
    else:
        # Batch 2
        pure_ai_text = txt
        
    # turn into tokens
    tokens_ai.extend(text_to_tokens(pure_ai_text))

print("Size of corpus in tokens:")
print(f"Pre-LLM Tokens: {len(tokens_pre)}")
print(f"Post-LLM Tokens: {len(tokens_post)}")
print(f"Gemma-4 AI Tokens: {len(tokens_ai)}\n")


# METRIC 1: MOVING AVERAGE TTR 

def calculate_true_mattr(token_list, window_size=50):
    if len(token_list) < window_size:
        return 1.0
    ttr_scores = []
    for i in range(len(token_list) - window_size + 1):
        window = token_list[i : i + window_size]
        ttr_scores.append(len(set(window)) / window_size)
    return round(np.mean(ttr_scores), 4)

print("Moving Average TTR")
print(f"Pre-LLM MATTR:  {calculate_true_mattr(tokens_pre)}")
print(f"Post-LLM MATTR: {calculate_true_mattr(tokens_post)}")
print(f"Gemma-4 MATTR:  {calculate_true_mattr(tokens_ai)}\n")


#  METRIC 2 ZIPF'S LAW SLOPE FROM RAW FREQUENCIES

def calculate_zipf_slope_from_raw(token_list):
    counts = Counter(token_list)
    freqs = sorted(counts.values(), reverse=True)
    
    ranks = np.arange(1, len(freqs) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    
    # linear regression (y = mx + c)
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    return round(slope, 3)

print("Zipf's Law Slope")
print(f"Pre-LLM Zipf Slope:  {calculate_zipf_slope_from_raw(tokens_pre)}")
print(f"Post-LLM Zipf Slope: {calculate_zipf_slope_from_raw(tokens_post)}")
print(f"Gemma-4 Zipf Slope:  {calculate_zipf_slope_from_raw(tokens_ai)}")
print("If the AI result has a steeper slope, ex: -1.15, it demonstrates lower lexical diversity and higher concentration)\n")


# METRIC 3: LOG-LIKELIHOOD (LL) 

def run_raw_log_likelihood(tokens_h, tokens_s, output_csv):
    counts_h = Counter(tokens_h)
    counts_s = Counter(tokens_s)
    
    all_words = set(counts_h.keys()) | set(counts_s.keys())
    N_h = len(tokens_h)
    N_s = len(tokens_s)
    
    rows = []
    for word in all_words:
        c = counts_h[word]
        d = counts_s[word]
        
        E1 = N_h * (c + d) / (N_h + N_s)
        E2 = N_s * (c + d) / (N_h + N_s)
        
        term1 = c * np.log(c / E1) if c > 0 else 0
        term2 = d * np.log(d / E2) if d > 0 else 0
        ll = 2 * (term1 + term2)
        
        if (d / N_s) < (c / N_h):
            ll = -ll
            
        rows.append({
            'Word': word, 'Freq_Human': c, 'Freq_AI': d,
            'LL_Score': round(ll, 2),
            'PMW_Human': round((c / N_h) * 1000000, 1),
            'PMW_AI': round((d / N_s) * 1000000, 1)
        })
        
    df_ll = pd.DataFrame(rows)
    df_ll.sort_values(by='LL_Score', ascending=False).to_csv(output_csv, index=False, encoding='utf-8-sig')

# executing log-likelihood comparisons
run_raw_log_likelihood(tokens_pre, tokens_ai, 'll_pre_vs_ai.csv')
run_raw_log_likelihood(tokens_post, tokens_ai, 'll_post_vs_ai.csv')

print("Finished calculating log-likelihood scores.")
