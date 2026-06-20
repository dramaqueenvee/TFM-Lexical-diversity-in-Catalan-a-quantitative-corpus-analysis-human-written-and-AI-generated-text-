import pandas as pd
import langid
import csv
import time
import requests
from sickle import Sickle


# function to retry requests.get in case of connection errors
original_get = requests.get

def retrying_get(*args, **kwargs):
    max_retries = 5
    backoff = 3
    
    # 1 second delay to avoid issues with the server
    time.sleep(1)
    
    for i in range(max_retries):
        try:
            return original_get(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            if i == max_retries - 1:
                raise e 
            
            print(f"\nThe TDX server has lost the connection. Retrying in {backoff} seconds... (Attempt {i+1}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2 


requests.get = retrying_get
# =====================================================================

def clean_text(text):
    if not text:
        return ""
    text_net = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return " ".join(text_net.split())


def harvest_and_clean_theses(target_count=150):
    print(f"Starting data collection (Target: {target_count} per corpus)...")
    
    sickle = Sickle('https://www.tdx.cat/oai/request')
    records = sickle.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True)
    
    pre_llm_data = []
    post_llm_data = []
    
    count_pre = 0
    count_post = 0

    for record in records:
        if count_pre >= target_count and count_post >= target_count:
            break
        
        meta = record.metadata
        
        # language filter 
        languages = meta.get('language', [])
        if not languages or not any(l and l.lower() in ['cat', 'ca', 'català', 'catalan'] for l in languages):
            continue

        # date filter
        dates = meta.get('date', [])
        if not dates: 
            continue
        try:
            year = int(dates[0][:4])
        except: 
            continue

        # ensuring the abstract is in Catalan with langid
        descriptions = meta.get('description', [])
        cat_abstract = ""
        for desc in descriptions:
            if not desc or len(desc) < 200: 
                continue
            lang, _ = langid.classify(desc)
            if lang == 'ca':
                cat_abstract = clean_text(desc)
                break
        
        if not cat_abstract: 
            continue

        # clean title and keywords
        titles = meta.get('title', [])
        raw_title = titles[0] if titles else ""
        clean_title = clean_text(raw_title)
        
        keywords_list = meta.get('subject', [])
        clean_keywords = clean_text(", ".join([str(k) for k in keywords_list if k is not None]))

        # structuring the entry
        entry = {
            'Title': clean_title,
            'Year': year,
            'Abstract': cat_abstract,
            'Keywords': clean_keywords
        }

        # classification based on year and count
        if 2010 <= year <= 2020 and count_pre < target_count:
            pre_llm_data.append(entry)
            count_pre += 1
            if count_pre % 10 == 0: 
                print(f"Progress: {count_pre}/{target_count} pre-LLM theses collected.")

        elif year >= 2023 and count_post < target_count:
            post_llm_data.append(entry)
            count_post += 1
            if count_post % 10 == 0: 
                print(f"Progress: {count_post}/{target_count} post-LLM theses collected.")

    # save into csv files
    if pre_llm_data:
        df_pre = pd.DataFrame(pre_llm_data)
        df_pre.to_csv('corpus_pre_llm.csv', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"Success: 'corpus_pre_llm.csv' generated with {len(df_pre)} files.")

    if post_llm_data:
        df_post = pd.DataFrame(post_llm_data)
        df_post.to_csv('corpus_post_llm.csv', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"✅ Success: 'corpus_post_llm.csv' generated with {len(df_post)} files.")


if __name__ == "__main__":
    harvest_and_clean_theses(target_count=150)