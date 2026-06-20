import pandas as pd
import csv
import time
from openai import OpenAI
from tqdm import tqdm

# connecting to OpenRouter API 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=""  
)


MODEL_NAME = "google/gemma-4-31b-it:free"

def gemma4(system_prompt, user_prompt):
    """Function to call the OpenRouter API with a retry mechanism in case of error"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=250  
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nTemporal API error: {e}. Waiting 5 seconds...")
        time.sleep(5)
        try:
            # second try
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except:
            return "[Generation error: No response from API after retry]"

# creating a variable to store the name of the input corpus file
name_corpus_input = 'corpus_pre_llm.csv' 

print(f"Reading the original corpus: {name_corpus_input}...")
df_pre = pd.read_csv(name_corpus_input)

# shuffle the rows randomly, with a fixed seed 
df_shuffled = df_pre.sample(frac=1, random_state=42).reset_index(drop=True)

# divide into 2 halves
half_A = df_shuffled.iloc[:75].copy()
half_B = df_shuffled.iloc[75:150].copy()

corpusAI = []

# BATCH 1 TEXT CONTINUATION (first 75 abstracts)

print("\nStarting text continuation")

system_prompt_A = (
    "Ets un investigador acadèmic fent una tesi doctoral en català. "
    "El teu objectiu és continuar la redacció d'un abstract de tesi doctoral a partir "
    "del fragment inicial proporcionat. Mantingues de forma estricta el mateix estil, "
    "rigor i terminologia que s'ha utilitzat a l'inici. Respon ÚNICAMENT amb la continuació "
    "del text de l'abstract (no afegeixis introduccions com 'aquí tens la continuació' ni cometes)."
)

for idx, row in tqdm(half_A.iterrows(), total=len(half_A)):
    abstract_original = str(row['Abstract'])
    
    # taking only 50 first words to give as context to the model
    words = abstract_original.split()
    firsthalf = " ".join(words[:50])
    
    user_prompt_A = (
        f"Títol: {row['Title']}\n\n"
        f"Fragment inicial de l'abstract (primeres 50 paraules):\n"
        f"\"{firsthalf}...\"\n\n"
        f"Continua redactant l'abstract a partir d'aquí fins a completar-lo de forma lògica."
    )
    
    # calling gemma to continue the text
    continuation_ai = gemma4(system_prompt_A, user_prompt_A)
    
    # result of the continuation
    complete_abstract_A = f"{firsthalf} / {continuation_ai}"
    
    corpusAI.append({
        'ID': row.get('ID', idx),
        'Title': row['Title'],
        'Batch': 'Text continuation',
        'Generated abstract': complete_abstract_A
    })
    time.sleep(0.5)  # to avoid rate limits

# BATCH 2 TEXT GENERATION (next 75 abstracts)

print("\nStarting Text generation from title and keywords")

system_prompt_B = (
    "Ets un investigador acadèmic fent una tesi doctoral en català. A partir del títol d'una tesi doctoral "
    "i de les seves paraules clau, has de redactar un abstract complet, formal i cohesionat en català. "
    "El resum ha de tenir una extensió aproximada d'unes 100 paraules. Respon ÚNICAMENT amb el text de l'abstract "
    "formal (no afegeixis encapçalaments, títols, ni salutacions cordials)."
)

for idx, row in tqdm(half_B.iterrows(), total=len(half_B)):

    # checking if keywords are empty 
    titol = row['Title']
    keywords_originals = row['Keywords']
    if pd.isna(keywords_originals) or str(keywords_originals).strip() == "" or str(keywords_originals).lower() == "nan":
        text_keywords = "No especificades per l'autor (basa't principalment en el títol per deduir el context)."
    else:
        text_keywords = keywords_originals
    
    user_prompt_B = (
        f"Títol : {titol}\n"
        f"Paraules clau: {text_keywords}\n\n"
        f"Genera un abstract acadèmic formal d'unes 100 paraules basat en aquestes dades."
    )
    
    # calling gemma to generate the abstract from title and keywords
    complete_abstract_B = gemma4(system_prompt_B, user_prompt_B)
    
    corpusAI.append({
        'ID': row.get('ID', idx + 75),
        'Title': row['Title'],
        'Batch': 'Text generation',
        'Generated abstract': complete_abstract_B
    })
    time.sleep(0.5) # to avoid rate limits

# saving the results in a new csv file 

df_artificial = pd.DataFrame(corpusAI)
df_artificial.to_csv('corpus_gemma4.csv', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

print(f"\nSuccess. Corpus saved correctly in 'corpus_gemma4.csv'.")
print(f"Total: {len(df_artificial)} abstracts generated (75 by text continuation and 75 by text generation).")