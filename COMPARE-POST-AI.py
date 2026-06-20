import pandas as pd

df_post = pd.read_csv('frequencies_post.csv')
df_ai = pd.read_csv('frequencies_ai.csv')

# turn into sets for easier comparison
set_post = set(df_post['Word'])
set_ai = set(df_ai['Word'])

# comparison groups 
new_words = set_ai - set_post     
lost_words = set_post - set_ai  
common_words = set_post & set_ai     

print("Comparison of lexical change")

print(f"New words at top 100 (AI): {len(new_words)}")
print(f"Lost words at top 100 (AI): {len(lost_words)}")
print(f"Common words at top 100 of both corpora: {len(common_words)}\n")

# new 15 words
print("TOP 15 NEW WORDS")
df_new = df_ai[df_ai['Word'].isin(new_words)].copy()
df_new = df_new.sort_values(by='Absolute frequency', ascending=False)
for idx, row in df_new.head(15).iterrows():
    print(f"• {row['Word']} — Freq: {row['Absolute frequency']} (PMW: {row['PMW']})")

print("\nTOP 15 LOST WORDS")
df_lost = df_post[df_post['Word'].isin(lost_words)].copy()
df_lost = df_lost.sort_values(by='Absolute frequency', ascending=False)
for idx, row in df_lost.head(15).iterrows():
    print(f"• {row['Word']} — Freq: {row['Absolute frequency']} (PMW: {row['PMW']})")

# 10 words with biggest increase in PMW among the common words
print("\nTOP 10 COMMON WORDS, major increment PMW")
# join in a single table
df_post_common = df_post[df_post['Word'].isin(common_words)][['Word', 'PMW']].rename(columns={'PMW': 'PMW_POST'})
df_ai_common = df_ai[df_ai['Word'].isin(common_words)][['Word', 'PMW']].rename(columns={'PMW': 'PMW_AI'})

# merging the common words with their PMW in both corpora
df_comp = pd.merge(df_post_common, df_ai_common, on='Word', suffixes=('_POST', '_AI'))

# calculating the difference and rounding later
df_comp['Difference_PMW'] = (df_comp['PMW_AI'] - df_comp['PMW_POST']).round(1)

#  rounding every column
df_comp['PMW_POST'] = df_comp['PMW_POST'].round(1)
df_comp['PMW_AI'] = df_comp['PMW_AI'].round(1)

# top 10 overused words in the AI-generated abstracts
print("\nTop 10 common words with largest PMW increment (AI OVER-REPRESENTATION)")
df_increase = df_comp.sort_values(by='Difference_PMW', ascending=False)
for idx, row in df_increase.head(10).iterrows():
    print(f"• {row['Word']}: +{row['Difference_PMW']} PMW (De {row['PMW_POST']} a {row['PMW_AI']})")

# top 10 words underrepresented in the AI-generated abstracts
print("\nTop 10 common words with largest PMW decrement (AI UNDER-REPRESENTATION)")
df_decrease = df_comp.sort_values(by='Difference_PMW', ascending=True)
for idx, row in df_decrease.head(10).iterrows():
    print(f"• {row['Word']}: {row['Difference_PMW']} PMW (De {row['PMW_POST']} a {row['PMW_AI']})")

# saving into csv sorted by PMW increment
df_increase.to_csv('comparison_post_ai.csv', index=False, encoding='utf-8-sig')
print("\nGenerated 'comparison_post_ai.csv' sorted by PMW increment.")
