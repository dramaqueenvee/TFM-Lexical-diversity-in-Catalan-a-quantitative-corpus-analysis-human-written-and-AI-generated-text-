import pandas as pd

df_pre = pd.read_csv('ranking_words_pre_llm.csv')
df_post = pd.read_csv('ranking_words_post_llm.csv')

#turn into sets for easier comparison
set_pre = set(df_pre['Word (lema)'])
set_post = set(df_post['Word (lema)'])

# comparison groups 
new_words = set_post - set_pre     
lost_words = set_pre - set_post  
common_words = set_pre & set_post     

print("Comparison of lexical change")

print(f"New words at top 100 (Post-LLM): {len(new_words)}")
print(f"Lost words at top 100 (Post-LLM): {len(lost_words)}")
print(f"Common words at top 100 of both corpora: {len(common_words)}\n")

#new 15 words that entered the Post-LLM ranking
print("TOP 15 NEW WORDS (entered the post-LLM ranking)")
df_new = df_post[df_post['Word (lema)'].isin(new_words)].copy()
df_new = df_new.sort_values(by='Absolute frequency', ascending=False)
for idx, row in df_new.head(15).iterrows():
    print(f"• {row['Word (lema)']} — Freq: {row['Absolute frequency']} (PMW: {row['PMW (per million words)']})")

print("\nTOP 15 LOST WORDS (removed from the post-LLM ranking)")
df_lost = df_pre[df_pre['Word (lema)'].isin(lost_words)].copy()
df_lost = df_lost.sort_values(by='Absolute frequency', ascending=False)
for idx, row in df_lost.head(15).iterrows():
    print(f"• {row['Word (lema)']} — Freq: {row['Absolute frequency']} (PMW: {row['PMW (per million words)']})")

# 10 words with biggest increase in PMW among the common words
print("\nTOP 10 NEW WORDS, major increment PMW")
# join in a single table
df_pre_comun = df_pre[df_pre['Word (lema)'].isin(common_words)][['Word (lema)', 'PMW (per million words)']].rename(columns={'PMW (per million words)': 'PMW_PRE'})
df_post_comun = df_post[df_post['Word (lema)'].isin(common_words)][['Word (lema)', 'PMW (per million words)']].rename(columns={'PMW (per million words)': 'PMW_POST'})

df_comp = pd.merge(df_pre_comun, df_post_comun, on='Word (lema)', suffixes=('_PRE', '_POST'))
df_comp['Difference_PMW'] = df_comp['PMW_POST'] - df_comp['PMW_PRE']

df_comp['PMW_PRE'] = df_comp['PMW_PRE'].round(1)
df_comp['PMW_POST'] = df_comp['PMW_POST'].round(1)
df_comp['Difference_PMW'] = (df_comp['PMW_POST'] - df_comp['PMW_PRE']).round(1)

# sort by the biggest increase in PMW
df_increase = df_comp.sort_values(by='Difference_PMW', ascending=False)
for idx, row in df_increase.head(10).iterrows():
    print(f"• {row['Word (lema)']} Increases: +{round(row['Difference_PMW'], 1)} PMW (De {row['PMW_PRE']} a {row['PMW_POST']})")

# save into csv the common words with their PMW in both corpora and the difference
df_comp.to_csv('comparison_both_corpora.csv', index=False, encoding='utf-8-sig')
print("\n Generated 'comparison_both_corpora.csv' correctly.")
