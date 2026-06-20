import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_pre = pd.read_csv('ll_pre_vs_ai.csv')
df_post = pd.read_csv('ll_post_vs_ai.csv')

# getting frequency columns and sorting them in descending order
freq_pre = sorted(df_pre['Freq_Human'].values, reverse=True)
freq_pre = [f for f in freq_pre if f > 0]

freq_post = sorted(df_post['Freq_Human'].values, reverse=True)
freq_post = [f for f in freq_post if f > 0]

# AI frequencies
freq_ai = sorted(df_post['Freq_AI'].values, reverse=True)
freq_ai = [f for f in freq_ai if f > 0]

# using plt to make the graph
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(8, 6))

# making the lines with log log scale
ax.loglog(range(1, len(freq_pre) + 1), freq_pre, 
          label='Pre-LLM (Slope: -0.974)', color='#1f77b4', alpha=0.7, linewidth=2.5)

ax.loglog(range(1, len(freq_post) + 1), freq_post, 
          label='Post-LLM (Slope: -0.978)', color='#aec7e8', alpha=0.7, linewidth=2.5)

ax.loglog(range(1, len(freq_ai) + 1), freq_ai, 
          label='Gemma-4 AI (Slope: -0.901)', color='#ff7f0e', alpha=0.8, linewidth=2.5)

# titles and labels
ax.set_title("Zipf's Law Distribution across Corpora (Log-Log Scale)", fontsize=13, weight='bold', pad=15)
ax.set_xlabel('Word Rank (log scale)', fontsize=11)
ax.set_ylabel('Word Frequency (log scale)', fontsize=11)


ax.grid(True, which="both", ls="--", color='#dddddd', alpha=0.5)

ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')

# ajdusting margins and saving plot in a png file
plt.tight_layout()
plt.savefig('zipf_law_log_log_plot.png', dpi=300)
print("Zipf's Law plot generated correctly and saved into 'zipf_law_log_log_plot.png'!")