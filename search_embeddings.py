# 1. load the embeddings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from create_embeddings import query

user_query = "competing with yourself is the only competition you need"
user_query_embedding = query(user_query)
user_query_embedding = np.array(user_query_embedding).reshape(1,-1)

embeddings_df = pd.read_csv("Embedded Data/quotes.csv")

# Specify columns to exclude
exclude_columns = ['author', 'quote']

# Select all columns except 'author' and 'quote'
selected_columns = embeddings_df.drop(exclude_columns, axis=1)
text_columns = embeddings_df.filter(exclude_columns, axis=1)
# Calculate cosine similarity
cosine_sim = cosine_similarity(selected_columns,user_query_embedding)

# Find the three biggest elements
three_biggest = np.argsort(cosine_sim, axis=0)[-3:].flatten()

results = text_columns.iloc[three_biggest]


for idx, result in results.iloc[::-1].iterrows():
    print(f"Similarity: {round(cosine_sim[idx][0],2)}; {result.author} said: '{result.quote}'\n\n")