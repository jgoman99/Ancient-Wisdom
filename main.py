import re
import numpy as np
import pandas as pd
from create_embeddings import query
with(open('Raw Data/quotes.txt', 'r', encoding='utf-8')) as f:
    quotes_text =  f.read()


# 1. Find all the quotes in the text
blocks = quotes_text.split('\n\n')
quotes, authors = zip(*[block.split('\n--') for block in blocks])

# 2. Cleaning
quotes = [re.sub(r'\n',' ',quote).strip() for quote in quotes]
authors = [author.strip() for author in authors]

# embed
embeddings = query(quotes)
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['author'] = authors
embeddings_df['quote'] = quotes

# 3. dump to csv (need to decide format, maybe jsonl?)
embeddings_df.to_csv("Embedded Data/quotes.csv", index=False)

