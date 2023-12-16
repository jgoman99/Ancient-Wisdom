import re
import numpy as np
import pandas as pd
import os
from create_embeddings import query
import time

def create_embeddings_df():


    # embed quotes
    if not os.path.exists("Embedded Data/quotes.csv"):
        print("Creating embeddings for quotes.txt")
        start_time = time.time()
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to create embeddings for quotes.csv: {elapsed_time} seconds")

    # embed quotes2 (much larger) 76k obs
    # Note: We need to pay attention to rate limit. We get like 50k per hour or something
    if not os.path.exists("Embedded Data/quotes2.csv"):
        start_time = time.time()
        print("Creating embeddings for quotes2.csv")
        df = pd.read_csv("Raw Data/quotes2.csv", sep = ';')

        quotes = df['QUOTE'].tolist()
        # embed
        batch_size = 100
        num_batches = len(quotes) // batch_size + 1

        embeddings = []
        batch_start_time = time.time()
        for i in range(num_batches):
            print(f"batch {i} of {num_batches}")
            batch_quotes = quotes[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = query(batch_quotes)
            embeddings.extend(batch_embeddings)
            batch_end_time = time.time()
            batch_elapsed_time = batch_end_time - batch_start_time
            print(f"Time to embed batch {i}: {batch_elapsed_time} seconds")
            batch_start_time = time.time()

        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df['author'] = df['AUTHOR']
        embeddings_df['quote'] = df['QUOTE']

        # 3. dump to csv (need to decide format, maybe jsonl?)
        embeddings_df.to_csv("Embedded Data/quotes2.csv", index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time to create embeddings for quotes2.csv: {elapsed_time} seconds")
