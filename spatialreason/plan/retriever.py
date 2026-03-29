import time
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import re
from .utils import standardize, standardize_category, change_name, process_retrieval_ducoment


class ToolRetriever:
    def __init__(self, corpus_tsv_path = "", model_path=""):
        self.corpus_tsv_path = corpus_tsv_path
        self.model_path = model_path
        self.corpus, self.corpus2tool = self.build_retrieval_corpus()
        self.embedder = self.build_retrieval_embedder()
        self.corpus_embeddings = self.build_corpus_embeddings()
        
    def build_retrieval_corpus(self):
        print("Building corpus...")
        print(f"DEBUG: Reading corpus from: {self.corpus_tsv_path}")

        # Read with explicit string dtype to prevent pandas from inferring types incorrectly
        documents_df = pd.read_csv(self.corpus_tsv_path, sep='\t', dtype=str, keep_default_na=False)

        print(f"DEBUG: Successfully read {len(documents_df)} rows from corpus")
        corpus, corpus2tool = process_retrieval_ducoment(documents_df)
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
        return corpus, corpus2tool

    def build_retrieval_embedder(self):
        print("Building embedder...")

        # Try to use local BERT model if available
        try:
            from pathlib import Path

            # Check if model_path is already a local path
            if Path(self.model_path).exists():
                print(f"🔧 Using specified local model path: {self.model_path}")
                embedder = SentenceTransformer(self.model_path)
            else:
                # Try to use local BERT model as fallback
                local_bert_path = Path('bert-base-uncased')
                if not local_bert_path.exists():
                    local_bert_path = Path.cwd() / 'bert-base-uncased'

                if local_bert_path.exists():
                    print(f"🔧 Using local BERT model for embedder: {local_bert_path}")
                    embedder = SentenceTransformer(str(local_bert_path))
                else:
                    print(f"⚠️ Local models not found, attempting to use: {self.model_path}")
                    embedder = SentenceTransformer(self.model_path)

        except Exception as e:
            print(f"❌ Failed to initialize sentence transformer: {e}")
            print("💡 Falling back to None - retrieval will be disabled")
            embedder = None

        return embedder
    
    def build_corpus_embeddings(self):
        print("Building corpus embeddings with embedder...")
        if self.embedder is None:
            print("⚠️ Embedder not available, returning None for corpus embeddings")
            return None

        try:
            corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)
            return corpus_embeddings
        except Exception as e:
            print(f"❌ Failed to build corpus embeddings: {e}")
            return None

    def retrieving(self, query, top_k=5, excluded_tools={}):
        print(f"Retrieving top {top_k} tools for query: '{query[:50]}...'")

        if self.embedder is None or self.corpus_embeddings is None:
            print("⚠️ Embedder or corpus embeddings not available, returning empty results")
            return []

        try:
            start = time.time()
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        except Exception as e:
            print(f"❌ Failed to encode query: {e}")
            return []

        # Increase search space to ensure diversity across toolkits
        search_multiplier = max(3, top_k // 2)  # Adaptive search space
        hits = util.semantic_search(query_embedding, self.corpus_embeddings,
                                  top_k=search_multiplier*top_k, score_function=util.cos_sim)

        retrieved_tools = []
        category_counts = {}  # Track tools per category for diversity

        for rank, hit in enumerate(hits[0]):
            if len(retrieved_tools) >= top_k:
                break

            category, tool_name, api_name = self.corpus2tool[self.corpus[hit['corpus_id']]].split('\t')
            category = standardize_category(category)
            tool_name = standardize(tool_name) # standardizing
            api_name = change_name(standardize(api_name)) # standardizing

            # Check exclusions
            if category in excluded_tools:
                if tool_name in excluded_tools[category]:
                    continue

            # Promote diversity by limiting tools per category (optional)
            category_counts[category] = category_counts.get(category, 0)

            tmp_dict = {
                "category": category,
                "tool_name": tool_name,
                "api_name": api_name,
                "score": hit['score']  # Include relevance score
            }
            retrieved_tools.append(tmp_dict)
            category_counts[category] += 1

        end_time = time.time()
        print(f"Retrieved {len(retrieved_tools)} tools in {end_time - start:.3f}s")
        print(f"Category distribution: {category_counts}")
        return retrieved_tools