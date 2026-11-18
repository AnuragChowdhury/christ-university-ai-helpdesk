"""
Enhanced RAG Engine integrating the full Christ University Handbook (handbook_20260724011902.pdf).
Includes page-aware chunking, vector indexing with TF-IDF/Dense embeddings,
citation preservation, and metadata enrichment.
"""
import os
import re
import json
from typing import List, Dict, Any
import numpy as np

class DocumentChunker:
    """Chunks documents into coherent passages with page and section metadata."""
    
    @staticmethod
    def chunk_extracted_pages(pages_data: List[Dict[str, Any]], max_words: int = 250, overlap: int = 50) -> List[Dict[str, Any]]:
        chunks = []
        chunk_id = 0
        
        for p in pages_data:
            page_num = p["page"]
            section = p.get("section", "General")
            text = p["text"]
            
            # Split text by paragraphs or double newlines
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            
            current_chunk_words = []
            for para in paragraphs:
                words = para.split()
                if len(current_chunk_words) + len(words) <= max_words:
                    current_chunk_words.extend(words)
                else:
                    if current_chunk_words:
                        chunk_text = " ".join(current_chunk_words)
                        chunks.append({
                            "id": f"chunk_{chunk_id}",
                            "page": page_num,
                            "section": section,
                            "text": chunk_text,
                            "word_count": len(current_chunk_words)
                        })
                        chunk_id += 1
                        # Retain overlap
                        current_chunk_words = current_chunk_words[-overlap:] + words
                    else:
                        # Single paragraph exceeds max_words, force slice
                        for i in range(0, len(words), max_words - overlap):
                            sub_text = " ".join(words[i:i + max_words])
                            chunks.append({
                                "id": f"chunk_{chunk_id}",
                                "page": page_num,
                                "section": section,
                                "text": sub_text,
                                "word_count": len(sub_text.split())
                            })
                            chunk_id += 1
                        current_chunk_words = []
                        
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "page": page_num,
                    "section": section,
                    "text": chunk_text,
                    "word_count": len(current_chunk_words)
                })
                chunk_id += 1
                
        return chunks


class LocalVectorStore:
    """
    In-memory vector store utilizing Cosine Similarity with Scikit-Learn TF-IDF or Dense SentenceTransformer.
    """
    def __init__(self, use_dense_embeddings: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.chunks: List[Dict[str, Any]] = []
        self.use_dense = use_dense_embeddings
        self.model_name = model_name
        self.dense_model = None
        self.vectors = None
        
        if self.use_dense:
            try:
                from sentence_transformers import SentenceTransformer
                self.dense_model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"[VectorStore] Dense model load failed ({e}), using TF-IDF.")
                self.use_dense = False

        if not self.use_dense:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15000)

    def index_documents(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        corpus = [f"Section: {c.get('section', '')}. {c['text']}" for c in chunks]
        if self.use_dense and self.dense_model:
            self.vectors = self.dense_model.encode(corpus, normalize_embeddings=True)
        else:
            self.vectors = self.tfidf.fit_transform(corpus).toarray()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        
        if self.use_dense and self.dense_model:
            q_vec = self.dense_model.encode([query], normalize_embeddings=True)[0]
            scores = np.dot(self.vectors, q_vec)
        else:
            q_vec = self.tfidf.transform([query]).toarray()[0]
            norm_q = np.linalg.norm(q_vec)
            if norm_q == 0:
                scores = np.zeros(len(self.chunks))
            else:
                norms = np.linalg.norm(self.vectors, axis=1)
                norms[norms == 0] = 1e-10
                scores = np.dot(self.vectors, q_vec) / (norms * norm_q)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })
        return results
