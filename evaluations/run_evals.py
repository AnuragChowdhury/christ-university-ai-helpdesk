"""
Comparative Benchmark and Evaluation Suite using the extracted official 109-page handbook.
Executes standard Christ University queries across all 3 models:
1. Microsoft Phi-3 Mini (Generative reasoning)
2. DistilBERT (Extractive QA & verification)
3. TinyLlama-1.1B (Edge generative synthesizer)

Evaluates accuracy, latency, groundedness, relevance, and LLMOps metrics with MLflow.
"""
import os
import json
import pandas as pd
from src.rag_engine import DocumentChunker, LocalVectorStore
from src.models import Phi3MiniBackend, DistilBertRerankerBackend, TinyLlamaSynthesizerBackend
from src.llmops_tracker import LLMOpsTracker

def run_evaluation_benchmark():
    print("=" * 70)
    print("Christ University Student Helpdesk - Comparative Model Evaluation")
    print("=" * 70)

    # 1. Load Extracted PDF Pages
    pages_json_path = os.path.join("data", "christ_university_pages.json")
    if os.path.exists(pages_json_path):
        with open(pages_json_path, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
        chunks = DocumentChunker.chunk_extracted_pages(pages_data, max_words=250, overlap=50)
        print(f"[RAG] Document ingested {len(chunks)} contextual chunks from 109-page handbook PDF.")
    else:
        print("[RAG] Extracted JSON not found, please run src/pdf_extractor.py first.")
        return
    
    vector_store = LocalVectorStore(use_dense_embeddings=False)
    vector_store.index_documents(chunks)
    print("[RAG] Vector index successfully built and ready for query retrieval.")

    # 2. Instantiate the Three Models
    phi3_model = Phi3MiniBackend(load_weights=False)
    distilbert_model = DistilBertRerankerBackend()
    tinyllama_model = TinyLlamaSynthesizerBackend()
    
    models = [
        ("Phi-3-Mini (3.8B)", phi3_model),
        ("DistilBERT-Extractor", distilbert_model),
        ("TinyLlama-1.1B", tinyllama_model)
    ]

    # 3. Test Queries
    test_queries = [
        {
            "query": "What is the minimum attendance required to write the End Semester Examinations?",
            "ground_truth": "Minimum 85% aggregate attendance is mandatory for End Semester Examination. Minimum 75% for Mid Semester Examination."
        },
        {
            "query": "Where is the Christ University Kengeri campus and which faculties are located there?",
            "ground_truth": "Located at Kanmanike, Kumbalgodu, Mysore Road. Houses School of Engineering and Technology and School of Architecture."
        },
        {
            "query": "What are the timings and digital database facilities at the Central Library?",
            "ground_truth": "Knowledge Centre open Monday to Saturday 8:00 AM to 9:00 PM, Sunday 9 AM to 4 PM during exams. Remote access to IEEE, ScienceDirect, Scopus."
        },
        {
            "query": "What is the dress code rule for male and female students and ID card policy?",
            "ground_truth": "Male: formal shirt tucked in, trousers, formal shoes. Female: formal salwar kameez with pinned dupatta. Smart ID card is mandatory at all times."
        },
        {
            "query": "What is the passing criteria for postgraduate students in CIA and ESE?",
            "ground_truth": "Minimum 40% in CIA and 40% in ESE separately, with an overall aggregate passing criteria of 50% for PG courses."
        }
    ]

    tracker = LLMOpsTracker()
    results = []

    for item in test_queries:
        query = item["query"]
        ground_truth = item["ground_truth"]
        print(f"\n--- Testing Query: '{query}' ---")
        
        # Retrieve relevant chunks via RAG
        retrieved = vector_store.search(query, top_k=3)
        context_str = "\n".join([f"- [Page {r['chunk'].get('page', 'N/A')}] {r['chunk']['text']}" for r in retrieved])
        
        for name, model in models:
            res = model.generate(query, context=context_str)
            response_text = res["response"]
            latency = res["latency_sec"]
            in_tok = res["input_tokens_est"]
            out_tok = res["output_tokens_est"]
            
            trace = tracker.log_inference(
                model_name=name,
                query=query,
                context=context_str,
                response=response_text,
                latency_sec=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
                ground_truth=ground_truth,
                metadata={"top_k_chunks": len(retrieved), "source": "handbook_20260724011902.pdf"}
            )
            
            m = trace["metrics"]
            results.append({
                "Model": name,
                "Query": query[:40] + "...",
                "Latency (s)": latency,
                "Groundedness": m["groundedness_score"],
                "Relevance": m["relevance_score"],
                "F1 Score": m["ground_truth_f1"],
                "Hallucination Risk": m["hallucination_risk"],
                "Response Sample": response_text[:80] + "..."
            })
            print(f"[{name}] Done (Latency: {latency:.4f}s, Groundedness: {m['groundedness_score']}, F1: {m['ground_truth_f1']})")

    df_results = pd.DataFrame(results)
    output_csv = os.path.join("evaluations", "model_comparison_results.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\n[LLMOps Benchmark] Saved detailed results to {output_csv}")
    
    summary = df_results.groupby("Model").agg({
        "Latency (s)": "mean",
        "Groundedness": "mean",
        "Relevance": "mean",
        "F1 Score": "mean",
        "Hallucination Risk": "mean"
    }).reset_index()
    
    print("\n" + "=" * 70)
    print("FINAL MODEL BENCHMARK COMPARISON SUMMARY (OFFICIAL HANDBOOK PDF)")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    summary_path = os.path.join("evaluations", "model_summary_metrics.json")
    summary.to_json(summary_path, orient="records", indent=2)
    print(f"[LLMOps Benchmark] Summary saved to {summary_path}")

if __name__ == "__main__":
    run_evaluation_benchmark()
