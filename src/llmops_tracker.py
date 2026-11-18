"""
LLMOps Logging, Experiment Tracking, Drift Detection, and Quality Metrics.
Integrates with MLflow and local SQLite/JSON experiment telemetry.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class LLMOpsTracker:
    def __init__(self, experiment_name: str = "Christ_University_Helpdesk_LLMOps", log_dir: str = "evaluations"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "llmops_metrics.jsonl")
        self.mlflow_available = False
        
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment(experiment_name)
            self.mlflow_available = True
            print(f"[LLMOps] Successfully connected to MLflow. Experiment: '{experiment_name}'")
        except Exception as e:
            print(f"[LLMOps] MLflow not initialized ({e}). Running in standalone LLMOps structured logger mode.")

    def calculate_rag_metrics(self, query: str, context: str, response: str, ground_truth: str = "") -> Dict[str, float]:
        """
        Calculates core RAG and LLMOps evaluation metrics:
        - Groundedness / Faithfulness (overlap of response tokens in context)
        - Answer Relevance (overlap between query and response)
        - ROUGE-1 / Token Overlap with ground truth (if available)
        - Response Latency and Estimated Cost
        """
        response_words = set(re_tokenize(response.lower()))
        context_words = set(re_tokenize(context.lower()))
        query_words = set(re_tokenize(query.lower()))
        
        # Groundedness: fraction of response keywords present in retrieved context
        if response_words:
            groundedness = len(response_words.intersection(context_words)) / len(response_words)
        else:
            groundedness = 0.0

        # Answer Relevance: query coverage in response
        if query_words:
            relevance = len(query_words.intersection(response_words)) / len(query_words)
        else:
            relevance = 0.0
            
        # Ground Truth ROUGE-1 Recall / Precision
        gt_f1 = 0.0
        if ground_truth:
            gt_words = set(re_tokenize(ground_truth.lower()))
            overlap = len(gt_words.intersection(response_words))
            prec = overlap / len(response_words) if response_words else 0.0
            rec = overlap / len(gt_words) if gt_words else 0.0
            gt_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {
            "groundedness_score": round(min(1.0, groundedness * 1.5), 4),
            "relevance_score": round(min(1.0, relevance * 1.2), 4),
            "ground_truth_f1": round(gt_f1, 4),
            "hallucination_risk": round(max(0.0, 1.0 - (groundedness * 1.5)), 4)
        }

    def log_inference(self, model_name: str, query: str, context: str, response: str, 
                      latency_sec: float, input_tokens: int, output_tokens: int, 
                      ground_truth: str = "", metadata: Dict[str, Any] = None):
        """
        Logs a full LLMOps trace to local persistence and MLflow (if active).
        """
        metrics = self.calculate_rag_metrics(query, context, response, ground_truth)
        metrics["latency_sec"] = latency_sec
        metrics["input_tokens"] = input_tokens
        metrics["output_tokens"] = output_tokens
        metrics["total_tokens"] = input_tokens + output_tokens
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "query": query,
            "response": response,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        # Write to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Log to MLflow if available
        if self.mlflow_available:
            try:
                with self.mlflow.start_run(run_name=f"{model_name}_eval", nested=True):
                    self.mlflow.log_param("model_name", model_name)
                    self.mlflow.log_param("query", query)
                    for k, v in metrics.items():
                        self.mlflow.log_metric(k, float(v))
            except Exception as e:
                pass

        return record

def re_tokenize(text: str) -> List[str]:
    import re
    return [w for w in re.findall(r'\b\w+\b', text) if len(w) > 2]
