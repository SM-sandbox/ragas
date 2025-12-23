#!/usr/bin/env python3
"""
Gold Standard Evaluation with Checkpointing and Parallel Execution.

Runs full RAG pipeline evaluation on gold corpus.
Checkpoints every 10 questions to resume on failure.
Supports parallel execution with ThreadPoolExecutor.

Usage:
  python scripts/eval/run_gold_eval.py --test           # Run 30 questions (5 per bucket)
  python scripts/eval/run_gold_eval.py                  # Run full 458 questions
  python scripts/eval/run_gold_eval.py --precision 12   # Use precision@12 instead of @25
  python scripts/eval/run_gold_eval.py --workers 5      # Parallel with 5 workers
"""

# Suppress verbose gRPC/absl logs BEFORE importing google libraries
import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")  # 0=INFO, 1=WARNING, 2=ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TensorFlow logs if present

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from google.oauth2 import id_token
from google.auth.transport.requests import Request

sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/gRAG_v3")

from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.core.config import QueryConfig
# Import gemini_client and config loader from lib
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.clients.gemini_client import generate_for_judge, get_model_info
from lib.core.config_loader import load_config, get_generator_config, get_judge_config
from lib.core.pricing import get_model_pricing, calculate_token_cost
# Old rate limiter (kept for reference, not deleted)
# from lib.core.rate_limiter import SmartRateLimiter, get_limiter_for_model
# New Smart Throttler from stem-pipeline integration
from lib.core.smart_throttler.rate_limiter import SmartRateLimiter, get_limiter_for_model
from lib.core.auth_manager import (
    get_auth_manager, refresh_adc_credentials, check_auth_valid,
    should_refresh_token, is_auth_error, AuthError
)

# Config
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_PATH = Path(__file__).parent.parent.parent / "clients_qa_gold" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "core_eval"
CHECKPOINT_INTERVAL = 10
DEFAULT_WORKERS = 50  # Optimal for 1 Cloud Run instance (balances parallelism vs contention)

# Cloud Run config
CLOUD_RUN_URL = "https://bfai-api-ppfq5ahfsq-ue.a.run.app"
CLOUD_PROJECT_ID = "bfai-prod"
CLOUD_SERVICE = "bfai-api"
CLOUD_REGION = "us-east1"
CLOUD_ENVIRONMENT = "production"
SA_KEY_PATH = Path(__file__).parent.parent.parent / "config" / "ragas-cloud-run-invoker.json"

# Retry config - no fallback, just retry
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2  # seconds, exponential backoff


def load_corpus(test_mode: bool = False):
    """Load corpus, optionally sampling 5 from each bucket for test mode."""
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    questions = data.get("questions", data)
    
    if test_mode:
        # Sample 5 from each of 6 buckets
        by_bucket = defaultdict(list)
        for q in questions:
            key = f"{q.get('question_type')}/{q.get('difficulty')}"
            by_bucket[key].append(q)
        
        sampled = []
        for key, qs in by_bucket.items():
            sampled.extend(qs[:5])
        print(f"TEST MODE: Sampled {len(sampled)} questions (5 per bucket)")
        return sampled
    
    return questions


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        for part in text.split("```"):
            if part.strip().startswith("{"):
                text = part.strip()
                break
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i+1])
    return json.loads(text)


class GoldEvaluator:
    def __init__(
        self,
        precision_k: int = 25,
        recall_k: int = None,  # New: configurable recall@K
        workers: int = None,
        generator_reasoning: str = None,
        cloud_mode: bool = False,
        model: str = None,
        config_type: str = "experiment",
        config_path: Path = None,
        num_questions: int = None,  # For run directory naming
    ):
        # Load config
        self.config = load_config(config_type=config_type, config_path=config_path)
        self.config_type = config_type
        
        # Extract generator and judge configs
        self.generator_config = get_generator_config(self.config)
        self.judge_config = get_judge_config(self.config)
        
        # Use provided values or fall back to config
        self.precision_k = precision_k if precision_k != 25 else self.config.get("retrieval", {}).get("precision_k", 25)
        self.recall_k = recall_k if recall_k is not None else self.config.get("retrieval", {}).get("recall_k", 100)
        self.workers = workers if workers is not None else self.config.get("execution", {}).get("workers", DEFAULT_WORKERS)
        self.generator_reasoning = generator_reasoning if generator_reasoning is not None else self.generator_config.get("reasoning_effort", "low")
        self.cloud_mode = cloud_mode
        self.model = model if model is not None else self.generator_config.get("model", "gemini-3-flash-preview")
        self.num_questions = num_questions  # Will be set when run() is called if not provided
        
        # Store judge config for use in _judge_answer
        self.judge_model_name = self.judge_config.get("model", "gemini-3-flash-preview")
        self.judge_temperature = self.judge_config.get("temperature", 0.0)
        self.judge_reasoning = self.judge_config.get("reasoning_effort", "low")
        self.judge_seed = self.judge_config.get("seed", 42)
        
        # Load pricing once at init (not per-question)
        self.generator_pricing = get_model_pricing(self.model)
        self.judge_pricing = get_model_pricing(self.judge_model_name)
        
        # Initialize rate limiter for the judge model (generator uses cloud orchestrator's limiter)
        self.rate_limiter = get_limiter_for_model(self.judge_model_name)
        
        self.lock = Lock()  # Thread-safe checkpoint access
        self.run_start_time = None
        
        # Auth management for long-running jobs
        self._auth_manager = get_auth_manager()
        self._last_token_refresh = 0
        self._fallback_count = 0
        self._fallback_threshold = 0.05  # Abort if >5% questions use fallback
        
        # Log config being used
        print(f"Config: {config_type}")
        print(f"Generator: {self.model}, reasoning={self.generator_reasoning}, temp={self.generator_config.get('temperature', 0.0)}")
        print(f"Judge: {self.judge_model_name}, reasoning={self.judge_reasoning}, temp={self.judge_temperature}")
        
        # Cloud mode setup
        if cloud_mode:
            print(f"CLOUD MODE: Using Cloud Run endpoint {CLOUD_RUN_URL}")
            # Load SA credentials for Cloud Run auth (don't set globally - judge needs default creds)
            from google.oauth2 import service_account
            self._cloud_credentials = service_account.IDTokenCredentials.from_service_account_file(
                str(SA_KEY_PATH), target_audience=CLOUD_RUN_URL
            )
            self._refresh_cloud_token()
            self.index_metadata = {"job_id": JOB_ID, "mode": "cloud", "endpoint": CLOUD_RUN_URL}
            self.orchestrator = {
                "endpoint": CLOUD_RUN_URL,
                "service": CLOUD_SERVICE,
                "project_id": CLOUD_PROJECT_ID,
                "environment": CLOUD_ENVIRONMENT,
                "region": CLOUD_REGION,
            }
            self.retriever = None
            self.ranker = None
            self.generator = None
        else:
            # Initialize local components
            print("LOCAL MODE: Initializing components...")
            jobs = get_jobs_config()
            job_config = jobs.get(JOB_ID, {})
            job_config["job_id"] = JOB_ID
            
            # Store index metadata for output
            self.index_metadata = {
                "job_id": JOB_ID,
                "deployed_index_id": job_config.get("deployed_index_id", ""),
                "last_build": job_config.get("last_build", ""),
                "chunks_indexed": job_config.get("chunks_indexed", 0),
                "document_count": job_config.get("document_count", 0),
                "embedding_model": job_config.get("embedding_model", ""),
                "embedding_dimension": job_config.get("embedding_dimension", 0),
                "mode": "local",
            }
            
            self.retriever = VectorSearchRetriever(job_config)
            self.ranker = GoogleRanker(project_id=PROJECT_ID)
            self.generator = GeminiAnswerGenerator()
            self.orchestrator = None  # Local mode has no orchestrator
        
        # Log model info
        model_info = get_model_info()
        self.judge_model = model_info['model_id']
        print(f"Judge model: {model_info['model_id']} ({model_info['status']})")
        print(f"Index: {JOB_ID}")
        
        # Run directory will be created when run() is called
        self.run_dir = None
        self.checkpoint_file = None
        self.results_file = None
    
    def _update_registry(self, output: dict) -> None:
        """Auto-update the checkpoint registry after each run."""
        try:
            registry_path = self.run_dir.parent / "registry.json"
            if not registry_path.exists():
                print(f"  Registry not found at {registry_path}, skipping auto-update")
                return
            
            with open(registry_path) as f:
                registry = json.load(f)
            
            # Extract run ID from folder name
            run_id = self.run_dir.name.split("__")[0]
            
            # Check if already in registry
            existing_ids = {e["id"] for e in registry["entries"]}
            if run_id in existing_ids:
                print(f"  {run_id} already in registry, skipping")
                return
            
            # Build registry entry from output
            metrics = output.get("metrics", {})
            config = output.get("config", {})
            index = output.get("index", {})
            timestamp = output.get("timestamp", "")
            
            entry = {
                "id": run_id,
                "date": timestamp.split("T")[0] if timestamp else "unknown",
                "mode": index.get("mode", "unknown"),
                "config_summary": f"p{config.get('precision_k', 25)}-3-flash-{config.get('generator_reasoning_effort', 'low')}",
                "model": config.get("generator_model", "unknown"),
                "precision_k": config.get("precision_k", 25),
                "questions": metrics.get("total", 0),
                "pass_rate": round(metrics.get("pass_rate", 0), 3),
                "fail_rate": round(metrics.get("fail_rate", 0), 3),
                "acceptable_rate": round(metrics.get("acceptable_rate", 0), 3),
                "folder": self.run_dir.name,
                "notes": "Auto-registered"
            }
            
            # Add and sort
            registry["entries"].append(entry)
            registry["entries"].sort(key=lambda x: int(x["id"][1:]))
            
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            
            print(f"  Registry updated: {run_id} added")
            
        except Exception as e:
            print(f"  Warning: Failed to update registry: {e}")
    
    def _generate_report(self, output: dict) -> None:
        """Auto-generate a markdown report after each run."""
        try:
            # Import the report generator
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
            from generate_checkpoint_report import generate_report, CHECKPOINTS_DIR
            
            # Load baseline data
            registry_path = self.run_dir.parent / "registry.json"
            baseline_data = None
            if registry_path.exists():
                with open(registry_path) as f:
                    registry = json.load(f)
                gold_baseline = registry.get("gold_baseline")
                if gold_baseline:
                    baseline_folder = CHECKPOINTS_DIR / gold_baseline.get("folder", "")
                    baseline_results = baseline_folder / "results.json"
                    if baseline_results.exists():
                        with open(baseline_results) as f:
                            baseline_data = json.load(f)
            
            # Generate report
            report_path = generate_report(self.results_file, None, baseline_data)
            print(f"  Report generated: {report_path}")
            
        except Exception as e:
            print(f"  Warning: Failed to generate report: {e}")
    
    def _create_run_directory(self, num_questions: int) -> Path:
        """
        Create a unique run directory for this evaluation.
        
        Naming convention:
        {TYPE}{ID}__{DATE}__{MODE}__p{K}-{MODEL}__q{QUESTIONS}
        
        Examples:
        - R001__2025-12-19__L__p25-flash-3__q458  (full run, local)
        - R002__2025-12-19__C__p25-flash-3__q30   (test run, cloud)
        """
        # Determine run type prefix based on config_type
        type_prefix = {
            "checkpoint": "C",
            "experiment": "E",
        }.get(self.config_type, "E")
        
        # Get base output directory from config
        client = self.config.get("client", "BFAI")
        base_dir = Path(__file__).parent.parent.parent / "clients_eval_data" / client
        
        # Determine subdirectory based on config type
        if self.config_type == "checkpoint":
            runs_dir = base_dir / "checkpoints"
        else:
            runs_dir = base_dir / "experiments"
        
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next ID by scanning existing directories
        existing = list(runs_dir.glob(f"{type_prefix}*"))
        existing_ids = []
        for d in existing:
            try:
                # Extract ID from folder name like "R001__..."
                id_part = d.name.split("__")[0]
                if id_part.startswith(type_prefix):
                    existing_ids.append(int(id_part[1:]))
            except (ValueError, IndexError):
                pass
        next_id = max(existing_ids, default=0) + 1
        
        # Build directory name
        # Format: {TYPE}{ID}__{DATE}__{local|cloud}__p{K}-{model}-{reasoning}__q{N}
        date_str = datetime.now().strftime("%Y-%m-%d")
        mode_str = "cloud" if self.cloud_mode else "local"
        
        # Model short name (e.g., "3-flash" from "gemini-3-flash-preview")
        model_short = self.model.replace("gemini-", "").replace("-preview", "")
        reasoning_short = self.generator_reasoning.lower()
        config_str = f"p{self.precision_k}-{model_short}-{reasoning_short}"
        
        dir_name = f"{type_prefix}{next_id:03d}__{date_str}__{mode_str}__{config_str}__q{num_questions}"
        
        run_dir = runs_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRun directory: {run_dir}")
        return run_dir
    
    def _refresh_cloud_token(self):
        """Refresh the Cloud Run ID token."""
        request = Request()
        self._cloud_credentials.refresh(request)
        self._cloud_token = self._cloud_credentials.token
    
    def _warmup_orchestrator(self, num_pings: int = 3, ping_interval: float = 2.0, wait_after: float = 5.0) -> dict:
        """
        Warm up the Cloud Run orchestrator before running evaluation.
        
        Sends multiple pings to ensure the instance is warm and ready.
        Returns warmup stats for logging.
        
        Args:
            num_pings: Number of health check pings to send
            ping_interval: Seconds between pings
            wait_after: Seconds to wait after pings before starting eval
            
        Returns:
            dict with warmup stats (times, status, etc.)
        """
        import time as time_module
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print("\n" + "="*60)
        print("CLOUD ORCHESTRATOR WARMUP")
        print("="*60)
        
        warmup_stats = {
            "phase": "warmup",
            "pings": [],
            "parallel_warmup": [],
            "ramp_up": [],
            "total_warmup_time": 0,
            "orchestrator_ready": False,
            "target_workers": self.workers,
        }
        
        warmup_start = time_module.time()
        
        # Phase 1: Sequential pings to wake up first instance
        print("\n  Phase 1: Sequential pings (wake up first instance)")
        print("  " + "-"*50)
        for i in range(num_pings):
            ping_num = i + 1
            print(f"  [{ping_num}/{num_pings}] Pinging orchestrator...", end=" ", flush=True)
            
            ping_start = time_module.time()
            try:
                # Use a lightweight health check endpoint or minimal query
                response = requests.get(
                    f"{CLOUD_RUN_URL}/health",
                    headers={"Authorization": f"Bearer {self._cloud_token}"},
                    timeout=60,
                )
                ping_time = time_module.time() - ping_start
                
                if response.status_code == 200:
                    print(f"OK ({ping_time:.2f}s)")
                    warmup_stats["pings"].append({"status": "ok", "time": ping_time})
                elif response.status_code == 404:
                    # No /health endpoint, try a minimal retrieve
                    print(f"no /health endpoint, trying /retrieve...", end=" ", flush=True)
                    ping_start = time_module.time()
                    response = requests.post(
                        f"{CLOUD_RUN_URL}/retrieve",
                        headers={"Authorization": f"Bearer {self._cloud_token}", "Content-Type": "application/json"},
                        json={"query": "warmup ping", "job_id": JOB_ID, "recall_top_k": 1},
                        timeout=60,
                    )
                    ping_time = time_module.time() - ping_start
                    print(f"OK ({ping_time:.2f}s)")
                    warmup_stats["pings"].append({"status": "ok", "time": ping_time})
                else:
                    print(f"HTTP {response.status_code} ({ping_time:.2f}s)")
                    warmup_stats["pings"].append({"status": f"http_{response.status_code}", "time": ping_time})
                    
            except requests.exceptions.Timeout:
                ping_time = time_module.time() - ping_start
                print(f"TIMEOUT ({ping_time:.2f}s) - instance likely cold starting")
                warmup_stats["pings"].append({"status": "timeout", "time": ping_time})
            except Exception as e:
                ping_time = time_module.time() - ping_start
                print(f"ERROR: {e} ({ping_time:.2f}s)")
                warmup_stats["pings"].append({"status": "error", "time": ping_time, "error": str(e)})
            
            # Wait between pings (except after last one)
            if i < num_pings - 1:
                print(f"  Waiting {ping_interval}s before next ping...")
                time_module.sleep(ping_interval)
        
        # Check if orchestrator is ready (at least one successful ping)
        successful_pings = [p for p in warmup_stats["pings"] if p["status"] == "ok"]
        warmup_stats["orchestrator_ready"] = len(successful_pings) > 0
        
        if not warmup_stats["orchestrator_ready"]:
            print(f"\n  ⚠ Orchestrator may not be ready - proceeding anyway")
            warmup_stats["total_warmup_time"] = time_module.time() - warmup_start
            return warmup_stats
        
        avg_ping = sum(p["time"] for p in successful_pings) / len(successful_pings)
        print(f"\n  ✓ First instance READY (avg ping: {avg_ping:.2f}s)")
        
        # Phase 2: Parallel warmup to spin up multiple instances
        if self.workers > 10:
            print("\n  Phase 2: Parallel warmup (spin up multiple instances)")
            print("  " + "-"*50)
            
            # Send parallel requests to force Cloud Run to scale up
            parallel_count = min(self.workers // 2, 20)  # Half of workers, max 20
            print(f"  Sending {parallel_count} parallel warmup requests...")
            
            def warmup_request(idx):
                try:
                    start = time_module.time()
                    response = requests.post(
                        f"{CLOUD_RUN_URL}/retrieve",
                        headers={"Authorization": f"Bearer {self._cloud_token}", "Content-Type": "application/json"},
                        json={"query": f"warmup parallel {idx}", "job_id": JOB_ID, "recall_top_k": 1},
                        timeout=60,
                    )
                    elapsed = time_module.time() - start
                    return {"idx": idx, "status": "ok" if response.status_code == 200 else f"http_{response.status_code}", "time": elapsed}
                except Exception as e:
                    return {"idx": idx, "status": "error", "time": 0, "error": str(e)}
            
            parallel_start = time_module.time()
            with ThreadPoolExecutor(max_workers=parallel_count) as executor:
                futures = [executor.submit(warmup_request, i) for i in range(parallel_count)]
                for future in as_completed(futures):
                    result = future.result()
                    warmup_stats["parallel_warmup"].append(result)
            
            parallel_time = time_module.time() - parallel_start
            parallel_ok = sum(1 for r in warmup_stats["parallel_warmup"] if r["status"] == "ok")
            print(f"  ✓ Parallel warmup complete: {parallel_ok}/{parallel_count} OK in {parallel_time:.1f}s")
        
        # Phase 3: Ramp-up with increasing concurrency
        if self.workers > 20:
            print("\n  Phase 3: Ramp-up (gradually increase concurrency)")
            print("  " + "-"*50)
            
            # Ramp up in stages: 10 -> 25 -> 50 -> target
            ramp_stages = [10, 25, 50]
            ramp_stages = [s for s in ramp_stages if s < self.workers]
            
            for stage_workers in ramp_stages:
                print(f"  Ramp stage: {stage_workers} concurrent requests...", end=" ", flush=True)
                
                stage_start = time_module.time()
                stage_results = []
                
                with ThreadPoolExecutor(max_workers=stage_workers) as executor:
                    futures = [executor.submit(warmup_request, i) for i in range(stage_workers)]
                    for future in as_completed(futures):
                        stage_results.append(future.result())
                
                stage_time = time_module.time() - stage_start
                stage_ok = sum(1 for r in stage_results if r["status"] == "ok")
                print(f"{stage_ok}/{stage_workers} OK in {stage_time:.1f}s")
                
                warmup_stats["ramp_up"].append({
                    "workers": stage_workers,
                    "ok_count": stage_ok,
                    "time": stage_time,
                })
                
                # Brief pause between stages
                time_module.sleep(1.0)
        
        # Final wait
        print(f"\n  Waiting {wait_after}s for full warmup...")
        time_module.sleep(wait_after)
        
        warmup_stats["total_warmup_time"] = time_module.time() - warmup_start
        print(f"\n  ✓ Total warmup time: {warmup_stats['total_warmup_time']:.1f}s")
        print("="*60 + "\n")
        
        return warmup_stats
    
    def _cloud_query(self, question: str) -> dict:
        """Query the Cloud Run endpoint."""
        payload = {
            "query": question,
            "job_id": JOB_ID,
            "top_k": self.precision_k,
            "model": self.model,  # Configurable model
            "reasoning_effort": self.generator_reasoning,  # Match local reasoning
        }
        response = requests.post(
            f"{CLOUD_RUN_URL}/query",
            headers={"Authorization": f"Bearer {self._cloud_token}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 401:
            # Token expired, refresh and retry
            self._refresh_cloud_token()
            response = requests.post(
                f"{CLOUD_RUN_URL}/query",
                headers={"Authorization": f"Bearer {self._cloud_token}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
        response.raise_for_status()
        return response.json()
    
    def _cloud_retrieve(self, question: str) -> list:
        """Get raw recall candidates from Cloud Run /retrieve endpoint."""
        response = requests.post(
            f"{CLOUD_RUN_URL}/retrieve",
            headers={"Authorization": f"Bearer {self._cloud_token}", "Content-Type": "application/json"},
            json={"query": question, "job_id": JOB_ID, "recall_top_k": 100},
            timeout=60,
        )
        if response.status_code == 401:
            self._refresh_cloud_token()
            response = requests.post(
                f"{CLOUD_RUN_URL}/retrieve",
                headers={"Authorization": f"Bearer {self._cloud_token}", "Content-Type": "application/json"},
                json={"query": question, "job_id": JOB_ID, "recall_top_k": 100},
                timeout=60,
            )
        response.raise_for_status()
        return response.json().get("chunks", [])
    
    def _judge_answer(self, question: str, ground_truth: str, answer: str, context: str) -> dict:
        """Judge answer quality using Gemini 3 Flash with structured JSON output.
        
        Returns dict with:
            - judgment: the scores and verdict
            - tokens: token usage for the judge call
            - metadata: model info
        """
        prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {answer}

Context (first 2000 chars): {context[:2000]}

Score 1-5 for each (5=best):
1. correctness - factually correct vs ground truth?
2. completeness - covers key points?
3. faithfulness - faithful to context, no hallucinations?
4. relevance - relevant to question?
5. clarity - clear and well-structured?

Respond with JSON containing: correctness, completeness, faithfulness, relevance, clarity, overall_score (all 1-5), and verdict (pass|partial|fail)."""
        
        try:
            # Acquire rate limiter capacity before making judge call
            # Estimate ~1000 tokens per judge call (prompt + response)
            self.rate_limiter.acquire_sync(estimated_tokens=1000)
            
            try:
                # Use gemini_client with config from loaded config file, return metadata for token tracking
                result = generate_for_judge(
                    prompt,
                    model=self.judge_model_name,
                    temperature=self.judge_temperature,
                    reasoning_effort=self.judge_reasoning,
                    seed=self.judge_seed,
                    return_metadata=True,
                )
                return result
            finally:
                # Always release the semaphore after the API call
                self.rate_limiter.release()
        except Exception as e:
            # Check if this is an auth error - raise instead of fallback
            if is_auth_error(e):
                raise AuthError(f"Authentication failed during judge call: {e}")
            
            # For non-auth errors, track fallback and return partial scores
            with self.lock:
                self._fallback_count += 1
            
            return {
                "judgment": {"correctness": 3, "completeness": 3, "faithfulness": 3, "relevance": 3, "clarity": 3, "overall_score": 3, "verdict": "partial", "parse_error": str(e)},
                "tokens": {"prompt": 0, "completion": 0, "thinking": 0, "total": 0, "cached": 0},
                "metadata": {"fallback": True},
            }
    
    def run_single_attempt(self, q: dict) -> dict:
        """Run single question through pipeline (one attempt, no retry)."""
        question = q.get("question", "")
        source_filenames = q.get("source_filenames", [])
        expected_source = source_filenames[0] if source_filenames else q.get("source_document", "")
        expected_source = expected_source.replace(".pdf", "").lower()
        ground_truth = q.get("ground_truth_answer", q.get("answer", ""))
        
        start = time.time()
        
        if self.cloud_mode:
            # Cloud mode: hit Cloud Run endpoints
            # 1. Get raw recall candidates for apples-to-apples comparison
            retrieve_start = time.time()
            raw_chunks = self._cloud_retrieve(question)
            retrieve_time = time.time() - retrieve_start
            
            # Calculate recall/MRR from raw 100 candidates (same as local)
            retrieved_docs = [c.get("doc_name", "").lower() for c in raw_chunks]
            recall_hit = any(expected_source in d for d in retrieved_docs)
            mrr = 0
            for rank, d in enumerate(retrieved_docs, 1):
                if expected_source in d:
                    mrr = 1.0 / rank
                    break
            
            # 2. Get answer from /query endpoint
            query_start = time.time()
            cloud_result = self._cloud_query(question)
            query_time = time.time() - query_start
            
            answer = cloud_result.get("answer", "")
            sources = cloud_result.get("sources", [])
            context = "\n\n".join([f"[{i+1}] {s.get('snippet', '')}" for i, s in enumerate(sources)])
            
            # Judge (returns judgment, tokens, and metadata)
            judge_start = time.time()
            judge_result = self._judge_answer(question, ground_truth, answer, context)
            judge_time = time.time() - judge_start
            
            total_time = time.time() - start
            
            # Extract generator tokens from cloud response metadata
            cloud_meta = cloud_result.get("metadata", {})
            generator_tokens = {
                "prompt": cloud_meta.get("prompt_tokens", 0) or 0,
                "completion": cloud_meta.get("completion_tokens", 0) or 0,
                "thinking": cloud_meta.get("thinking_tokens", 0) or 0,
                "total": cloud_meta.get("total_tokens", 0) or 0,
                "cached": cloud_meta.get("cached_content_tokens", 0) or 0,
            }
            judge_tokens = judge_result.get("tokens", {"prompt": 0, "completion": 0, "thinking": 0, "total": 0, "cached": 0})
            
            # Calculate cost estimate
            gen_cost = calculate_token_cost(generator_tokens, self.generator_pricing)
            judge_cost = calculate_token_cost(judge_tokens, self.judge_pricing)
            cost_estimate = {
                "generator": gen_cost,
                "judge": judge_cost,
                "total": round(gen_cost + judge_cost, 8),
            }
            
            return {
                "question_id": q.get("question_id", ""),
                "question_type": q.get("question_type", ""),
                "difficulty": q.get("difficulty", ""),
                "question_text": question,
                "expected_answer": ground_truth,
                "source_documents": source_filenames,
                "generated_answer": answer,
                "recall_hit": recall_hit,
                "mrr": mrr,
                "judgment": judge_result.get("judgment", judge_result),
                "time": total_time,
                "timing": {
                    "cloud_retrieve": retrieve_time,
                    "cloud_query": query_time,
                    "judge": judge_time,
                    "total": total_time,
                },
                "generator_tokens": generator_tokens,
                "judge_tokens": judge_tokens,
                "cost_estimate_usd": cost_estimate,
                "llm_metadata": {
                    "model": cloud_meta.get("model", "cloud"),
                    "mode": "cloud",
                    "has_citations": cloud_meta.get("has_citations", False),
                    "reasoning_effort": cloud_meta.get("reasoning_effort", "low"),
                },
                "orchestrator": self.orchestrator,
                "answer_length": len(answer),
                "retrieval_candidates": len(raw_chunks),
            }
        
        # Local mode: use gRAG_v3 pipeline directly
        # Store retrieval config for output
        retrieval_config = {
            "recall_top_k": self.recall_k,
            "precision_top_n": self.precision_k,
            "enable_hybrid": True,
            "enable_reranking": True,
            "rrf_alpha": 0.5,  # 50/50 hybrid weighting
            "ranking_model": "semantic-ranker-512@latest",
        }
        config = QueryConfig(
            recall_top_k=retrieval_config["recall_top_k"],
            precision_top_n=retrieval_config["precision_top_n"],
            enable_hybrid=retrieval_config["enable_hybrid"],
            enable_reranking=retrieval_config["enable_reranking"],
            job_id=JOB_ID,
            reasoning_effort=self.generator_reasoning,
        )
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(question, config)
        chunks = list(retrieval_result.chunks)
        retrieval_time = time.time() - retrieval_start
        
        # Check recall@100
        retrieved_docs = [c.doc_name.lower() if c.doc_name else "" for c in chunks]
        recall_hit = any(expected_source in d for d in retrieved_docs)
        
        # MRR
        mrr = 0
        for rank, d in enumerate(retrieved_docs, 1):
            if expected_source in d:
                mrr = 1.0 / rank
                break
        
        # Rerank
        rerank_start = time.time()
        reranked = self.ranker.rank(question, chunks, self.precision_k)
        rerank_time = time.time() - rerank_start
        
        # Generate
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(reranked)])
        gen_start = time.time()
        gen_result = self.generator.generate(query=question, context=context, config=config)
        answer = gen_result.answer_text
        gen_time = time.time() - gen_start
        
        # Judge (returns judgment, tokens, and metadata)
        judge_start = time.time()
        judge_result = self._judge_answer(question, ground_truth, answer, context)
        judge_time = time.time() - judge_start
        
        total_time = time.time() - start
        
        # Extract generator tokens
        generator_tokens = {
            "prompt": gen_result.prompt_tokens,
            "completion": gen_result.completion_tokens,
            "thinking": gen_result.thinking_tokens,
            "total": gen_result.total_tokens,
            "cached": gen_result.cached_content_tokens,
        }
        
        # Extract judge tokens from result
        judge_tokens = judge_result.get("tokens", {"prompt": 0, "completion": 0, "thinking": 0, "total": 0, "cached": 0})
        
        # Calculate cost estimate using pre-loaded pricing (no per-question lookup)
        gen_cost = calculate_token_cost(generator_tokens, self.generator_pricing)
        judge_cost = calculate_token_cost(judge_tokens, self.judge_pricing)
        cost_estimate = {
            "generator": gen_cost,
            "judge": judge_cost,
            "total": round(gen_cost + judge_cost, 8),
        }
        
        # Build comprehensive per-question result with all metadata
        return {
            # Identity
            "question_id": q.get("question_id", ""),
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            
            # Input (for debugging)
            "question_text": question,
            "expected_answer": ground_truth,
            "source_documents": source_filenames,
            
            # Output
            "generated_answer": answer,
            "retrieved_doc_names": retrieved_docs[:self.precision_k],  # Top-k after rerank
            "context_char_count": len(context),
            
            # Evaluation
            "recall_hit": recall_hit,
            "mrr": mrr,
            "judgment": judge_result.get("judgment", judge_result),  # Handle both old and new format
            
            # Performance
            "time": total_time,
            "timing": {
                "retrieval": retrieval_time,
                "rerank": rerank_time,
                "generation": gen_time,
                "judge": judge_time,
                "total": total_time,
            },
            
            # Token usage (separate for generator and judge)
            "generator_tokens": generator_tokens,
            "judge_tokens": judge_tokens,
            
            # Cost estimate (USD)
            "cost_estimate_usd": cost_estimate,
            
            # Generator LLM metadata
            "llm_metadata": {
                "model": gen_result.model,
                "model_version": gen_result.model_version,
                "finish_reason": gen_result.finish_reason,
                "reasoning_effort": gen_result.reasoning_effort,
                "used_fallback": gen_result.used_fallback,
                "avg_logprobs": gen_result.avg_logprobs,
                "response_id": gen_result.response_id,
                "temperature": gen_result.temperature,
                "has_citations": gen_result.has_citations,
            },
            
            # Index metadata
            "index": {
                "job_id": JOB_ID,
                "deployed_index_id": self.index_metadata.get("deployed_index_id", ""),
                "embedding_model": self.index_metadata.get("embedding_model", ""),
                "embedding_dimension": self.index_metadata.get("embedding_dimension", 0),
            },
            
            # Retrieval config
            "retrieval_config": retrieval_config,
            
            # Generator config (note: seed not passed to gRAG_v3, uses temp=0.0 for reproducibility)
            "generator_config": {
                "model": self.model,
                "reasoning_effort": self.generator_reasoning,
                "temperature": self.generator_config.get("temperature", 0.0),
                "max_output_tokens": self.generator_config.get("max_output_tokens", 8192),
            },
            
            # Judge config
            "judge_config": {
                "model": self.judge_model_name,
                "reasoning_effort": self.judge_reasoning,
                "temperature": self.judge_temperature,
                "seed": self.judge_seed,
            },
            
            # Counts
            "answer_length": len(answer),
            "retrieval_candidates": len(chunks),
        }
    
    def run_single(self, q: dict) -> dict:
        """Run single question with retry logic (no fallback)."""
        qid = q.get("question_id", "unknown")
        last_error = None
        error_phase = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self.run_single_attempt(q)
                # Success - add retry metadata
                result["retry_info"] = {
                    "attempts": attempt,
                    "recovered": attempt > 1,
                    "error": None,
                }
                return result
                
            except Exception as e:
                last_error = str(e)
                error_phase = self._detect_error_phase(e)
                
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(delay)
                    continue
        
        # All retries exhausted - return error result
        return {
            "question_id": qid,
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "error": last_error,
            "error_phase": error_phase,
            "retry_info": {
                "attempts": MAX_RETRIES,
                "recovered": False,
                "error": last_error,
            },
        }
    
    def _detect_error_phase(self, error: Exception) -> str:
        """Detect which phase caused the error based on error message."""
        error_str = str(error).lower()
        if "retriev" in error_str or "vector" in error_str or "index" in error_str:
            return "retrieval"
        elif "rerank" in error_str or "rank" in error_str:
            return "rerank"
        elif "generat" in error_str or "gemini" in error_str or "model" in error_str:
            return "generation"
        elif "judge" in error_str or "evaluat" in error_str:
            return "judge"
        return "unknown"
    
    def _check_fallback_threshold(self, total_processed: int) -> None:
        """Check if fallback rate exceeds threshold and abort if so."""
        if total_processed < 10:
            return  # Don't check until we have enough samples
        
        fallback_rate = self._fallback_count / total_processed
        if fallback_rate > self._fallback_threshold:
            raise AuthError(
                f"Aborting: Fallback rate ({fallback_rate:.1%}) exceeds threshold ({self._fallback_threshold:.1%}). "
                f"{self._fallback_count} of {total_processed} questions used fallback scores. "
                "This indicates systematic failures - check auth and API status."
            )
    
    def _maybe_refresh_token(self) -> None:
        """Refresh ADC token if needed (every 45 minutes)."""
        if should_refresh_token():
            try:
                refresh_adc_credentials()
                self._last_token_refresh = time.time()
                print("  >> ADC token refreshed")
            except Exception as e:
                print(f"  >> WARNING: Token refresh failed: {e}")
    
    def run(self, questions: list):
        """Run evaluation with checkpointing and optional parallelism."""
        self.run_start_time = time.time()
        num_questions = len(questions)
        
        # PRE-FLIGHT: Validate auth before starting
        print("\n" + "="*60)
        print("PRE-FLIGHT AUTH CHECK")
        print("="*60)
        try:
            check_auth_valid()
            self._last_token_refresh = time.time()
            print("✓ ADC credentials validated and refreshed")
        except AuthError as e:
            print(f"✗ Auth check failed: {e}")
            print("\nRun: gcloud auth application-default login")
            raise
        
        # Reset fallback counter
        self._fallback_count = 0
        
        # Create unique run directory for this evaluation
        self.run_dir = self._create_run_directory(num_questions)
        self.checkpoint_file = self.run_dir / "checkpoint.json"
        self.results_file = self.run_dir / "results.json"
        
        # Warm up orchestrator in cloud mode before starting parallel execution
        warmup_stats = None
        if self.cloud_mode and self.workers > 1:
            warmup_stats = self._warmup_orchestrator(num_pings=3, ping_interval=2.0, wait_after=5.0)
        
        print(f"\n{'='*60}")
        print(f"GOLD STANDARD EVAL - Precision@{self.precision_k}")
        print(f"Questions: {num_questions}")
        print(f"Workers: {self.workers}")
        if warmup_stats:
            print(f"Warmup: {warmup_stats['total_warmup_time']:.1f}s ({'ready' if warmup_stats['orchestrator_ready'] else 'not ready'})")
        print(f"{'='*60}\n")
        
        # Load checkpoint (fresh directory = no checkpoint)
        completed = {}
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                for r in json.load(f):
                    completed[r["question_id"]] = r
            print(f"Resuming from checkpoint: {len(completed)} done")
        
        # Filter out already completed
        pending = [q for q in questions if q.get("question_id", "") not in completed]
        print(f"Pending: {len(pending)} questions")
        
        results = list(completed.values())
        processed = len(completed)
        
        if self.workers == 1:
            # Sequential mode
            for i, q in enumerate(pending):
                qid = q.get("question_id", f"q_{processed + i}")
                try:
                    # Mid-run token refresh check
                    self._maybe_refresh_token()
                    
                    result = self.run_single(q)
                    results.append(result)
                    completed[qid] = result
                    verdict = result.get("judgment", {}).get("verdict", "?")
                    result_time = result.get("time", 0)
                    print(f"[{processed + i + 1}/{len(questions)}] {qid}: {verdict} ({result_time:.1f}s)")
                    
                    # Check fallback threshold
                    self._check_fallback_threshold(processed + i + 1)
                    
                    if (processed + i + 1) % CHECKPOINT_INTERVAL == 0:
                        with open(self.checkpoint_file, 'w') as f:
                            json.dump(list(completed.values()), f, indent=2)
                        print(f"  >> Checkpoint saved ({len(completed)} questions)")
                except AuthError as e:
                    # Auth errors are fatal - abort the run
                    print(f"\n{'='*60}")
                    print(f"FATAL: Authentication error - aborting run")
                    print(f"Error: {e}")
                    print(f"{'='*60}\n")
                    raise
                except Exception as e:
                    print(f"[{processed + i + 1}/{len(questions)}] {qid}: ERROR - {e}")
                    results.append({"question_id": qid, "error": str(e)})
        else:
            # Parallel mode with ThreadPoolExecutor
            print(f"Starting parallel execution with {self.workers} workers...")
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.run_single, q): q for q in pending}
                
                for future in as_completed(futures):
                    q = futures[future]
                    qid = q.get("question_id", "unknown")
                    try:
                        result = future.result(timeout=120)
                        with self.lock:
                            results.append(result)
                            completed[qid] = result
                            processed += 1
                            verdict = result.get("judgment", {}).get("verdict", "?")
                            result_time = result.get("time", 0)
                            print(f"[{processed}/{len(questions)}] {qid}: {verdict} ({result_time:.1f}s)")
                            
                            # Mid-run token refresh (thread-safe via auth_manager)
                            self._maybe_refresh_token()
                            
                            # Check fallback threshold
                            self._check_fallback_threshold(processed)
                            
                            if processed % CHECKPOINT_INTERVAL == 0:
                                with open(self.checkpoint_file, 'w') as f:
                                    json.dump(list(completed.values()), f, indent=2)
                                print(f"  >> Checkpoint saved ({len(completed)} questions)")
                    except AuthError as e:
                        # Auth errors are fatal - abort the run
                        print(f"\n{'='*60}")
                        print(f"FATAL: Authentication error - aborting run")
                        print(f"Error: {e}")
                        print(f"{'='*60}\n")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        raise
                    except Exception as e:
                        print(f"[?/{len(questions)}] {qid}: ERROR - {e}")
                        with self.lock:
                            results.append({"question_id": qid, "error": str(e)})
        
        # Final save
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(completed.values()), f, indent=2)
        
        # Calculate run duration
        run_duration = time.time() - self.run_start_time if self.run_start_time else 0
        
        # Calculate metrics
        valid = [r for r in results if "judgment" in r]
        
        # Verdict counts
        pass_count = sum(1 for r in valid if r.get("judgment", {}).get("verdict") == "pass")
        partial_count = sum(1 for r in valid if r.get("judgment", {}).get("verdict") == "partial")
        fail_count = sum(1 for r in valid if r.get("judgment", {}).get("verdict") == "fail")
        
        metrics = {
            "precision_k": self.precision_k,
            "total": len(questions),
            "completed": len(valid),
            "recall_at_100": sum(1 for r in valid if r.get("recall_hit")) / len(valid) if valid else 0,
            "mrr": sum(r.get("mrr", 0) for r in valid) / len(valid) if valid else 0,
            "pass_rate": pass_count / len(valid) if valid else 0,
            "partial_rate": partial_count / len(valid) if valid else 0,
            "fail_rate": fail_count / len(valid) if valid else 0,
            "acceptable_rate": (pass_count + partial_count) / len(valid) if valid else 0,
        }
        
        # Dimension averages
        for dim in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]:
            scores = [r.get("judgment", {}).get(dim, 0) for r in valid if r.get("judgment", {}).get(dim)]
            metrics[f"{dim}_avg"] = sum(scores) / len(scores) if scores else 0
        
        # Aggregate tokens
        token_totals = {
            "prompt_total": sum(r.get("tokens", {}).get("prompt", 0) for r in valid),
            "completion_total": sum(r.get("tokens", {}).get("completion", 0) for r in valid),
            "thinking_total": sum(r.get("tokens", {}).get("thinking", 0) for r in valid),
            "cached_total": sum(r.get("tokens", {}).get("cached", 0) for r in valid),
        }
        token_totals["total"] = token_totals["prompt_total"] + token_totals["completion_total"] + token_totals["thinking_total"]
        
        # Aggregate latency
        latency = {
            "total_avg_s": sum(r.get("time", 0) for r in valid) / len(valid) if valid else 0,
            "total_min_s": min((r.get("time", 999) for r in valid), default=0),
            "total_max_s": max((r.get("time", 0) for r in valid), default=0),
            "by_phase": {
                "retrieval_avg_s": sum(r.get("timing", {}).get("retrieval", 0) for r in valid) / len(valid) if valid else 0,
                "rerank_avg_s": sum(r.get("timing", {}).get("rerank", 0) for r in valid) / len(valid) if valid else 0,
                "generation_avg_s": sum(r.get("timing", {}).get("generation", 0) for r in valid) / len(valid) if valid else 0,
                "judge_avg_s": sum(r.get("timing", {}).get("judge", 0) for r in valid) / len(valid) if valid else 0,
            },
        }
        
        # Aggregate answer stats
        answer_lengths = [r.get("answer_length", 0) for r in valid if r.get("answer_length")]
        answer_stats = {
            "avg_length_chars": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            "min_length_chars": min(answer_lengths) if answer_lengths else 0,
            "max_length_chars": max(answer_lengths) if answer_lengths else 0,
        }
        
        # Finish reason distribution
        finish_reasons = {}
        for r in valid:
            reason = r.get("llm_metadata", {}).get("finish_reason", "UNKNOWN")
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        
        # Get generator model from first result
        generator_model = valid[0].get("llm_metadata", {}).get("model", "unknown") if valid else "unknown"
        
        # Aggregate retry stats
        retry_stats = {
            "total_questions": len(questions),
            "succeeded_first_try": sum(1 for r in results if r.get("retry_info", {}).get("attempts", 1) == 1 and not r.get("error")),
            "succeeded_after_retry": sum(1 for r in results if r.get("retry_info", {}).get("recovered", False)),
            "failed_all_retries": sum(1 for r in results if r.get("error") and r.get("retry_info", {}).get("attempts", 0) >= MAX_RETRIES),
            "total_retry_attempts": sum(r.get("retry_info", {}).get("attempts", 1) for r in results),
        }
        retry_stats["avg_attempts"] = retry_stats["total_retry_attempts"] / len(results) if results else 1.0
        
        # Aggregate errors by phase
        error_results = [r for r in results if r.get("error")]
        errors = {
            "total_errors": len(error_results),
            "by_phase": {
                "retrieval": sum(1 for r in error_results if r.get("error_phase") == "retrieval"),
                "rerank": sum(1 for r in error_results if r.get("error_phase") == "rerank"),
                "generation": sum(1 for r in error_results if r.get("error_phase") == "generation"),
                "judge": sum(1 for r in error_results if r.get("error_phase") == "judge"),
            },
            "error_messages": [r.get("error", "") for r in error_results[:10]],  # First 10 errors
        }
        
        # Skipped questions (those with errors that couldn't be recovered)
        skipped = {
            "count": len(error_results),
            "reasons": {
                "missing_ground_truth": 0,  # Would be caught earlier
                "invalid_question": 0,
                "timeout": sum(1 for r in error_results if "timeout" in r.get("error", "").lower()),
            },
            "question_ids": [r.get("question_id", "") for r in error_results],
        }
        
        # Breakdown by question type
        breakdown_by_type = {}
        for qtype in ["single_hop", "multi_hop"]:
            type_results = [r for r in valid if r.get("question_type") == qtype]
            if type_results:
                type_pass = sum(1 for r in type_results if r.get("judgment", {}).get("verdict") == "pass")
                type_partial = sum(1 for r in type_results if r.get("judgment", {}).get("verdict") == "partial")
                type_fail = sum(1 for r in type_results if r.get("judgment", {}).get("verdict") == "fail")
                breakdown_by_type[qtype] = {
                    "total": len(type_results),
                    "pass": type_pass,
                    "partial": type_partial,
                    "fail": type_fail,
                    "pass_rate": type_pass / len(type_results) if type_results else 0,
                }
        
        # Breakdown by difficulty
        breakdown_by_difficulty = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in valid if r.get("difficulty") == diff]
            if diff_results:
                diff_pass = sum(1 for r in diff_results if r.get("judgment", {}).get("verdict") == "pass")
                diff_partial = sum(1 for r in diff_results if r.get("judgment", {}).get("verdict") == "partial")
                diff_fail = sum(1 for r in diff_results if r.get("judgment", {}).get("verdict") == "fail")
                breakdown_by_difficulty[diff] = {
                    "total": len(diff_results),
                    "pass": diff_pass,
                    "partial": diff_partial,
                    "fail": diff_fail,
                    "pass_rate": diff_pass / len(diff_results) if diff_results else 0,
                }
        
        # 6-bucket breakdown: type x difficulty with full metrics including all dimensions
        quality_by_bucket = {}
        for qtype in ["single_hop", "multi_hop"]:
            quality_by_bucket[qtype] = {}
            for diff in ["easy", "medium", "hard"]:
                bucket_results = [r for r in valid if r.get("question_type") == qtype and r.get("difficulty") == diff]
                if bucket_results:
                    bucket_pass = sum(1 for r in bucket_results if r.get("judgment", {}).get("verdict") == "pass")
                    bucket_partial = sum(1 for r in bucket_results if r.get("judgment", {}).get("verdict") == "partial")
                    bucket_fail = sum(1 for r in bucket_results if r.get("judgment", {}).get("verdict") == "fail")
                    bucket_recall = sum(1 for r in bucket_results if r.get("recall_hit")) / len(bucket_results)
                    bucket_mrr = sum(r.get("mrr", 0) for r in bucket_results) / len(bucket_results)
                    
                    # Calculate all dimension score averages
                    bucket_overall = [r.get("judgment", {}).get("overall_score", 0) for r in bucket_results]
                    bucket_correctness = [r.get("judgment", {}).get("correctness", 0) for r in bucket_results]
                    bucket_completeness = [r.get("judgment", {}).get("completeness", 0) for r in bucket_results]
                    bucket_faithfulness = [r.get("judgment", {}).get("faithfulness", 0) for r in bucket_results]
                    bucket_relevance = [r.get("judgment", {}).get("relevance", 0) for r in bucket_results]
                    bucket_clarity = [r.get("judgment", {}).get("clarity", 0) for r in bucket_results]
                    
                    n = len(bucket_results)
                    quality_by_bucket[qtype][diff] = {
                        "count": n,
                        "pass_rate": bucket_pass / n,
                        "partial_rate": bucket_partial / n,
                        "fail_rate": bucket_fail / n,
                        "recall_at_100": bucket_recall,
                        "mrr": bucket_mrr,
                        "overall_score_avg": sum(bucket_overall) / n if bucket_overall else 0,
                        "correctness_avg": sum(bucket_correctness) / n if bucket_correctness else 0,
                        "completeness_avg": sum(bucket_completeness) / n if bucket_completeness else 0,
                        "faithfulness_avg": sum(bucket_faithfulness) / n if bucket_faithfulness else 0,
                        "relevance_avg": sum(bucket_relevance) / n if bucket_relevance else 0,
                        "clarity_avg": sum(bucket_clarity) / n if bucket_clarity else 0,
                    }
        
        # Build comprehensive output
        output = {
            "schema_version": "1.1",
            "timestamp": datetime.now().isoformat(),
            "client": "BFAI",
            "index": self.index_metadata,
            "config": {
                "generator_model": generator_model,
                "generator_reasoning_effort": self.generator_reasoning,
                "generator_seed": self.generator_config.get("seed", 42),
                "judge_model": self.judge_model,
                "judge_reasoning_effort": self.judge_reasoning,
                "judge_seed": self.judge_seed,
                "precision_k": self.precision_k,
                "recall_k": self.recall_k,
                "workers": self.workers,
                "temperature": self.generator_config.get("temperature", 0.0),
            },
            "metrics": metrics,
            "latency": latency,
            "tokens": token_totals,
            "answer_stats": answer_stats,
            "quality": {
                "finish_reason_distribution": finish_reasons,
                "fallback_rate": sum(1 for r in valid if r.get("llm_metadata", {}).get("used_fallback")) / len(valid) if valid else 0,
            },
            "execution": {
                "run_duration_seconds": run_duration,
                "questions_per_second": len(valid) / run_duration if run_duration > 0 else 0,
                "workers": self.workers,
                "rate_limiter": self.rate_limiter.get_usage(),
            },
            "retry_stats": retry_stats,
            "errors": errors,
            "skipped": skipped,
            "breakdown_by_type": breakdown_by_type,
            "breakdown_by_difficulty": breakdown_by_difficulty,
            "quality_by_bucket": quality_by_bucket,
            "orchestrator": self.orchestrator,
            "warmup": warmup_stats,
            "cost_summary_usd": {
                "generator_total": sum(r.get("cost_estimate_usd", {}).get("generator", 0) for r in valid),
                "judge_total": sum(r.get("cost_estimate_usd", {}).get("judge", 0) for r in valid),
                "total": sum(r.get("cost_estimate_usd", {}).get("total", 0) for r in valid),
                "avg_per_question": sum(r.get("cost_estimate_usd", {}).get("total", 0) for r in valid) / len(valid) if valid else 0,
            },
            "results": results,
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Auto-update registry
        self._update_registry(output)
        
        # Auto-generate report
        self._generate_report(output)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS - Precision@{self.precision_k}")
        print(f"{'='*60}")
        print(f"Completed: {metrics['completed']}/{metrics['total']}")
        print(f"Recall@100: {metrics['recall_at_100']:.1%}")
        print(f"MRR: {metrics['mrr']:.3f}")
        print(f"Pass Rate: {metrics['pass_rate']:.1%}")
        print(f"Partial Rate: {metrics['partial_rate']:.1%}")
        print(f"Fail Rate: {metrics['fail_rate']:.1%}")
        print(f"Overall Score: {metrics.get('overall_score_avg', 0):.2f}/5")
        print(f"\nTokens: {token_totals['total']:,} total ({token_totals['prompt_total']:,} in, {token_totals['completion_total']:,} out)")
        print(f"Latency: {latency['total_avg_s']:.2f}s avg ({run_duration:.1f}s total)")
        print(f"\nResults saved: {self.results_file}")
        print(f"{'='*60}")
        
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode: 30 questions")
    parser.add_argument("--precision", type=int, default=25, help="Precision@K (default: 25)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}, use 1 for sequential)")
    parser.add_argument("--quick", type=int, default=0, help="Quick test: run N questions only")
    parser.add_argument("--generator-reasoning", type=str, default="low", choices=["low", "high"],
                        help="Reasoning effort for generator: low (fast) or high (more thinking)")
    parser.add_argument("--cloud", action="store_true", 
                        help="Cloud mode: hit Cloud Run endpoint instead of local gRAG_v3 pipeline")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Generator model (default: gemini-3-flash-preview)")
    args = parser.parse_args()
    
    questions = load_corpus(test_mode=args.test)
    
    # Quick mode for testing parallelism
    if args.quick > 0:
        questions = questions[:args.quick]
        print(f"QUICK MODE: Running {len(questions)} questions only")
    
    evaluator = GoldEvaluator(
        precision_k=args.precision, 
        workers=args.workers, 
        generator_reasoning=args.generator_reasoning,
        cloud_mode=args.cloud,
        model=args.model
    )
    evaluator.run(questions)


if __name__ == "__main__":
    main()
