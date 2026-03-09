"""
Enhanced Evaluation Runner
Orchestrates production evaluation with complete metrics, caching, and reporting
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.production_evaluator import ProductionEvaluator, ReRanker
from llm_server.services.llm_client import LLMClient
from eval.metrics_reporter import generate_production_report


def create_reports_directory():
    """Create reports directory if it doesn't exist"""
    reports_dir = Path("eval/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def run_full_evaluation(
    use_caching: bool = True,
    use_reranking: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run complete evaluation pipeline
    
    Args:
        use_caching: Enable result and embedding caching
        use_reranking: Enable LLM-based re-ranking
        verbose: Print progress
    
    Returns:
        Complete results dictionary
    """
    
    print("\n" + "="*80)
    print("PRODUCTION EVALUATION - COMPLETE PIPELINE")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  - Caching:      {use_caching}")
    print(f"  - Re-ranking:   {use_reranking}")
    print(f"  - Verbose:      {verbose}")
    print()
    
    # Create reports directory
    reports_dir = create_reports_directory()
    
    # Run evaluation
    evaluator = ProductionEvaluator(
        ground_truth_path=r"E:\Agmentis\Scalable_FAISS_Store\eval\ground_truth.json",
        use_caching=use_caching,
        use_reranking=use_reranking
    )
    # If re-ranking requested, attach a ReRanker backed by the LLM client
    if use_reranking:
        try:
            llm = LLMClient()
            reranker = ReRanker(llm)
            evaluator.retriever.reranker = reranker
            evaluator.retriever.use_reranking = True
        except Exception as e:
            print(f"Warning: failed to initialize re-ranker: {e}")
    
    results = evaluator.evaluate(verbose=verbose)
    
    # Generate reports
    generate_production_report(results)
    
    return results


def run_comparison_evaluation() -> None:
    """
    Run comparative evaluation: with vs without caching & re-ranking
    """
    
    print("\n" + "="*80)
    print("COMPARATIVE EVALUATION - OPTIMIZATION IMPACT")
    print("="*80)
    
    scenarios = [
        {"name": "Baseline (No Optimization)", "caching": False, "reranking": False},
        {"name": "With Caching", "caching": True, "reranking": False},
        {"name": "With Re-ranking", "caching": False, "reranking": True},
        {"name": "Full Optimization", "caching": True, "reranking": True},
    ]
    
    results_dict = {}
    
    for scenario in scenarios:
        print(f"\n--- Running: {scenario['name']} ---")
        
        results = run_full_evaluation(
            use_caching=scenario["caching"],
            use_reranking=scenario["reranking"],
            verbose=False
        )
        
        results_dict[scenario["name"]] = {
            "metrics": results["metrics"],
            "latency": results["latency"],
            "caching": results["caching"]
        }
    
    # Save comparison
    with open("eval/reports/comparison_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Scenario':<30} {'Recall@3':<12} {'Top-1':<12} {'MRR':<10} {'Latency(ms)':<12}")
    print("-" * 80)
    
    for scenario_name, data in results_dict.items():
        metrics = data["metrics"]
        latency = data["latency"].get("end_to_end", {})
        
        mean_latency = latency.get("mean", 0)
        
        print(f"{scenario_name:<30} {metrics['recall_at_3']:.1%}          {metrics['top_1_accuracy']:.1%}          {metrics['mrr']:.3f}      {mean_latency:.1f}ms")
    
    print("\n✓ Comparison results saved to eval/reports/comparison_results.json")


if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "compare":
            # Run comparative evaluation
            run_comparison_evaluation()
        else:
            # Run single evaluation with default settings
            results = run_full_evaluation(
                use_caching=True,
                use_reranking=False,
                verbose=True
            )
            print("\n✓ Evaluation completed successfully!")
            print(f"Results keys: {results.keys()}")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
