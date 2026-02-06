#!/usr/bin/env python3
"""
Run all feature engineering pipelines for FX pairs.

This script orchestrates the execution of all three feature engineering modules:
- Technical Indicators
- Mean Reversion
- Regime Detection

Usage:
    python run_all_features.py
"""

import time
from technical_indicators_feature_engineering import main as run_technical_indicators
from mean_reversion_feature_engineering import main as run_mean_reversion
from regime_detection_features import main as run_regime_detection


def main():
    """Run all feature engineering pipelines sequentially."""
    
    pipelines = [
        ("Technical Indicators", run_technical_indicators),
        ("Mean Reversion", run_mean_reversion),
        ("Regime Detection", run_regime_detection),
    ]
    
    total_start = time.time()
    results = {}
    
    for name, pipeline_func in pipelines:
        print(f"\n{'#'*80}")
        print(f"# Starting {name} Feature Engineering")
        print(f"{'#'*80}\n")
        
        start = time.time()
        try:
            pipeline_func()
            elapsed = time.time() - start
            results[name] = ("✅ Success", elapsed)
            print(f"\n{name} completed in {elapsed:.1f} seconds")
        except Exception as e:
            elapsed = time.time() - start
            results[name] = (f"❌ Failed: {e}", elapsed)
            print(f"\n{name} failed after {elapsed:.1f} seconds: {e}")
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print("Feature Engineering Summary")
    print(f"{'='*80}")
    for name, (status, elapsed) in results.items():
        print(f"  {name}: {status} ({elapsed:.1f}s)")
    print(f"\nTotal time: {total_elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
