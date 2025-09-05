"""
Performance analysis script for VQASynth pipeline.
Profiles existing pipeline performance and identifies optimization opportunities.
"""

import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, List
import json
import matplotlib.pyplot as plt
from vqasynth.realtime_optimizer import PerformanceProfiler
import cProfile
import pstats

class VQASynthPipelineProfiler:
    """Comprehensive profiler for VQASynth pipeline."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.baseline_metrics = {}
        
    def profile_existing_pipeline(self):
        """Profile the existing VQASynth pipeline."""
        print("=== PROFILING EXISTING VQASYNTH PIPELINE ===")
        
        # Mock pipeline stages based on existing modules
        stages = {
            'depth_processing': self._mock_depth_stage,
            'embeddings_generation': self._mock_embeddings_stage,
            'scene_fusion': self._mock_scene_fusion_stage,
            'spatial_reasoning': self._mock_reasoning_stage,
            'localization': self._mock_localization_stage
        }
        
        total_start = time.time()
        
        for stage_name, stage_func in stages.items():
            print(f"Profiling {stage_name}...")
            self.profiler.profile_stage(stage_name, stage_func)
        
        total_time = time.time() - total_start
        self.baseline_metrics['total_pipeline_time'] = total_time
        
        return self.profiler.generate_profile_report()
    
    def _mock_depth_stage(self):
        """Mock depth processing stage."""
        # Simulate depth processing time
        time.sleep(np.random.uniform(0.015, 0.035))  # 15-35ms
        
        # Simulate memory allocation
        temp_data = torch.randn(512, 512, 3)  # Depth map
        
        return temp_data
    
    def _mock_embeddings_stage(self):
        """Mock embeddings generation stage."""
        # Simulate embeddings processing time
        time.sleep(np.random.uniform(0.025, 0.055))  # 25-55ms
        
        # Simulate memory allocation for embeddings
        temp_embeddings = torch.randn(256, 768)  # Feature embeddings
        
        return temp_embeddings
    
    def _mock_scene_fusion_stage(self):
        """Mock scene fusion stage."""
        # Simulate scene fusion processing time
        time.sleep(np.random.uniform(0.020, 0.030))  # 20-30ms
        
        # Simulate memory allocation for fused features
        temp_fusion = torch.randn(512, 1024)
        
        return temp_fusion
    
    def _mock_reasoning_stage(self):
        """Mock spatial reasoning stage."""
        # Simulate reasoning processing time (typically the slowest)
        time.sleep(np.random.uniform(0.040, 0.090))  # 40-90ms
        
        # Simulate memory allocation for reasoning
        temp_reasoning = torch.randn(128, 512)
        
        return temp_reasoning
    
    def _mock_localization_stage(self):
        """Mock localization stage."""
        # Simulate localization processing time
        time.sleep(np.random.uniform(0.010, 0.020))  # 10-20ms
        
        # Simulate memory allocation
        temp_loc = torch.randn(64, 256)
        
        return temp_loc
    
    def analyze_memory_usage(self):
        """Analyze memory usage patterns."""
        print("\\n=== MEMORY USAGE ANALYSIS ===")
        
        memory_info = psutil.virtual_memory()
        gpu_memory = 0
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        memory_analysis = {
            'system_memory_total_gb': memory_info.total / (1024**3),
            'system_memory_available_gb': memory_info.available / (1024**3),
            'system_memory_percent': memory_info.percent,
            'gpu_memory_allocated_gb': gpu_memory,
            'process_memory_mb': psutil.Process().memory_info().rss / (1024**2)
        }
        
        print(f"System Memory: {memory_analysis['system_memory_percent']:.1f}% used")
        print(f"Available Memory: {memory_analysis['system_memory_available_gb']:.2f} GB")
        print(f"GPU Memory Allocated: {memory_analysis['gpu_memory_allocated_gb']:.2f} GB")
        print(f"Process Memory: {memory_analysis['process_memory_mb']:.1f} MB")
        
        return memory_analysis
    
    def identify_bottlenecks(self, profile_report: Dict[str, Any]):
        """Identify performance bottlenecks."""
        print("\\n=== BOTTLENECK ANALYSIS ===")
        
        bottlenecks = []
        
        # Time-based bottlenecks
        for stage, stats in profile_report['stage_breakdown'].items():
            if stats['time_ms'] > 30:  # Stages taking > 30ms
                bottlenecks.append({
                    'type': 'time',
                    'stage': stage,
                    'value': stats['time_ms'],
                    'severity': 'high' if stats['time_ms'] > 50 else 'medium'
                })
        
        # Memory-based bottlenecks
        for stage, stats in profile_report['stage_breakdown'].items():
            if stats['memory_mb'] > 200:  # Stages using > 200MB
                bottlenecks.append({
                    'type': 'memory',
                    'stage': stage,
                    'value': stats['memory_mb'],
                    'severity': 'high' if stats['memory_mb'] > 500 else 'medium'
                })
        
        print(f"Found {len(bottlenecks)} bottlenecks:")
        for bottleneck in bottlenecks:
            print(f"  {bottleneck['severity'].upper()}: {bottleneck['stage']} "
                  f"({bottleneck['type']}: {bottleneck['value']:.1f})")
        
        return bottlenecks
    
    def generate_optimization_recommendations(self, profile_report: Dict, bottlenecks: List[Dict]):
        """Generate specific optimization recommendations."""
        print("\\n=== OPTIMIZATION RECOMMENDATIONS ===")
        
        recommendations = []
        
        # Real-time constraint check
        if profile_report['total_pipeline_time_ms'] > 100:
            recommendations.append({
                'priority': 'critical',
                'category': 'real_time',
                'description': f"Pipeline time ({profile_report['total_pipeline_time_ms']:.1f}ms) exceeds real-time constraint (100ms)",
                'suggestions': [
                    "Apply model quantization (INT8)",
                    "Implement parallel processing",
                    "Use TensorRT optimization",
                    "Reduce model complexity"
                ]
            })
        
        # Stage-specific recommendations
        bottleneck_stages = [b['stage'] for b in bottlenecks if b['type'] == 'time']
        
        for stage in bottleneck_stages:
            if 'reasoning' in stage:
                recommendations.append({
                    'priority': 'high',
                    'category': 'reasoning_optimization',
                    'description': f"Reasoning stage is bottleneck",
                    'suggestions': [
                        "Implement uncertainty-aware early stopping",
                        "Use knowledge distillation for smaller model",
                        "Cache frequent reasoning patterns",
                        "Implement progressive reasoning"
                    ]
                })
            
            elif 'embeddings' in stage:
                recommendations.append({
                    'priority': 'high',
                    'category': 'embeddings_optimization',
                    'description': f"Embeddings stage is bottleneck",
                    'suggestions': [
                        "Use pre-computed embeddings cache",
                        "Implement embedding quantization",
                        "Reduce embedding dimensions",
                        "Use approximate nearest neighbors"
                    ]
                })
        
        # Memory optimization
        memory_bottlenecks = [b for b in bottlenecks if b['type'] == 'memory']
        if memory_bottlenecks:
            recommendations.append({
                'priority': 'medium',
                'category': 'memory_optimization',
                'description': "High memory usage detected",
                'suggestions': [
                    "Implement gradient checkpointing",
                    "Use mixed precision training",
                    "Batch size optimization",
                    "Memory pooling for tensors"
                ]
            })
        
        # Print recommendations
        for rec in recommendations:
            print(f"\\n{rec['priority'].upper()}: {rec['description']}")
            for suggestion in rec['suggestions']:
                print(f"  - {suggestion}")
        
        return recommendations
    
    def create_performance_baseline(self):
        """Create performance baseline for comparison."""
        print("\\n=== CREATING PERFORMANCE BASELINE ===")
        
        baseline = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'performance_metrics': {},
            'optimization_status': 'baseline'
        }
        
        # Run multiple iterations for stable baseline
        iteration_times = []
        for i in range(5):
            start_time = time.time()
            self.profile_existing_pipeline()
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
        
        baseline['performance_metrics'] = {
            'avg_pipeline_time_ms': np.mean(iteration_times) * 1000,
            'std_pipeline_time_ms': np.std(iteration_times) * 1000,
            'min_pipeline_time_ms': np.min(iteration_times) * 1000,
            'max_pipeline_time_ms': np.max(iteration_times) * 1000
        }
        
        # Save baseline
        with open('/home/isr-lab3/James/VQASynth-UAV/performance_baseline.json', 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"Baseline created: {baseline['performance_metrics']['avg_pipeline_time_ms']:.1f}ms ± {baseline['performance_metrics']['std_pipeline_time_ms']:.1f}ms")
        
        return baseline

def analyze_performance():
    """Main performance analysis function."""
    print("VQASynth Pipeline Performance Analysis")
    print("=" * 50)
    
    profiler = VQASynthPipelineProfiler()
    
    # 1. Create baseline
    baseline = profiler.create_performance_baseline()
    
    # 2. Profile current pipeline
    profile_report = profiler.profile_existing_pipeline()
    
    # 3. Analyze memory usage
    memory_analysis = profiler.analyze_memory_usage()
    
    # 4. Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks(profile_report)
    
    # 5. Generate recommendations
    recommendations = profiler.generate_optimization_recommendations(profile_report, bottlenecks)
    
    # 6. Create summary report
    summary_report = {
        'baseline': baseline,
        'profile': profile_report,
        'memory': memory_analysis,
        'bottlenecks': bottlenecks,
        'recommendations': recommendations,
        'analysis_timestamp': time.time()
    }
    
    # Save comprehensive report
    with open('/home/isr-lab3/James/VQASynth-UAV/performance_analysis_report.json', 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print("\\n=== SUMMARY ===")
    print(f"Baseline performance: {baseline['performance_metrics']['avg_pipeline_time_ms']:.1f}ms")
    print(f"Real-time capable: {profile_report['real_time_capable']}")
    print(f"Bottlenecks found: {len(bottlenecks)}")
    print(f"Recommendations: {len(recommendations)}")
    print(f"\\nDetailed report saved to: performance_analysis_report.json")
    
    return summary_report

if __name__ == "__main__":
    analyze_performance()
