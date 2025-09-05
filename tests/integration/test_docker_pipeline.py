#!/usr/bin/env python3
"""
Comprehensive VQASynth inference time testing script
Supports both Docker pipeline and standalone component testing
"""

import os
import sys
import time
import subprocess
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class VQASynthTester:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_path = self.project_root / "config" / "config.yaml"
        self.results = {}
        
    def load_config(self) -> Dict:
        """Load the YAML configuration"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return {}
            
    def check_docker_setup(self) -> bool:
        """Check if Docker and Docker Compose are available"""
        try:
            # Check Docker
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return False
                
            # Check Docker Compose
            result = subprocess.run(["docker", "compose", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
            
        except FileNotFoundError:
            return False
            
    def check_gpu_access(self) -> Dict:
        """Check GPU availability"""
        gpu_info = {"available": False, "count": 0, "memory": "N/A"}
        
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=count,memory.total", 
                                   "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info["available"] = True
                gpu_info["count"] = len(lines)
                if lines:
                    memory = lines[0].split(',')[1].strip()
                    gpu_info["memory"] = f"{int(memory)/1024:.1f}GB"
                    
        except FileNotFoundError:
            pass
            
        return gpu_info
        
    def prepare_test_data(self) -> str:
        """Prepare test dataset for pipeline"""
        # Create a temporary output directory
        output_dir = tempfile.mkdtemp(prefix="vqasynth_test_")
        
        # Create minimal test dataset structure
        dataset_dir = Path(output_dir) / "test_dataset"
        dataset_dir.mkdir()
        
        # Copy example image
        example_img = self.project_root / "examples" / "assets" / "warehouse_rgb.jpg"
        if example_img.exists():
            shutil.copy(example_img, dataset_dir / "test_image.jpg")
            
        return output_dir
        
    def test_docker_pipeline(self) -> Dict:
        """Test the full Docker pipeline and measure time"""
        print("🐳 Testing Docker Pipeline...")
        
        if not self.check_docker_setup():
            return {"error": "Docker or Docker Compose not available"}
            
        # Prepare test data
        test_output_dir = self.prepare_test_data()
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["OUTPUT_DIR"] = test_output_dir
            env["HF_TOKEN"] = os.path.expanduser("~/.cache/huggingface/token")
            
            # Build base image
            print("📦 Building base image...")
            start_time = time.time()
            
            build_result = subprocess.run([
                "docker", "build", "-f", "docker/base_image/Dockerfile", 
                "-t", "vqasynth:base", "."
            ], capture_output=True, text=True, cwd=self.project_root)
            
            build_time = time.time() - start_time
            
            if build_result.returncode != 0:
                return {"error": f"Docker build failed: {build_result.stderr}"}
                
            print(f"✅ Base image built in {build_time:.2f}s")
            
            # Run pipeline
            print("🚀 Running pipeline...")
            pipeline_start = time.time()
            
            pipeline_result = subprocess.run([
                "docker", "compose", "-f", "pipelines/spatialvqa.yaml", 
                "up", "--build"
            ], capture_output=True, text=True, cwd=self.project_root, env=env)
            
            pipeline_time = time.time() - pipeline_start
            
            return {
                "build_time": build_time,
                "pipeline_time": pipeline_time,
                "total_time": build_time + pipeline_time,
                "success": pipeline_result.returncode == 0,
                "output": pipeline_result.stdout[-1000:] if pipeline_result.stdout else "",
                "error": pipeline_result.stderr[-1000:] if pipeline_result.stderr else ""
            }
            
        except Exception as e:
            return {"error": str(e)}
        finally:
            # Cleanup
            shutil.rmtree(test_output_dir, ignore_errors=True)
            
    def test_individual_stages(self) -> Dict:
        """Test individual Docker stages"""
        print("🔍 Testing Individual Stages...")
        
        stages = [
            "embeddings_stage",
            "filter_stage", 
            "location_refinement_stage",
            "scene_fusion_stage",
            "prompt_stage"
        ]
        
        stage_results = {}
        
        for stage in stages:
            print(f"  📊 Testing {stage}...")
            dockerfile = f"docker/{stage}/Dockerfile"
            
            if not (self.project_root / dockerfile).exists():
                stage_results[stage] = {"error": "Dockerfile not found"}
                continue
                
            try:
                start_time = time.time()
                
                # Build stage
                result = subprocess.run([
                    "docker", "build", "-f", dockerfile,
                    "-t", f"vqasynth:{stage}", "."
                ], capture_output=True, text=True, cwd=self.project_root)
                
                build_time = time.time() - start_time
                
                stage_results[stage] = {
                    "build_time": build_time,
                    "success": result.returncode == 0,
                    "error": result.stderr[-500:] if result.stderr else ""
                }
                
            except Exception as e:
                stage_results[stage] = {"error": str(e)}
                
        return stage_results
        
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🎯 VQASynth Comprehensive Testing")
        print("=" * 60)
        
        # System info
        gpu_info = self.check_gpu_access()
        docker_available = self.check_docker_setup()
        
        print(f"🖥️  GPU Available: {gpu_info['available']}")
        if gpu_info['available']:
            print(f"🎮 GPU Count: {gpu_info['count']}")
            print(f"💾 GPU Memory: {gpu_info['memory']}")
        print(f"🐳 Docker Available: {docker_available}")
        print()
        
        total_start = time.time()
        
        # Test 1: Docker Pipeline (full end-to-end)
        if docker_available:
            pipeline_results = self.test_docker_pipeline()
            self.results["pipeline"] = pipeline_results
            
            if "error" not in pipeline_results:
                print(f"✅ Full pipeline completed in {pipeline_results['total_time']:.2f}s")
                print(f"   📦 Build: {pipeline_results['build_time']:.2f}s")
                print(f"   🚀 Pipeline: {pipeline_results['pipeline_time']:.2f}s")
            else:
                print(f"❌ Pipeline failed: {pipeline_results['error']}")
        else:
            print("⚠️  Skipping Docker pipeline test (Docker not available)")
            
        print()
        
        # Test 2: Individual stages
        if docker_available:
            stage_results = self.test_individual_stages()
            self.results["stages"] = stage_results
            
            print("📊 Stage Build Times:")
            for stage, result in stage_results.items():
                if "build_time" in result:
                    status = "✅" if result["success"] else "❌"
                    print(f"   {status} {stage}: {result['build_time']:.2f}s")
                else:
                    print(f"   ❌ {stage}: {result.get('error', 'Unknown error')}")
        
        total_time = time.time() - total_start
        
        print()
        print("📈 Summary")
        print("=" * 60)
        print(f"🕐 Total test time: {total_time:.2f}s")
        
        # Inference time estimates
        if "pipeline" in self.results and "pipeline_time" in self.results["pipeline"]:
            pipeline_time = self.results["pipeline"]["pipeline_time"]
            print(f"⚡ End-to-end pipeline: {pipeline_time:.2f}s")
            print(f"🏃 Estimated per-image processing: {pipeline_time:.2f}s")
            
        print("\n💡 Recommendations:")
        if not gpu_info["available"]:
            print("   - Install NVIDIA GPU drivers and CUDA for better performance")
        if not docker_available:
            print("   - Install Docker and Docker Compose for full pipeline testing")
            
        return self.results

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("🚀 Quick test mode - Docker pipeline only")
        tester = VQASynthTester()
        if tester.check_docker_setup():
            results = tester.test_docker_pipeline()
            print(f"Pipeline result: {results}")
        else:
            print("❌ Docker not available for quick test")
    else:
        tester = VQASynthTester()
        results = tester.run_comprehensive_test()
        
if __name__ == "__main__":
    main()
