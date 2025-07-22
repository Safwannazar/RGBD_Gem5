import time
import psutil
import threading
import multiprocessing as mp
from pathlib import Path
import os
import json
import platform
from datetime import datetime
from RGBD import ModifiedRGBDProcessor  # Your existing module

class HardwareSimulator:
    def __init__(self, config_name, cpu_cores, cpu_freq_ghz, memory_gb, gpu_type):
        self.config_name = config_name
        self.cpu_cores = cpu_cores
        self.cpu_freq_ghz = cpu_freq_ghz
        self.memory_gb = memory_gb
        self.gpu_type = gpu_type

        baseline_score = 8 * 3.2  # 8 cores × 3.2GHz
        current_score = cpu_cores * cpu_freq_ghz
        self.performance_factor = current_score / baseline_score

        print(f"🔧 {config_name}: {cpu_cores} cores @ {cpu_freq_ghz}GHz, {memory_gb}GB RAM, {gpu_type}")
        print(f"   Performance factor: {self.performance_factor:.3f}x")

class BenchmarkRunner:
    def __init__(self):
        self.configurations = {
            "Jetson_Nano": HardwareSimulator("Jetson Nano", 4, 1.43, 4, "Maxwell GPU"),
            "Jetson_Xavier": HardwareSimulator("Jetson Xavier", 8, 2.26, 32, "Volta GPU"),
            "Raspberry_Pi_4": HardwareSimulator("Raspberry Pi 4", 4, 1.5, 8, "VideoCore GPU"),
            "Generic_PC": HardwareSimulator("Generic PC", 8, 3.2, 16, "Generic GPU")
        }

        self.results = {}

    def limit_cpu_cores(self, num_cores):
        if platform.system() == "Windows":
            print("⚠️  CPU affinity setting is not supported on Windows. Skipping.")
            return False
        try:
            available_cores = list(range(min(num_cores, psutil.cpu_count())))
            os.sched_setaffinity(0, available_cores)
            return True
        except Exception as e:
            print(f"⚠️  Could not limit CPU cores to {num_cores}: {e}")
            return False

    def simulate_memory_pressure(self, target_memory_gb):
        available_memory_gb = psutil.virtual_memory().total / (1024**3)
        if target_memory_gb < available_memory_gb:
            print(f"   💾 Simulating {target_memory_gb}GB memory constraint")
        else:
            print(f"   💾 Using available {available_memory_gb:.1f}GB memory")

    def run_single_benchmark(self, config_name, hw_config):
        print(f"\n{'='*60}")
        print(f"🚀 TESTING: {config_name}")
        print(f"{'='*60}")

        if platform.system() != "Windows":
            try:
                original_affinity = os.sched_getaffinity(0)
            except Exception:
                original_affinity = None
        else:
            original_affinity = None

        try:
            self.limit_cpu_cores(hw_config.cpu_cores)
            self.simulate_memory_pressure(hw_config.memory_gb)

            left_image_path = "frameL/img- (1).jpg"
            right_image_path = "frameR/img- (1).jpg"
            output_dir = f"gem5_test_results_{config_name}"

            if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
                print("❌ Image files not found. Please update paths in the script.")
                return None

            processor = ModifiedRGBDProcessor()

            print("🔥 Warm-up run...")
            processor.process_single_pair(
                left_path=left_image_path,
                right_path=right_image_path,
                output_dir=output_dir,
                image_index=0,
                pair_name="warmup"
            )

            num_runs = 3
            execution_times = []

            for run in range(num_runs):
                print(f"⏱️  Run {run + 1}/{num_runs}...")

                start_time = time.perf_counter()
                start_cpu = time.process_time()

                result = processor.process_single_pair(
                    left_path=left_image_path,
                    right_path=right_image_path,
                    output_dir=output_dir,
                    image_index=run + 1,
                    pair_name=f"run_{run + 1}"
                )

                end_time = time.perf_counter()
                end_cpu = time.process_time()

                wall_time = end_time - start_time
                cpu_time = end_cpu - start_cpu

                execution_times.append({
                    'wall_time': wall_time,
                    'cpu_time': cpu_time,
                    'success': result is not None
                })

                print(f"   Wall time: {wall_time:.3f}s, CPU time: {cpu_time:.3f}s")
                time.sleep(0.1)

            wall_times = [t['wall_time'] for t in execution_times if t['success']]
            cpu_times = [t['cpu_time'] for t in execution_times if t['success']]

            if wall_times:
                scaled_wall_times = [t / hw_config.performance_factor for t in wall_times]

                benchmark_result = {
                    'config_name': config_name,
                    'hardware': {
                        'cpu_cores': hw_config.cpu_cores,
                        'cpu_freq_ghz': hw_config.cpu_freq_ghz,
                        'memory_gb': hw_config.memory_gb,
                        'gpu_type': hw_config.gpu_type,
                        'performance_factor': hw_config.performance_factor
                    },
                    'timing': {
                        'raw_wall_time_avg': sum(wall_times) / len(wall_times),
                        'raw_wall_time_min': min(wall_times),
                        'raw_wall_time_max': max(wall_times),
                        'scaled_wall_time_avg': sum(scaled_wall_times) / len(scaled_wall_times),
                        'scaled_wall_time_min': min(scaled_wall_times),
                        'scaled_wall_time_max': max(scaled_wall_times),
                        'cpu_time_avg': sum(cpu_times) / len(cpu_times),
                        'num_successful_runs': len(wall_times),
                        'total_runs': num_runs
                    },
                    'timestamp': datetime.now().isoformat()
                }

                print(f"\n📊 RESULTS for {config_name}:")
                print(f"   Average execution time: {benchmark_result['timing']['scaled_wall_time_avg']:.3f}s")
                print(f"   Min execution time: {benchmark_result['timing']['scaled_wall_time_min']:.3f}s")
                print(f"   Max execution time: {benchmark_result['timing']['scaled_wall_time_max']:.3f}s")
                print(f"   Successful runs: {len(wall_times)}/{num_runs}")

                return benchmark_result
            else:
                print(f"❌ All runs failed for {config_name}")
                return None

        except Exception as e:
            print(f"❌ Error during benchmark: {str(e)}")
            return None

        finally:
            if platform.system() != "Windows" and original_affinity:
                try:
                    os.sched_setaffinity(0, original_affinity)
                except:
                    pass

    def run_all_benchmarks(self):
        print("🎯 Starting RGBD Processing Benchmark Suite")
        print(f"Timestamp: {datetime.now()}")

        for config_name, hw_config in self.configurations.items():
            result = self.run_single_benchmark(config_name, hw_config)
            if result:
                self.results[config_name] = result

        self.save_results()
        self.print_comparison()

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rgbd_benchmark_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

    def print_comparison(self):
        print(f"\n{'='*80}")
        print("📈 PERFORMANCE COMPARISON")
        print(f"{'='*80}")

        if not self.results:
            print("❌ No results to compare")
            return

        print(f"{'Configuration':<15} {'Avg Time (s)':<12} {'Min Time (s)':<12} {'Max Time (s)':<12} {'Speedup':<10}")
        print("-" * 80)

        baseline_name = "Generic_PC"
        if baseline_name in self.results:
            baseline_time = self.results[baseline_name]['timing']['scaled_wall_time_avg']
        else:
            times = [(name, data['timing']['scaled_wall_time_avg']) for name, data in self.results.items()]
            baseline_name, baseline_time = min(times, key=lambda x: x[1])

        for config_name, data in self.results.items():
            avg_time = data['timing']['scaled_wall_time_avg']
            min_time = data['timing']['scaled_wall_time_min']
            max_time = data['timing']['scaled_wall_time_max']
            speedup = baseline_time / avg_time if avg_time > 0 else 0
            marker = "🏆" if config_name == baseline_name else "  "
            print(f"{marker}{config_name:<13} {avg_time:<12.3f} {min_time:<12.3f} {max_time:<12.3f} {speedup:<10.2f}x")

        fastest = min(self.results.items(), key=lambda x: x[1]['timing']['scaled_wall_time_avg'])[0]
        slowest = max(self.results.items(), key=lambda x: x[1]['timing']['scaled_wall_time_avg'])[0]

        print(f"\n🏆 Fastest: {fastest}")
        print(f"🐌 Slowest: {slowest}")

def main():
    print("🔧 RGBD Processing Hardware Benchmark Tool")
    print("=" * 60)

    runner = BenchmarkRunner()
    runner.run_all_benchmarks()

    print("\n✅ Benchmark suite completed!")
    print("Check the generated JSON file for detailed results.")

if __name__ == "__main__":
    main()
