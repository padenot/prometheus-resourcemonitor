#!/usr/bin/env python3
"""
Test script for the resource monitor using stress-ng to generate load.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from resource_monitor import SystemResourceMonitor


def run_stress_test(duration=60, poll_interval=0.1):
    """Run a test with various stress-ng workloads.
    
    Args:
        duration: Total test duration in seconds
        poll_interval: Polling interval in seconds (e.g., 0.1 for 10Hz)
    """
    
    print(f"Starting resource monitor test...")
    print(f"  Duration: {duration} seconds")
    print(f"  Polling interval: {poll_interval} seconds ({int(1/poll_interval)}Hz)")
    
    # Create monitor with specified polling interval
    monitor = SystemResourceMonitor(poll_interval=poll_interval, metadata={
        'test': 'stress-ng workload test',
        'platform': 'macOS',
        'duration': duration,
        'poll_interval': poll_interval
    })
    
    # Start monitoring
    monitor.start()
    print("Monitor started, beginning test phases...")
    
    # Calculate phase durations based on total duration
    # For a short test (<=10s), just do baseline and CPU stress
    if duration <= 10:
        baseline_duration = duration * 0.3  # 30% baseline
        cpu_duration = duration * 0.7       # 70% CPU stress
        
        # Phase 1: Baseline
        print(f"\n📊 Phase 1: Baseline measurement ({baseline_duration:.1f} seconds)...")
        monitor.begin_phase("baseline")
        time.sleep(baseline_duration)
        monitor.finish_phase("baseline")
        
        # Phase 2: CPU stress
        print(f"\n🔥 Phase 2: CPU stress ({cpu_duration:.1f} seconds)...")
        monitor.begin_phase("cpu_stress")
        monitor.record_event("cpu_stress_start")
        
        cpu_proc = subprocess.Popen([
            'stress-ng', 
            '--cpu', '4',
            '--cpu-load', '80',
            '--timeout', f'{int(cpu_duration)}s',
            '--metrics-brief'
        ])
        
        monitor.begin_marker("stress", "CPU stress test", timestamp=time.time())
        cpu_proc.wait()
        monitor.end_marker("stress", "CPU stress test", timestamp=time.time())
        monitor.record_event("cpu_stress_end")
        monitor.finish_phase("cpu_stress")
        
    else:
        # For longer tests, do all phases
        # Distribute time across phases
        phase_time = duration / 6
        baseline_duration = phase_time
        cpu_duration = phase_time * 1.5  # CPU gets a bit more time
        memory_duration = phase_time * 1.5  # Memory gets a bit more time
        io_duration = phase_time
        combined_duration = phase_time
        cooldown_duration = phase_time
        
        # Adjust if necessary to fit duration
        total = baseline_duration + cpu_duration + memory_duration + io_duration + combined_duration + cooldown_duration
        if total > duration:
            scale = duration / total
            baseline_duration *= scale
            cpu_duration *= scale
            memory_duration *= scale
            io_duration *= scale
            combined_duration *= scale
            cooldown_duration *= scale
        
        # Phase 1: Baseline (idle)
        print(f"\n📊 Phase 1: Baseline measurement ({baseline_duration:.1f} seconds)...")
        monitor.begin_phase("baseline")
        time.sleep(baseline_duration)
        monitor.finish_phase("baseline")
        
        # Phase 2: CPU stress
        print(f"\n🔥 Phase 2: CPU stress ({cpu_duration:.1f} seconds)...")
        monitor.begin_phase("cpu_stress")
        monitor.record_event("cpu_stress_start")
        
        cpu_proc = subprocess.Popen([
            'stress-ng', 
            '--cpu', '4',
            '--cpu-load', '80',
            '--timeout', f'{int(cpu_duration)}s',
            '--metrics-brief'
        ])
        
        monitor.begin_marker("stress", "CPU stress test started", timestamp=time.time())
        time.sleep(cpu_duration / 2)
        monitor.record_event("cpu_stress_midpoint")
        time.sleep(cpu_duration / 2)
        monitor.end_marker("stress", "CPU stress test started", timestamp=time.time())
        
        cpu_proc.wait()
        monitor.record_event("cpu_stress_end")
        monitor.finish_phase("cpu_stress")
        
        # Phase 3: Memory stress
        print(f"\n💾 Phase 3: Memory stress ({memory_duration:.1f} seconds)...")
        monitor.begin_phase("memory_stress")
        monitor.record_event("memory_stress_start")
        
        mem_proc = subprocess.Popen([
            'stress-ng',
            '--vm', '2',
            '--vm-bytes', '256M',
            '--timeout', f'{int(memory_duration)}s',
            '--metrics-brief'
        ])
        
        monitor.begin_marker("stress", "Memory stress test", timestamp=time.time())
        mem_proc.wait()
        monitor.end_marker("stress", "Memory stress test", timestamp=time.time())
        
        monitor.record_event("memory_stress_end")
        monitor.finish_phase("memory_stress")
        
        # Phase 4: IO stress
        print(f"\n💿 Phase 4: I/O stress ({io_duration:.1f} seconds)...")
        monitor.begin_phase("io_stress")
        monitor.record_event("io_stress_start")
        
        io_proc = subprocess.Popen([
            'stress-ng',
            '--io', '2',
            '--timeout', f'{int(io_duration)}s',
            '--metrics-brief'
        ])
        
        monitor.begin_marker("stress", "IO stress test", timestamp=time.time())
        io_proc.wait()
        monitor.end_marker("stress", "IO stress test", timestamp=time.time())
        
        monitor.record_event("io_stress_end")
        monitor.finish_phase("io_stress")
        
        # Phase 5: Combined stress
        print(f"\n🎯 Phase 5: Combined stress ({combined_duration:.1f} seconds)...")
        monitor.begin_phase("combined_stress")
        monitor.record_event("combined_stress_start")
        
        combined_proc = subprocess.Popen([
            'stress-ng',
            '--cpu', '2',
            '--vm', '1',
            '--vm-bytes', '128M',
            '--io', '1',
            '--timeout', f'{int(combined_duration)}s',
            '--metrics-brief'
        ])
        
        with monitor.phase("nested_phase_example"):
            time.sleep(min(5, combined_duration / 2))
            monitor.record_event("combined_stress_midpoint")
        
        combined_proc.wait()
        monitor.record_event("combined_stress_end")
        monitor.finish_phase("combined_stress")
        
        # Phase 6: Cool down
        print(f"\n❄️ Phase 6: Cool down ({cooldown_duration:.1f} seconds)...")
        monitor.begin_phase("cooldown")
        time.sleep(cooldown_duration)
        monitor.finish_phase("cooldown")
    
    # Stop monitoring
    print("\n✅ Stopping monitor and collecting results...")
    monitor.stop()
    
    return monitor


def save_results(monitor, output_dir="test_output"):
    """Save the monitoring results in various formats."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save as dict (JSON)
    print("\n💾 Saving results...")
    
    dict_output = output_dir / "metrics_dict.json"
    with open(dict_output, 'w') as f:
        json.dump(monitor.as_dict(), f, indent=2, default=str)
    print(f"  ✓ Saved dict format to {dict_output}")
    
    # Save as Firefox Profiler format
    profile_output = output_dir / "profile.json"
    with open(profile_output, 'w') as f:
        json.dump(monitor.as_profile(), f, indent=2)
    print(f"  ✓ Saved Firefox Profiler format to {profile_output}")
    
    # Print summary statistics
    print("\n📈 Test Summary:")
    print(f"  • Total duration: {monitor.end_time - monitor.start_time:.2f} seconds")
    print(f"  • Measurements collected: {len(monitor.measurements)}")
    print(f"  • Phases recorded: {len(monitor.phases)}")
    print(f"  • Events recorded: {len(monitor.events)}")
    print(f"  • Markers recorded: {len(monitor.markers)}")
    
    # Print phase durations
    print("\n⏱️ Phase Durations:")
    for phase_name, (start, end) in monitor.phases.items():
        duration = end - start
        print(f"  • {phase_name}: {duration:.2f}s")
    
    # Print some sample metrics from each phase
    print("\n📊 Sample Metrics by Phase:")
    for phase_name in monitor.phases.keys():
        metrics = list(monitor.phase_usage(phase_name))
        if metrics:
            # Get average CPU usage for this phase
            cpu_percents = []
            for m in metrics:
                if m.cpu_percent:
                    avg_cpu = sum(m.cpu_percent) / len(m.cpu_percent)
                    cpu_percents.append(avg_cpu)
            
            if cpu_percents:
                avg_phase_cpu = sum(cpu_percents) / len(cpu_percents)
                print(f"  • {phase_name}: avg CPU = {avg_phase_cpu:.1f}%")
    
    print(f"\n🎉 Test complete! Open {profile_output} in the Firefox Profiler to visualize.")
    print("   Visit: https://profiler.firefox.com/ and load the JSON file")


def main():
    """Main test function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Test resource monitor with stress-ng workloads',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=10,
        help='Total test duration in seconds'
    )
    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=0.1,
        help='Polling interval in seconds (0.1 = 10Hz, 0.01 = 100Hz)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='test_output',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        print("❌ Duration must be positive")
        return 1
    
    if args.interval <= 0 or args.interval > args.duration:
        print("❌ Invalid polling interval")
        return 1
    
    try:
        # Check if stress-ng is available
        result = subprocess.run(['which', 'stress-ng'], capture_output=True)
        if result.returncode != 0:
            print("❌ stress-ng is not installed. Please install it first:")
            print("   brew install stress-ng")
            return 1
        
        print("🚀 Resource Monitor Test with stress-ng")
        print("=" * 50)
        
        # Run the test with specified parameters
        monitor = run_stress_test(duration=args.duration, poll_interval=args.interval)
        
        # Save and display results
        save_results(monitor, output_dir=args.output_dir)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())