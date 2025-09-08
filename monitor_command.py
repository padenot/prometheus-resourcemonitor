#!/usr/bin/env python3
"""
Monitor system resources while running a specific command.
Automatically stops monitoring when the command completes.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from resource_monitor import SystemResourceMonitor


def main():
    """Main entry point for command monitoring."""
    parser = argparse.ArgumentParser(
        description='Monitor resources while running a command',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='Command to run while monitoring'
    )
    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=0.1,
        help='Polling interval in seconds (0.1 = 10Hz)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='profiles',
        help='Output directory for saved profiles'
    )
    parser.add_argument(
        '--phase',
        type=str,
        help='Mark the entire command execution as a phase with this name'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        print("❌ No command specified")
        print("Usage: monitor_command.py [options] -- command [args...]")
        print("Example: monitor_command.py -- make -j8")
        return 1
    
    # Remove '--' if present at the start of command
    if args.command[0] == '--':
        args.command = args.command[1:]
    
    if not args.command:
        print("❌ No command specified after --")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for this session
    start_timestamp = datetime.now()
    timestamp_str = start_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Prepare command string for display
    command_str = ' '.join(args.command)
    
    print("=" * 60)
    print("🚀 Resource Monitor for Command Execution")
    print("=" * 60)
    print(f"Command: {command_str}")
    print(f"Start time: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Polling rate: {int(1/args.interval)}Hz")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Initialize monitor
    monitor = SystemResourceMonitor(
        poll_interval=args.interval,
        metadata={
            'session': 'command',
            'command': command_str,
            'timestamp': timestamp_str,
            'hostname': Path.home().name,
            'poll_interval': args.interval
        }
    )
    
    # Start monitoring
    monitor.start()
    print("📊 Monitoring started...")
    
    # Start the command with a phase if specified
    if args.phase:
        monitor.begin_phase(args.phase)
        monitor.record_event(f"{args.phase}_start")
    
    monitor.record_event("command_start")
    
    # Run the command
    print(f"🔧 Running: {command_str}")
    print("-" * 60)
    
    try:
        # Run command and capture output
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Terminating command...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return_code = -1
        monitor.record_event("command_interrupted")
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        return_code = 1
        monitor.record_event("command_error")
    
    print("-" * 60)
    
    # Record completion
    monitor.record_event("command_end")
    if args.phase:
        monitor.record_event(f"{args.phase}_end")
        monitor.finish_phase(args.phase)
    
    # Stop monitoring
    monitor.stop()
    
    # Generate filenames with timestamp and command
    safe_command = command_str.replace('/', '_').replace(' ', '_')[:50]
    profile_filename = f"profile_{timestamp_str}_{safe_command}.json"
    dict_filename = f"metrics_{timestamp_str}_{safe_command}.json"
    
    profile_path = output_dir / profile_filename
    dict_path = output_dir / dict_filename
    
    print("\n💾 Saving results...")
    
    # Save Firefox Profiler format
    try:
        profile_data = monitor.as_profile()
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        print(f"  ✓ Saved profile: {profile_path}")
    except Exception as e:
        print(f"  ✗ Error saving profile: {e}")
    
    # Save dict format
    try:
        dict_data = monitor.as_dict()
        with open(dict_path, 'w') as f:
            json.dump(dict_data, f, indent=2, default=str)
        print(f"  ✓ Saved metrics: {dict_path}")
    except Exception as e:
        print(f"  ✗ Error saving metrics: {e}")
    
    # Print summary
    print("\n📊 Monitoring Summary:")
    print(f"  • Command: {command_str}")
    print(f"  • Return code: {return_code}")
    print(f"  • Duration: {monitor.end_time - monitor.start_time:.1f} seconds")
    print(f"  • Measurements: {len(monitor.measurements)}")
    
    # Calculate average metrics
    if monitor.measurements:
        cpu_values = [sum(m.cpu_percent) for m in monitor.measurements if hasattr(m, 'cpu_percent') and m.cpu_percent]
        mem_values = [m.virt.used / (1024**3) for m in monitor.measurements if hasattr(m, 'virt') and m.virt]
        
        if cpu_values:
            print(f"  • Average CPU: {sum(cpu_values) / len(cpu_values):.1f}%")
        if mem_values:
            print(f"  • Average Memory: {sum(mem_values) / len(mem_values):.1f} GB")
    
    print(f"\n🎉 Profile saved! Open in Firefox Profiler:")
    print(f"   https://profiler.firefox.com/")
    print(f"   Load file: {profile_path}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())