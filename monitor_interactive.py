#!/usr/bin/env python3
"""
Interactive resource monitor that runs continuously until Ctrl+C.
Saves profile with timestamp in filename.
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from resource_monitor import SystemResourceMonitor, load_config


class InteractiveMonitor:
    """Interactive monitoring session handler."""
    
    def __init__(self, poll_interval=0.1, output_dir="profiles", config=None):
        self.config = config
        self.poll_interval = poll_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.monitor = None
        self.start_timestamp = None
        self.interrupted = False
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if self.interrupted:
            # Force exit if pressed twice
            print("\n\n⚠️ Force quitting...")
            sys.exit(1)
            
        self.interrupted = True
        print("\n\n🛑 Interrupt received! Stopping monitor and saving data...")
        print("   (Press Ctrl+C again to force quit)")
        
        if self.monitor:
            self.stop_and_save()
            sys.exit(0)
    
    def start(self):
        """Start interactive monitoring."""
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Create timestamp for this session
        self.start_timestamp = datetime.now()
        timestamp_str = self.start_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        print("=" * 60)
        print("🚀 Interactive Resource Monitor")
        print("=" * 60)
        print(f"Start time: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Polling rate: {int(1/self.poll_interval)}Hz ({self.poll_interval}s interval)")
        print(f"Output directory: {self.output_dir}")
        print(f"Session ID: {timestamp_str}")
        print("-" * 60)
        print("📊 Monitoring system resources...")
        print("   Press Ctrl+C to stop and save the profile")
        print("-" * 60)
        
        # Initialize monitor
        self.monitor = SystemResourceMonitor(
            poll_interval=self.poll_interval,
            metadata={
                'session': 'interactive',
                'timestamp': timestamp_str,
                'hostname': Path.home().name,
                'poll_interval': self.poll_interval
            },
            config=self.config
        )
        
        # Start monitoring
        self.monitor.start()
        
        # Keep running until interrupted
        try:
            while not self.interrupted:
                # Print periodic status updates
                elapsed = time.time() - self.start_timestamp.timestamp()
                measurements = len(self.monitor.collector.metrics_data)
                
                # Update status line every second
                print(f"\r⏱️  Elapsed: {elapsed:.1f}s | 📈 Measurements: {measurements} | 💾 Press Ctrl+C to save", 
                      end='', flush=True)
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            # This shouldn't happen as we handle SIGINT, but just in case
            self.signal_handler(None, None)
    
    def stop_and_save(self):
        """Stop monitoring and save results."""
        if not self.monitor:
            return
            
        # Stop the monitor
        self.monitor.stop()
        
        # Generate filename with timestamp
        timestamp_str = self.start_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        profile_filename = f"profile_{timestamp_str}.json"
        dict_filename = f"metrics_{timestamp_str}.json"
        
        profile_path = self.output_dir / profile_filename
        dict_path = self.output_dir / dict_filename
        
        print("\n\n💾 Saving results...")
        
        # Save Firefox Profiler format
        try:
            profile_data = self.monitor.as_profile()
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            print(f"  ✓ Saved profile: {profile_path}")
        except Exception as e:
            print(f"  ✗ Error saving profile: {e}")
        
        # Save dict format
        try:
            dict_data = self.monitor.as_dict()
            with open(dict_path, 'w') as f:
                json.dump(dict_data, f, indent=2, default=str)
            print(f"  ✓ Saved metrics: {dict_path}")
        except Exception as e:
            print(f"  ✗ Error saving metrics: {e}")
        
        # Print summary
        print("\n📊 Session Summary:")
        print(f"  • Duration: {self.monitor.end_time - self.monitor.start_time:.1f} seconds")
        print(f"  • Measurements: {len(self.monitor.measurements)}")
        print(f"  • Phases: {len(self.monitor.phases)}")
        print(f"  • Events: {len(self.monitor.events)}")
        
        # Calculate average metrics
        if self.monitor.measurements:
            cpu_values = [sum(m.cpu_percent) for m in self.monitor.measurements if hasattr(m, 'cpu_percent') and m.cpu_percent]
            mem_values = [m.virt.used / (1024**3) for m in self.monitor.measurements if hasattr(m, 'virt') and m.virt]
            
            if cpu_values:
                print(f"  • Average CPU: {sum(cpu_values) / len(cpu_values):.1f}%")
            if mem_values:
                print(f"  • Average Memory: {sum(mem_values) / len(mem_values):.1f} GB")
        
        print(f"\n🎉 Profile saved! Open in Firefox Profiler:")
        print(f"   https://profiler.firefox.com/")
        print(f"   Load file: {profile_path}")
        

def main():
    """Main entry point for interactive monitoring."""
    parser = argparse.ArgumentParser(
        description='Interactive resource monitor - runs until Ctrl+C',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default='profiles',
        help='Output directory for saved profiles'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.interval <= 0:
        print("❌ Invalid polling interval")
        return 1
    
    # Load configuration if specified
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return 1
        config = load_config(config_path)
        print(f"✓ Loaded configuration from: {config_path}")
    
    # Create and run monitor
    monitor = InteractiveMonitor(
        poll_interval=args.interval,
        output_dir=args.output_dir,
        config=config
    )
    
    try:
        monitor.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())