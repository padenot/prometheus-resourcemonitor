#!/usr/bin/env python3
"""
List available collectors for node_exporter on the current system.
Shows which collectors are enabled/disabled and their current status.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Set

from resource_monitor import PrometheusExporter, load_config


def get_collector_info(port: int) -> Dict[str, List[str]]:
    """Query node_exporter to get collector information."""
    collectors = {
        'enabled': [],
        'disabled': [],
        'failed': []
    }
    
    # Query the metrics endpoint
    try:
        url = f"http://localhost:{port}/metrics"
        with urllib.request.urlopen(url, timeout=5) as response:
            content = response.read().decode('utf-8')
            
        # Parse collector success/failure metrics
        for line in content.split('\n'):
            if line.startswith('node_scrape_collector_success'):
                # Extract collector name and status
                # Format: node_scrape_collector_success{collector="cpu"} 1
                if '{collector="' in line:
                    collector = line.split('{collector="')[1].split('"')[0]
                    value = float(line.split('}')[1].strip())
                    
                    if value == 1:
                        collectors['enabled'].append(collector)
                    else:
                        collectors['failed'].append(collector)
                        
    except Exception as e:
        print(f"Error querying node_exporter: {e}")
    
    return collectors


def list_all_collectors() -> List[str]:
    """Get a list of all known collectors for node_exporter."""
    # This is a comprehensive list of collectors as of node_exporter v1.7.0
    # Some may not be available on all platforms
    return [
        'arp',           # ARP statistics (Linux)
        'bcache',        # Block cache statistics (Linux)
        'bonding',       # Network bonding (Linux)
        'boottime',      # System boot time
        'btrfs',         # Btrfs filesystem statistics (Linux)
        'buddyinfo',     # Memory fragmentation (Linux)
        'conntrack',     # Connection tracking (Linux)
        'cpu',           # CPU statistics
        'cpufreq',       # CPU frequency (Linux)
        'diskstats',     # Disk I/O statistics
        'dmi',           # DMI/SMBIOS data (Linux)
        'drbd',          # DRBD statistics (Linux)
        'edac',          # Error detection and correction (Linux)
        'entropy',       # Entropy available (Linux)
        'exec',          # Execute scripts for metrics
        'fibrechannel',  # Fibre Channel statistics (Linux)
        'filefd',        # File descriptor statistics (Linux)
        'filesystem',    # Filesystem statistics
        'hwmon',         # Hardware monitoring (Linux)
        'infiniband',    # Infiniband statistics (Linux)
        'interrupts',    # Interrupt statistics (Linux)
        'ipvs',          # IP Virtual Server stats (Linux)
        'ksmd',          # Kernel samepage merging (Linux)
        'lnstat',        # Linux network statistics (Linux)
        'loadavg',       # Load average
        'logind',        # Systemd login stats (Linux)
        'mdadm',         # Software RAID (Linux)
        'meminfo',       # Memory information
        'meminfo_numa',  # NUMA memory information (Linux)
        'mountstats',    # Mount statistics (Linux)
        'netclass',      # Network interface info (Linux)
        'netdev',        # Network device statistics
        'netstat',       # Network statistics (Linux)
        'nfs',           # NFS client statistics (Linux)
        'nfsd',          # NFS server statistics (Linux)
        'ntp',           # NTP statistics
        'nvme',          # NVMe statistics (Linux)
        'os',            # OS release info (Linux)
        'perf',          # Perf statistics (Linux)
        'powersupplyclass', # Power supply info (Linux)
        'pressure',      # Pressure stall information (Linux)
        'processes',     # Process statistics (Linux)
        'qdisc',         # Queueing discipline (Linux)
        'rapl',          # Power consumption (Linux)
        'runit',         # Runit service stats
        'schedstat',     # Scheduler statistics (Linux)
        'selinux',       # SELinux statistics (Linux)
        'slabinfo',      # Slab allocator stats (Linux)
        'sockstat',      # Socket statistics (Linux)
        'softnet',       # Software network stats (Linux)
        'stat',          # Various statistics
        'supervisord',   # Supervisord service stats
        'systemd',       # Systemd service stats (Linux)
        'tapestats',     # Tape statistics (Linux)
        'tcpstat',       # TCP statistics (Linux)
        'textfile',      # Custom metrics from files
        'thermal',       # Thermal zone stats (Linux)
        'thermal_zone',  # Thermal zones (Linux)
        'time',          # System time
        'timex',         # Time synchronization
        'udp_queues',    # UDP queue stats (Linux)
        'uname',         # System information
        'vmstat',        # Virtual memory statistics
        'wifi',          # WiFi statistics (Linux)
        'xfs',           # XFS filesystem stats (Linux)
        'zfs',           # ZFS filesystem stats (Linux/FreeBSD)
        'zoneinfo',      # Memory zone info (Linux)
    ]


def test_collectors(collectors_to_test: List[str] = None) -> Dict[str, str]:
    """Test which collectors work on this system."""
    results = {}
    exporter = PrometheusExporter()
    
    if collectors_to_test is None:
        collectors_to_test = list_all_collectors()
    
    print(f"Testing {len(collectors_to_test)} collectors...")
    print("This may take a moment...\n")
    
    # Test each collector individually
    for collector in collectors_to_test:
        try:
            port = exporter._find_free_port()
            binary_path = exporter._get_exporter_path("node_exporter")
            
            # Start node_exporter with only this collector enabled
            args = [
                str(binary_path),
                f"--web.listen-address=:{port}",
                f"--collector.{collector}",
                "--collector.disable-defaults"
            ]
            
            # Start the process
            process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(0.5)
            
            # Check if it started successfully
            if process.poll() is None:
                # Still running, try to query it
                try:
                    url = f"http://localhost:{port}/metrics"
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if response.status == 200:
                            results[collector] = "available"
                        else:
                            results[collector] = "error"
                except:
                    results[collector] = "error"
                    
                # Stop the process
                process.terminate()
                process.wait(timeout=2)
            else:
                # Process exited, check stderr for reason
                stderr = process.stderr.read()
                if "unknown collector" in stderr.lower() or "couldn't create" in stderr.lower():
                    results[collector] = "unavailable"
                else:
                    results[collector] = "error"
                    
        except Exception as e:
            results[collector] = "error"
            
        # Progress indicator
        sys.stdout.write(f"\rTested: {len(results)}/{len(collectors_to_test)}")
        sys.stdout.flush()
    
    print("\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='List and test available node_exporter collectors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test all collectors to see which work on this system'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file to show current settings'
    )
    parser.add_argument(
        '--running',
        action='store_true',
        help='Check currently running node_exporter instance'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )
    
    args = parser.parse_args()
    
    if args.running:
        # Try to find a running node_exporter
        print("Looking for running node_exporter instance...")
        
        # Try common ports
        for port in [9100, 9200]:  # Common node_exporter ports
            try:
                collectors = get_collector_info(port)
                if collectors['enabled']:
                    print(f"\n✓ Found node_exporter on port {port}")
                    print(f"\nEnabled collectors ({len(collectors['enabled'])}):")
                    for c in sorted(collectors['enabled']):
                        print(f"  ✓ {c}")
                    
                    if collectors['failed']:
                        print(f"\nFailed collectors ({len(collectors['failed'])}):")
                        for c in sorted(collectors['failed']):
                            print(f"  ✗ {c}")
                    
                    return 0
            except:
                continue
        
        # Start our own instance temporarily
        print("No running instance found. Starting temporary instance...")
        exporter = PrometheusExporter()
        port = exporter.start_exporter("node_exporter")
        time.sleep(2)
        
        collectors = get_collector_info(port)
        exporter.stop_all()
        
        if args.json:
            print(json.dumps(collectors, indent=2))
        else:
            print(f"\nEnabled collectors ({len(collectors['enabled'])}):")
            for c in sorted(collectors['enabled']):
                print(f"  ✓ {c}")
            
            if collectors['failed']:
                print(f"\nFailed collectors ({len(collectors['failed'])}):")
                for c in sorted(collectors['failed']):
                    print(f"  ✗ {c}")
    
    elif args.test:
        # Test all collectors
        results = test_collectors()
        
        available = [c for c, status in results.items() if status == "available"]
        unavailable = [c for c, status in results.items() if status == "unavailable"]
        errors = [c for c, status in results.items() if status == "error"]
        
        if args.json:
            print(json.dumps({
                'available': sorted(available),
                'unavailable': sorted(unavailable),
                'errors': sorted(errors)
            }, indent=2))
        else:
            print(f"Available collectors ({len(available)}):")
            for c in sorted(available):
                print(f"  ✓ {c}")
            
            print(f"\nUnavailable collectors ({len(unavailable)}):")
            for c in sorted(unavailable):
                print(f"  ✗ {c}")
            
            if errors:
                print(f"\nErrors ({len(errors)}):")
                for c in sorted(errors):
                    print(f"  ⚠️  {c}")
    
    elif args.config:
        # Show configuration
        config = load_config(Path(args.config) if args.config != "default" else None)
        
        if args.json:
            print(json.dumps(config, indent=2))
        else:
            node_config = config.get('exporters', {}).get('node_exporter', {})
            
            print("Configuration settings:")
            print("\nExplicitly enabled collectors:")
            for c in node_config.get('enabled_collectors', []):
                print(f"  + {c}")
            if not node_config.get('enabled_collectors'):
                print("  (none - using defaults)")
            
            print("\nExplicitly disabled collectors:")
            for c in node_config.get('disabled_collectors', []):
                print(f"  - {c}")
            if not node_config.get('disabled_collectors'):
                print("  (none)")
    
    else:
        # Just list all known collectors
        all_collectors = list_all_collectors()
        
        if args.json:
            print(json.dumps(all_collectors, indent=2))
        else:
            print(f"Known node_exporter collectors ({len(all_collectors)}):\n")
            
            # Group by availability
            linux_only = []
            cross_platform = ['cpu', 'diskstats', 'filesystem', 'loadavg', 
                            'meminfo', 'netdev', 'time', 'uname', 'boottime']
            
            for c in sorted(all_collectors):
                if c in cross_platform:
                    print(f"  • {c} (cross-platform)")
                else:
                    print(f"  • {c} (platform-specific)")
            
            print("\nUse --test to see which collectors work on your system")
            print("Use --running to see currently enabled collectors")
            print("Use --config <file> to see configuration settings")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())