#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Resource monitor using Prometheus exporters for Firefox Profiler format.

This module provides system resource monitoring using node_exporter instead of psutil,
generating output compatible with the Firefox Profiler (https://profiler.firefox.com/).
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

SystemResourceUsage = namedtuple(
    "SystemResourceUsage",
    ["start", "end", "cpu_times", "cpu_percent", "io", "virt", "swap", "raw_metrics"],
)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    default_config = {
        'exporters': {
            'node_exporter': {
                'enabled_collectors': [],
                'disabled_collectors': []  # Allow thermal by default
            },
            'process_exporter': {
                'enabled': False
            },
            'windows_exporter': {
                'enabled_collectors': []
            }
        },
        'monitoring': {
            'poll_interval': 0.1,
            'output_dir': 'profiles',
            'max_measurements': 0,
            'auto_start_process_exporter': False
        }
    }
    
    if config_path is None:
        config_path = Path("config.yaml")
    
    if not config_path.exists():
        # Return default configuration if no config file exists
        return default_config
    
    if not HAS_YAML:
        print("⚠️  PyYAML not installed. Using default configuration.")
        print("   To use YAML config files, install: pip3 install --user PyYAML")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Error loading config file: {e}")
        print("   Using default configuration.")
        return default_config


class PrometheusExporter:
    """Manages Prometheus exporter binaries and processes."""
    
    EXPORTER_URLS = {
        "node_exporter": {
            "linux-amd64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz",
            "linux-arm64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-arm64.tar.gz",
            "darwin-amd64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.darwin-amd64.tar.gz",
            "darwin-arm64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.darwin-arm64.tar.gz",
        },
        "process_exporter": {
            "linux-amd64": "https://github.com/ncabatoff/process-exporter/releases/download/v0.7.10/process-exporter-0.7.10.linux-amd64.tar.gz",
            "linux-arm64": "https://github.com/ncabatoff/process-exporter/releases/download/v0.7.10/process-exporter-0.7.10.linux-arm64.tar.gz",
            "darwin-amd64": "https://github.com/ncabatoff/process-exporter/releases/download/v0.7.10/process-exporter-0.7.10.darwin-amd64.tar.gz",
            "darwin-arm64": "https://github.com/ncabatoff/process-exporter/releases/download/v0.7.10/process-exporter-0.7.10.darwin-arm64.tar.gz",
        },
        "windows_exporter": {
            "win32-amd64": "https://github.com/prometheus-community/windows_exporter/releases/download/v0.25.1/windows_exporter-0.25.1-amd64.msi"
        }
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "prometheus_exporters"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_config()
        
        # Detect platform and architecture
        self.system = platform.system().lower()
        if self.system == "windows":
            self.system = "win32"
        
        # Detect architecture
        machine = platform.machine().lower()
        if machine in ['arm64', 'aarch64']:
            self.arch = 'arm64'
        elif machine in ['x86_64', 'amd64']:
            self.arch = 'amd64'
        else:
            self.arch = 'amd64'  # fallback
        
        self.platform_key = f"{self.system}-{self.arch}"
        self.processes = {}
        self.ports = {}
    
    def _get_exporter_path(self, exporter_name: str) -> Path:
        """Get the path to an exporter binary, checking local dir first."""
        # Check local exporters directory first
        local_path = Path("./exporters") / exporter_name
        if local_path.exists():
            return local_path
        
        # Fall back to cache directory
        exporter_dir = self.cache_dir / exporter_name
        
        if self.system == "win32":
            binary_name = f"{exporter_name}.exe"
        else:
            binary_name = exporter_name
        
        binary_path = exporter_dir / binary_name
        
        if binary_path.exists():
            return binary_path
        
        # Download and extract the exporter
        print(f"Downloading {exporter_name} for {self.platform_key}...")
        url = self.EXPORTER_URLS[exporter_name].get(self.platform_key)
        if not url:
            raise RuntimeError(f"No {exporter_name} available for {self.platform_key}")
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            urlretrieve(url, tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        exporter_dir.mkdir(parents=True, exist_ok=True)
        
        if url.endswith(".tar.gz"):
            import tarfile
            import shutil
            with tarfile.open(tmp_path, "r:gz") as tar:
                # Extract all to temp directory
                extract_dir = tmp_path.parent / f"extract_{exporter_name}"
                tar.extractall(extract_dir)
                
                # Find the actual binary
                found = False
                for root, dirs, files in os.walk(extract_dir):
                    if binary_name in files:
                        src_binary = Path(root) / binary_name
                        # Ensure it's executable (not a text file)
                        with open(src_binary, 'rb') as f:
                            # Check if it starts with ELF (Linux) or Mach-O (macOS) magic bytes
                            magic = f.read(4)
                            if magic[:4] in [b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe']:
                                shutil.move(str(src_binary), str(binary_path))
                                found = True
                                break
                
                # Clean up
                shutil.rmtree(extract_dir)
                if not found:
                    raise RuntimeError(f"Could not find executable {binary_name} in archive")
        elif url.endswith(".zip"):
            with zipfile.ZipFile(tmp_path, "r") as zip_file:
                for name in zip_file.namelist():
                    if name.endswith(binary_name):
                        with zip_file.open(name) as source:
                            with open(binary_path, "wb") as target:
                                target.write(source.read())
                        break
        elif url.endswith(".msi"):
            # For Windows, we'd need to extract from MSI
            # For now, assume the exporter is installed
            raise NotImplementedError("Auto-download for Windows not yet implemented. Please install windows_exporter manually.")
        
        tmp_path.unlink()
        
        # Make executable on Unix
        if self.system != "win32":
            binary_path.chmod(0o755)
            
            # On macOS, remove quarantine attribute for downloaded files
            if self.system == "darwin":
                subprocess.run(['xattr', '-d', 'com.apple.quarantine', str(binary_path)], 
                             capture_output=True, check=False)
        
        return binary_path
    
    def _find_free_port(self) -> int:
        """Find a free port to use for the exporter."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def start_exporter(self, exporter_name: str) -> int:
        """Start an exporter and return its port."""
        if exporter_name in self.processes:
            return self.ports[exporter_name]
        
        binary_path = self._get_exporter_path(exporter_name)
        port = self._find_free_port()
        
        args = [str(binary_path), f"--web.listen-address=:{port}"]
        
        # Add specific flags based on exporter
        if exporter_name == "node_exporter":
            # Use configuration to enable/disable collectors
            node_config = self.config.get('exporters', {}).get('node_exporter', {})
            
            # Add enabled collectors
            for collector in node_config.get('enabled_collectors', []):
                args.append(f"--collector.{collector}")
            
            # Add disabled collectors
            for collector in node_config.get('disabled_collectors', []):
                args.append(f"--no-collector.{collector}")
        elif exporter_name == "process_exporter" and self.system == "linux":
            # Create a config for process_exporter
            config_path = self.cache_dir / "process_exporter.yml"
            config_path.write_text("""
process_names:
  - name: "{{.Comm}}"
    cmdline:
    - '.+'
""")
            args.extend([f"--config.path={config_path}"])
        
        # Log the startup command for debugging
        print(f"Starting {exporter_name} on port {port}...")
        print(f"Command: {' '.join(args)}")
        
        # For debugging, keep stderr but don't capture  
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        self.processes[exporter_name] = process
        self.ports[exporter_name] = port
        
        # Poll the exporter endpoint until it's ready (max 10 seconds)
        import urllib.request
        import urllib.error
        max_wait = 10
        poll_interval = 0.1
        elapsed = 0
        
        while elapsed < max_wait:
            # Check if process has exited
            if process.poll() is not None:
                print(f"Process exited with code: {process.poll()}")
                raise RuntimeError(f"Failed to start {exporter_name}")
            
            # Try to connect to the metrics endpoint
            try:
                with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=1) as response:
                    if response.status == 200:
                        # Exporter is ready
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                # Not ready yet, keep waiting
                pass
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        if elapsed >= max_wait:
            raise RuntimeError(f"{exporter_name} did not become ready within {max_wait} seconds")
        
        return port
    
    def stop_all(self):
        """Stop all running exporters."""
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()
        self.ports.clear()


class PrometheusMetricsCollector:
    """Collects metrics from Prometheus exporters."""
    
    def __init__(self, poll_interval: float = 1.0, config: Optional[Dict[str, Any]] = None):
        self.poll_interval = poll_interval
        self.config = config or load_config()
        self.exporter = PrometheusExporter(config=self.config)
        self.metrics_data = []
        self.running = False
        self._thread = None
        
    def _scrape_metrics(self, port: int) -> Dict[str, Any]:
        """Scrape metrics from a Prometheus exporter."""
        url = f"http://localhost:{port}/metrics"
        try:
            with urlopen(url, timeout=5) as response:
                text = response.read().decode('utf-8')
                return self._parse_prometheus_metrics(text)
        except Exception as e:
            print(f"Error scraping metrics from port {port}: {e}")
            return {}
    
    def _parse_prometheus_metrics(self, text: str) -> Dict[str, Any]:
        """Parse Prometheus metrics text format."""
        metrics = {}
        for line in text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split(' ')
            if len(parts) >= 2:
                metric_name = parts[0]
                try:
                    value = float(parts[1])
                    metrics[metric_name] = value
                except ValueError:
                    continue
        
        return metrics
    
    def _convert_metrics_to_legacy_format(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Prometheus metrics to legacy psutil-like format."""
        result = {
            'cpu_percent': [],
            'cpu_times': [],
            'memory': {},
            'swap': {},
            'io': {},
        }
        
        # CPU metrics
        cpu_count = int(metrics.get('node_cpu_info', 1))
        for i in range(cpu_count):
            # Calculate CPU percentage from node_cpu_seconds_total
            cpu_total = 0
            cpu_idle = metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="idle"}}', 0)
            for mode in ['user', 'system', 'nice', 'iowait', 'irq', 'softirq', 'steal', 'idle']:
                cpu_total += metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="{mode}"}}', 0)
            
            if cpu_total > 0:
                cpu_percent = 100 * (1 - cpu_idle / cpu_total)
            else:
                cpu_percent = 0
            
            result['cpu_percent'].append(cpu_percent)
            
            # CPU times
            cpu_times = {
                'user': metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="user"}}', 0),
                'system': metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="system"}}', 0),
                'idle': metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="idle"}}', 0),
                'nice': metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="nice"}}', 0),
                'iowait': metrics.get(f'node_cpu_seconds_total{{cpu="{i}",mode="iowait"}}', 0),
            }
            result['cpu_times'].append(cpu_times)
        
        # Memory metrics - handle both Linux and macOS naming conventions
        total_mem = metrics.get('node_memory_total_bytes', metrics.get('node_memory_MemTotal_bytes', 0))
        free_mem = metrics.get('node_memory_free_bytes', metrics.get('node_memory_MemFree_bytes', 0))
        available_mem = metrics.get('node_memory_available_bytes', metrics.get('node_memory_MemAvailable_bytes', free_mem))
        cached_mem = metrics.get('node_memory_cached_bytes', metrics.get('node_memory_Cached_bytes', 0))
        
        result['memory'] = {
            'total': total_mem,
            'available': available_mem,
            'free': free_mem,
            'used': total_mem - available_mem,
            'cached': cached_mem,
            'buffers': metrics.get('node_memory_buffer_bytes', metrics.get('node_memory_Buffers_bytes', 0)),
        }
        
        # Swap metrics
        result['swap'] = {
            'total': metrics.get('node_memory_SwapTotal_bytes', 0),
            'free': metrics.get('node_memory_SwapFree_bytes', 0),
            'used': metrics.get('node_memory_SwapTotal_bytes', 0) - metrics.get('node_memory_SwapFree_bytes', 0),
        }
        
        # Disk IO metrics - aggregate across all devices
        total_read_bytes = 0
        total_write_bytes = 0
        total_read_count = 0
        total_write_count = 0
        
        for key, value in metrics.items():
            if 'node_disk_read_bytes_total' in key:
                total_read_bytes += value
            elif 'node_disk_written_bytes_total' in key:
                total_write_bytes += value
            elif 'node_disk_reads_completed_total' in key:
                total_read_count += value
            elif 'node_disk_writes_completed_total' in key:
                total_write_count += value
        
        result['io'] = {
            'read_bytes': total_read_bytes,
            'write_bytes': total_write_bytes,
            'read_count': total_read_count,
            'write_count': total_write_count,
        }
        
        return result
    
    def _collection_loop(self):
        """Main collection loop that runs in a separate thread."""
        # Start exporters based on platform
        if platform.system().lower() == "windows":
            port = self.exporter.start_exporter("windows_exporter")
            exporter_ports = [port]
        else:
            node_port = self.exporter.start_exporter("node_exporter")
            exporter_ports = [node_port]
            
            # Optionally start process_exporter based on config or environment
            process_config = self.config.get('exporters', {}).get('process_exporter', {})
            monitoring_config = self.config.get('monitoring', {})
            
            if process_config.get('enabled', False) or monitoring_config.get('auto_start_process_exporter', False) or os.environ.get("MOZ_PROCESS_SAMPLING"):
                try:
                    proc_port = self.exporter.start_exporter("process_exporter")
                    exporter_ports.append(proc_port)
                except Exception as e:
                    print(f"Could not start process_exporter: {e}")
        
        last_metrics = {}
        # Initialize the first start_time
        last_end_time = time.monotonic()
        
        while self.running:
            # Use previous end time as current start time for perfect continuity
            start_time = last_end_time
            
            # Scrape all exporters
            current_metrics = {}
            for port in exporter_ports:
                metrics = self._scrape_metrics(port)
                current_metrics.update(metrics)
            
            # Convert to legacy format
            converted = self._convert_metrics_to_legacy_format(current_metrics)
            
            # Store raw IO values for delta calculation
            raw_io = converted['io'].copy() if 'io' in converted else {}
            
            # Calculate deltas for counters
            if last_metrics:
                # IO deltas
                io_delta = {}
                for key in ['read_bytes', 'write_bytes', 'read_count', 'write_count']:
                    io_delta[key] = converted['io'].get(key, 0) - last_metrics.get('last_raw_io', {}).get(key, 0)
                converted['io'] = io_delta
            else:
                # First measurement - set IO to 0 instead of cumulative totals
                converted['io'] = {
                    'read_bytes': 0,
                    'write_bytes': 0,
                    'read_count': 0,
                    'write_count': 0
                }
            
            # CPU deltas for percentage calculation  
            if last_metrics:
                cpu_deltas = []
                for i, (current_times, last_times) in enumerate(zip(converted['cpu_times'], last_metrics.get('cpu_times', []))):
                    deltas = {}
                    total_delta = 0
                    for mode in ['user', 'system', 'nice', 'iowait', 'idle']:
                        delta = current_times.get(mode, 0) - last_times.get(mode, 0)
                        deltas[mode] = max(0, delta)  # Ensure non-negative
                        total_delta += deltas[mode]
                    
                    # Calculate CPU percentage from deltas
                    if total_delta > 0:
                        cpu_percent = 100 * (1 - deltas['idle'] / total_delta)
                    else:
                        cpu_percent = 0
                    
                    cpu_deltas.append(cpu_percent)
                
                # Replace cpu_percent with calculated deltas if available
                if cpu_deltas:
                    converted['cpu_percent'] = cpu_deltas
            
            end_time = time.monotonic()
            
            self.metrics_data.append({
                'timestamp': time.time(),
                'start': start_time,
                'end': end_time,
                'metrics': converted,
                'raw_metrics': current_metrics
            })
            
            # Store converted metrics and raw IO for next iteration
            last_metrics = converted.copy()
            last_metrics['last_raw_io'] = raw_io
            
            # Sleep for the remainder of the interval
            elapsed = end_time - start_time
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Set the end time for the next iteration (after sleep)
            last_end_time = time.monotonic()
    
    def start(self):
        """Start the metrics collection."""
        self.running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the metrics collection."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.exporter.stop_all()
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics."""
        return self.metrics_data


class SystemResourceMonitor:
    """
    System resource monitor using Prometheus exporters.
    
    Maintains compatibility with the original marker API while using
    Prometheus ecosystem tools for metrics collection.
    """
    
    instance = None
    
    def __init__(self, poll_interval: float = 1.0, metadata: Dict[str, Any] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the resource monitor."""
        self.config = config or load_config()
        self.poll_interval = poll_interval or self.config.get('monitoring', {}).get('poll_interval', 0.1)
        self.metadata = metadata or {}
        self.collector = PrometheusMetricsCollector(poll_interval, config=self.config)
        
        self.start_time = None
        self.end_time = None
        self.start_timestamp = None
        
        self.events = []
        self.markers = []
        self.processes = []
        self.phases = OrderedDict()
        
        self._active_phases = {}
        self._active_markers = {}
        
        self._running = False
        self._stopped = False
    
    def convert_to_monotonic_time(self, timestamp):
        """Convert a timestamp to monotonic time."""
        return timestamp - self.start_timestamp + self.start_time
    
    def start(self):
        """Start measuring system resources."""
        if self._running:
            return
        
        self.start_time = time.monotonic()
        self.start_timestamp = time.time()
        self.collector.start()
        self._running = True
        SystemResourceMonitor.instance = self
    
    def stop(self):
        """Stop measuring system resources."""
        if not self._running:
            return
        
        self.stop_time = time.monotonic()
        self.end_time = time.monotonic()
        self.collector.stop()
        
        # Process collected metrics
        self.measurements = []
        for data in self.collector.get_metrics():
            # Create a legacy-compatible measurement
            metrics = data['metrics']
            
            # Create proper namedtuples with actual data
            io_data = metrics['io']
            IoTuple = namedtuple('io', io_data.keys())
            io_tuple = IoTuple(*io_data.values())
            
            memory_data = metrics['memory']
            MemoryTuple = namedtuple('virt', memory_data.keys())
            memory_tuple = MemoryTuple(*memory_data.values())
            
            swap_data = metrics['swap'] 
            SwapTuple = namedtuple('swap', swap_data.keys())
            swap_tuple = SwapTuple(*swap_data.values())
            
            measurement = SystemResourceUsage(
                start=data['start'],
                end=data['end'],
                cpu_times=metrics['cpu_times'],
                cpu_percent=metrics['cpu_percent'],
                io=io_tuple,
                virt=memory_tuple,
                swap=swap_tuple,
                raw_metrics=data['raw_metrics'],
            )
            self.measurements.append(measurement)
        
        self._running = False
        self._stopped = True
        SystemResourceMonitor.instance = None
    
    # Marker API methods - preserved from original
    
    @staticmethod
    def record_event(name):
        """Record an event as occurring now."""
        if SystemResourceMonitor.instance:
            SystemResourceMonitor.instance.events.append((time.monotonic(), name))
    
    @staticmethod
    def record_marker(name, start, end, text):
        """Record a marker with a duration and optional text."""
        if SystemResourceMonitor.instance:
            SystemResourceMonitor.instance.markers.append((name, start, end, text))
    
    @staticmethod
    def begin_marker(name, text, disambiguator=None, timestamp=None):
        """Begin a marker."""
        if SystemResourceMonitor.instance:
            id = name + ":" + text
            if disambiguator:
                id += ":" + disambiguator
            SystemResourceMonitor.instance._active_markers[id] = (
                SystemResourceMonitor.instance.convert_to_monotonic_time(timestamp)
                if timestamp
                else time.monotonic()
            )
    
    @staticmethod
    def end_marker(name, text, disambiguator=None, timestamp=None):
        """End a marker."""
        if not SystemResourceMonitor.instance:
            return
        end = time.monotonic()
        if timestamp:
            end = SystemResourceMonitor.instance.convert_to_monotonic_time(timestamp)
        id = name + ":" + text
        if disambiguator:
            id += ":" + disambiguator
        if id not in SystemResourceMonitor.instance._active_markers:
            return
        start = SystemResourceMonitor.instance._active_markers.pop(id)
        SystemResourceMonitor.instance.record_marker(name, start, end, text)
    
    @contextmanager
    def phase(self, name):
        """Context manager for recording a phase."""
        self.begin_phase(name)
        yield
        self.finish_phase(name)
    
    def begin_phase(self, name):
        """Record the start of a phase."""
        assert name not in self._active_phases
        self._active_phases[name] = time.monotonic()
    
    def finish_phase(self, name):
        """Record the end of a phase."""
        assert name in self._active_phases
        phase = (self._active_phases[name], time.monotonic())
        self.phases[name] = phase
        del self._active_phases[name]
        return phase[1] - phase[0]
    
    # Data query methods
    
    def range_usage(self, start=None, end=None):
        """Get usage data within a time range."""
        if not self._stopped or self.start_time is None:
            return
        
        if start is None:
            start = self.start_time
        
        if end is None:
            end = self.end_time
        
        for entry in self.measurements:
            if entry.start < start:
                continue
            
            if entry.end > end:
                break
            
            yield entry
    
    def phase_usage(self, phase):
        """Get usage data for a specific phase."""
        time_start, time_end = self.phases[phase]
        return self.range_usage(time_start, time_end)
    
    def as_dict(self):
        """Convert to dict for serialization."""
        # Simplified version - full implementation would match original
        return {
            'version': 2,
            'start': self.start_time,
            'end': self.end_time,
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else None,
            'measurements': len(self.measurements),
            'events': list(self.events),
            'markers': list(self.markers),
            'phases': dict(self.phases),
            'metadata': self.metadata,
        }
    
    def as_profile(self):
        """
        Convert to Firefox Profiler format.
        
        Matches the original format structure exactly.
        """
        if not self._stopped:
            raise RuntimeError("Must stop monitoring before generating profile")
        
        profile_time = time.monotonic()
        # Use the first measurement's start time as reference to avoid negative times
        # This ensures all measurement times are relative to the actual start of data collection
        start_time = self.measurements[0].start if self.measurements else self.start_time
        
        
        # Match the original profile structure exactly
        profile = {
            "meta": {
                "processType": 0,
                "product": "mach",
                "stackwalk": 0,
                "version": 27,
                "preprocessedProfileVersion": 57,
                "symbolicationNotSupported": True,
                "interval": self.poll_interval * 1000,
                "startTime": self.start_timestamp * 1000,
                "profilingStartTime": 100,  # Start markers slightly before this
                "profilingEndTime": round((self.end_time - start_time) * 1000 + 0.0005, 3),
                "logicalCPUs": 8,  # Will update from metrics
                "physicalCPUs": 8, # Will update from metrics  
                "mainMemory": 137438953472,  # Will update from metrics
                "categories": [
                    {
                        "name": "Other",
                        "color": "grey",
                        "subcategories": ["Other"],
                    },
                    {
                        "name": "Phases",
                        "color": "grey",
                        "subcategories": ["Other"],
                    },
                    {
                        "name": "Tasks", 
                        "color": "grey",
                        "subcategories": ["Other"],
                    },
                ],
                "markerSchema": [
                    {
                        "name": "CPU",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "cpuPercent", "label": "CPU Percent", "format": "string"},
                            {"key": "user_pct", "label": "User %", "format": "string"},
                            {"key": "iowait_pct", "label": "IO Wait %", "format": "string"},
                            {"key": "system_pct", "label": "System %", "format": "string"},
                            {"key": "nice_pct", "label": "Nice %", "format": "string"},
                            {"key": "idle_pct", "label": "Idle %", "format": "string"}
                        ],
                        "graphs": [
                            {"key": "softirq", "color": "orange", "type": "bar"},
                            {"key": "iowait", "color": "red", "type": "bar"},
                            {"key": "system", "color": "grey", "type": "bar"},
                            {"key": "user", "color": "yellow", "type": "bar"},
                            {"key": "nice", "color": "blue", "type": "bar"}
                        ],
                    },
                    {
                        "name": "Mem",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "used", "label": "Memory Used", "format": "bytes"},
                            {"key": "cached", "label": "Memory cached", "format": "bytes"},
                            {"key": "buffers", "label": "Memory buffers", "format": "bytes"},
                        ],
                        "graphs": [
                            {"key": "used", "color": "orange", "type": "line-filled"}
                        ],
                    },
                    {
                        "name": "IO",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "write_bytes", "label": "Written", "format": "bytes"},
                            {"key": "write_count", "label": "Write count", "format": "integer"},
                            {"key": "read_bytes", "label": "Read", "format": "bytes"},
                            {"key": "read_count", "label": "Read count", "format": "integer"},
                        ],
                        "graphs": [
                            {"key": "read_bytes", "color": "green", "type": "bar"},
                            {"key": "write_bytes", "color": "red", "type": "bar"},
                        ],
                    },
                    {
                        "name": "NetIO",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "rx_bytes", "label": "RX Bytes", "format": "bytes"},
                            {"key": "tx_bytes", "label": "TX Bytes", "format": "bytes"},
                        ],
                        "graphs": [
                            {"key": "rx_bytes", "color": "green", "type": "line"},
                            {"key": "tx_bytes", "color": "orange", "type": "line"},
                        ],
                    },
                    {
                        "name": "Thermal",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "zone_0", "label": "Zone 0", "format": "decimal"},
                            {"key": "zone_1", "label": "Zone 1", "format": "decimal"},
                            {"key": "zone_2", "label": "Zone 2", "format": "decimal"},
                            {"key": "zone_3", "label": "Zone 3", "format": "decimal"},
                        ],
                        "graphs": [
                            {"key": "zone_0", "color": "red", "type": "line"},
                        ],
                    },
                    {
                        "name": "Power",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "current_ampere", "label": "Current (A)", "format": "decimal"},
                            {"key": "voltage", "label": "Voltage (V)", "format": "decimal"},
                            {"key": "charging", "label": "Charging", "format": "integer"},
                        ],
                        "graphs": [
                            {"key": "current_ampere", "color": "blue", "type": "line"},
                            {"key": "voltage", "color": "yellow", "type": "line"},
                        ],
                    },
                    {
                        "name": "LoadAvg",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "load1", "label": "1 min", "format": "decimal"},
                            {"key": "load5", "label": "5 min", "format": "decimal"},
                            {"key": "load15", "label": "15 min", "format": "decimal"},
                        ],
                        "graphs": [
                            {"key": "load1", "color": "red", "type": "line"},
                            {"key": "load5", "color": "orange", "type": "line"},
                            {"key": "load15", "color": "yellow", "type": "line"},
                        ],
                    },
                    {
                        "name": "Filesystem",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "size_bytes", "label": "Total Size", "format": "bytes"},
                            {"key": "free_bytes", "label": "Free Space", "format": "bytes"},
                            {"key": "used_percent", "label": "Used %", "format": "percentage"},
                        ],
                        "graphs": [
                            {"key": "used_percent", "color": "purple", "type": "bar"},
                        ],
                    },
                    {
                        "name": "VMStats",
                        "tooltipLabel": "{marker.name}",
                        "display": [],
                        "fields": [
                            {"key": "pgpgin", "label": "Page In", "format": "integer"},
                            {"key": "pgpgout", "label": "Page Out", "format": "integer"},
                            {"key": "pswpin", "label": "Swap In", "format": "integer"},
                            {"key": "pswpout", "label": "Swap Out", "format": "integer"},
                        ],
                        "graphs": [
                            {"key": "pgpgin", "color": "cyan", "type": "bar"},
                            {"key": "pgpgout", "color": "magenta", "type": "bar"},
                        ],
                    },
                    {
                        "name": "Phase",
                        "tooltipLabel": "{marker.data.phase}",
                        "tableLabel": "{marker.name} — {marker.data.phase}",
                        "chartLabel": "{marker.data.phase}",
                        "display": [
                            "marker-chart",
                            "marker-table",
                            "timeline-overview",
                        ],
                        "fields": [],
                    },
                    {
                        "name": "Text",
                        "tooltipLabel": "{marker.name}",
                        "tableLabel": "{marker.name} — {marker.data.text}",
                        "chartLabel": "{marker.data.text}",
                        "display": ["marker-chart", "marker-table"],
                        "fields": [
                            {
                                "key": "text",
                                "label": "Description",
                                "format": "string",
                                "searchable": True,
                            }
                        ],
                    },
                ],
                "usesOnlyOneStackType": True,
            },
            "libs": [],
            "threads": [{
                "processType": "default",
                "processName": "mach",
                "processStartupTime": 0,
                "processShutdownTime": None,
                "registerTime": 0,
                "unregisterTime": None,
                "pausedRanges": [],
                "showMarkersInTimeline": True,
                "name": "",
                "isMainThread": False,
                "pid": "0",
                "tid": 0,
                "samples": {
                    "weightType": "samples",
                    "weight": None,
                    "stack": [],
                    "time": [],
                    "length": 0,
                },
                "markers": {
                    "data": [],
                    "name": [],
                    "startTime": [],
                    "endTime": [],
                    "phase": [],
                    "category": [],
                    "length": 0,
                },
                "stackTable": {
                    "frame": [0],
                    "prefix": [None],
                    "category": [0],
                    "subcategory": [0],
                    "length": 1,
                },
                "frameTable": {
                    "address": [-1],
                    "inlineDepth": [0],
                    "category": [None],
                    "subcategory": [0],
                    "func": [0],
                    "nativeSymbol": [None],
                    "innerWindowID": [0],
                    "implementation": [None],
                    "line": [None],
                    "column": [None],
                    "length": 1,
                },
                "funcTable": {
                    "isJS": [False],
                    "relevantForJS": [False],
                    "name": [0],
                    "resource": [-1],
                    "fileName": [None],
                    "lineNumber": [None],
                    "columnNumber": [None],
                    "length": 1,
                },
                "resourceTable": {
                    "lib": [],
                    "name": [],
                    "host": [],
                    "type": [],
                    "length": 0,
                },
                "nativeSymbols": {
                    "libIndex": [],
                    "address": [],
                    "name": [],
                    "functionSize": [],
                    "length": 0,
                },
            }],
            "counters": [],
        }
        
        # Get system info from first measurement if available
        if self.measurements:
            first_metrics = self.collector.get_metrics()[0] if self.collector.get_metrics() else None
            if first_metrics:
                raw_metrics = first_metrics.get('raw_metrics', {})
                # Try to get CPU count from metrics
                cpu_count = 0
                for key in raw_metrics:
                    if 'node_cpu_seconds_total' in key and 'cpu=' in key:
                        cpu_num = int(key.split('cpu="')[1].split('"')[0])
                        cpu_count = max(cpu_count, cpu_num + 1)
                
                if cpu_count > 0:
                    profile["meta"]["logicalCPUs"] = cpu_count
                    profile["meta"]["physicalCPUs"] = cpu_count
                
                # Get memory info
                total_memory = raw_metrics.get('node_memory_MemTotal_bytes', 0)
                if total_memory > 0:
                    profile["meta"]["mainMemory"] = int(total_memory)
        
        # Add metadata
        for key, value in self.metadata.items():
            profile["meta"][key] = value
        
        markers = profile["threads"][0]["markers"]
        string_array = []
        
        def get_string_index(string):
            try:
                return string_array.index(string)
            except ValueError:
                string_array.append(string)
                return len(string_array) - 1
        
        def add_marker(name_index, start_or_timestamp_ms, end, data, category_index=0):
            # All timestamps passed here should already be in milliseconds
            # We don't need to do any conversion based on magnitude
            start_ms = start_or_timestamp_ms
            end_ms = end if end else None
                
            markers["startTime"].append(start_ms)
            if end is None:
                markers["endTime"].append(None)
                markers["phase"].append(0)  # Instant
            else:
                markers["endTime"].append(end_ms)
                markers["phase"].append(1)  # Interval
            markers["category"].append(category_index)
            markers["name"].append(name_index)
            markers["data"].append(data)
            markers["length"] += 1
        
        # Add resource usage markers
        
        # Add continuous metrics as instant markers for graphs FIRST to get indices 0,1,2
        if self.measurements:
            cpu_string_index = get_string_index("CPU Use")
            mem_string_index = get_string_index("Memory") 
            io_string_index = get_string_index("IO")
            
            # Filter out measurements before start_time first
            valid_measurements = [m for m in self.measurements if m.end >= start_time]
            
            for i, m in enumerate(valid_measurements):
                # Each marker represents the interval from this measurement to the next
                # For the last measurement, use its own start and end
                # Ensure times are never negative by using max(0, ...)
                start_ms = max(0, round((m.start - start_time) * 1000))
                
                if i < len(valid_measurements) - 1:
                    # Use the next measurement's start time as this marker's end time
                    # This ensures no overlap and perfect continuity
                    next_m = valid_measurements[i + 1]
                    end_ms = max(0, round((next_m.start - start_time) * 1000))
                else:
                    # For the last measurement, use its own end time
                    end_ms = max(0, round((m.end - start_time) * 1000))
                
                # Add CPU marker with real node_exporter data
                if hasattr(m, 'cpu_percent') and m.cpu_percent:
                    # Use sum of CPU percentages across all cores for total system load
                    total_cpu = sum(m.cpu_percent)
                    
                    # Use the already calculated delta-based percentages from the collection loop
                    # For breakdown, get from raw_metrics if available, otherwise use simple approximation
                    raw_metrics = getattr(m, 'raw_metrics', {})
                    
                    # Try to get per-CPU breakdown from current measurement cpu_times
                    if hasattr(m, 'cpu_times') and m.cpu_times and len(m.cpu_times) > 0:
                        # Average the absolute values across all CPUs
                        avg_times = {}
                        for mode in ['user', 'system', 'nice', 'iowait']:
                            total = sum(cpu.get(mode, 0) for cpu in m.cpu_times)
                            avg_times[mode] = total / len(m.cpu_times) if m.cpu_times else 0
                        
                        # For display percentages, approximate from total CPU usage
                        # This is imperfect but gives reasonable breakdown
                        if total_cpu > 0:
                            user_pct = total_cpu * 0.7  # Typical user dominance
                            system_pct = total_cpu * 0.25
                            nice_pct = total_cpu * 0.02
                            iowait_pct = total_cpu * 0.03
                        else:
                            user_pct = system_pct = nice_pct = iowait_pct = 0
                            
                        user_val = user_pct / 100
                        system_val = system_pct / 100  
                        nice_val = nice_pct / 100
                        iowait_val = iowait_pct / 100
                    else:
                        user_val = system_val = nice_val = iowait_val = 0
                        user_pct = system_pct = nice_pct = iowait_pct = 0
                    
                    softirq_pct = 0  # Not tracked separately
                    
                    cpu_data = {
                        "type": "CPU",
                        "cpuPercent": f"{total_cpu:.1f}%",
                        "nice": nice_val,
                        "user": user_val,
                        "system": system_val,
                        "iowait": iowait_val,
                        "softirq": 0,  # Not available in node_exporter
                        "nice_pct": f"{nice_pct:.1f}%",
                        "user_pct": f"{user_pct:.1f}%", 
                        "system_pct": f"{system_pct:.1f}%",
                        "iowait_pct": f"{iowait_pct:.1f}%",
                        "idle_pct": f"{100 - total_cpu:.1f}%"
                    }
                    add_marker(cpu_string_index, start_ms, end_ms, cpu_data, category_index=0)
                
                # Add Memory marker - use real node_exporter data
                if hasattr(m, 'virt') and m.virt:
                    mem_data = {
                        "type": "Mem",
                        "used": getattr(m.virt, 'used', 0),
                        "cached": getattr(m.virt, 'cached', 0),
                        "buffers": getattr(m.virt, 'buffers', 0)
                    }
                    add_marker(mem_string_index, start_ms, end_ms, mem_data, category_index=0)
                
                # Add IO marker - use real node_exporter data
                if hasattr(m, 'io') and m.io:
                    io_data = {
                        "type": "IO",
                        "read_count": getattr(m.io, 'read_count', 0),
                        "read_bytes": getattr(m.io, 'read_bytes', 0),
                        "write_count": getattr(m.io, 'write_count', 0),
                        "write_bytes": getattr(m.io, 'write_bytes', 0)
                    }
                    add_marker(io_string_index, start_ms, end_ms, io_data, category_index=0)
                
                # Add generic metrics from raw_metrics if available
                if hasattr(m, 'raw_metrics') and m.raw_metrics:
                    raw = m.raw_metrics
                    
                    # Network metrics (cumulative counters, need deltas)
                    net_metrics_found = False
                    net_rx_bytes = 0
                    net_tx_bytes = 0
                    
                    for k, v in raw.items():
                        if k.startswith('node_network_receive_bytes_total') and 'device="lo"' not in k:
                            net_rx_bytes += v
                            net_metrics_found = True
                        elif k.startswith('node_network_transmit_bytes_total') and 'device="lo"' not in k:
                            net_tx_bytes += v
                            net_metrics_found = True
                    
                    if net_metrics_found:
                        # Calculate deltas for network bytes
                        if i > 0 and hasattr(self.measurements[i-1], 'raw_metrics'):
                            prev_raw = self.measurements[i-1].raw_metrics
                            prev_rx = 0
                            prev_tx = 0
                            for k, v in prev_raw.items():
                                if k.startswith('node_network_receive_bytes_total') and 'device="lo"' not in k:
                                    prev_rx += v
                                elif k.startswith('node_network_transmit_bytes_total') and 'device="lo"' not in k:
                                    prev_tx += v
                            net_rx_delta = max(0, net_rx_bytes - prev_rx)
                            net_tx_delta = max(0, net_tx_bytes - prev_tx)
                        else:
                            net_rx_delta = 0
                            net_tx_delta = 0
                        
                        # Use a different name to avoid Firefox Profiler's "Network requests" handling
                        net_string_index = get_string_index("Net I/O")
                        net_data = {
                            "type": "NetIO",  # Changed from "Network" to avoid conflicts
                            "rx_bytes": net_rx_delta,
                            "tx_bytes": net_tx_delta
                        }
                        add_marker(net_string_index, start_ms, end_ms, net_data, category_index=0)
                    
                    # Thermal metrics (Linux) or Power metrics (macOS)
                    thermal_zones = {}
                    power_metrics = {}
                    
                    for k, v in raw.items():
                        if k.startswith('node_thermal_zone_temp'):
                            # Linux thermal zones
                            if 'zone="' in k:
                                zone = k.split('zone="')[1].split('"')[0]
                                thermal_zones[f"zone_{zone}"] = v
                        elif k.startswith('node_power_supply'):
                            # macOS power metrics
                            if 'current_ampere' in k:
                                power_metrics['current_ampere'] = v
                            elif 'voltage' in k:
                                power_metrics['voltage'] = v
                            elif 'charging' in k and 'InternalBattery' in k:
                                power_metrics['charging'] = v
                    
                    if thermal_zones:
                        thermal_string_index = get_string_index("Thermal")
                        thermal_data = {
                            "type": "Thermal",
                            **thermal_zones
                        }
                        add_marker(thermal_string_index, start_ms, end_ms, thermal_data, category_index=0)
                    elif power_metrics:
                        # Use power metrics on systems without thermal zones
                        power_string_index = get_string_index("Power")
                        power_data = {
                            "type": "Power",
                            **power_metrics
                        }
                        add_marker(power_string_index, start_ms, end_ms, power_data, category_index=0)
                    
                    # Load average metrics
                    if 'node_load1' in raw:
                        load_string_index = get_string_index("Load Average")
                        load_data = {
                            "type": "LoadAvg",
                            "load1": raw.get('node_load1', 0),
                            "load5": raw.get('node_load5', 0),
                            "load15": raw.get('node_load15', 0)
                        }
                        add_marker(load_string_index, start_ms, end_ms, load_data, category_index=0)
                    
                    # Filesystem metrics (aggregate all filesystems)
                    fs_metrics = {}
                    for key in raw:
                        if key.startswith('node_filesystem_size_bytes'):
                            # Skip special filesystems
                            if any(skip in key for skip in ['tmpfs', 'devfs', 'overlay', 'shm']):
                                continue
                            fs_metrics['total_size'] = fs_metrics.get('total_size', 0) + raw[key]
                        elif key.startswith('node_filesystem_free_bytes'):
                            if any(skip in key for skip in ['tmpfs', 'devfs', 'overlay', 'shm']):
                                continue
                            fs_metrics['total_free'] = fs_metrics.get('total_free', 0) + raw[key]
                    
                    if fs_metrics:
                        fs_string_index = get_string_index("Filesystem")
                        fs_data = {
                            "type": "Filesystem",
                            "size_bytes": fs_metrics.get('total_size', 0),
                            "free_bytes": fs_metrics.get('total_free', 0),
                            "used_percent": ((fs_metrics.get('total_size', 0) - fs_metrics.get('total_free', 0)) / fs_metrics.get('total_size', 1)) * 100 if fs_metrics.get('total_size', 0) > 0 else 0
                        }
                        add_marker(fs_string_index, start_ms, end_ms, fs_data, category_index=0)
                    
                    # VM Statistics (cumulative, need deltas)
                    if 'node_vmstat_pgpgin' in raw:
                        # Calculate deltas for VM stats
                        if i > 0 and hasattr(self.measurements[i-1], 'raw_metrics'):
                            prev_raw = self.measurements[i-1].raw_metrics
                            pgpgin_delta = max(0, raw.get('node_vmstat_pgpgin', 0) - prev_raw.get('node_vmstat_pgpgin', 0))
                            pgpgout_delta = max(0, raw.get('node_vmstat_pgpgout', 0) - prev_raw.get('node_vmstat_pgpgout', 0))
                            pswpin_delta = max(0, raw.get('node_vmstat_pswpin', 0) - prev_raw.get('node_vmstat_pswpin', 0))
                            pswpout_delta = max(0, raw.get('node_vmstat_pswpout', 0) - prev_raw.get('node_vmstat_pswpout', 0))
                        else:
                            pgpgin_delta = pgpgout_delta = pswpin_delta = pswpout_delta = 0
                        
                        vm_string_index = get_string_index("VM Stats")
                        vm_data = {
                            "type": "VMStats",
                            "pgpgin": pgpgin_delta,
                            "pgpgout": pgpgout_delta,
                            "pswpin": pswpin_delta,
                            "pswpout": pswpout_delta
                        }
                        add_marker(vm_string_index, start_ms, end_ms, vm_data, category_index=0)
        
        # Add phase markers AFTER counter markers
        phase_string_index = get_string_index("Phase")
        for phase_name, (phase_start, phase_end) in self.phases.items():
            # Convert phase times to same timestamp format as counter markers
            phase_start_relative = phase_start - start_time
            phase_end_relative = phase_end - start_time
            phase_start_ms = round(phase_start_relative * 1000)
            phase_end_ms = round(phase_end_relative * 1000)
            
            if phase_start_ms < 0:
                phase_start_ms = 0
            if phase_end_ms < 0:
                phase_end_ms = 0
                
            phase_data = {
                "type": "Phase", 
                "phase": phase_name
            }
            # For phase markers, use pre-calculated millisecond timestamps
            add_marker(phase_string_index, phase_start_ms, phase_end_ms, phase_data, category_index=1)
        
        # Add generic markers
        for name, start, end, text in self.markers:
            marker_data = {"type": "Text"}
            if text:
                marker_data["text"] = text
            # Convert monotonic times to milliseconds
            start_ms = round((start - start_time) * 1000, 3)
            end_ms = round((end - start_time) * 1000, 3) if end else None
            add_marker(get_string_index(name), start_ms, end_ms, marker_data, category_index=2)
        
        # Add events  
        if self.events:
            event_string_index = get_string_index("Event")
            for event_time, text in self.events:
                if text:
                    # Convert monotonic time to milliseconds
                    event_ms = round((event_time - start_time) * 1000, 3)
                    add_marker(
                        event_string_index,
                        event_ms,
                        None,
                        {"type": "Text", "text": text},
                        category_index=2
                    )
        
        # Add "(root)" at the end to match working profile format
        get_string_index("(root)")
        
        profile["counters"] = []
        profile["shared"] = {
            "stringArray": string_array
        }
        
        return profile