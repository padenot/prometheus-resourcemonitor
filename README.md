# Prometheus-based System Resource Monitor for Firefox Profiler

fork of in-tree resourcemonitor.py that uses prometheus tooling.

## Files

### Core Modules
- `resource_monitor.py` - Main monitoring engine with Prometheus exporter management and Firefox Profiler format generation
- `monitor_interactive.py` - Interactive monitoring mode that runs until Ctrl+C, saves timestamped profiles
- `monitor_command.py` - Monitors system resources while executing a specific command
- `test_monitor.py` - Generates synthetic workloads using stress-ng for testing monitoring capabilities

### Utilities
- `download_exporters.py` - Automatically downloads platform-specific Prometheus exporters (node_exporter, etc.)
- `list_collectors.py` - Lists, tests, and shows information about available Prometheus collectors
- `show_collectors.sh` - Quick shell script to display currently enabled collectors
- `fix_macos_security.sh` - Removes macOS quarantine attributes from downloaded exporters

### Configuration Files
- `config.example.yaml` - Template configuration file showing available options
- `config_minimal.yaml` - Minimal collector configuration for low-overhead monitoring
- `config_extended.yaml` - Extended configuration enabling additional collectors for comprehensive metrics

### Documentation
- `README.md` - This file
- `LICENSE` - Mozilla Public License 2.0
- `requirements.txt` - Optional Python dependencies (PyYAML)
- `reference_psutil_implementation.py` - Original psutil-based implementation for reference

## License

MPL2.0, this is forked from firefox.
