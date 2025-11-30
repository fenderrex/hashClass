# Meshtastic Signal Layer Monitor - Quick Guide

## Overview

Simple debugging tool for Meshtastic that monitors multiple serial ports simultaneously and records comprehensive channel data and timing metrics.

## Features

✅ **Multi-Port Monitoring** - Monitor ACM* and USB* ports simultaneously
✅ **Comprehensive Metrics** - Track timing, throughput, and statistics
✅ **Raw Packet Logging** - Record all packets (even failed decodes)
✅ **ACK Tracking** - Measure ACK response times
✅ **Configurable Buffers** - Adjust buffer sizes as needed
✅ **JSON Export** - Export all captured data

## Quick Start

### Auto-Detect All Devices

```bash
python meshtastic_monitor.py --auto
```

### Monitor Specific Ports

```bash
python meshtastic_monitor.py --ports /dev/ttyACM0,/dev/ttyUSB0
```

### Monitor with Custom Settings

```bash
# Large buffer + ACK tracking
python meshtastic_monitor.py --auto --buffer 8192 --require-ack

# Different baud rate + faster stats
python meshtastic_monitor.py --ports /dev/ttyUSB1 -b 57600 --interval 5
```

## Metrics Tracked

### Channel Metrics

| Metric | Description |
|--------|-------------|
| **node_id** | Unique node identifier |
| **node_name** | Node display name |
| **frequency** | Operating frequency |
| **location** | GPS coordinates (lat/lon/alt) |

### Timing Metrics

| Metric | Description |
|--------|-------------|
| **uptime** | Monitor uptime in seconds |
| **last_received_time** | Timestamp of last RX |
| **last_broadcast_time** | Timestamp of last TX |
| **time_since_last_rx** | Seconds since last receive |
| **time_since_last_tx** | Seconds since last transmit |
| **avg_ack_time** | Average ACK response time (ms) |
| **avg_broadcast_time** | Average broadcast duration |
| **avg_repeat_time** | Average repeat interval |
| **avg_mqtt_time** | Average MQTT roundtrip time |

### Counters

| Counter | Description |
|---------|-------------|
| **total_messages** | Total messages received |
| **decoded_messages** | Successfully decoded messages |
| **failed_decodes** | Failed decode attempts |
| **total_broadcasts** | Total transmissions |
| **total_acks** | Total ACKs received |

## CLI Options

```
usage: meshtastic_monitor.py [-h] [--ports PORTS] [--auto]
                              [--baudrate BAUDRATE] [--buffer BUFFER]
                              [--require-ack] [--interval INTERVAL]
                              [--export EXPORT]

Options:
  --ports, -p       Comma-separated list of serial ports
  --auto, -a        Auto-detect Meshtastic ports
  --baudrate, -b    Serial baud rate (default: 115200)
  --buffer          Buffer size in bytes (default: 4096)
  --require-ack     Require ACK between data blocks
  --interval, -i    Stats display interval (default: 10s)
  --export, -e      Export packets to JSON file
```

## Example Output

```
Auto-detecting Meshtastic devices...
Found device: /dev/ttyACM0 - Meshtastic Device
Found device: /dev/ttyUSB0 - CH340 Serial
✓ Connected to /dev/ttyACM0 @ 115200 baud
✓ Connected to /dev/ttyUSB0 @ 115200 baud
✓ Monitoring started on /dev/ttyACM0
✓ Monitoring started on /dev/ttyUSB0

✓ Monitoring 2 port(s)

Press Ctrl+C to stop monitoring...
Stats will be displayed every 10 seconds

================================================================================
Meshtastic Multi-Port Monitor - Statistics
================================================================================

📡 /dev/ttyACM0
--------------------------------------------------------------------------------
  Connection: ✓ Connected
  Uptime: 10.45s

  Messages:
    Total: 23
    Decoded: 20
    Failed: 3
    Broadcasts: 5
    ACKs: 4

  Timing:
    Avg ACK time: 125.50ms
    Time since last RX: 2.34s
    Time since last TX: 5.67s

  Buffers:
    RX: 145/4096 (3.5%)
    TX: 67/4096 (1.6%)
    Packets logged: 23
    Pending ACKs: 1

📡 /dev/ttyUSB0
--------------------------------------------------------------------------------
  Connection: ✓ Connected
  Uptime: 10.45s

  Messages:
    Total: 18
    Decoded: 18
    Failed: 0
    Broadcasts: 3
    ACKs: 2

  Timing:
    Avg ACK time: 98.25ms
    Time since last RX: 1.12s
    Time since last TX: 8.45s

  Buffers:
    RX: 112/4096 (2.7%)
    TX: 45/4096 (1.1%)
    Packets logged: 18
    Pending ACKs: 0
```

## Exporting Data

### Export All Packets to JSON

```bash
python meshtastic_monitor.py --auto --export packets.json
```

Press Ctrl+C when done, and all packets will be exported.

### JSON Format

```json
[
  {
    "timestamp": 1701360000.123,
    "time_str": "2023-11-30T12:00:00.123000",
    "port": "/dev/ttyACM0",
    "direction": "rx",
    "data_hex": "940123456789abcdef",
    "data_len": 9,
    "decoded": true,
    "decode_error": "",
    "from_node": "!12345678",
    "to_node": "!ffffffff",
    "message_type": "TEXT_MESSAGE",
    "payload_hex": "48656c6c6f"
  }
]
```

## Python API Usage

```python
from meshtastic_monitor import MeshtasticMultiMonitor

# Create monitor
monitor = MeshtasticMultiMonitor(
    buffer_size=8192,
    require_ack=True
)

# Add ports
monitor.add_port('/dev/ttyACM0', baudrate=115200)
monitor.add_port('/dev/ttyUSB0', baudrate=115200)

# Or auto-detect
ports = monitor.auto_detect_ports()
for port in ports:
    monitor.add_port(port)

# Start monitoring
monitor.start_all()

try:
    while True:
        time.sleep(10)

        # Get stats
        stats = monitor.get_all_stats()
        print(stats)

except KeyboardInterrupt:
    pass
finally:
    monitor.stop_all()
    monitor.export_packets('my_packets.json')
```

## Advanced Usage

### Access Individual Port Monitors

```python
# Get specific monitor
monitor_acm = monitor.monitors['/dev/ttyACM0']

# Send data
monitor_acm.send_data(b'\x94\x01\x02\x03', wait_ack=True)

# Get buffer data
rx_data = list(monitor_acm.rx_buffer)
tx_data = list(monitor_acm.tx_buffer)

# Access raw packets
for packet in monitor_acm.raw_packets:
    print(f"{packet.timestamp}: {packet.data.hex()}")
```

### Custom Buffer Sizes

```python
# Different buffer sizes per port
monitor = MeshtasticMultiMonitor(buffer_size=4096)
monitor.monitors['/dev/ttyACM0'].rx_buffer = deque(maxlen=8192)
monitor.monitors['/dev/ttyUSB0'].rx_buffer = deque(maxlen=16384)
```

## Troubleshooting

### No Devices Found

```bash
# Check for devices manually
ls /dev/tty* | grep -E "(ACM|USB)"

# Check permissions
sudo usermod -a -G dialout $USER
# Logout and login again
```

### Permission Denied

```bash
# Add to dialout group
sudo usermod -a -G dialout $USER

# Or run with sudo (not recommended)
sudo python meshtastic_monitor.py --auto
```

### High Failed Decodes

- The monitor uses simplified packet parsing
- Real Meshtastic uses protobuf encoding
- Failed decodes are still logged for analysis
- Check raw packet hex in JSON export

### Buffer Overflow

If buffers fill up:
- Increase buffer size: `--buffer 16384`
- Reduce monitoring interval
- Export and clear data periodically

## Integration with FTP Protocol

The monitor can work alongside the FTP protocol:

```python
from meshtastic_monitor import MeshtasticMultiMonitor
from meshtastic_ftp import MeshtasticFTP, Packet

# Monitor traffic
monitor = MeshtasticMultiMonitor()
monitor.add_port('/dev/ttyACM0')
monitor.start_all()

# Also use for FTP
ftp = MeshtasticFTP(base_path="/data")
# ... FTP operations ...

# Monitor captures all traffic including FTP packets
```

## Tips

1. **Start Simple**: Use `--auto` first to detect devices
2. **Monitor Before FTP**: Run monitor to understand traffic patterns
3. **Export Regularly**: Export data before stopping to preserve logs
4. **Check Timing**: Use ACK times to tune FTP retry logic
5. **Buffer Sizing**: Start with 4096, increase if overflow occurs
6. **Multiple Nodes**: Monitor multiple nodes to see mesh behavior

## Dependencies

```bash
pip install pyserial
```

That's it! No other dependencies required.

## Examples

### Example 1: Quick Check

```bash
# Just check what's happening
python meshtastic_monitor.py --auto --interval 5
```

### Example 2: Long-Term Logging

```bash
# Run overnight, export in morning
nohup python meshtastic_monitor.py --auto --export overnight.json &

# Stop with:
pkill -f meshtastic_monitor
```

### Example 3: Dual-Port Debug

```bash
# Monitor two nodes simultaneously
python meshtastic_monitor.py \
  --ports /dev/ttyACM0,/dev/ttyACM1 \
  --buffer 8192 \
  --require-ack \
  --interval 3 \
  --export dual_node.json
```

---

**Version:** 1.0
**Compatible with:** Meshtastic FTP Protocol v1.0
**Last Updated:** 2025-11-30
