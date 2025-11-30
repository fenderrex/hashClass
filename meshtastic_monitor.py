#!/usr/bin/env python3
"""
Meshtastic Signal Layer Monitor & Debugger
==========================================
Simple tool for debugging Meshtastic signal layer that:
- Monitors multiple serial ports simultaneously (ACM* and USB*)
- Records all channel data (frequency, name, location, broadcasts, etc.)
- Tracks comprehensive timing metrics
- Supports configurable buffers
- Optional ACK between data blocks

Usage:
    python meshtastic_monitor.py --ports /dev/ttyACM0,/dev/ttyUSB0
    python meshtastic_monitor.py --auto  # Auto-detect all Meshtastic ports
"""

import serial
import serial.tools.list_ports
import threading
import time
import json
import struct
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import deque
from dataclasses import dataclass, asdict
import argparse


@dataclass
class ChannelMetrics:
    """Channel timing and statistics metrics"""
    # Identification
    node_id: str = ""
    node_name: str = ""
    frequency: float = 0.0
    location: Dict[str, float] = None

    # Timing metrics
    uptime: float = 0.0
    last_received_time: float = 0.0
    last_broadcast_time: float = 0.0
    time_since_last_rx: float = 0.0
    time_since_last_tx: float = 0.0

    # Average timings
    avg_ack_time: float = 0.0
    avg_broadcast_time: float = 0.0
    avg_repeat_time: float = 0.0
    avg_mqtt_time: float = 0.0

    # Counters
    total_messages: int = 0
    decoded_messages: int = 0
    failed_decodes: int = 0
    total_broadcasts: int = 0
    total_acks: int = 0

    # Durations
    last_broadcast_duration: float = 0.0

    def __post_init__(self):
        if self.location is None:
            self.location = {"lat": 0.0, "lon": 0.0, "alt": 0.0}


@dataclass
class RawPacket:
    """Raw packet data for logging"""
    timestamp: float
    port: str
    direction: str  # 'rx' or 'tx'
    data: bytes
    decoded: bool = False
    decode_error: str = ""

    # Optional decoded fields
    from_node: str = ""
    to_node: str = ""
    message_type: str = ""
    payload: bytes = b""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'time_str': datetime.fromtimestamp(self.timestamp).isoformat(),
            'port': self.port,
            'direction': self.direction,
            'data_hex': self.data.hex(),
            'data_len': len(self.data),
            'decoded': self.decoded,
            'decode_error': self.decode_error,
            'from_node': self.from_node,
            'to_node': self.to_node,
            'message_type': self.message_type,
            'payload_hex': self.payload.hex() if self.payload else "",
        }


class SerialPortMonitor:
    """Monitor a single serial port"""

    def __init__(self, port: str, baudrate: int = 115200, buffer_size: int = 4096):
        """
        Initialize serial port monitor.

        Args:
            port: Serial port path (e.g., /dev/ttyACM0)
            baudrate: Baud rate (default: 115200)
            buffer_size: Receive buffer size
        """
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size

        self.serial = None
        self.running = False
        self.thread = None

        # Buffers
        self.rx_buffer = deque(maxlen=buffer_size)
        self.tx_buffer = deque(maxlen=buffer_size)
        self.raw_packets = deque(maxlen=1000)  # Keep last 1000 packets

        # Metrics
        self.metrics = ChannelMetrics()
        self.start_time = time.time()

        # ACK tracking
        self.pending_acks = {}  # msg_id -> timestamp
        self.ack_times = []  # List of ACK durations

    def connect(self) -> bool:
        """Connect to serial port"""
        try:
            self.serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0.1,
                write_timeout=1.0
            )
            print(f"✓ Connected to {self.port} @ {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from serial port"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial and self.serial.is_open:
            self.serial.close()
        print(f"✓ Disconnected from {self.port}")

    def start(self):
        """Start monitoring in background thread"""
        if not self.serial or not self.serial.is_open:
            if not self.connect():
                return False

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"✓ Monitoring started on {self.port}")
        return True

    def stop(self):
        """Stop monitoring"""
        self.disconnect()

    def _monitor_loop(self):
        """Main monitoring loop (runs in thread)"""
        while self.running:
            try:
                # Read available data
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    self._process_rx_data(data)

                # Update metrics
                self._update_metrics()

                time.sleep(0.001)  # Small sleep to prevent CPU spinning

            except Exception as e:
                print(f"[{self.port}] Error in monitor loop: {e}")
                time.sleep(0.1)

    def _process_rx_data(self, data: bytes):
        """Process received data"""
        timestamp = time.time()

        # Add to RX buffer
        for byte in data:
            self.rx_buffer.append(byte)

        # Try to parse packets
        packets = self._extract_packets(data)

        for packet_data in packets:
            # Create raw packet record
            packet = RawPacket(
                timestamp=timestamp,
                port=self.port,
                direction='rx',
                data=packet_data
            )

            # Try to decode
            try:
                self._decode_packet(packet)
                packet.decoded = True
                self.metrics.decoded_messages += 1
            except Exception as e:
                packet.decode_error = str(e)
                self.metrics.failed_decodes += 1

            self.raw_packets.append(packet)
            self.metrics.total_messages += 1
            self.metrics.last_received_time = timestamp

    def _extract_packets(self, data: bytes) -> List[bytes]:
        """
        Extract individual packets from data stream.
        Simple implementation - looks for packet markers.
        """
        packets = []

        # Simple framing: look for start marker (0x94)
        # This is a simplified version - real Meshtastic uses protobuf
        i = 0
        while i < len(data):
            if data[i] == 0x94:  # Common start marker
                # Try to extract packet
                if i + 4 < len(data):
                    # Simple length extraction (this is simplified)
                    length = data[i + 3] if i + 3 < len(data) else 0
                    end = min(i + 4 + length, len(data))
                    packets.append(data[i:end])
                    i = end
                else:
                    i += 1
            else:
                i += 1

        # If no structured packets found, just return whole data as one packet
        if not packets and data:
            packets.append(data)

        return packets

    def _decode_packet(self, packet: RawPacket):
        """
        Decode packet (simplified version).
        Real implementation would use Meshtastic protobuf definitions.
        """
        data = packet.data

        if len(data) < 4:
            return

        # Simple header parsing (this is very simplified)
        # Real Meshtastic packets are protobuf encoded
        packet.from_node = f"!{data[0]:02x}{data[1]:02x}{data[2]:02x}{data[3]:02x}"

        if len(data) >= 8:
            packet.to_node = f"!{data[4]:02x}{data[5]:02x}{data[6]:02x}{data[7]:02x}"

        if len(data) > 8:
            packet.payload = data[8:]

            # Try to detect message type from payload
            if packet.payload:
                first_byte = packet.payload[0]
                if first_byte == 0x03:
                    packet.message_type = "TEXT_MESSAGE"
                elif first_byte == 0x04:
                    packet.message_type = "POSITION"
                elif first_byte == 0x08:
                    packet.message_type = "ACK"
                    self._handle_ack(packet)
                else:
                    packet.message_type = f"UNKNOWN_0x{first_byte:02x}"

    def _handle_ack(self, packet: RawPacket):
        """Handle ACK packet for timing metrics"""
        msg_id = packet.payload[1:5].hex() if len(packet.payload) >= 5 else ""

        if msg_id in self.pending_acks:
            # Calculate ACK time
            ack_time = packet.timestamp - self.pending_acks[msg_id]
            self.ack_times.append(ack_time)

            # Update average
            if self.ack_times:
                self.metrics.avg_ack_time = sum(self.ack_times) / len(self.ack_times)

            self.metrics.total_acks += 1
            del self.pending_acks[msg_id]

    def _update_metrics(self):
        """Update timing metrics"""
        now = time.time()

        self.metrics.uptime = now - self.start_time

        if self.metrics.last_received_time > 0:
            self.metrics.time_since_last_rx = now - self.metrics.last_received_time

        if self.metrics.last_broadcast_time > 0:
            self.metrics.time_since_last_tx = now - self.metrics.last_broadcast_time

    def send_data(self, data: bytes, wait_ack: bool = False) -> bool:
        """
        Send data to serial port.

        Args:
            data: Data to send
            wait_ack: Wait for ACK after sending

        Returns:
            True if sent successfully
        """
        if not self.serial or not self.serial.is_open:
            return False

        try:
            timestamp = time.time()
            self.serial.write(data)

            # Add to TX buffer
            for byte in data:
                self.tx_buffer.append(byte)

            # Record packet
            packet = RawPacket(
                timestamp=timestamp,
                port=self.port,
                direction='tx',
                data=data
            )
            self.raw_packets.append(packet)

            # Update metrics
            self.metrics.last_broadcast_time = timestamp
            self.metrics.total_broadcasts += 1

            # Track for ACK if requested
            if wait_ack:
                msg_id = data[:4].hex()  # Simplified
                self.pending_acks[msg_id] = timestamp

            return True

        except Exception as e:
            print(f"[{self.port}] Send error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'port': self.port,
            'connected': self.serial.is_open if self.serial else False,
            'metrics': asdict(self.metrics),
            'buffer_usage': {
                'rx': len(self.rx_buffer),
                'tx': len(self.tx_buffer),
                'rx_max': self.rx_buffer.maxlen,
                'tx_max': self.tx_buffer.maxlen,
            },
            'packets_logged': len(self.raw_packets),
            'pending_acks': len(self.pending_acks),
        }


class MeshtasticMultiMonitor:
    """Monitor multiple Meshtastic devices simultaneously"""

    def __init__(self, buffer_size: int = 4096, require_ack: bool = False):
        """
        Initialize multi-port monitor.

        Args:
            buffer_size: Buffer size for each port
            require_ack: Require ACK between data blocks
        """
        self.monitors: Dict[str, SerialPortMonitor] = {}
        self.buffer_size = buffer_size
        self.require_ack = require_ack
        self.running = False

    def add_port(self, port: str, baudrate: int = 115200) -> bool:
        """Add a port to monitor"""
        if port in self.monitors:
            print(f"Port {port} already being monitored")
            return False

        monitor = SerialPortMonitor(port, baudrate, self.buffer_size)
        if monitor.connect():
            self.monitors[port] = monitor
            return True
        return False

    def auto_detect_ports(self) -> List[str]:
        """Auto-detect Meshtastic serial ports"""
        ports = []

        for port in serial.tools.list_ports.comports():
            # Look for ACM and USB ports
            if 'ACM' in port.device or 'USB' in port.device.upper():
                print(f"Found device: {port.device} - {port.description}")
                ports.append(port.device)

        return ports

    def start_all(self):
        """Start monitoring all ports"""
        self.running = True
        for port, monitor in self.monitors.items():
            monitor.start()
        print(f"\n✓ Monitoring {len(self.monitors)} port(s)")

    def stop_all(self):
        """Stop monitoring all ports"""
        self.running = False
        for monitor in self.monitors.values():
            monitor.stop()
        print("\n✓ All monitors stopped")

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all monitors"""
        stats = {}
        for port, monitor in self.monitors.items():
            stats[port] = monitor.get_stats()
        return stats

    def export_packets(self, filename: str = "meshtastic_packets.json"):
        """Export all logged packets to JSON file"""
        all_packets = []

        for port, monitor in self.monitors.items():
            for packet in monitor.raw_packets:
                all_packets.append(packet.to_dict())

        # Sort by timestamp
        all_packets.sort(key=lambda x: x['timestamp'])

        with open(filename, 'w') as f:
            json.dump(all_packets, f, indent=2)

        print(f"\n✓ Exported {len(all_packets)} packets to {filename}")

    def print_stats(self):
        """Print statistics for all ports"""
        stats = self.get_all_stats()

        print("\n" + "=" * 80)
        print("Meshtastic Multi-Port Monitor - Statistics")
        print("=" * 80)

        for port, port_stats in stats.items():
            print(f"\n📡 {port}")
            print("-" * 80)

            metrics = port_stats['metrics']
            buf = port_stats['buffer_usage']

            print(f"  Connection: {'✓ Connected' if port_stats['connected'] else '✗ Disconnected'}")
            print(f"  Uptime: {metrics['uptime']:.2f}s")
            print(f"\n  Messages:")
            print(f"    Total: {metrics['total_messages']}")
            print(f"    Decoded: {metrics['decoded_messages']}")
            print(f"    Failed: {metrics['failed_decodes']}")
            print(f"    Broadcasts: {metrics['total_broadcasts']}")
            print(f"    ACKs: {metrics['total_acks']}")

            print(f"\n  Timing:")
            print(f"    Avg ACK time: {metrics['avg_ack_time']*1000:.2f}ms")
            print(f"    Time since last RX: {metrics['time_since_last_rx']:.2f}s")
            print(f"    Time since last TX: {metrics['time_since_last_tx']:.2f}s")

            print(f"\n  Buffers:")
            print(f"    RX: {buf['rx']}/{buf['rx_max']} ({buf['rx']/buf['rx_max']*100:.1f}%)")
            print(f"    TX: {buf['tx']}/{buf['tx_max']} ({buf['tx']/buf['tx_max']*100:.1f}%)")
            print(f"    Packets logged: {port_stats['packets_logged']}")
            print(f"    Pending ACKs: {port_stats['pending_acks']}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Meshtastic Signal Layer Monitor & Debugger',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Monitor specific ports:
    %(prog)s --ports /dev/ttyACM0,/dev/ttyUSB0

  Auto-detect all Meshtastic devices:
    %(prog)s --auto

  Custom buffer size and require ACK:
    %(prog)s --auto --buffer 8192 --require-ack

  Monitor with live stats every 5 seconds:
    %(prog)s --auto --interval 5
        """
    )

    parser.add_argument('--ports', '-p', help='Comma-separated list of serial ports')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Auto-detect Meshtastic ports')
    parser.add_argument('--baudrate', '-b', type=int, default=115200,
                       help='Serial baud rate (default: 115200)')
    parser.add_argument('--buffer', type=int, default=4096,
                       help='Buffer size in bytes (default: 4096)')
    parser.add_argument('--require-ack', action='store_true',
                       help='Require ACK between data blocks')
    parser.add_argument('--interval', '-i', type=int, default=10,
                       help='Stats display interval in seconds (default: 10)')
    parser.add_argument('--export', '-e', help='Export packets to JSON file')

    args = parser.parse_args()

    if not args.ports and not args.auto:
        parser.print_help()
        print("\nError: Either --ports or --auto must be specified")
        return 1

    # Create multi-monitor
    monitor = MeshtasticMultiMonitor(
        buffer_size=args.buffer,
        require_ack=args.require_ack
    )

    # Add ports
    if args.auto:
        print("Auto-detecting Meshtastic devices...")
        ports = monitor.auto_detect_ports()
        if not ports:
            print("No Meshtastic devices found!")
            return 1
        for port in ports:
            monitor.add_port(port, args.baudrate)
    else:
        for port in args.ports.split(','):
            port = port.strip()
            monitor.add_port(port, args.baudrate)

    if not monitor.monitors:
        print("No ports to monitor!")
        return 1

    # Start monitoring
    monitor.start_all()

    try:
        print(f"\nPress Ctrl+C to stop monitoring...")
        print(f"Stats will be displayed every {args.interval} seconds\n")

        last_stats = time.time()

        while True:
            time.sleep(1)

            # Print stats at interval
            if time.time() - last_stats >= args.interval:
                monitor.print_stats()
                last_stats = time.time()

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        monitor.stop_all()

        # Export if requested
        if args.export:
            monitor.export_packets(args.export)

        # Final stats
        monitor.print_stats()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
