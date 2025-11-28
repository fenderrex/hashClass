#!/usr/bin/env python3
"""
Meshtastic USB Terminal with Piping Support
============================================
Enhanced terminal with channel filtering and data piping capabilities.
Messages are strictly ordered by timestamp for reliable piping.

Usage:
    # Interactive mode (full UI)
    python3 meshtastic_pipe.py

    # Pipe mode - all messages
    python3 meshtastic_pipe.py --pipe

    # Filter by channel
    python3 meshtastic_pipe.py --pipe --channel Ch0
    python3 meshtastic_pipe.py --pipe --channel PM

    # Multiple channels
    python3 meshtastic_pipe.py --pipe --channel Ch0,Ch1,PM

    # Different output formats
    python3 meshtastic_pipe.py --pipe --format json
    python3 meshtastic_pipe.py --pipe --format csv
    python3 meshtastic_pipe.py --pipe --format text

    # Pipe to other programs
    python3 meshtastic_pipe.py --pipe --channel Ch0 | grep "keyword"
    python3 meshtastic_pipe.py --pipe --format json | jq '.text'
    python3 meshtastic_pipe.py --pipe --channel PM > private_messages.log

Examples:
    # Monitor only private messages
    python3 meshtastic_pipe.py --pipe --channel PM

    # Get all Ch0 messages as JSON
    python3 meshtastic_pipe.py --pipe --channel Ch0 --format json

    # Filter and search
    python3 meshtastic_pipe.py --pipe | grep "emergency"

Dependencies:
    pip install meshtastic pyserial
"""

import curses
import time
import threading
import queue
import math
import json
import csv
import argparse
import sys
import os
import glob
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Set

try:
    import meshtastic
    import meshtastic.serial_interface
    from meshtastic.protobuf import portnums_pb2, mesh_pb2
except ImportError:
    print("Error: meshtastic library not found!", file=sys.stderr)
    print("Install with: pip install meshtastic pyserial", file=sys.stderr)
    sys.exit(1)


@dataclass
class Message:
    """Represents a received message with strict ordering"""
    timestamp: str
    timestamp_unix: float  # For strict ordering
    channel: str
    sender: str
    sender_id: str  # Full node ID
    sender_name: Optional[str] = None  # User-friendly name if available
    text: str = ""
    is_own: bool = False
    packet_type: str = "USER"  # USER, MQTT, MESH
    airtime_ms: float = 0.0
    hop_count: int = 0  # Number of hops
    snr: float = 0.0  # Signal-to-noise ratio
    rssi: int = 0  # Received signal strength
    raw_packet: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for JSON output"""
        return {
            'timestamp': self.timestamp,
            'timestamp_unix': self.timestamp_unix,
            'channel': self.channel,
            'sender': self.sender,
            'sender_id': self.sender_id,
            'sender_name': self.sender_name,
            'text': self.text,
            'is_own': self.is_own,
            'packet_type': self.packet_type,
            'airtime_ms': self.airtime_ms,
            'hop_count': self.hop_count,
            'snr': self.snr,
            'rssi': self.rssi
        }

    def to_csv_row(self):
        """Convert to CSV row"""
        return [
            self.timestamp,
            self.channel,
            self.sender,
            self.sender_id,
            self.sender_name or "",
            self.packet_type,
            self.text,
            str(self.is_own),
            f"{self.airtime_ms:.2f}",
            str(self.hop_count),
            f"{self.snr:.1f}",
            str(self.rssi)
        ]

    def to_text(self):
        """Convert to human-readable text with sender info"""
        me_tag = "[ME] " if self.is_own else ""
        type_tag = f"[{self.packet_type[0]}] " if self.packet_type != "USER" else ""

        # Build sender display
        sender_display = self.sender
        if self.sender_name:
            sender_display = f"{self.sender_name}({self.sender})"

        # Add hop count if > 0
        hop_info = f" ↔{self.hop_count}" if self.hop_count > 0 else ""

        # Add signal info if available
        signal_info = ""
        if self.rssi != 0:
            signal_info = f" [{self.rssi}dBm"
            if self.snr != 0:
                signal_info += f" SNR:{self.snr:.1f}]"
            else:
                signal_info += "]"

        return f"{self.timestamp} {type_tag}{me_tag}{self.channel:4s} {sender_display:12s}{hop_info}{signal_info}: {self.text}"


@dataclass
class ChannelStats:
    """Statistics for a channel"""
    message_count: int = 0
    user_messages: int = 0
    mqtt_messages: int = 0
    mesh_overhead: int = 0
    total_airtime_ms: float = 0.0


class MeshtasticPipe:
    """Meshtastic terminal with piping support"""

    # Color pairs (for UI mode)
    COLOR_PM = 1
    COLOR_CH0 = 2
    COLOR_CH1 = 3
    COLOR_CH2 = 4
    COLOR_MQTT = 5
    COLOR_MESH = 6
    COLOR_ME = 7
    COLOR_DUTY_OK = 8
    COLOR_DUTY_WARN = 9
    COLOR_DUTY_DANGER = 10
    COLOR_HEADER = 11

    # LoRa airtime estimation constants
    LORA_PREAMBLE_MS = 50
    LORA_BYTE_MS = 5.0

    def __init__(self, pipe_mode=False, channel_filter=None, output_format='text'):
        self.interface: Optional[meshtastic.serial_interface.SerialInterface] = None
        self.messages: deque = deque(maxlen=10000)  # Larger buffer for piping
        self.message_queue: queue.Queue = queue.Queue()
        self.running = True
        self.input_buffer = ""
        self.current_channel = 0
        self.my_node_id = None

        # Pipe mode settings
        self.pipe_mode = pipe_mode
        self.channel_filter: Set[str] = set(channel_filter) if channel_filter else set()
        self.output_format = output_format

        # CSV writer (if needed)
        self.csv_writer = None
        if pipe_mode and output_format == 'csv':
            import io
            self.csv_writer = csv.writer(sys.stdout)
            # Write header
            self.csv_writer.writerow(['timestamp', 'channel', 'sender', 'sender_id', 'sender_name', 'type', 'text', 'is_own', 'airtime_ms', 'hop_count', 'snr', 'rssi'])
            sys.stdout.flush()

        # Node database for name lookup
        self.node_db: Dict[int, str] = {}  # node_id -> name

        # Statistics
        self.channel_stats: Dict[str, ChannelStats] = defaultdict(ChannelStats)
        self.total_airtime_ms = 0.0
        self.start_time = time.time()

        # Window size (for UI mode)
        self.height = 0
        self.width = 0

        # Message lock for thread safety
        self.message_lock = threading.Lock()

    def find_meshtastic_ports(self) -> List[str]:
        """Find all potential Meshtastic USB devices"""
        ports = []

        # Linux/Mac USB serial ports
        for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/cu.usbserial*', '/dev/cu.usbmodem*']:
            ports.extend(glob.glob(pattern))

        # Windows COM ports
        if sys.platform == 'win32':
            try:
                import serial.tools.list_ports
                for port in serial.tools.list_ports.comports():
                    if 'USB' in port.description or 'Serial' in port.description:
                        ports.append(port.device)
            except:
                pass

        return sorted(ports)

    def connect_to_device(self) -> bool:
        """Auto-connect to first available Meshtastic device"""
        ports = self.find_meshtastic_ports()

        if not ports:
            return False

        for port in ports:
            try:
                if self.pipe_mode:
                    print(f"Connecting to {port}...", file=sys.stderr)

                self.interface = meshtastic.serial_interface.SerialInterface(port)
                self.interface.onReceive = self.on_receive

                # Get our node ID
                if self.interface.myInfo:
                    self.my_node_id = self.interface.myInfo.my_node_num

                if self.pipe_mode:
                    print(f"Connected to {port}", file=sys.stderr)
                    if self.my_node_id:
                        print(f"Node ID: !{self.my_node_id:08x}", file=sys.stderr)
                    if self.channel_filter:
                        print(f"Filtering channels: {', '.join(self.channel_filter)}", file=sys.stderr)
                    print(f"Output format: {self.output_format}", file=sys.stderr)
                    print("---", file=sys.stderr)

                return True
            except Exception as e:
                if self.pipe_mode:
                    print(f"Failed to connect to {port}: {e}", file=sys.stderr)
                continue

        return False

    def estimate_airtime(self, payload_size: int) -> float:
        """Estimate airtime in milliseconds for a packet"""
        return self.LORA_PREAMBLE_MS + (payload_size * self.LORA_BYTE_MS)

    def get_duty_cycle_percent(self) -> float:
        """Calculate current duty cycle percentage"""
        elapsed_sec = time.time() - self.start_time
        if elapsed_sec == 0:
            return 0.0
        elapsed_ms = elapsed_sec * 1000
        return (self.total_airtime_ms / elapsed_ms) * 100 if elapsed_ms > 0 else 0.0

    def should_output_message(self, msg: Message) -> bool:
        """Check if message should be output based on channel filter"""
        if not self.channel_filter:
            return True
        return msg.channel in self.channel_filter

    def output_message(self, msg: Message):
        """Output message in pipe mode"""
        if not self.should_output_message(msg):
            return

        try:
            if self.output_format == 'json':
                print(json.dumps(msg.to_dict()), flush=True)
            elif self.output_format == 'csv':
                if self.csv_writer:
                    self.csv_writer.writerow(msg.to_csv_row())
                    sys.stdout.flush()
            else:  # text
                print(msg.to_text(), flush=True)
        except BrokenPipeError:
            # Handle broken pipe gracefully
            self.running = False
            sys.exit(0)
        except Exception as e:
            print(f"Output error: {e}", file=sys.stderr)

    def on_receive(self, packet, interface):
        """Callback for received packets"""
        try:
            # Get timestamp immediately for ordering
            timestamp_unix = time.time()
            timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Determine packet type and channel
            portnum = packet.get('decoded', {}).get('portnum')
            channel_index = packet.get('channel', 0)

            # Determine if it's a PM
            if channel_index == 0 and packet.get('to') != 0xFFFFFFFF:
                channel_name = "PM"
            else:
                channel_name = f"Ch{channel_index}"

            # Get sender info
            from_id = packet.get('from')
            is_own = from_id == self.my_node_id
            sender_short = "ME" if is_own else f"!{from_id:08x}"[-4:]
            sender_id = f"!{from_id:08x}" if from_id else "unknown"

            # Try to get sender name from node database
            sender_name = None
            if from_id and from_id in self.node_db:
                sender_name = self.node_db[from_id]
            elif self.interface and hasattr(self.interface, 'nodes'):
                # Try to get from interface's node DB
                for node_id, node in self.interface.nodes.items():
                    if node_id == from_id:
                        user_info = node.get('user', {})
                        if 'longName' in user_info:
                            sender_name = user_info['longName']
                            self.node_db[from_id] = sender_name
                        elif 'shortName' in user_info:
                            sender_name = user_info['shortName']
                            self.node_db[from_id] = sender_name
                        break

            # Get signal metrics
            hop_count = packet.get('hopLimit', 0) - packet.get('hopStart', 0) if 'hopLimit' in packet else 0
            if hop_count < 0:
                hop_count = 0

            rssi = packet.get('rxRssi', 0)
            snr = packet.get('rxSnr', 0.0)

            # Determine packet type
            packet_type = "MESH"
            if portnum == portnums_pb2.TEXT_MESSAGE_APP:
                packet_type = "USER"
            elif portnum == portnums_pb2.POSITION_APP:
                packet_type = "MESH"
            elif 'mqtt' in str(packet).lower():
                packet_type = "MQTT"

            # Get message text
            text = ""
            if 'decoded' in packet and 'text' in packet['decoded']:
                text = packet['decoded']['text']
            elif 'decoded' in packet and 'payload' in packet['decoded']:
                payload = packet['decoded']['payload']
                if isinstance(payload, bytes):
                    try:
                        text = payload.decode('utf-8')
                    except:
                        text = f"[Binary: {len(payload)} bytes]"
                else:
                    text = str(payload)
            else:
                text = f"[{portnum}]"

            # Estimate airtime
            packet_size = len(str(packet).encode('utf-8'))
            airtime = self.estimate_airtime(packet_size)

            # Create message with strict ordering
            msg = Message(
                timestamp=timestamp_str,
                timestamp_unix=timestamp_unix,
                channel=channel_name,
                sender=sender_short,
                sender_id=sender_id,
                sender_name=sender_name,
                text=text,
                is_own=is_own,
                packet_type=packet_type,
                airtime_ms=airtime,
                hop_count=hop_count,
                snr=snr,
                rssi=rssi,
                raw_packet=packet
            )

            # Update statistics
            with self.message_lock:
                stats = self.channel_stats[channel_name]
                stats.message_count += 1
                stats.total_airtime_ms += airtime

                if packet_type == "USER":
                    stats.user_messages += 1
                elif packet_type == "MQTT":
                    stats.mqtt_messages += 1
                else:
                    stats.mesh_overhead += 1

                self.total_airtime_ms += airtime

            # In pipe mode, output immediately with proper ordering
            if self.pipe_mode:
                self.output_message(msg)
            else:
                # In UI mode, queue for display
                self.message_queue.put(msg)

        except Exception as e:
            if self.pipe_mode:
                print(f"Parse error: {e}", file=sys.stderr)
            else:
                error_msg = Message(
                    timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    timestamp_unix=time.time(),
                    channel="SYS",
                    sender="ERR",
                    sender_id="!00000000",
                    sender_name="System",
                    text=f"Parse error: {str(e)}",
                    packet_type="MESH",
                    hop_count=0,
                    snr=0.0,
                    rssi=0
                )
                self.message_queue.put(error_msg)

    def send_message(self, text: str, channel: int = 0, dest: Optional[str] = None):
        """Send a message"""
        if not self.interface:
            return

        try:
            if dest:
                # Private message
                if dest.startswith('!'):
                    dest = dest[1:]
                dest_id = int(dest, 16)
                self.interface.sendText(text, destinationId=dest_id, channelIndex=channel)
            else:
                # Broadcast to channel
                self.interface.sendText(text, channelIndex=channel)

            # Add to our message list
            timestamp_unix = time.time()
            msg = Message(
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                timestamp_unix=timestamp_unix,
                channel=f"Ch{channel}" if not dest else "PM",
                sender="ME",
                sender_id=f"!{self.my_node_id:08x}" if self.my_node_id else "!00000000",
                sender_name="You",
                text=text,
                is_own=True,
                packet_type="USER",
                airtime_ms=self.estimate_airtime(len(text)),
                hop_count=0,
                snr=0.0,
                rssi=0
            )

            if self.pipe_mode:
                self.output_message(msg)
            else:
                self.message_queue.put(msg)

        except Exception as e:
            if self.pipe_mode:
                print(f"Send error: {e}", file=sys.stderr)

    def run_pipe_mode(self):
        """Run in pipe mode (non-interactive)"""
        print("Press Ctrl+C to stop...", file=sys.stderr)
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.interface:
                self.interface.close()

    # UI mode functions (same as original terminal)
    def handle_command(self, cmd: str):
        """Handle special commands"""
        parts = cmd.split(None, 1)
        command = parts[0].lower()

        if command == '/quit' or command == '/q':
            self.running = False
        elif command == '/pm' and len(parts) > 1:
            args = parts[1].split(None, 1)
            if len(args) >= 2:
                node_id = args[0]
                message = args[1]
                self.send_message(message, dest=node_id)
        elif command == '/ch' and len(parts) > 1:
            args = parts[1].split(None, 1)
            try:
                channel = int(args[0])
                if len(args) >= 2:
                    self.send_message(args[1], channel=channel)
                else:
                    self.current_channel = channel
            except ValueError:
                pass

    def draw_progress_bar(self, stdscr, y: int, x: int, width: int, percent: float, label: str):
        """Draw a progress bar"""
        try:
            filled = int((percent / 100.0) * width)
            filled = max(0, min(width, filled))

            if percent < 15:
                color = curses.color_pair(self.COLOR_DUTY_OK)
            elif percent < 20:
                color = curses.color_pair(self.COLOR_DUTY_WARN)
            else:
                color = curses.color_pair(self.COLOR_DUTY_DANGER)

            stdscr.addstr(y, x, label[:15].ljust(15))
            bar_x = x + 16
            if bar_x + width < self.width:
                stdscr.addstr(y, bar_x, "[", color)
                if filled > 0:
                    stdscr.addstr(y, bar_x + 1, "=" * filled, color | curses.A_BOLD)
                if filled < width - 2:
                    stdscr.addstr(y, bar_x + 1 + filled, " " * (width - 2 - filled))
                stdscr.addstr(y, bar_x + width - 1, "]", color)
                pct_text = f" {percent:5.2f}%"
                if bar_x + width + len(pct_text) < self.width:
                    stdscr.addstr(y, bar_x + width, pct_text, color | curses.A_BOLD)
        except:
            pass

    def draw_header(self, stdscr):
        """Draw header with statistics"""
        try:
            header_color = curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD
            title = "═══ MESHTASTIC USB TERMINAL (PIPE MODE) ═══"
            stdscr.addstr(0, (self.width - len(title)) // 2, title, header_color)

            device_info = f"Device: {self.interface.stream.port if self.interface else 'None'}"
            if self.my_node_id:
                device_info += f" | Node: !{self.my_node_id:08x}"
            stdscr.addstr(1, 0, device_info[:self.width-1])

            counts = []
            for ch_name in sorted(self.channel_stats.keys()):
                stats = self.channel_stats[ch_name]
                counts.append(f"{ch_name}:{stats.message_count}")
            count_str = "Messages: " + " ".join(counts)
            stdscr.addstr(2, 0, count_str[:self.width-1])

            total_user = sum(s.user_messages for s in self.channel_stats.values())
            total_mqtt = sum(s.mqtt_messages for s in self.channel_stats.values())
            total_mesh = sum(s.mesh_overhead for s in self.channel_stats.values())
            traffic = f"User:{total_user} MQTT:{total_mqtt} Mesh:{total_mesh}"
            stdscr.addstr(3, 0, traffic[:self.width-1])

            duty = self.get_duty_cycle_percent()
            airtime_str = f"Airtime: {self.total_airtime_ms/1000:.2f}s"
            stdscr.addstr(4, 0, airtime_str[:self.width-1])

            bar_width = min(40, self.width - 35)
            if bar_width > 10:
                self.draw_progress_bar(stdscr, 5, 0, bar_width, duty, "Duty Cycle:")

            stdscr.addstr(6, 0, "─" * (self.width - 1))
        except:
            pass

    def draw_messages(self, stdscr, start_y: int, height: int):
        """Draw message list"""
        try:
            display_msgs = list(self.messages)[-height:]
            for i, msg in enumerate(display_msgs):
                y = start_y + i
                if y >= self.height - 2:
                    break

                if msg.is_own:
                    color = curses.color_pair(self.COLOR_ME) | curses.A_BOLD
                elif msg.channel == "PM":
                    color = curses.color_pair(self.COLOR_PM)
                elif msg.channel == "Ch0":
                    color = curses.color_pair(self.COLOR_CH0)
                elif msg.channel == "Ch1":
                    color = curses.color_pair(self.COLOR_CH1)
                else:
                    color = curses.color_pair(self.COLOR_CH2)

                me_tag = "[ME] " if msg.is_own else ""
                type_indicator = ""
                if msg.packet_type == "MQTT":
                    type_indicator = "[M] "
                    color = curses.color_pair(self.COLOR_MQTT)
                elif msg.packet_type == "MESH":
                    type_indicator = "[~] "
                    color = curses.color_pair(self.COLOR_MESH)

                line = f"{msg.timestamp} {type_indicator}{me_tag}{msg.channel:4s} {msg.sender:6s}: {msg.text}"
                if len(line) > self.width - 1:
                    line = line[:self.width-4] + "..."
                stdscr.addstr(y, 0, line, color)
        except:
            pass

    def draw_input(self, stdscr):
        """Draw input line"""
        try:
            y = self.height - 2
            stdscr.addstr(y, 0, "─" * (self.width - 1))
            y = self.height - 1
            prompt = f"Ch{self.current_channel}> "
            stdscr.addstr(y, 0, prompt)
            input_x = len(prompt)
            display_input = self.input_buffer
            max_input_len = self.width - input_x - 1
            if len(display_input) > max_input_len:
                display_input = display_input[-max_input_len:]
            stdscr.addstr(y, input_x, display_input + "_")
        except:
            pass

    def init_colors(self):
        """Initialize color pairs"""
        curses.init_pair(self.COLOR_PM, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_CH0, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_CH1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_CH2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_MQTT, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_MESH, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_ME, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_DUTY_OK, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_DUTY_WARN, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_DUTY_DANGER, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_HEADER, curses.COLOR_CYAN, curses.COLOR_BLACK)

    def update_screen_size(self, stdscr):
        """Update screen dimensions"""
        self.height, self.width = stdscr.getmaxyx()

    def process_messages(self):
        """Process queued messages"""
        while not self.message_queue.empty():
            try:
                msg = self.message_queue.get_nowait()
                self.messages.append(msg)
            except queue.Empty:
                break

    def run_ui_mode(self, stdscr):
        """Run in UI mode (interactive)"""
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)

        self.init_colors()
        self.update_screen_size(stdscr)

        stdscr.clear()
        stdscr.addstr(0, 0, "Searching for Meshtastic devices...")
        stdscr.refresh()

        if not self.connect_to_device():
            stdscr.clear()
            stdscr.addstr(0, 0, "ERROR: No Meshtastic device found!")
            stdscr.addstr(1, 0, "Please connect a device and try again.")
            stdscr.addstr(2, 0, "Press any key to exit...")
            stdscr.nodelay(False)
            stdscr.getch()
            return

        last_draw = time.time()

        while self.running:
            current_time = time.time()
            self.process_messages()

            try:
                key = stdscr.getch()
                if key != -1:
                    if key == 27:
                        self.running = False
                    elif key in (curses.KEY_BACKSPACE, 127, 8):
                        if self.input_buffer:
                            self.input_buffer = self.input_buffer[:-1]
                    elif key == 10 or key == 13:
                        if self.input_buffer:
                            if self.input_buffer.startswith('/'):
                                self.handle_command(self.input_buffer)
                            else:
                                self.send_message(self.input_buffer, channel=self.current_channel)
                            self.input_buffer = ""
                    elif key == curses.KEY_RESIZE:
                        self.update_screen_size(stdscr)
                    elif 32 <= key <= 126:
                        self.input_buffer += chr(key)
            except:
                pass

            if current_time - last_draw >= 0.05:
                stdscr.clear()
                self.update_screen_size(stdscr)
                self.draw_header(stdscr)
                msg_start_y = 7
                msg_height = self.height - msg_start_y - 2
                self.draw_messages(stdscr, msg_start_y, msg_height)
                self.draw_input(stdscr)
                stdscr.refresh()
                last_draw = current_time

        if self.interface:
            self.interface.close()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Meshtastic USB Terminal with Piping Support',
        epilog='''
Examples:
  # Interactive UI mode
  %(prog)s

  # Pipe all messages
  %(prog)s --pipe

  # Pipe only private messages
  %(prog)s --pipe --channel PM

  # Pipe multiple channels as JSON
  %(prog)s --pipe --channel Ch0,Ch1 --format json

  # Pipe and filter
  %(prog)s --pipe | grep "emergency"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--pipe', action='store_true',
                        help='Run in pipe mode (non-interactive, output to stdout)')

    parser.add_argument('--channel', '--channels', type=str,
                        help='Filter by channel(s). Comma-separated list. Examples: PM, Ch0, "Ch0,Ch1,PM"')

    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text',
                        help='Output format for pipe mode (default: text)')

    parser.add_argument('--port', type=str,
                        help='Specific serial port to use (otherwise auto-detect)')

    return parser.parse_args()


def main():
    """Entry point"""
    args = parse_args()

    # Parse channel filter
    channel_filter = None
    if args.channel:
        channel_filter = [ch.strip() for ch in args.channel.split(',')]

    # Create terminal instance
    terminal = MeshtasticPipe(
        pipe_mode=args.pipe,
        channel_filter=channel_filter,
        output_format=args.format
    )

    try:
        if args.pipe:
            # Pipe mode: connect and stream to stdout
            if not terminal.connect_to_device():
                print("ERROR: No Meshtastic device found!", file=sys.stderr)
                print("Connect a device and try again.", file=sys.stderr)
                sys.exit(1)
            terminal.run_pipe_mode()
        else:
            # UI mode: run curses interface
            curses.wrapper(terminal.run_ui_mode)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        # Handle pipe being closed
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
