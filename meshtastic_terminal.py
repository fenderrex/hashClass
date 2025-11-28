#!/usr/bin/env python3
"""
Meshtastic USB Terminal
=======================
A comprehensive terminal interface for Meshtastic devices with real-time
message monitoring, channel tracking, duty cycle monitoring, and more.

Features:
- Auto-detect USB Meshtastic devices
- Real-time message monitoring
- Color-coded channels (PM=Magenta, Ch0=Green, Ch1=Yellow, Ch2+=Cyan)
- MQTT overhead tracking
- Channel utilization monitoring
- 25% duty cycle limit monitoring with color warnings
- Airtime usage display (milliseconds)
- Progress bar for duty cycle
- Interactive messaging (/pm, /ch, /quit)
- Per-channel message counts
- [ME] tag for own messages
- Fast updates (7+ times per second)
- Preserved user input during screen updates

Usage:
    python3 meshtastic_terminal.py

Dependencies:
    pip install meshtastic pyserial
"""

import curses
import time
import threading
import queue
import math
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import glob
import sys
import os

try:
    import meshtastic
    import meshtastic.serial_interface
    from meshtastic.protobuf import portnums_pb2, mesh_pb2
except ImportError:
    print("Error: meshtastic library not found!")
    print("Install with: pip install meshtastic pyserial")
    sys.exit(1)


@dataclass
class Message:
    """Represents a received message"""
    timestamp: str
    channel: str
    sender: str
    text: str
    is_own: bool = False
    packet_type: str = "USER"  # USER, MQTT, MESH
    airtime_ms: float = 0.0


@dataclass
class ChannelStats:
    """Statistics for a channel"""
    message_count: int = 0
    user_messages: int = 0
    mqtt_messages: int = 0
    mesh_overhead: int = 0
    total_airtime_ms: float = 0.0


class MeshtasticTerminal:
    """Main terminal application"""

    # Color pairs
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

    # LoRa airtime estimation constants (rough estimates)
    LORA_PREAMBLE_MS = 50
    LORA_BYTE_MS = 5.0

    def __init__(self):
        self.interface: Optional[meshtastic.serial_interface.SerialInterface] = None
        self.messages: deque = deque(maxlen=1000)
        self.message_queue: queue.Queue = queue.Queue()
        self.running = True
        self.input_buffer = ""
        self.current_channel = 0
        self.my_node_id = None

        # Statistics
        self.channel_stats: Dict[str, ChannelStats] = defaultdict(ChannelStats)
        self.total_airtime_ms = 0.0
        self.start_time = time.time()

        # Window size
        self.height = 0
        self.width = 0

    def find_meshtastic_ports(self) -> List[str]:
        """Find all potential Meshtastic USB devices"""
        ports = []

        # Linux/Mac USB serial ports
        for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/cu.usbserial*', '/dev/cu.usbmodem*']:
            ports.extend(glob.glob(pattern))

        # Windows COM ports (if running on Windows)
        if sys.platform == 'win32':
            import serial.tools.list_ports
            for port in serial.tools.list_ports.comports():
                if 'USB' in port.description or 'Serial' in port.description:
                    ports.append(port.device)

        return sorted(ports)

    def connect_to_device(self) -> bool:
        """Auto-connect to first available Meshtastic device"""
        ports = self.find_meshtastic_ports()

        if not ports:
            return False

        for port in ports:
            try:
                self.interface = meshtastic.serial_interface.SerialInterface(port)
                self.interface.onReceive = self.on_receive

                # Get our node ID
                if self.interface.myInfo:
                    self.my_node_id = self.interface.myInfo.my_node_num

                return True
            except Exception as e:
                continue

        return False

    def estimate_airtime(self, payload_size: int) -> float:
        """Estimate airtime in milliseconds for a packet"""
        # This is a rough estimate - actual airtime depends on:
        # - Spreading factor (SF)
        # - Bandwidth
        # - Coding rate
        # - Preamble length
        # Typical LoRa medium range: ~200ms for 50 bytes
        return self.LORA_PREAMBLE_MS + (payload_size * self.LORA_BYTE_MS)

    def get_duty_cycle_percent(self) -> float:
        """Calculate current duty cycle percentage"""
        elapsed_sec = time.time() - self.start_time
        if elapsed_sec == 0:
            return 0.0

        # Duty cycle = (total airtime / elapsed time) * 100
        elapsed_ms = elapsed_sec * 1000
        return (self.total_airtime_ms / elapsed_ms) * 100 if elapsed_ms > 0 else 0.0

    def on_receive(self, packet, interface):
        """Callback for received packets"""
        try:
            # Determine packet type and channel
            portnum = packet.get('decoded', {}).get('portnum')
            channel_index = packet.get('channel', 0)

            # Determine if it's a PM (channel 0 is typically primary)
            if channel_index == 0 and packet.get('to') != 0xFFFFFFFF:
                channel_name = "PM"
            else:
                channel_name = f"Ch{channel_index}"

            # Get sender info
            from_id = packet.get('from')
            is_own = from_id == self.my_node_id

            sender = "ME" if is_own else f"!{from_id:08x}"[-4:]

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

            # Create message
            msg = Message(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                channel=channel_name,
                sender=sender,
                text=text,
                is_own=is_own,
                packet_type=packet_type,
                airtime_ms=airtime
            )

            # Update statistics
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

            # Add to queue
            self.message_queue.put(msg)

        except Exception as e:
            # Debug: create error message
            error_msg = Message(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                channel="SYS",
                sender="ERR",
                text=f"Parse error: {str(e)}",
                packet_type="MESH"
            )
            self.message_queue.put(error_msg)

    def send_message(self, text: str, channel: int = 0, dest: Optional[str] = None):
        """Send a message"""
        if not self.interface:
            return

        try:
            if dest:
                # Private message - parse node ID
                if dest.startswith('!'):
                    dest = dest[1:]
                dest_id = int(dest, 16)
                self.interface.sendText(text, destinationId=dest_id, channelIndex=channel)
            else:
                # Broadcast to channel
                self.interface.sendText(text, channelIndex=channel)

            # Add to our message list
            msg = Message(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                channel=f"Ch{channel}" if not dest else "PM",
                sender="ME",
                text=text,
                is_own=True,
                packet_type="USER",
                airtime_ms=self.estimate_airtime(len(text))
            )
            self.message_queue.put(msg)

        except Exception as e:
            error_msg = Message(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                channel="SYS",
                sender="ERR",
                text=f"Send error: {str(e)}",
                packet_type="MESH"
            )
            self.message_queue.put(error_msg)

    def handle_command(self, cmd: str):
        """Handle special commands"""
        parts = cmd.split(None, 1)
        command = parts[0].lower()

        if command == '/quit' or command == '/q':
            self.running = False

        elif command == '/pm' and len(parts) > 1:
            # /pm <node_id> <message>
            args = parts[1].split(None, 1)
            if len(args) >= 2:
                node_id = args[0]
                message = args[1]
                self.send_message(message, dest=node_id)

        elif command == '/ch' and len(parts) > 1:
            # /ch <channel_num> [message]
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

            # Choose color based on duty cycle
            if percent < 15:
                color = curses.color_pair(self.COLOR_DUTY_OK)
            elif percent < 20:
                color = curses.color_pair(self.COLOR_DUTY_WARN)
            else:
                color = curses.color_pair(self.COLOR_DUTY_DANGER)

            # Draw label
            stdscr.addstr(y, x, label[:15].ljust(15))

            # Draw bar
            bar_x = x + 16
            if bar_x + width < self.width:
                stdscr.addstr(y, bar_x, "[", color)
                if filled > 0:
                    stdscr.addstr(y, bar_x + 1, "=" * filled, color | curses.A_BOLD)
                if filled < width - 2:
                    stdscr.addstr(y, bar_x + 1 + filled, " " * (width - 2 - filled))
                stdscr.addstr(y, bar_x + width - 1, "]", color)

                # Draw percentage
                pct_text = f" {percent:5.2f}%"
                if bar_x + width + len(pct_text) < self.width:
                    stdscr.addstr(y, bar_x + width, pct_text, color | curses.A_BOLD)
        except:
            pass

    def draw_header(self, stdscr):
        """Draw header with statistics"""
        try:
            header_color = curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD

            # Line 0: Title
            title = "═══ MESHTASTIC USB TERMINAL ═══"
            stdscr.addstr(0, (self.width - len(title)) // 2, title, header_color)

            # Line 1: Device info
            device_info = f"Device: {self.interface.stream.port if self.interface else 'None'}"
            if self.my_node_id:
                device_info += f" | Node: !{self.my_node_id:08x}"
            stdscr.addstr(1, 0, device_info[:self.width-1])

            # Line 2: Channel message counts
            counts = []
            for ch_name in sorted(self.channel_stats.keys()):
                stats = self.channel_stats[ch_name]
                counts.append(f"{ch_name}:{stats.message_count}")

            count_str = "Messages: " + " ".join(counts)
            stdscr.addstr(2, 0, count_str[:self.width-1])

            # Line 3: Traffic breakdown
            total_user = sum(s.user_messages for s in self.channel_stats.values())
            total_mqtt = sum(s.mqtt_messages for s in self.channel_stats.values())
            total_mesh = sum(s.mesh_overhead for s in self.channel_stats.values())

            traffic = f"User:{total_user} MQTT:{total_mqtt} Mesh:{total_mesh}"
            stdscr.addstr(3, 0, traffic[:self.width-1])

            # Line 4: Airtime and duty cycle
            duty = self.get_duty_cycle_percent()
            airtime_str = f"Airtime: {self.total_airtime_ms/1000:.2f}s"
            stdscr.addstr(4, 0, airtime_str[:self.width-1])

            # Line 5: Duty cycle progress bar
            bar_width = min(40, self.width - 35)
            if bar_width > 10:
                self.draw_progress_bar(stdscr, 5, 0, bar_width, duty, "Duty Cycle:")

            # Line 6: Separator
            stdscr.addstr(6, 0, "─" * (self.width - 1))

        except:
            pass

    def draw_messages(self, stdscr, start_y: int, height: int):
        """Draw message list"""
        try:
            # Get messages to display (most recent at bottom)
            display_msgs = list(self.messages)[-height:]

            for i, msg in enumerate(display_msgs):
                y = start_y + i
                if y >= self.height - 2:
                    break

                # Choose color based on channel
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

                # Format message
                me_tag = "[ME] " if msg.is_own else ""

                # Packet type indicator
                type_indicator = ""
                if msg.packet_type == "MQTT":
                    type_indicator = "[M] "
                    color = curses.color_pair(self.COLOR_MQTT)
                elif msg.packet_type == "MESH":
                    type_indicator = "[~] "
                    color = curses.color_pair(self.COLOR_MESH)

                line = f"{msg.timestamp} {type_indicator}{me_tag}{msg.channel:4s} {msg.sender:6s}: {msg.text}"

                # Truncate to fit
                if len(line) > self.width - 1:
                    line = line[:self.width-4] + "..."

                stdscr.addstr(y, 0, line, color)

        except:
            pass

    def draw_input(self, stdscr):
        """Draw input line"""
        try:
            y = self.height - 2

            # Input separator
            stdscr.addstr(y, 0, "─" * (self.width - 1))

            # Input prompt
            y = self.height - 1
            prompt = f"Ch{self.current_channel}> "
            stdscr.addstr(y, 0, prompt)

            # Input text (with cursor)
            input_x = len(prompt)
            display_input = self.input_buffer

            # Truncate if too long
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

    def run(self, stdscr):
        """Main UI loop"""
        # Initialize
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(50)  # 50ms timeout for fast updates (20 Hz)

        self.init_colors()
        self.update_screen_size(stdscr)

        # Try to connect
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

        # Main loop
        last_draw = time.time()

        while self.running:
            current_time = time.time()

            # Process incoming messages
            self.process_messages()

            # Handle input (non-blocking)
            try:
                key = stdscr.getch()

                if key != -1:  # Key was pressed
                    if key == 27:  # ESC
                        self.running = False
                    elif key in (curses.KEY_BACKSPACE, 127, 8):  # Backspace
                        if self.input_buffer:
                            self.input_buffer = self.input_buffer[:-1]
                    elif key == 10 or key == 13:  # Enter
                        if self.input_buffer:
                            if self.input_buffer.startswith('/'):
                                self.handle_command(self.input_buffer)
                            else:
                                self.send_message(self.input_buffer, channel=self.current_channel)
                            self.input_buffer = ""
                    elif key == curses.KEY_RESIZE:
                        self.update_screen_size(stdscr)
                    elif 32 <= key <= 126:  # Printable characters
                        self.input_buffer += chr(key)
            except:
                pass

            # Redraw screen (fast updates - ~20 Hz)
            if current_time - last_draw >= 0.05:  # 50ms = 20 Hz
                stdscr.clear()

                self.update_screen_size(stdscr)

                # Draw components
                self.draw_header(stdscr)

                # Calculate message area
                msg_start_y = 7
                msg_height = self.height - msg_start_y - 2

                self.draw_messages(stdscr, msg_start_y, msg_height)
                self.draw_input(stdscr)

                stdscr.refresh()
                last_draw = current_time

        # Cleanup
        if self.interface:
            self.interface.close()


def main():
    """Entry point"""
    terminal = MeshtasticTerminal()

    try:
        curses.wrapper(terminal.run)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
