#!/usr/bin/env python3
"""
Meshtastic FTP Client
======================
Command-line client for Meshtastic FTP protocol with serial support.

Usage:
    python meshtastic_ftp_client.py --port /dev/ttyUSB0 list /path
    python meshtastic_ftp_client.py --port /dev/ttyUSB0 get remote.txt local.txt
    python meshtastic_ftp_client.py --port /dev/ttyUSB0 put local.txt remote.txt
    python meshtastic_ftp_client.py --port /dev/ttyUSB0 delete remote.txt
    python meshtastic_ftp_client.py --port /dev/ttyUSB0 mkdir /new_dir

For testing without hardware:
    python meshtastic_ftp_client.py --simulate list /
"""

import sys
import time
import struct
import json
import argparse
from typing import Optional, List
from pathlib import Path

# Import from meshtastic_ftp module
from meshtastic_ftp import (
    Packet, Command, CRC16, MeshtasticFTP,
    START_MARKER, END_MARKER, MAX_PAYLOAD_SIZE,
    MAX_RETRIES, TIMEOUT_SECONDS
)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Run: pip install pyserial")
    print("Running in simulation mode only.")


class FTPClient:
    """
    Meshtastic FTP Client with serial communication and retry logic.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, simulate: bool = False):
        """
        Initialize FTP client.

        Args:
            port: Serial port (e.g., /dev/ttyUSB0, COM3)
            baudrate: Serial baud rate
            simulate: Use simulation mode (no actual serial)
        """
        self.seq_num = 0
        self.simulate = simulate or not SERIAL_AVAILABLE

        if self.simulate:
            print("[SIMULATION MODE]")
            # Create simulated server for testing
            self.server = MeshtasticFTP(base_path="/tmp/meshtastic_test")
            import os
            os.makedirs("/tmp/meshtastic_test", exist_ok=True)
            self.serial = None
        else:
            if not port:
                raise ValueError("Serial port required when not in simulation mode")
            self.serial = serial.Serial(port, baudrate, timeout=TIMEOUT_SECONDS)
            self.server = None

        self.receive_buffer = b''

    def _next_seq(self) -> int:
        """Get next sequence number"""
        seq = self.seq_num
        self.seq_num = (self.seq_num + 1) & 0xFFFF
        return seq

    def _send_packet(self, packet: Packet) -> bool:
        """
        Send packet over serial.

        Args:
            packet: Packet to send

        Returns:
            True if sent successfully
        """
        encoded = packet.encode()

        if self.simulate:
            print(f"[TX] {packet}")
            print(f"     Size: {len(encoded)} bytes, CRC: OK")
            return True

        try:
            self.serial.write(encoded)
            self.serial.flush()
            print(f"[TX] {packet}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send: {e}")
            return False

    def _receive_packet(self, timeout: float = TIMEOUT_SECONDS) -> Optional[Packet]:
        """
        Receive packet from serial.

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Received packet or None
        """
        if self.simulate:
            # In simulation mode, we don't actually receive over serial
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Read available data
            if self.serial.in_waiting > 0:
                self.receive_buffer += self.serial.read(self.serial.in_waiting)

            # Look for start marker
            start_idx = self.receive_buffer.find(START_MARKER)
            if start_idx == -1:
                # No start marker yet, keep waiting
                time.sleep(0.01)
                continue

            # Discard data before start marker
            if start_idx > 0:
                self.receive_buffer = self.receive_buffer[start_idx:]

            # Check if we have minimum packet size
            if len(self.receive_buffer) < 11:
                time.sleep(0.01)
                continue

            # Extract length field
            length = struct.unpack('>H', self.receive_buffer[2:4])[0]
            total_length = 11 + length

            # Wait for complete packet
            if len(self.receive_buffer) < total_length:
                time.sleep(0.01)
                continue

            # Extract packet
            packet_data = self.receive_buffer[:total_length]
            self.receive_buffer = self.receive_buffer[total_length:]

            # Decode packet
            packet = Packet.decode(packet_data)
            if packet:
                print(f"[RX] {packet}")
                return packet
            else:
                print("[ERROR] Invalid packet received (CRC failed)")
                # Continue looking for next packet

        return None

    def _send_receive_retry(self, packet: Packet, expect_multi: bool = False) -> Optional[List[Packet]]:
        """
        Send packet and wait for response with retry logic.

        Args:
            packet: Packet to send
            expect_multi: Expect multiple response packets (for GET)

        Returns:
            List of response packets or None if failed
        """
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                print(f"[RETRY] Attempt {attempt + 1}/{MAX_RETRIES}")

            # Send packet
            if not self._send_packet(packet):
                continue

            if self.simulate:
                # In simulation mode, process locally
                responses = self.server.process_packet(packet)
                for resp in responses:
                    print(f"[RX] {resp}")
                return responses

            # Wait for response
            responses = []
            first_response = self._receive_packet()

            if not first_response:
                print("[ERROR] Timeout waiting for response")
                continue

            responses.append(first_response)

            # If expecting multiple packets (GET command), keep receiving
            if expect_multi and first_response.cmd == Command.INFO:
                info = json.loads(first_response.payload.decode('utf-8'))
                num_chunks = info.get('chunks', 0)

                for i in range(num_chunks):
                    data_packet = self._receive_packet()
                    if not data_packet:
                        print(f"[ERROR] Timeout waiting for chunk {i+1}/{num_chunks}")
                        break
                    responses.append(data_packet)

                if len(responses) == num_chunks + 1:
                    return responses  # Got all chunks

            elif not expect_multi:
                return responses

        print("[ERROR] Max retries exceeded")
        return None

    def list_directory(self, path: str = '.') -> bool:
        """
        List directory contents.

        Args:
            path: Directory path to list

        Returns:
            True if successful
        """
        print(f"\nListing directory: {path}")
        print("-" * 60)

        packet = Packet(Command.LIST, self._next_seq(), path.encode('utf-8'))
        responses = self._send_receive_retry(packet)

        if not responses:
            return False

        resp = responses[0]
        if resp.cmd == Command.ACK:
            entries = json.loads(resp.payload.decode('utf-8'))
            print(f"{'Type':<6} {'Size':<12} {'Name'}")
            print("-" * 60)
            for entry in entries:
                type_str = entry['type'].upper()
                size_str = str(entry['size']) if entry['type'] == 'file' else '-'
                print(f"{type_str:<6} {size_str:<12} {entry['name']}")
            print(f"\nTotal: {len(entries)} items")
            return True
        else:
            print(f"Error: {resp.payload.decode('utf-8')}")
            return False

    def get_file(self, remote_path: str, local_path: str) -> bool:
        """
        Download file from remote.

        Args:
            remote_path: Remote file path
            local_path: Local file path to save

        Returns:
            True if successful
        """
        print(f"\nDownloading: {remote_path} -> {local_path}")
        print("-" * 60)

        packet = Packet(Command.GET, self._next_seq(), remote_path.encode('utf-8'))
        responses = self._send_receive_retry(packet, expect_multi=True)

        if not responses:
            return False

        # First packet should be INFO
        if responses[0].cmd != Command.INFO:
            if responses[0].cmd == Command.NACK:
                print(f"Error: {responses[0].payload.decode('utf-8')}")
            else:
                print("Error: Expected INFO packet")
            return False

        info = json.loads(responses[0].payload.decode('utf-8'))
        print(f"File: {info['name']}")
        print(f"Size: {info['size']} bytes")
        print(f"Chunks: {info['chunks']}")

        # Collect data from DATA packets
        file_data = b''
        for i, resp in enumerate(responses[1:], 1):
            if resp.cmd != Command.DATA:
                print(f"Error: Expected DATA packet, got {Command(resp.cmd).name}")
                return False

            file_data += resp.payload
            print(f"Received chunk {i}/{info['chunks']} ({len(resp.payload)} bytes)")

        # Verify size
        if len(file_data) != info['size']:
            print(f"Error: Size mismatch! Expected {info['size']}, got {len(file_data)}")
            return False

        # Write to file
        try:
            with open(local_path, 'wb') as f:
                f.write(file_data)
            print(f"\nFile saved: {local_path}")
            print(f"Total size: {len(file_data)} bytes")
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False

    def put_file(self, local_path: str, remote_path: str) -> bool:
        """
        Upload file to remote.

        Args:
            local_path: Local file path
            remote_path: Remote file path

        Returns:
            True if successful
        """
        print(f"\nUploading: {local_path} -> {remote_path}")
        print("-" * 60)

        # Read local file
        try:
            with open(local_path, 'rb') as f:
                file_data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return False

        print(f"File size: {len(file_data)} bytes")

        # Check if we need to split into multiple packets
        max_file_size = MAX_PAYLOAD_SIZE - 100  # Leave room for path

        if len(file_data) > max_file_size:
            print(f"Warning: File too large for single packet ({len(file_data)} > {max_file_size})")
            print("Consider implementing chunked upload")
            return False

        # Build PUT payload
        path_bytes = remote_path.encode('utf-8')
        put_payload = struct.pack('>H', len(path_bytes)) + path_bytes + file_data

        packet = Packet(Command.PUT, self._next_seq(), put_payload)
        responses = self._send_receive_retry(packet)

        if not responses:
            return False

        resp = responses[0]
        if resp.cmd == Command.ACK:
            print(f"Success: {resp.payload.decode('utf-8')}")
            return True
        else:
            print(f"Error: {resp.payload.decode('utf-8')}")
            return False

    def delete_file(self, path: str) -> bool:
        """
        Delete file on remote.

        Args:
            path: File path to delete

        Returns:
            True if successful
        """
        print(f"\nDeleting: {path}")
        print("-" * 60)

        packet = Packet(Command.DELETE, self._next_seq(), path.encode('utf-8'))
        responses = self._send_receive_retry(packet)

        if not responses:
            return False

        resp = responses[0]
        if resp.cmd == Command.ACK:
            print(f"Success: {resp.payload.decode('utf-8')}")
            return True
        else:
            print(f"Error: {resp.payload.decode('utf-8')}")
            return False

    def make_directory(self, path: str) -> bool:
        """
        Create directory on remote.

        Args:
            path: Directory path to create

        Returns:
            True if successful
        """
        print(f"\nCreating directory: {path}")
        print("-" * 60)

        packet = Packet(Command.MKDIR, self._next_seq(), path.encode('utf-8'))
        responses = self._send_receive_retry(packet)

        if not responses:
            return False

        resp = responses[0]
        if resp.cmd == Command.ACK:
            print(f"Success: {resp.payload.decode('utf-8')}")
            return True
        else:
            print(f"Error: {resp.payload.decode('utf-8')}")
            return False

    def close(self):
        """Close serial connection"""
        if self.serial and not self.simulate:
            self.serial.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Meshtastic FTP Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List directory:
    %(prog)s --port /dev/ttyUSB0 list /data

  Download file:
    %(prog)s --port /dev/ttyUSB0 get /data/log.txt ./log.txt

  Upload file:
    %(prog)s --port /dev/ttyUSB0 put ./config.json /data/config.json

  Delete file:
    %(prog)s --port /dev/ttyUSB0 delete /data/old.txt

  Create directory:
    %(prog)s --port /dev/ttyUSB0 mkdir /data/backups

  Simulation mode (no hardware):
    %(prog)s --simulate list /
        """
    )

    parser.add_argument('--port', '-p', help='Serial port (e.g., /dev/ttyUSB0, COM3)')
    parser.add_argument('--baudrate', '-b', type=int, default=115200,
                       help='Serial baud rate (default: 115200)')
    parser.add_argument('--simulate', '-s', action='store_true',
                       help='Simulation mode (no actual serial)')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # LIST command
    list_parser = subparsers.add_parser('list', help='List directory contents')
    list_parser.add_argument('path', nargs='?', default='.', help='Directory path')

    # GET command
    get_parser = subparsers.add_parser('get', help='Download file')
    get_parser.add_argument('remote', help='Remote file path')
    get_parser.add_argument('local', help='Local file path')

    # PUT command
    put_parser = subparsers.add_parser('put', help='Upload file')
    put_parser.add_argument('local', help='Local file path')
    put_parser.add_argument('remote', help='Remote file path')

    # DELETE command
    del_parser = subparsers.add_parser('delete', help='Delete file')
    del_parser.add_argument('path', help='File path to delete')

    # MKDIR command
    mkdir_parser = subparsers.add_parser('mkdir', help='Create directory')
    mkdir_parser.add_argument('path', help='Directory path to create')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create client
    try:
        client = FTPClient(
            port=args.port,
            baudrate=args.baudrate,
            simulate=args.simulate
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Execute command
    success = False
    try:
        if args.command == 'list':
            success = client.list_directory(args.path)

        elif args.command == 'get':
            success = client.get_file(args.remote, args.local)

        elif args.command == 'put':
            success = client.put_file(args.local, args.remote)

        elif args.command == 'delete':
            success = client.delete_file(args.path)

        elif args.command == 'mkdir':
            success = client.make_directory(args.path)

    finally:
        client.close()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
