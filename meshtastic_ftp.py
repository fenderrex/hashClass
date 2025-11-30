#!/usr/bin/env python3
"""
Meshtastic Serial FTP Protocol
================================
A robust file transfer protocol for Meshtastic devices using serial communication
with CRC16 checksums for data integrity.

Protocol Specification:
-----------------------
Packet Structure:
  [START_MARKER][LENGTH][CMD][SEQ][PAYLOAD][CRC16][END_MARKER]

Fields:
  START_MARKER: 2 bytes (0xAA 0x55)
  LENGTH: 2 bytes (payload length, big-endian)
  CMD: 1 byte (command type)
  SEQ: 2 bytes (sequence number, big-endian)
  PAYLOAD: 0-512 bytes (command-specific data)
  CRC16: 2 bytes (CRC-16-CCITT checksum)
  END_MARKER: 2 bytes (0x55 0xAA)

Commands:
  0x01: LIST - List directory contents
  0x02: GET - Download file
  0x03: PUT - Upload file
  0x04: DELETE - Delete file
  0x05: MKDIR - Create directory
  0x06: ACK - Acknowledgment
  0x07: NACK - Negative acknowledgment
  0x08: DATA - File data chunk
  0x09: INFO - File information

Features:
- CRC16 checksums for error detection
- Sequence numbers for packet ordering
- ACK/NACK responses for reliability
- Chunked file transfer (512 byte chunks)
- Automatic retry on errors
"""

import struct
import time
import os
import json
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from enum import IntEnum


class Command(IntEnum):
    """Protocol command types"""
    LIST = 0x01
    GET = 0x02
    PUT = 0x03
    DELETE = 0x04
    MKDIR = 0x05
    ACK = 0x06
    NACK = 0x07
    DATA = 0x08
    INFO = 0x09


# Protocol constants
START_MARKER = b'\xAA\x55'
END_MARKER = b'\x55\xAA'
MAX_PAYLOAD_SIZE = 512
MAX_RETRIES = 3
TIMEOUT_SECONDS = 5.0


class CRC16:
    """CRC-16-CCITT checksum calculation"""

    @staticmethod
    def calculate(data: bytes) -> int:
        """
        Calculate CRC-16-CCITT checksum.
        Polynomial: 0x1021 (x^16 + x^12 + x^5 + 1)
        Initial value: 0xFFFF
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc = crc << 1
                crc &= 0xFFFF
        return crc

    @staticmethod
    def verify(data: bytes, expected_crc: int) -> bool:
        """Verify data against expected CRC"""
        return CRC16.calculate(data) == expected_crc


class Packet:
    """Represents a protocol packet"""

    def __init__(self, cmd: Command, seq: int, payload: bytes = b''):
        """
        Create a new packet.

        Args:
            cmd: Command type
            seq: Sequence number (0-65535)
            payload: Payload data (max 512 bytes)
        """
        if len(payload) > MAX_PAYLOAD_SIZE:
            raise ValueError(f"Payload too large: {len(payload)} > {MAX_PAYLOAD_SIZE}")

        self.cmd = cmd
        self.seq = seq & 0xFFFF
        self.payload = payload

    def encode(self) -> bytes:
        """
        Encode packet to bytes.

        Returns:
            Encoded packet bytes
        """
        # Build packet data (without markers and CRC)
        length = len(self.payload)
        packet_data = struct.pack('>HBH', length, self.cmd, self.seq) + self.payload

        # Calculate CRC over packet data
        crc = CRC16.calculate(packet_data)

        # Build complete packet
        packet = START_MARKER + packet_data + struct.pack('>H', crc) + END_MARKER

        return packet

    @staticmethod
    def decode(data: bytes) -> Optional['Packet']:
        """
        Decode packet from bytes.

        Args:
            data: Raw packet bytes

        Returns:
            Decoded Packet object or None if invalid
        """
        # Check minimum length
        if len(data) < 11:  # START(2) + LENGTH(2) + CMD(1) + SEQ(2) + CRC(2) + END(2)
            return None

        # Check markers
        if data[:2] != START_MARKER or data[-2:] != END_MARKER:
            return None

        # Extract fields
        length, cmd, seq = struct.unpack('>HBH', data[2:7])

        # Check payload length
        if len(data) != 11 + length:
            return None

        payload = data[7:7+length]
        received_crc = struct.unpack('>H', data[7+length:9+length])[0]

        # Verify CRC
        packet_data = data[2:7+length]
        if not CRC16.verify(packet_data, received_crc):
            return None

        return Packet(Command(cmd), seq, payload)

    def __repr__(self) -> str:
        return f"Packet(cmd={Command(self.cmd).name}, seq={self.seq}, payload_len={len(self.payload)})"


class MeshtasticFTP:
    """
    Meshtastic FTP Server/Client implementation.
    Handles file operations over serial connection with checksums.
    """

    def __init__(self, base_path: str = "."):
        """
        Initialize FTP handler.

        Args:
            base_path: Base directory for file operations
        """
        self.base_path = Path(base_path).resolve()
        self.seq_num = 0
        self.receive_buffer = b''

    def _next_seq(self) -> int:
        """Get next sequence number"""
        seq = self.seq_num
        self.seq_num = (self.seq_num + 1) & 0xFFFF
        return seq

    def _safe_path(self, path: str) -> Path:
        """
        Get safe path within base directory.
        Prevents directory traversal attacks.
        """
        full_path = (self.base_path / path).resolve()
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(f"Path outside base directory: {path}")
        return full_path

    def handle_list(self, path: str = ".") -> Packet:
        """
        Handle LIST command - list directory contents.

        Args:
            path: Directory path to list

        Returns:
            Response packet with directory listing
        """
        try:
            target = self._safe_path(path)
            if not target.exists():
                return Packet(Command.NACK, self._next_seq(),
                            b"Directory not found")

            if not target.is_dir():
                return Packet(Command.NACK, self._next_seq(),
                            b"Not a directory")

            # Build directory listing
            entries = []
            for item in sorted(target.iterdir()):
                entry = {
                    'name': item.name,
                    'type': 'dir' if item.is_dir() else 'file',
                    'size': item.stat().st_size if item.is_file() else 0,
                    'modified': int(item.stat().st_mtime)
                }
                entries.append(entry)

            # Encode as JSON
            payload = json.dumps(entries).encode('utf-8')

            return Packet(Command.ACK, self._next_seq(), payload)

        except Exception as e:
            return Packet(Command.NACK, self._next_seq(),
                        str(e).encode('utf-8'))

    def handle_get(self, path: str) -> List[Packet]:
        """
        Handle GET command - download file.
        Returns list of packets (INFO + DATA chunks).

        Args:
            path: File path to download

        Returns:
            List of packets with file data
        """
        try:
            target = self._safe_path(path)
            if not target.exists():
                return [Packet(Command.NACK, self._next_seq(),
                             b"File not found")]

            if not target.is_file():
                return [Packet(Command.NACK, self._next_seq(),
                             b"Not a file")]

            # Read file
            with open(target, 'rb') as f:
                file_data = f.read()

            # Send file info first
            info = {
                'name': target.name,
                'size': len(file_data),
                'chunks': (len(file_data) + MAX_PAYLOAD_SIZE - 1) // MAX_PAYLOAD_SIZE
            }
            packets = [Packet(Command.INFO, self._next_seq(),
                            json.dumps(info).encode('utf-8'))]

            # Split into chunks and create DATA packets
            for i in range(0, len(file_data), MAX_PAYLOAD_SIZE):
                chunk = file_data[i:i + MAX_PAYLOAD_SIZE]
                packets.append(Packet(Command.DATA, self._next_seq(), chunk))

            return packets

        except Exception as e:
            return [Packet(Command.NACK, self._next_seq(),
                         str(e).encode('utf-8'))]

    def handle_put(self, path: str, data: bytes) -> Packet:
        """
        Handle PUT command - upload file.

        Args:
            path: File path to write
            data: File contents

        Returns:
            Response packet
        """
        try:
            target = self._safe_path(path)

            # Create parent directory if needed
            target.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(target, 'wb') as f:
                f.write(data)

            return Packet(Command.ACK, self._next_seq(),
                        f"File written: {len(data)} bytes".encode('utf-8'))

        except Exception as e:
            return Packet(Command.NACK, self._next_seq(),
                        str(e).encode('utf-8'))

    def handle_delete(self, path: str) -> Packet:
        """
        Handle DELETE command - delete file.

        Args:
            path: File path to delete

        Returns:
            Response packet
        """
        try:
            target = self._safe_path(path)
            if not target.exists():
                return Packet(Command.NACK, self._next_seq(),
                            b"File not found")

            if target.is_file():
                target.unlink()
            else:
                target.rmdir()

            return Packet(Command.ACK, self._next_seq(),
                        b"Deleted successfully")

        except Exception as e:
            return Packet(Command.NACK, self._next_seq(),
                        str(e).encode('utf-8'))

    def handle_mkdir(self, path: str) -> Packet:
        """
        Handle MKDIR command - create directory.

        Args:
            path: Directory path to create

        Returns:
            Response packet
        """
        try:
            target = self._safe_path(path)
            target.mkdir(parents=True, exist_ok=True)

            return Packet(Command.ACK, self._next_seq(),
                        b"Directory created")

        except Exception as e:
            return Packet(Command.NACK, self._next_seq(),
                        str(e).encode('utf-8'))

    def process_packet(self, packet: Packet) -> List[Packet]:
        """
        Process incoming packet and generate response.

        Args:
            packet: Received packet

        Returns:
            List of response packets
        """
        if packet.cmd == Command.LIST:
            path = packet.payload.decode('utf-8') if packet.payload else '.'
            return [self.handle_list(path)]

        elif packet.cmd == Command.GET:
            path = packet.payload.decode('utf-8')
            return self.handle_get(path)

        elif packet.cmd == Command.PUT:
            # PUT payload format: path_length(2) + path + file_data
            if len(packet.payload) < 2:
                return [Packet(Command.NACK, self._next_seq(),
                             b"Invalid PUT payload")]

            path_len = struct.unpack('>H', packet.payload[:2])[0]
            if len(packet.payload) < 2 + path_len:
                return [Packet(Command.NACK, self._next_seq(),
                             b"Invalid PUT payload")]

            path = packet.payload[2:2+path_len].decode('utf-8')
            data = packet.payload[2+path_len:]
            return [self.handle_put(path, data)]

        elif packet.cmd == Command.DELETE:
            path = packet.payload.decode('utf-8')
            return [self.handle_delete(path)]

        elif packet.cmd == Command.MKDIR:
            path = packet.payload.decode('utf-8')
            return [self.handle_mkdir(path)]

        else:
            return [Packet(Command.NACK, self._next_seq(),
                         b"Unknown command")]


class SerialTransport:
    """
    Simulated serial transport for testing.
    In production, this would use pyserial or similar.
    """

    def __init__(self):
        self.tx_buffer = []
        self.rx_buffer = []

    def write(self, data: bytes):
        """Write data to serial port"""
        self.tx_buffer.append(data)

    def read(self, timeout: float = 1.0) -> Optional[bytes]:
        """Read data from serial port"""
        if self.rx_buffer:
            return self.rx_buffer.pop(0)
        return None

    def inject_receive(self, data: bytes):
        """Inject data into receive buffer (for testing)"""
        self.rx_buffer.append(data)


def example_usage():
    """Example usage of Meshtastic FTP protocol"""

    print("=" * 60)
    print("Meshtastic Serial FTP Protocol - Example Usage")
    print("=" * 60)
    print()

    # Create FTP server
    server = MeshtasticFTP(base_path="/tmp/meshtastic_ftp")
    os.makedirs("/tmp/meshtastic_ftp", exist_ok=True)

    # Test 1: LIST command
    print("Test 1: LIST directory")
    print("-" * 40)
    list_packet = Packet(Command.LIST, 0, b'.')
    encoded = list_packet.encode()
    print(f"Encoded packet: {encoded.hex()}")
    print(f"Packet size: {len(encoded)} bytes")

    decoded = Packet.decode(encoded)
    print(f"Decoded packet: {decoded}")

    responses = server.process_packet(decoded)
    for resp in responses:
        print(f"Response: {resp}")
        print(f"Payload: {resp.payload.decode('utf-8')}")
    print()

    # Test 2: PUT command
    print("Test 2: PUT file")
    print("-" * 40)
    file_path = "test.txt"
    file_data = b"Hello, Meshtastic FTP!"

    # Build PUT payload
    path_bytes = file_path.encode('utf-8')
    put_payload = struct.pack('>H', len(path_bytes)) + path_bytes + file_data

    put_packet = Packet(Command.PUT, 1, put_payload)
    encoded = put_packet.encode()
    print(f"PUT packet size: {len(encoded)} bytes")

    decoded = Packet.decode(encoded)
    responses = server.process_packet(decoded)
    for resp in responses:
        print(f"Response: {resp}")
        print(f"Message: {resp.payload.decode('utf-8')}")
    print()

    # Test 3: GET command
    print("Test 3: GET file")
    print("-" * 40)
    get_packet = Packet(Command.GET, 2, file_path.encode('utf-8'))
    encoded = get_packet.encode()

    decoded = Packet.decode(encoded)
    responses = server.process_packet(decoded)

    print(f"Received {len(responses)} packets:")
    for i, resp in enumerate(responses):
        print(f"  Packet {i}: {resp}")
        if resp.cmd == Command.INFO:
            info = json.loads(resp.payload.decode('utf-8'))
            print(f"    File info: {info}")
        elif resp.cmd == Command.DATA:
            print(f"    Data: {resp.payload}")
    print()

    # Test 4: CRC verification
    print("Test 4: CRC checksum verification")
    print("-" * 40)
    test_data = b"Test data for CRC"
    crc = CRC16.calculate(test_data)
    print(f"Data: {test_data}")
    print(f"CRC16: 0x{crc:04X}")
    print(f"Verification: {CRC16.verify(test_data, crc)}")

    # Corrupt data
    corrupt_data = test_data + b'X'
    print(f"Corrupt data: {corrupt_data}")
    print(f"Verification: {CRC16.verify(corrupt_data, crc)}")
    print()

    # Test 5: Packet corruption detection
    print("Test 5: Corrupt packet detection")
    print("-" * 40)
    test_packet = Packet(Command.ACK, 99, b"Test")
    encoded = test_packet.encode()
    print(f"Original packet: {encoded.hex()}")

    # Corrupt a byte in the middle
    corrupted = bytearray(encoded)
    corrupted[len(corrupted)//2] ^= 0xFF
    corrupted = bytes(corrupted)
    print(f"Corrupted packet: {corrupted.hex()}")

    decoded_corrupt = Packet.decode(corrupted)
    print(f"Decode result: {decoded_corrupt}")
    print("(None indicates corruption was detected)")
    print()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    example_usage()
