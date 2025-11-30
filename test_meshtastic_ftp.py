#!/usr/bin/env python3
"""
Comprehensive Test Suite for Meshtastic FTP Protocol
=====================================================
Tests all protocol features including:
- Packet encoding/decoding
- CRC checksums
- All FTP commands
- Error handling
- Corruption detection
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

from meshtastic_ftp import (
    Packet, Command, CRC16, MeshtasticFTP,
    START_MARKER, END_MARKER, MAX_PAYLOAD_SIZE
)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def record(self, name: str, passed: bool, message: str = ""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"✓ {name}")
        else:
            self.failed += 1
            print(f"✗ {name}: {message}")

    def summary(self):
        print("\n" + "=" * 70)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, message in self.tests:
                if not passed:
                    print(f"  - {name}: {message}")


def test_crc16(results: TestResults):
    """Test CRC16 checksum calculation"""
    print("\n" + "=" * 70)
    print("Testing CRC16 Checksums")
    print("=" * 70)

    # Test 1: Basic CRC calculation
    data = b"Hello, World!"
    crc = CRC16.calculate(data)
    results.record(
        "CRC16 calculation",
        crc > 0,
        "CRC should be non-zero"
    )

    # Test 2: CRC verification
    results.record(
        "CRC16 verification (valid)",
        CRC16.verify(data, crc),
        "Valid CRC should verify"
    )

    # Test 3: Corrupted data detection
    corrupted = data + b"X"
    results.record(
        "CRC16 corruption detection",
        not CRC16.verify(corrupted, crc),
        "Corrupted data should not verify"
    )

    # Test 4: Deterministic
    crc2 = CRC16.calculate(data)
    results.record(
        "CRC16 deterministic",
        crc == crc2,
        "Same data should produce same CRC"
    )

    # Test 5: Empty data
    empty_crc = CRC16.calculate(b"")
    results.record(
        "CRC16 empty data",
        empty_crc == 0xFFFF,
        "Empty data CRC should be 0xFFFF"
    )


def test_packet_encoding(results: TestResults):
    """Test packet encoding and decoding"""
    print("\n" + "=" * 70)
    print("Testing Packet Encoding/Decoding")
    print("=" * 70)

    # Test 1: Simple ACK packet
    packet = Packet(Command.ACK, 42, b"OK")
    encoded = packet.encode()
    results.record(
        "Packet encoding",
        len(encoded) > 0 and encoded.startswith(START_MARKER),
        "Encoded packet should start with START_MARKER"
    )

    # Test 2: Decode valid packet
    decoded = Packet.decode(encoded)
    results.record(
        "Packet decoding",
        decoded is not None and decoded.cmd == Command.ACK,
        "Decoded packet should match original"
    )

    # Test 3: Sequence number preservation
    results.record(
        "Sequence number preservation",
        decoded.seq == 42,
        f"Expected seq 42, got {decoded.seq if decoded else 'None'}"
    )

    # Test 4: Payload preservation
    results.record(
        "Payload preservation",
        decoded.payload == b"OK",
        f"Expected payload b'OK', got {decoded.payload if decoded else 'None'}"
    )

    # Test 5: Maximum payload size
    try:
        large_payload = b"X" * MAX_PAYLOAD_SIZE
        large_packet = Packet(Command.DATA, 100, large_payload)
        encoded_large = large_packet.encode()
        decoded_large = Packet.decode(encoded_large)
        results.record(
            "Maximum payload size",
            decoded_large is not None and len(decoded_large.payload) == MAX_PAYLOAD_SIZE,
            "Should handle maximum payload size"
        )
    except Exception as e:
        results.record("Maximum payload size", False, str(e))

    # Test 6: Empty payload
    empty_packet = Packet(Command.LIST, 0, b"")
    encoded_empty = empty_packet.encode()
    decoded_empty = Packet.decode(encoded_empty)
    results.record(
        "Empty payload",
        decoded_empty is not None and len(decoded_empty.payload) == 0,
        "Should handle empty payload"
    )

    # Test 7: Corrupted packet detection
    corrupted = bytearray(encoded)
    corrupted[len(corrupted) // 2] ^= 0xFF
    decoded_corrupt = Packet.decode(bytes(corrupted))
    results.record(
        "Corrupted packet detection",
        decoded_corrupt is None,
        "Corrupted packet should fail to decode"
    )

    # Test 8: Incomplete packet
    incomplete = encoded[:len(encoded) // 2]
    decoded_incomplete = Packet.decode(incomplete)
    results.record(
        "Incomplete packet detection",
        decoded_incomplete is None,
        "Incomplete packet should fail to decode"
    )

    # Test 9: Wrong markers
    wrong_start = b"\xFF\xFF" + encoded[2:]
    decoded_wrong = Packet.decode(wrong_start)
    results.record(
        "Wrong start marker detection",
        decoded_wrong is None,
        "Wrong start marker should fail"
    )


def test_ftp_commands(results: TestResults):
    """Test FTP command handlers"""
    print("\n" + "=" * 70)
    print("Testing FTP Commands")
    print("=" * 70)

    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="meshtastic_ftp_test_")

    try:
        server = MeshtasticFTP(base_path=test_dir)

        # Test 1: LIST empty directory
        list_packet = Packet(Command.LIST, 0, b".")
        responses = server.process_packet(list_packet)
        results.record(
            "LIST empty directory",
            len(responses) == 1 and responses[0].cmd == Command.ACK,
            "LIST should return ACK"
        )

        # Test 2: MKDIR
        mkdir_packet = Packet(Command.MKDIR, 1, b"test_dir")
        responses = server.process_packet(mkdir_packet)
        results.record(
            "MKDIR command",
            len(responses) == 1 and responses[0].cmd == Command.ACK,
            "MKDIR should return ACK"
        )

        # Test 3: Verify directory created
        test_path = Path(test_dir) / "test_dir"
        results.record(
            "MKDIR creates directory",
            test_path.exists() and test_path.is_dir(),
            "Directory should exist after MKDIR"
        )

        # Test 4: PUT file
        import struct
        file_content = b"Test file content for Meshtastic FTP"
        file_path = "test_file.txt"
        path_bytes = file_path.encode('utf-8')
        put_payload = struct.pack('>H', len(path_bytes)) + path_bytes + file_content

        put_packet = Packet(Command.PUT, 2, put_payload)
        responses = server.process_packet(put_packet)
        results.record(
            "PUT command",
            len(responses) == 1 and responses[0].cmd == Command.ACK,
            "PUT should return ACK"
        )

        # Test 5: Verify file created
        file_path_full = Path(test_dir) / file_path
        results.record(
            "PUT creates file",
            file_path_full.exists() and file_path_full.is_file(),
            "File should exist after PUT"
        )

        # Test 6: Verify file content
        if file_path_full.exists():
            with open(file_path_full, 'rb') as f:
                content = f.read()
            results.record(
                "PUT file content",
                content == file_content,
                f"Expected {file_content}, got {content}"
            )
        else:
            results.record("PUT file content", False, "File not created")

        # Test 7: LIST non-empty directory
        list_packet = Packet(Command.LIST, 3, b".")
        responses = server.process_packet(list_packet)
        results.record(
            "LIST non-empty directory",
            len(responses) == 1 and responses[0].cmd == Command.ACK,
            "LIST should return ACK"
        )

        # Test 8: GET file
        get_packet = Packet(Command.GET, 4, file_path.encode('utf-8'))
        responses = server.process_packet(get_packet)
        results.record(
            "GET command",
            len(responses) >= 2 and responses[0].cmd == Command.INFO,
            f"GET should return INFO + DATA packets, got {len(responses)} packets"
        )

        # Test 9: Verify GET file content
        if len(responses) >= 2:
            import json
            info = json.loads(responses[0].payload.decode('utf-8'))
            data_packets = [r for r in responses[1:] if r.cmd == Command.DATA]
            retrieved_content = b"".join(p.payload for p in data_packets)

            results.record(
                "GET file content",
                retrieved_content == file_content,
                f"Expected {len(file_content)} bytes, got {len(retrieved_content)} bytes"
            )
        else:
            results.record("GET file content", False, "No data packets received")

        # Test 10: DELETE file
        delete_packet = Packet(Command.DELETE, 5, file_path.encode('utf-8'))
        responses = server.process_packet(delete_packet)
        results.record(
            "DELETE command",
            len(responses) == 1 and responses[0].cmd == Command.ACK,
            "DELETE should return ACK"
        )

        # Test 11: Verify file deleted
        results.record(
            "DELETE removes file",
            not file_path_full.exists(),
            "File should not exist after DELETE"
        )

        # Test 12: GET non-existent file
        get_fail_packet = Packet(Command.GET, 6, b"nonexistent.txt")
        responses = server.process_packet(get_fail_packet)
        results.record(
            "GET non-existent file",
            len(responses) == 1 and responses[0].cmd == Command.NACK,
            "GET non-existent file should return NACK"
        )

        # Test 13: DELETE non-existent file
        delete_fail_packet = Packet(Command.DELETE, 7, b"nonexistent.txt")
        responses = server.process_packet(delete_fail_packet)
        results.record(
            "DELETE non-existent file",
            len(responses) == 1 and responses[0].cmd == Command.NACK,
            "DELETE non-existent file should return NACK"
        )

        # Test 14: Path traversal protection
        traversal_packet = Packet(Command.GET, 8, b"../../../etc/passwd")
        responses = server.process_packet(traversal_packet)
        results.record(
            "Path traversal protection",
            len(responses) == 1 and responses[0].cmd == Command.NACK,
            "Path traversal should be blocked"
        )

    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


def test_large_file_transfer(results: TestResults):
    """Test large file transfer with chunking"""
    print("\n" + "=" * 70)
    print("Testing Large File Transfer")
    print("=" * 70)

    test_dir = tempfile.mkdtemp(prefix="meshtastic_ftp_large_")

    try:
        server = MeshtasticFTP(base_path=test_dir)

        # Create file that will result in multiple GET chunks
        # Keep PUT within limits, but large enough to test GET chunking
        large_content = b"X" * 1024
        file_path = "large_file.bin"
        path_bytes = file_path.encode('utf-8')

        # First create the file directly for GET testing
        file_full_path = Path(test_dir) / file_path
        with open(file_full_path, 'wb') as f:
            f.write(large_content)

        results.record(
            "Large file setup",
            file_full_path.exists(),
            "Test file should be created"
        )

        # GET large file
        get_packet = Packet(Command.GET, 1, file_path.encode('utf-8'))
        responses = server.process_packet(get_packet)

        results.record(
            "GET large file",
            len(responses) >= 2 and responses[0].cmd == Command.INFO,
            f"GET large file should return multiple packets, got {len(responses)}"
        )

        # Verify chunking
        if len(responses) >= 2:
            import json
            info = json.loads(responses[0].payload.decode('utf-8'))
            expected_chunks = (len(large_content) + MAX_PAYLOAD_SIZE - 1) // MAX_PAYLOAD_SIZE

            results.record(
                "Large file chunking",
                info['chunks'] == expected_chunks,
                f"Expected {expected_chunks} chunks, got {info['chunks']}"
            )

            # Reconstruct file
            data_packets = [r for r in responses[1:] if r.cmd == Command.DATA]
            reconstructed = b"".join(p.payload for p in data_packets)

            results.record(
                "Large file reconstruction",
                reconstructed == large_content,
                f"Expected {len(large_content)} bytes, got {len(reconstructed)} bytes"
            )
        else:
            results.record("Large file chunking", False, "No INFO packet")
            results.record("Large file reconstruction", False, "No data packets")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_protocol_edge_cases(results: TestResults):
    """Test protocol edge cases and error conditions"""
    print("\n" + "=" * 70)
    print("Testing Protocol Edge Cases")
    print("=" * 70)

    test_dir = tempfile.mkdtemp(prefix="meshtastic_ftp_edge_")

    try:
        server = MeshtasticFTP(base_path=test_dir)

        # Test 1: Invalid command
        invalid_packet = Packet(0xFF, 0, b"test")
        try:
            responses = server.process_packet(invalid_packet)
            results.record(
                "Invalid command handling",
                len(responses) == 1 and responses[0].cmd == Command.NACK,
                "Invalid command should return NACK"
            )
        except Exception as e:
            results.record("Invalid command handling", False, str(e))

        # Test 2: Oversized payload attempt
        try:
            oversized = Packet(Command.ACK, 0, b"X" * (MAX_PAYLOAD_SIZE + 1))
            results.record("Oversized payload rejection", False, "Should have raised ValueError")
        except ValueError:
            results.record("Oversized payload rejection", True)

        # Test 3: Sequence number rollover
        high_seq_packet = Packet(Command.ACK, 0xFFFF, b"test")
        encoded = high_seq_packet.encode()
        decoded = Packet.decode(encoded)
        results.record(
            "High sequence number",
            decoded is not None and decoded.seq == 0xFFFF,
            "Should handle max sequence number"
        )

        # Test 4: Binary data in payload
        binary_data = bytes(range(256))
        binary_packet = Packet(Command.DATA, 0, binary_data)
        encoded = binary_packet.encode()
        decoded = Packet.decode(encoded)
        results.record(
            "Binary data in payload",
            decoded is not None and decoded.payload == binary_data,
            "Should handle all byte values in payload"
        )

        # Test 5: UTF-8 in payload
        utf8_data = "Hello 世界 🌍".encode('utf-8')
        utf8_packet = Packet(Command.ACK, 0, utf8_data)
        encoded = utf8_packet.encode()
        decoded = Packet.decode(encoded)
        results.record(
            "UTF-8 in payload",
            decoded is not None and decoded.payload == utf8_data,
            "Should handle UTF-8 in payload"
        )

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def main():
    """Run all tests"""
    print("=" * 70)
    print("Meshtastic FTP Protocol Test Suite")
    print("=" * 70)

    results = TestResults()

    test_crc16(results)
    test_packet_encoding(results)
    test_ftp_commands(results)
    test_large_file_transfer(results)
    test_protocol_edge_cases(results)

    results.summary()

    return 0 if results.failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
