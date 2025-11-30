# Meshtastic Serial FTP Protocol Documentation

## Overview

The Meshtastic Serial FTP Protocol is a robust file transfer system designed for Meshtastic devices using serial communication. It provides reliable file operations with CRC-16 checksums for data integrity, making it ideal for low-bandwidth, potentially unreliable communication channels.

## Features

✅ **CRC-16 Checksums** - Every packet includes CRC-16-CCITT checksum for error detection
✅ **Sequence Numbers** - Packet ordering and tracking
✅ **Chunked Transfer** - Large files split into 512-byte chunks
✅ **Retry Logic** - Automatic retry on transmission errors
✅ **Path Security** - Directory traversal protection
✅ **Full FTP Operations** - LIST, GET, PUT, DELETE, MKDIR
✅ **Binary Safe** - Handles all file types correctly

## Protocol Specification

### Packet Structure

Every packet follows this binary format:

```
[START_MARKER][LENGTH][CMD][SEQ][PAYLOAD][CRC16][END_MARKER]
     2 bytes   2 bytes 1 byte 2 bytes 0-512 bytes 2 bytes    2 bytes
```

#### Field Descriptions

| Field | Size | Type | Description |
|-------|------|------|-------------|
| START_MARKER | 2 bytes | Fixed | `0xAA 0x55` - Packet start marker |
| LENGTH | 2 bytes | Big-endian uint16 | Payload length (0-512) |
| CMD | 1 byte | uint8 | Command type (see below) |
| SEQ | 2 bytes | Big-endian uint16 | Sequence number (0-65535) |
| PAYLOAD | 0-512 bytes | Binary | Command-specific data |
| CRC16 | 2 bytes | Big-endian uint16 | CRC-16-CCITT checksum |
| END_MARKER | 2 bytes | Fixed | `0x55 0xAA` - Packet end marker |

**Total packet size**: 11 + payload_length bytes

### Commands

| Code | Name | Description |
|------|------|-------------|
| 0x01 | LIST | List directory contents |
| 0x02 | GET | Download file |
| 0x03 | PUT | Upload file |
| 0x04 | DELETE | Delete file or directory |
| 0x05 | MKDIR | Create directory |
| 0x06 | ACK | Acknowledgment (success) |
| 0x07 | NACK | Negative acknowledgment (error) |
| 0x08 | DATA | File data chunk |
| 0x09 | INFO | File information |

### CRC-16 Checksum

Uses **CRC-16-CCITT** algorithm:
- Polynomial: `0x1021` (x¹⁶ + x¹² + x⁵ + 1)
- Initial value: `0xFFFF`
- Computed over: LENGTH + CMD + SEQ + PAYLOAD (excludes markers and CRC itself)

### Command Details

#### LIST Command (0x01)

**Request:**
- CMD: 0x01
- PAYLOAD: Directory path (UTF-8 string, default: ".")

**Response:**
- CMD: 0x06 (ACK) or 0x07 (NACK)
- PAYLOAD: JSON array of file entries (on success) or error message (on failure)

**Example Response Payload (JSON):**
```json
[
  {
    "name": "file.txt",
    "type": "file",
    "size": 1024,
    "modified": 1638360000
  },
  {
    "name": "subdir",
    "type": "dir",
    "size": 0,
    "modified": 1638360100
  }
]
```

#### GET Command (0x02)

**Request:**
- CMD: 0x02
- PAYLOAD: File path (UTF-8 string)

**Response (multiple packets):**

1. **INFO Packet** (0x09):
   - Payload: JSON with file metadata
   ```json
   {
     "name": "file.txt",
     "size": 1024,
     "chunks": 2
   }
   ```

2. **DATA Packets** (0x08):
   - One or more packets with file data
   - Each packet contains up to 512 bytes
   - Packets sent in sequence

**Error Response:**
- CMD: 0x07 (NACK)
- PAYLOAD: Error message (UTF-8 string)

#### PUT Command (0x03)

**Request:**
- CMD: 0x03
- PAYLOAD: `[path_length(2)][path][file_data]`
  - path_length: uint16 big-endian
  - path: UTF-8 string
  - file_data: Binary file content

**Response:**
- CMD: 0x06 (ACK) or 0x07 (NACK)
- PAYLOAD: Success message or error message

**Limitations:**
- Total payload must be ≤ 512 bytes
- For larger files, send multiple PUT commands with offsets (future enhancement)

#### DELETE Command (0x04)

**Request:**
- CMD: 0x04
- PAYLOAD: File/directory path (UTF-8 string)

**Response:**
- CMD: 0x06 (ACK) or 0x07 (NACK)
- PAYLOAD: Success message or error message

#### MKDIR Command (0x05)

**Request:**
- CMD: 0x05
- PAYLOAD: Directory path (UTF-8 string)

**Response:**
- CMD: 0x06 (ACK) or 0x07 (NACK)
- PAYLOAD: Success message or error message

## Implementation

### Files

1. **meshtastic_ftp.py** - Core protocol implementation
   - `CRC16` class - Checksum calculation
   - `Packet` class - Packet encoding/decoding
   - `MeshtasticFTP` class - Server-side command handlers
   - `SerialTransport` class - Serial communication wrapper

2. **meshtastic_ftp_client.py** - Command-line client
   - Full CLI interface
   - Retry logic
   - Progress reporting
   - Simulation mode for testing

3. **test_meshtastic_ftp.py** - Comprehensive test suite
   - 37 test cases
   - CRC verification tests
   - Packet encoding/decoding tests
   - All command tests
   - Edge case tests

### Usage Examples

#### Server Side

```python
from meshtastic_ftp import MeshtasticFTP, Packet

# Create FTP server
server = MeshtasticFTP(base_path="/path/to/files")

# Process incoming packet
packet = Packet.decode(received_data)
if packet:
    responses = server.process_packet(packet)
    for response in responses:
        send_over_serial(response.encode())
```

#### Client Side (Python API)

```python
from meshtastic_ftp_client import FTPClient

# Create client
client = FTPClient(port="/dev/ttyUSB0", baudrate=115200)

# List directory
client.list_directory("/data")

# Download file
client.get_file("/data/log.txt", "./log.txt")

# Upload file
client.put_file("./config.json", "/data/config.json")

# Delete file
client.delete_file("/data/old.txt")

# Create directory
client.make_directory("/data/backups")

client.close()
```

#### Client Side (CLI)

```bash
# List directory
python meshtastic_ftp_client.py --port /dev/ttyUSB0 list /data

# Download file
python meshtastic_ftp_client.py --port /dev/ttyUSB0 get /data/log.txt ./log.txt

# Upload file
python meshtastic_ftp_client.py --port /dev/ttyUSB0 put ./config.json /data/config.json

# Delete file
python meshtastic_ftp_client.py --port /dev/ttyUSB0 delete /data/old.txt

# Create directory
python meshtastic_ftp_client.py --port /dev/ttyUSB0 mkdir /data/backups

# Simulation mode (no hardware)
python meshtastic_ftp_client.py --simulate list /
```

## Hardware Setup

### Requirements

- Meshtastic device with serial interface
- USB-to-Serial adapter (if needed)
- Python 3.7+
- pyserial library

### Serial Connection

```bash
# Install pyserial
pip install pyserial

# Find serial port
# Linux: /dev/ttyUSB0, /dev/ttyACM0
# macOS: /dev/cu.usbserial-*
# Windows: COM1, COM2, etc.

# Test connection
python meshtastic_ftp_client.py --port /dev/ttyUSB0 list /
```

### Typical Baud Rates

- 9600 bps - Low speed, high reliability
- 57600 bps - Medium speed
- 115200 bps - High speed (recommended)
- 921600 bps - Very high speed

## Error Handling

### Error Detection

1. **CRC Verification**
   - Every packet CRC is verified
   - Corrupted packets are discarded
   - No ACK sent for corrupted packets

2. **Timeout Detection**
   - 5-second timeout per packet
   - Automatic retry (up to 3 attempts)
   - Error reported after max retries

3. **Path Validation**
   - Directory traversal attacks blocked
   - Paths restricted to base directory
   - Invalid paths return NACK

### Error Codes

Errors are reported via NACK packets with UTF-8 error messages:

- "File not found"
- "Directory not found"
- "Not a file"
- "Not a directory"
- "Path outside base directory"
- "Invalid PUT payload"
- "Unknown command"

## Performance Characteristics

### Throughput

With 115200 baud serial:
- Raw throughput: ~11.5 KB/s
- Effective throughput: ~9-10 KB/s (with protocol overhead)
- 512-byte chunk transfer: ~50ms per chunk

### Overhead

Per packet overhead:
- Protocol headers: 11 bytes (2.1% for 512-byte payload)
- CRC: 2 bytes (included in headers)
- Markers: 4 bytes (included in headers)

### Reliability

- CRC-16 error detection: 99.998% error detection rate
- Retry mechanism: 3 attempts
- Total failure rate: < 0.01% (with good serial connection)

## Security Considerations

### Path Traversal Protection

All file operations validate paths to prevent directory traversal:

```python
# Blocked: attempts to access parent directories
"/../../etc/passwd"  # Returns NACK
"../../../secret"    # Returns NACK

# Allowed: paths within base directory
"data/config.json"   # OK
"/logs/app.log"      # OK
```

### Access Control

- All operations restricted to configured base_path
- No execution of files
- Read/write only within allowed directory tree

### Recommendations

1. Run server with minimal privileges
2. Use dedicated directory for FTP operations
3. Implement file size limits
4. Add authentication layer for production use
5. Monitor for unusual activity

## Testing

### Run Test Suite

```bash
# Run all tests
python test_meshtastic_ftp.py

# Expected output: 37 passed, 0 failed
```

### Test Coverage

- ✅ CRC-16 calculation and verification
- ✅ Packet encoding/decoding
- ✅ All FTP commands (LIST, GET, PUT, DELETE, MKDIR)
- ✅ Large file transfer with chunking
- ✅ Error detection and handling
- ✅ Path traversal protection
- ✅ Binary data handling
- ✅ UTF-8 support
- ✅ Edge cases and error conditions

### Example Test Output

```
======================================================================
Test Results: 37 passed, 0 failed
======================================================================
```

## Future Enhancements

### Planned Features

1. **Chunked PUT** - Support uploading files > 512 bytes
2. **Resume Transfer** - Resume interrupted file transfers
3. **Compression** - Optional compression for text files
4. **Authentication** - Username/password or token-based auth
5. **Encryption** - Optional AES encryption layer
6. **Batch Operations** - Multiple file operations in one command
7. **File Attributes** - Preserve permissions and timestamps
8. **Streaming** - Stream large files without loading into memory

### Protocol Extensions

Future protocol versions may add:
- Command 0x0A: RESUME - Resume partial transfer
- Command 0x0B: AUTH - Authentication
- Command 0x0C: STAT - Get file statistics
- Command 0x0D: RENAME - Rename file

## Troubleshooting

### Common Issues

**1. "Permission denied" errors**
```bash
# Linux: Add user to dialout group
sudo usermod -a -G dialout $USER
# Logout and login again
```

**2. "Serial port not found"**
```bash
# List available ports
ls /dev/tty*    # Linux/macOS
mode            # Windows

# Check device is connected
dmesg | grep tty    # Linux
system_profiler SPUSBDataType    # macOS
```

**3. "CRC verification failed"**
- Check baud rate matches on both ends
- Verify serial cable quality
- Reduce baud rate if errors persist
- Check for electromagnetic interference

**4. "Timeout waiting for response"**
- Increase timeout value
- Check serial connection
- Verify device is powered and running
- Test with simulation mode first

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This protocol implementation is provided as-is for use with Meshtastic devices and other embedded systems.

## References

- [CRC-16-CCITT Algorithm](https://en.wikipedia.org/wiki/Cyclic_redundancy_check)
- [Meshtastic Project](https://meshtastic.org/)
- [Serial Communication Best Practices](https://en.wikipedia.org/wiki/Serial_communication)
- [File Transfer Protocol Concepts](https://en.wikipedia.org/wiki/File_Transfer_Protocol)

---

**Version:** 1.0
**Last Updated:** 2025-11-30
**Author:** Meshtastic FTP Protocol Implementation
