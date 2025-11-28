#!/usr/bin/env python3
"""
Test script to list available Meshtastic USB devices
"""

import glob
import sys

def find_meshtastic_ports():
    """Find all potential Meshtastic USB devices"""
    ports = []

    # Linux/Mac USB serial ports
    for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/cu.usbserial*', '/dev/cu.usbmodem*']:
        found = glob.glob(pattern)
        if found:
            print(f"Found devices matching {pattern}:")
            for port in found:
                print(f"  - {port}")
                ports.extend(found)

    # Windows COM ports (if running on Windows)
    if sys.platform == 'win32':
        try:
            import serial.tools.list_ports
            print("\nScanning Windows COM ports...")
            for port in serial.tools.list_ports.comports():
                if 'USB' in port.description or 'Serial' in port.description:
                    print(f"  - {port.device}: {port.description}")
                    ports.append(port.device)
        except ImportError:
            print("pyserial not installed, skipping Windows COM port scan")

    return sorted(ports)

if __name__ == "__main__":
    print("Scanning for Meshtastic USB devices...\n")

    ports = find_meshtastic_ports()

    if ports:
        print(f"\n✓ Found {len(ports)} potential device(s)")
        print("\nTo connect, run:")
        print("  python3 meshtastic_terminal.py")
    else:
        print("\n✗ No USB serial devices found")
        print("\nTroubleshooting:")
        print("  1. Connect your Meshtastic device via USB")
        print("  2. Check the USB cable supports data (not charge-only)")
        print("  3. On Linux, add your user to 'dialout' group:")
        print("     sudo usermod -a -G dialout $USER")
        print("  4. Verify device shows up with: ls /dev/ttyUSB* /dev/ttyACM*")
