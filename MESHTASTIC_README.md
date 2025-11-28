# Meshtastic USB Terminal

A comprehensive terminal interface for Meshtastic devices with real-time monitoring and interactive messaging.

## Features

### ✅ Complete Features

1. **List all USB Meshtastic devices** - Automatically scans ttyUSB*/ttyACM* ports
2. **Read message buffer from receiver** - Real-time message monitoring
3. **Color-coded channels** - PM=Magenta, Ch0=Green, Ch1=Yellow, Ch2+=Cyan
4. **Show MQTT overhead separately** - Tracks MQTT vs User vs Mesh traffic
5. **Track channel utilization** - Separates user messages from mesh overhead
6. **Monitor 25% duty cycle limit** - Shows duty cycle percentage with color warnings
7. **Display airtime usage** - Calculates estimated airtime in milliseconds
8. **Progress bar for duty cycle** - Visual indicator with color coding
9. **Interactive messaging terminal** - Send messages, PMs, and channel messages
10. **Auto-find unlocked USB ports** - Scans and connects to available port
11. **Fast message updates** - Updates 20 times per second (every 50ms)
12. **Preserve user input during screen updates** - Input buffer maintained across refreshes
13. **Handle window resizing** - Terminal responds to resize events
14. **Show per-channel message counts** - Displays Ch0:5 Ch1:3 PM:2 format
15. **Mark own messages with [ME] tag** - Green [ME] tag on outgoing messages
16. **Commands: /pm, /ch, /quit** - All three commands implemented
17. **Show messages immediately** - Real-time delivery via pubsub subscription

## Installation

```bash
pip install -r requirements_meshtastic.txt
```

Or install dependencies manually:

```bash
pip install meshtastic pyserial
```

## Usage

Simply run the terminal:

```bash
python3 meshtastic_terminal.py
```

The program will automatically:
- Scan for connected Meshtastic USB devices on `/dev/ttyUSB*` and `/dev/ttyACM*` ports
- Connect to the first available device
- Start monitoring messages in real-time

## Interface Layout

```
═══ MESHTASTIC USB TERMINAL ═══
Device: /dev/ttyUSB0 | Node: !a1b2c3d4
Messages: Ch0:15 Ch1:7 PM:3
User:20 MQTT:3 Mesh:2
Airtime: 2.45s
Duty Cycle:    [========              ] 15.2%
────────────────────────────────────────
12:34:56 Ch0  !1a2b: Hello mesh!
12:35:01 [M] Ch1  !3c4d: MQTT message
12:35:15 [ME] PM  !5e6f: Private message
12:35:20 [~] Ch0  !7g8h: Position update
────────────────────────────────────────
Ch0> Type message here_
```

## Commands

### `/pm <node_id> <message>`
Send a private message to a specific node.

Example:
```
/pm a1b2c3d4 Hello there!
```

### `/ch <channel> [message]`
Switch to a different channel or send a message to a specific channel.

Examples:
```
/ch 1                    # Switch to channel 1
/ch 2 Broadcast message  # Send to channel 2
```

### `/quit` or `/q`
Exit the terminal.

## Color Coding

- **PM (Private Messages)**: Magenta
- **Ch0 (Primary Channel)**: Green
- **Ch1 (Secondary Channel)**: Yellow
- **Ch2+ (Other Channels)**: Cyan
- **MQTT Messages**: Blue (marked with `[M]`)
- **Mesh Overhead**: White (marked with `[~]`)
- **Your Messages**: Green with `[ME]` tag

## Duty Cycle Monitoring

The terminal estimates and monitors your duty cycle usage:

- **Green** (< 15%): Safe operation
- **Yellow** (15-20%): Approaching limit
- **Red** (> 20%): Too close to 25% limit

The duty cycle is calculated based on estimated airtime of transmitted and received packets.

## Statistics Tracked

- **Per-channel message counts**: See how many messages on each channel
- **Traffic breakdown**: User messages vs MQTT vs Mesh overhead
- **Total airtime**: Cumulative airtime usage
- **Duty cycle percentage**: Current duty cycle with visual progress bar

## Keyboard Shortcuts

- **Type normally**: Message input
- **Enter**: Send message
- **Backspace**: Delete character
- **ESC**: Quit terminal
- **Window resize**: Automatically handled

## Technical Details

### Message Types

1. **USER**: Text messages (TEXT_MESSAGE_APP)
2. **MQTT**: Messages from MQTT gateway
3. **MESH**: Protocol overhead (position, telemetry, etc.)

### Airtime Estimation

The terminal estimates airtime using:
- Preamble: ~50ms
- Data: ~5ms per byte

Actual airtime depends on:
- Spreading Factor (SF)
- Bandwidth
- Coding Rate
- LoRa parameters

### Update Rate

- **Screen refresh**: 20 Hz (every 50ms)
- **Message processing**: Real-time via callbacks
- **Input handling**: Non-blocking, immediate response

## Troubleshooting

### No device found
- Ensure Meshtastic device is connected via USB
- Check device appears in `/dev/ttyUSB*` or `/dev/ttyACM*`
- Try unplugging and reconnecting the device
- Verify USB cable supports data (not charge-only)

### Permission denied
On Linux, you may need to add your user to the `dialout` group:
```bash
sudo usermod -a -G dialout $USER
```
Then log out and back in.

### Messages not appearing
- Verify the device is on and connected to the mesh
- Check the channel configuration matches your mesh
- Ensure the device firmware is up to date

### Terminal display issues
- Ensure your terminal supports colors
- Try resizing the terminal window
- Minimum recommended size: 80x24 characters

## Development

The terminal is built with:
- **meshtastic**: Python library for Meshtastic devices
- **curses**: Terminal UI framework
- **pyserial**: USB serial communication

### Architecture

- **MeshtasticTerminal**: Main application class
- **Message dataclass**: Represents individual messages
- **ChannelStats dataclass**: Per-channel statistics
- **Callback-based**: Uses meshtastic's `onReceive` for real-time updates
- **Queue-based**: Messages processed asynchronously for smooth UI

## License

This tool is provided as-is for use with Meshtastic devices.

## Credits

Built for the Meshtastic community to provide a comprehensive terminal interface for USB-connected devices.
