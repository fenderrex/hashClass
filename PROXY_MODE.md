# Meshtastic Proxy Mode - PuTTY for Mesh

Bidirectional proxy mode turns your Meshtastic device into a serial-like terminal, similar to PuTTY but for mesh networks.

## Quick Start

```bash
# Basic proxy mode - type messages, see all received
python3 meshtastic_pipe.py --pipe --proxy

# Specify send channel
python3 meshtastic_pipe.py --pipe --proxy --send-channel 0

# Pipe input from file
cat messages.txt | python3 meshtastic_pipe.py --pipe --proxy

# Echo messages through
echo "Hello mesh!" | python3 meshtastic_pipe.py --pipe --proxy
```

## How It Works

**Bidirectional Communication:**
- **stdin → mesh**: Anything you type goes to the mesh network
- **mesh → stdout**: All received messages appear on stdout

This allows you to use Meshtastic like a serial terminal or pipe data through it.

## Use Cases

### 1. Interactive Chat Terminal

```bash
python3 meshtastic_pipe.py --pipe --proxy
```

Type messages and press Enter. See all incoming messages in real-time.

### 2. Send Scripted Messages

```bash
# Send a list of messages
cat << EOF | python3 meshtastic_pipe.py --pipe --proxy
Hello mesh!
Testing 1 2 3
Anyone copy?
EOF
```

### 3. Automated Announcements

```bash
# Send system status every 5 minutes
while true; do
  echo "System status: $(uptime)" | python3 meshtastic_pipe.py --pipe --proxy
  sleep 300
done
```

### 4. Bridge to Other Programs

```bash
# Receive from mesh, process, send back
python3 meshtastic_pipe.py --pipe --proxy | \
  grep "status" | \
  awk '{print "Got status request"}' | \
  python3 meshtastic_pipe.py --pipe --proxy
```

### 5. Two-Device Communication Test

```bash
# Device 1 (sends and receives)
python3 meshtastic_pipe.py --pipe --proxy

# Device 2 (sends and receives)
python3 meshtastic_pipe.py --pipe --proxy

# Type on either terminal to send, see messages on both
```

### 6. Log All Traffic While Sending

```bash
# Receive all, log to file, also send from stdin
python3 meshtastic_pipe.py --pipe --proxy | tee mesh_log.txt
```

### 7. Different Channels

```bash
# Send on channel 1
python3 meshtastic_pipe.py --pipe --proxy --send-channel 1

# Send on channel 2
python3 meshtastic_pipe.py --pipe --proxy --send-channel 2
```

### 8. Filter Received, Send on Different Channel

```bash
# Receive from Ch0, send on Ch1
python3 meshtastic_pipe.py --pipe --channel Ch0 --proxy --send-channel 1
```

## JSON Mode with Proxy

```bash
# Receive as JSON, send plain text
python3 meshtastic_pipe.py --pipe --proxy --format json
```

Incoming messages are JSON, but anything you type gets sent as plain text.

## Comparison to PuTTY

| Feature | PuTTY (Serial) | Meshtastic Proxy |
|---------|---------------|-----------------|
| Bidirectional | ✓ | ✓ |
| Type to send | ✓ | ✓ |
| See received | ✓ | ✓ |
| Multiple channels | ✗ | ✓ (--send-channel) |
| Filter input | ✗ | ✓ (--channel) |
| JSON output | ✗ | ✓ (--format json) |
| Signal info | ✗ | ✓ (RSSI, SNR, hops) |
| Sender names | ✗ | ✓ (node database) |

## Advanced Examples

### Network Bridge

Connect two mesh networks via internet:

```bash
# Mesh A → Internet → Mesh B
ssh user@mesh-a "python3 meshtastic_pipe.py --pipe --proxy" | \
  ssh user@mesh-b "python3 meshtastic_pipe.py --pipe --proxy"
```

### Database Logging with Sending

```bash
# Log to database AND send responses
python3 meshtastic_pipe.py --pipe --proxy --format json | tee >(
  jq -r '{ts:.timestamp, msg:.text}' | \
  while read line; do
    echo "$line" | sqlite3 mesh.db ".import /dev/stdin messages"
  done
)
```

### Automated Responder

```bash
# Auto-reply to specific messages
python3 meshtastic_pipe.py --pipe --proxy | \
  grep -i "ping" | \
  while read line; do
    echo "Pong!"
  done | \
  python3 meshtastic_pipe.py --pipe --proxy
```

### Weather Station

```bash
# Broadcast weather every hour
while true; do
  curl "wttr.in/?format=%l:+%C+%t" | python3 meshtastic_pipe.py --pipe --proxy
  sleep 3600
done
```

### Emergency Broadcast

```bash
# Send emergency message, log all responses
echo "EMERGENCY: Need assistance at coordinates X,Y" | \
  python3 meshtastic_pipe.py --pipe --proxy | \
  tee emergency_responses.log
```

## Troubleshooting

### Messages not sending

Check that:
1. Device is connected: `python3 test_meshtastic_devices.py`
2. Channel number is correct (0-7)
3. Device is not in client mode (needs to be able to send)
4. LoRa settings match your mesh

### Not seeing received messages

1. Make sure you're not filtering them out with `--channel`
2. Check device is actually receiving (try Meshtastic app)
3. Verify baud rate and serial connection

### Two devices not hearing each other

1. Check both are on same channel
2. Verify LoRa settings match (freq, spread, bandwidth)
3. Ensure they're in range
4. Check duty cycle limits (25% max)

## Technical Details

### Message Flow

```
stdin → stdin_reader_thread → send_message() → Meshtastic interface → Radio
Radio → Meshtastic interface → on_receive() → output_message() → stdout
```

### Thread Safety

- stdin reader runs in separate thread
- Message queue for thread-safe communication
- All sends go through meshtastic library (thread-safe)

### Buffering

- 10,000 message buffer
- Immediate output (flush after each message)
- Line-buffered input (send on Enter)

## Exit

- **Ctrl+C**: Clean shutdown
- **EOF** on stdin: Closes stdin reader, program continues receiving
- **Broken pipe**: Clean exit if stdout closes

## See Also

- `PIPING_GUIDE.md` - Advanced piping and filtering
- `MESHTASTIC_README.md` - Full terminal features
- `examples_meshtastic_pipe.sh` - More examples
