# Meshtastic USB Terminal - Piping Guide

Complete guide to piping and filtering Meshtastic messages with strict message ordering.

## Understanding Message Sources

Each message shows **where it's coming from** with detailed information:

### Message Format

```
12:34:56.789 Ch0  Alice(!a1b2) ↔2 [-85dBm SNR:8.5]: Hello mesh!
└─timestamp  │   │    │        │   └─signal info    └─message text
             │   │    │        └─hop count (relayed twice)
             │   │    └─short node ID
             │   └─friendly name (if available)
             └─channel
```

### Fields Explained

- **Timestamp**: `HH:MM:SS.mmm` with millisecond precision for ordering
- **Channel**: `Ch0`, `Ch1`, `PM` (private message), etc.
- **Sender Name**: User-friendly name like `Alice` (if available)
- **Sender ID**: Short node ID like `!a1b2` (last 4 hex digits)
- **Hop Count**: `↔2` means relayed through 2 hops (direct = no indicator)
- **Signal Info**: `[-85dBm SNR:8.5]` shows signal strength and quality
- **Message Text**: The actual message content

### Packet Type Indicators

- **(no indicator)**: USER message (text chat)
- **[M]**: MQTT message (from MQTT gateway)
- **[~]**: MESH overhead (position, telemetry, etc.)
- **[ME]**: Your own messages

## Basic Usage

### Interactive Mode (Full UI)

```bash
python3 meshtastic_pipe.py
```

Shows full terminal UI with statistics, duty cycle, and color coding.

### Pipe Mode (Stream to stdout)

```bash
python3 meshtastic_pipe.py --pipe
```

Streams all messages to stdout in real-time for piping to other programs.

## Channel Filtering

### Filter Single Channel

```bash
# Only private messages
python3 meshtastic_pipe.py --pipe --channel PM

# Only primary channel
python3 meshtastic_pipe.py --pipe --channel Ch0

# Only secondary channel
python3 meshtastic_pipe.py --pipe --channel Ch1
```

### Filter Multiple Channels

```bash
# Channels 0 and 1
python3 meshtastic_pipe.py --pipe --channel Ch0,Ch1

# All channels except private messages
python3 meshtastic_pipe.py --pipe --channel Ch0,Ch1,Ch2,Ch3
```

## Output Formats

### Text Format (Default)

Human-readable with full context:

```bash
python3 meshtastic_pipe.py --pipe --format text
```

Example output:
```
12:34:56.123 Ch0  Alice(!a1b2) ↔1 [-78dBm SNR:9.2]: Anyone copy?
12:34:58.456 Ch0  Bob(!c3d4) [-82dBm SNR:7.5]: I copy you Alice!
12:35:01.789 PM   Charlie(!e5f6) ↔2 [-90dBm SNR:5.1]: Hey Bob, PM
```

### JSON Format

Machine-readable with all metadata:

```bash
python3 meshtastic_pipe.py --pipe --format json
```

Example output:
```json
{"timestamp":"12:34:56.123","timestamp_unix":1234567890.123,"channel":"Ch0","sender":"!a1b2","sender_id":"!a1b2c3d4","sender_name":"Alice","text":"Anyone copy?","is_own":false,"packet_type":"USER","airtime_ms":125.5,"hop_count":1,"snr":9.2,"rssi":-78}
```

### CSV Format

Spreadsheet-compatible:

```bash
python3 meshtastic_pipe.py --pipe --format csv > messages.csv
```

Columns: `timestamp,channel,sender,sender_id,sender_name,type,text,is_own,airtime_ms,hop_count,snr,rssi`

## Seeing Where Messages Come From

### Identify Senders

```bash
# See all unique senders
python3 meshtastic_pipe.py --pipe | awk '{print $4}' | sort -u

# Count messages per sender
python3 meshtastic_pipe.py --pipe | awk '{print $4}' | sort | uniq -c | sort -rn

# Watch for specific sender
python3 meshtastic_pipe.py --pipe | grep 'Alice'
```

### Track Signal Quality

```bash
# Show only messages with signal info
python3 meshtastic_pipe.py --pipe | grep 'dBm'

# Extract signal strength data
python3 meshtastic_pipe.py --pipe --format json | jq '{sender:.sender_name, rssi:.rssi, snr:.snr}'

# Alert on weak signals (< -95 dBm)
python3 meshtastic_pipe.py --pipe | grep -E '\[-9[5-9]dBm|\[-[0-9]{3}dBm'
```

### Monitor Hop Counts

```bash
# Show multi-hop messages only
python3 meshtastic_pipe.py --pipe | grep '↔'

# Count hops per message
python3 meshtastic_pipe.py --pipe --format json | jq '.hop_count'

# Find messages that traveled far (3+ hops)
python3 meshtastic_pipe.py --pipe --format json | jq 'select(.hop_count >= 3)'
```

### Node Discovery

```bash
# See all nodes with names
python3 meshtastic_pipe.py --pipe --format json | jq -r '"\(.sender_name // "Unknown") (\(.sender_id))"' | sort -u

# Build node map
python3 meshtastic_pipe.py --pipe --format json | jq -r '{id:.sender_id, name:.sender_name, channel:.channel}' | jq -s 'unique_by(.id)'
```

## Practical Examples

### 1. Log Only Direct Messages

```bash
python3 meshtastic_pipe.py --pipe --channel PM > private_chat.log
```

### 2. Search All Messages for Keywords

```bash
python3 meshtastic_pipe.py --pipe | grep -i 'emergency\|sos\|help'
```

### 3. Monitor Specific User

```bash
# By name
python3 meshtastic_pipe.py --pipe | grep 'Alice'

# By node ID
python3 meshtastic_pipe.py --pipe | grep '!a1b2'
```

### 4. Real-time Notifications

```bash
# Desktop notifications on keywords
python3 meshtastic_pipe.py --pipe | grep -i 'meeting' | while read msg; do
  notify-send "Mesh Message" "$msg"
done
```

### 5. Channel-specific Logs

```bash
# Separate file per channel
python3 meshtastic_pipe.py --pipe --format json | jq -r '"\(.channel)|\(.timestamp)|\(.sender_name)|\(.text)"' | while IFS='|' read ch ts sender text; do
  echo "[$ts] $sender: $text" >> "log_${ch}.txt"
done
```

### 6. Message Statistics

```bash
# Count by channel
python3 meshtastic_pipe.py --pipe | awk '{print $3}' | sort | uniq -c

# Count by sender
python3 meshtastic_pipe.py --pipe | awk '{print $4}' | sort | uniq -c | sort -rn

# Messages per minute
python3 meshtastic_pipe.py --pipe | awk '{print substr($1,1,5)}' | uniq -c
```

### 7. Signal Quality Analysis

```bash
# Extract RSSI values
python3 meshtastic_pipe.py --pipe --format json | jq -r '[.timestamp, .sender_name, .rssi] | @csv' > signal_log.csv

# Find weakest signals
python3 meshtastic_pipe.py --pipe --format json | jq 'select(.rssi < -90) | {sender:.sender_name, rssi:.rssi, text:.text}'
```

### 8. Network Topology Mapping

```bash
# Build hop count map (which nodes are how far away)
python3 meshtastic_pipe.py --pipe --format json | jq '{sender:.sender_name, id:.sender_id, hops:.hop_count, rssi:.rssi}' | jq -s 'group_by(.id) | map({node:.[0].sender, avg_hops:(map(.hops)|add/length), avg_rssi:(map(.rssi)|add/length)})'
```

### 9. Filter by Message Type

```bash
# Only user messages (not overhead)
python3 meshtastic_pipe.py --pipe --format json | jq 'select(.packet_type == "USER")'

# Only MQTT messages
python3 meshtastic_pipe.py --pipe | grep '\[M\]'

# Only mesh overhead
python3 meshtastic_pipe.py --pipe | grep '\[~\]'
```

### 10. Database Integration

```bash
# SQLite logging
python3 meshtastic_pipe.py --pipe --format csv | sqlite3 mesh.db ".import /dev/stdin messages"

# PostgreSQL logging (requires psql)
python3 meshtastic_pipe.py --pipe --format csv | psql -d meshdb -c "COPY messages FROM STDIN WITH CSV HEADER"
```

### 11. Network Forwarding

```bash
# Send to remote syslog
python3 meshtastic_pipe.py --pipe | logger -t meshtastic -n syslog.example.com

# Broadcast on network
python3 meshtastic_pipe.py --pipe | nc -l -p 9999

# WebSocket streaming (with websocat)
python3 meshtastic_pipe.py --pipe | websocat -s 8080
```

### 12. Alert on Patterns

```bash
# Alert on multiple keywords
python3 meshtastic_pipe.py --pipe | grep -E 'urgent|emergency|help' | while read line; do
  echo "$line" | mail -s "Mesh Alert" admin@example.com
done
```

### 13. Message Replay

```bash
# Save all messages
python3 meshtastic_pipe.py --pipe --format json > mesh_archive.jsonl

# Replay later
cat mesh_archive.jsonl | jq -r '.text'
```

### 14. Traffic Analysis

```bash
# Airtime usage by sender
python3 meshtastic_pipe.py --pipe --format json | jq -r '[.sender_name, .airtime_ms] | @csv' | awk -F, '{sum[$1]+=$2; count[$1]++} END {for(s in sum) print s, sum[s]"ms total", count[s]"msgs", sum[s]/count[s]"ms avg"}' | sort -k2 -rn
```

### 15. Timestamp Conversion

```bash
# Add full date to timestamps
python3 meshtastic_pipe.py --pipe | while read line; do
  echo "[$(date '+%Y-%m-%d')] $line"
done
```

## Advanced Filtering with jq

### Extract Specific Fields

```bash
# Just sender and text
python3 meshtastic_pipe.py --pipe --format json | jq -r '"\(.sender_name): \(.text)"'

# Full sender info
python3 meshtastic_pipe.py --pipe --format json | jq '{who:.sender_name, from:.sender_id, signal:.rssi, hops:.hop_count, message:.text}'
```

### Conditional Filtering

```bash
# Messages from direct neighbors (0 hops)
python3 meshtastic_pipe.py --pipe --format json | jq 'select(.hop_count == 0)'

# Strong signals only (> -80 dBm)
python3 meshtastic_pipe.py --pipe --format json | jq 'select(.rssi > -80)'

# Long messages (> 50 chars)
python3 meshtastic_pipe.py --pipe --format json | jq 'select(.text | length > 50)'
```

### Aggregations

```bash
# Messages per hour
python3 meshtastic_pipe.py --pipe --format json | jq -r '.timestamp[:2]' | uniq -c

# Average hop count
python3 meshtastic_pipe.py --pipe --format json | jq -s 'map(.hop_count) | add/length'

# Signal quality stats
python3 meshtastic_pipe.py --pipe --format json | jq -s 'map(.rssi) | {min:min, max:max, avg:(add/length)}'
```

## Message Ordering Guarantee

All messages include `timestamp_unix` field for strict chronological ordering:

```bash
# Ensure proper ordering even if piping to slow consumer
python3 meshtastic_pipe.py --pipe --format json | jq -s 'sort_by(.timestamp_unix) | .[]'
```

Messages are output immediately when received, maintaining real-time order.

## Performance Tips

1. **Use channel filters** to reduce noise:
   ```bash
   --channel PM  # much faster than piping all and grepping
   ```

2. **Choose appropriate format**:
   - `text`: Human reading
   - `json`: Complex processing
   - `csv`: Spreadsheets/databases

3. **Handle broken pipes** gracefully - program exits cleanly if pipe closes

4. **Buffer management** - 10,000 message buffer ensures no loss

## Troubleshooting

### No messages appearing

```bash
# Check connection
python3 test_meshtastic_devices.py

# Verify device is receiving
python3 meshtastic_pipe.py  # interactive mode first
```

### Wrong channel

```bash
# List all channels
python3 meshtastic_pipe.py --pipe | awk '{print $3}' | sort -u
```

### Signal info missing

Some packets may not include RSSI/SNR. This is normal for:
- Your own messages (not received, so no signal)
- Some packet types

### Sender names not showing

Names come from the node database. They appear after:
- Node sends a message
- Your device receives node info packet
- May show just ID initially, then name later

## See Also

- `MESHTASTIC_README.md` - General terminal documentation
- `examples_meshtastic_pipe.sh` - Quick reference examples
- `test_meshtastic_devices.py` - Device detection tool
