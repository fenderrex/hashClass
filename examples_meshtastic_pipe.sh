#!/bin/bash
# Meshtastic Pipe Examples
# ========================
# This file shows various ways to use meshtastic_pipe.py

echo "Meshtastic USB Terminal - Piping Examples"
echo "=========================================="
echo ""

# Example 1: Monitor all messages in real-time
echo "Example 1: Monitor ALL messages"
echo "Command: python3 meshtastic_pipe.py --pipe"
echo ""

# Example 2: Monitor only private messages
echo "Example 2: Monitor ONLY private messages"
echo "Command: python3 meshtastic_pipe.py --pipe --channel PM"
echo ""

# Example 3: Monitor specific channel
echo "Example 3: Monitor ONLY channel 0"
echo "Command: python3 meshtastic_pipe.py --pipe --channel Ch0"
echo ""

# Example 4: Monitor multiple channels
echo "Example 4: Monitor channels 0 and 1"
echo "Command: python3 meshtastic_pipe.py --pipe --channel Ch0,Ch1"
echo ""

# Example 5: Search for keywords
echo "Example 5: Search for keyword 'help' in all messages"
echo "Command: python3 meshtastic_pipe.py --pipe | grep -i 'help'"
echo ""

# Example 6: Save to log file
echo "Example 6: Save ALL messages to log file"
echo "Command: python3 meshtastic_pipe.py --pipe > mesh_log.txt"
echo ""

# Example 7: Save PM only to log
echo "Example 7: Save ONLY private messages to log"
echo "Command: python3 meshtastic_pipe.py --pipe --channel PM > private_messages.log"
echo ""

# Example 8: JSON output
echo "Example 8: Get messages as JSON"
echo "Command: python3 meshtastic_pipe.py --pipe --format json"
echo ""

# Example 9: JSON with jq filtering
echo "Example 9: Extract just the message text using jq"
echo "Command: python3 meshtastic_pipe.py --pipe --format json | jq -r '.text'"
echo ""

# Example 10: CSV output
echo "Example 10: Get messages as CSV for spreadsheet"
echo "Command: python3 meshtastic_pipe.py --pipe --format csv > messages.csv"
echo ""

# Example 11: Filter by sender
echo "Example 11: Show only messages from specific sender"
echo "Command: python3 meshtastic_pipe.py --pipe | grep '!a1b2:'"
echo ""

# Example 12: Count messages per channel
echo "Example 12: Count messages per channel (run for a while, then Ctrl+C)"
echo "Command: python3 meshtastic_pipe.py --pipe | awk '{print \$3}' | sort | uniq -c"
echo ""

# Example 13: Alert on keywords
echo "Example 13: Alert on emergency keywords"
echo "Command: python3 meshtastic_pipe.py --pipe | grep -i 'emergency\\|sos\\|help' | while read line; do echo \"\$line\"; notify-send \"Mesh Alert\" \"\$line\"; done"
echo ""

# Example 14: Separate logs by channel
echo "Example 14: Separate log files per channel"
echo "Command: python3 meshtastic_pipe.py --pipe --format json | jq -r '\"\\(.channel) \\(.timestamp) \\(.sender): \\(.text)\"' | while IFS=' ' read channel rest; do echo \"\$rest\" >> \"log_\${channel}.txt\"; done"
echo ""

# Example 15: Monitor and timestamp
echo "Example 15: Add system timestamp to each message"
echo "Command: python3 meshtastic_pipe.py --pipe | while read line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done"
echo ""

# Example 16: Forward to network
echo "Example 16: Forward messages over network (netcat)"
echo "Command: python3 meshtastic_pipe.py --pipe | nc -l 9999"
echo ""

# Example 17: Database logging
echo "Example 17: Log to SQLite database"
cat << 'SQLEOF'
Command:
python3 meshtastic_pipe.py --pipe --format json | while read line; do
  echo "INSERT INTO messages VALUES (json('$line'));" | sqlite3 mesh.db
done
SQLEOF
echo ""

# Example 18: Color-coded real-time monitoring
echo "Example 18: Color-coded monitoring with grep"
echo "Command: python3 meshtastic_pipe.py --pipe | grep --color=always -E 'PM|Ch0|Ch1|\$'"
echo ""

# Example 19: Watch for specific user
echo "Example 19: Monitor specific user's messages only"
echo "Command: python3 meshtastic_pipe.py --pipe | grep '!a1b2'"
echo ""

# Example 20: Statistics
echo "Example 20: Live statistics"
cat << 'STATSEOF'
Command:
python3 meshtastic_pipe.py --pipe | awk '{
  channel=$3;
  sender=$4;
  msg_count++;
  channel_count[channel]++;
  sender_count[sender]++;
  if(msg_count % 10 == 0) {
    print "=== STATS ===";
    print "Total:", msg_count;
    for(c in channel_count) print "  " c ":", channel_count[c];
  }
}'
STATSEOF
echo ""

echo "For more information, see MESHTASTIC_README.md"
