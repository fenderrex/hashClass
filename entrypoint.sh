#!/bin/sh
set -e

# Start virtual X server for headless GUI apps
Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 &

# Give X some time to start
sleep 1

# If args provided, pass them to the script, else default to abbb1.png
if [ "$#" -gt 0 ]; then
	exec python star.py "$@"
else
	exec python star.py abbb1.png
fi
