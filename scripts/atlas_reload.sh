#!/usr/bin/env bash
# Reload the Atlas agents we temporarily unloaded for the AJAR overnight run.
# Run this in the morning when you're ready to chat / work normally again.

set -euo pipefail

echo "[atlas-reload] reloading heavy launchd agents..."
launchctl load ~/Library/LaunchAgents/com.edward.mlx-server.plist
launchctl load ~/Library/LaunchAgents/com.omlx.app.plist
launchctl load ~/Library/LaunchAgents/com.edward.mlx-embeddings.plist
launchctl load ~/Library/LaunchAgents/ai.openclaw.node.plist
launchctl load ~/Library/LaunchAgents/com.edward.omlx-model-sync.plist
echo "[atlas-reload] done. Verify with: launchctl list | grep -iE '(mlx|omlx|openclaw)'"
