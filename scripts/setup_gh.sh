#!/bin/bash
# Re-establish the GitHub CLI for this repo, if it ever disappears.
# Home directory persists across VM replacements; /usr/bin does not,
# so a persistent copy lives in ~/workspace/bin/gh when installed here.
# Auth itself is NOT stored in this script. If gh is not authenticated,
# this script prints the device-flow step to complete in a browser
# already signed in to GitHub, then re-checks auth.
set -e
GH_BIN=""
if command -v gh >/dev/null 2>&1; then GH_BIN="gh"; fi
if [ -x "$HOME/workspace/bin/gh" ]; then GH_BIN="$HOME/workspace/bin/gh"; fi
if [ -z "$GH_BIN" ]; then
  echo "gh not found, downloading a persistent copy to ~/workspace/bin/gh"
  mkdir -p "$HOME/workspace/bin"
  # Latest stable at time of writing; adjust if the release moves.
  curl -fsSL -o /tmp/gh.tar.gz https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz
  tar -xzf /tmp/gh.tar.gz -C /tmp
  cp /tmp/gh_2.102.0_linux_amd64/bin/gh "$HOME/workspace/bin/gh"
  chmod +x "$HOME/workspace/bin/gh"
  GH_BIN="$HOME/workspace/bin/gh"
fi
echo "gh: $($GH_BIN --version | head -1)"
if "$GH_BIN" auth status >/dev/null 2>&1; then
  echo "gh is authenticated"
  "$GH_BIN" auth status
else
  echo "gh is NOT authenticated."
  echo "Run: $GH_BIN auth login --hostname github.com --git-protocol https --web"
  echo "Then approve the device code at https://github.com/login/device"
  echo "in a browser already signed in to GitHub as alxspiker."
  exit 1
fi
