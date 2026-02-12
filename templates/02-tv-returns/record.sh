#!/usr/bin/env bash
# Record the tv_returns demo GIF.
# Runs setup (clean build, restore PII) BEFORE VHS starts recording.
set -e
cd "$(dirname "$0")"

echo "Resetting demo state..."
rm -rf build
sed -i 's/A customer/john.smith@gmail.com/g' fixtures/vendor_offers.json 2>/dev/null || true

echo "Recording..."
vhs tape.tape
