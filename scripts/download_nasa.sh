#!/usr/bin/env bash
# Download the NASA Battery Dataset (.mat files) from the official PHM S3 bucket.
# Total size: ~200 MB compressed, ~600 MB uncompressed.
#
# Usage:  bash scripts/download_nasa.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_DIR/data/nasa"

mkdir -p "$TARGET_DIR"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

ZIP_URL="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
ZIP_FILE="$TMP_DIR/nasa_battery.zip"

echo "Downloading NASA Battery Dataset (~200 MB)..."
echo "  URL: $ZIP_URL"
echo "  Target: $TARGET_DIR"
curl -fL --progress-bar -o "$ZIP_FILE" "$ZIP_URL"

echo "Unzipping..."
unzip -q -o "$ZIP_FILE" -d "$TMP_DIR"

# The zip extracts to "5. Battery Data Set/" containing 6 sub-zips
DATA_DIR="$TMP_DIR/5. Battery Data Set"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: expected '5. Battery Data Set' directory in zip"
    exit 1
fi

# Move the top-level directory into target
rm -rf "$TARGET_DIR/5. Battery Data Set"
mv "$DATA_DIR" "$TARGET_DIR/"

# Unzip each sub-archive
cd "$TARGET_DIR/5. Battery Data Set"
for z in *.zip; do
    echo "  Unpacking $z..."
    dir="${z%.zip}"
    mkdir -p "$dir"
    unzip -q -o "$z" -d "$dir"
done

# Count .mat files
N_MAT=$(find "$TARGET_DIR" -name "*.mat" | wc -l)
echo ""
echo "Done. $N_MAT .mat files extracted to $TARGET_DIR"
echo "Directory structure:"
find "$TARGET_DIR" -maxdepth 2 -type d | sort
