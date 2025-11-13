#!/bin/bash

VERIFIER="verifier.py"

# Directories
INPUT_DIR="inputs"
OUTPUT_DIR="Outputs_To_Verify"

# Report file
REPORT_FILE="report.txt"

# Loop through downloaded files and rename
for FILE in "$OUTPUT_DIR"/output_from_*_to_*.txt; do
    [ -e "$FILE" ] || continue

    BASENAME=$(basename "$FILE")

    # Extract the second number
    NEW_NUM=$(echo "$BASENAME" | sed -E 's/output_from_[0-9]+_to_([0-9]+)\.txt/\1/')

    NEW_NAME="output_${NEW_NUM}.txt"
    NEW_PATH="${OUTPUT_DIR}/${NEW_NAME}"

    echo "Renaming: $BASENAME → $NEW_NAME"
    mv "$FILE" "$NEW_PATH"
done

# Loop through all new-style output files
for OUTPUT_FILE in "$OUTPUT_DIR"/output_*.txt; do
    [ -e "$OUTPUT_FILE" ] || continue

    BASENAME=$(basename "$OUTPUT_FILE")

    # Extract the numeric ID (i.e. 1128 from output_1128.txt)
    ID=$(echo "$BASENAME" | sed -E 's/output_([0-9]+)\.txt/\1/')

    INPUT_FILE="$INPUT_DIR/input_group${ID}.txt"

    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "No matching input file for $OUTPUT_FILE" >> "$REPORT_FILE"
        continue
    fi

    echo "Verifying $OUTPUT_FILE against $INPUT_FILE..."

    # Run verifier
    VERIFIER_OUTPUT=$(python3 "$VERIFIER" "$INPUT_FILE" "$OUTPUT_FILE" 2>&1)

    LAST_LINE=$(echo "$VERIFIER_OUTPUT" | tail -n 1)

    # Append result to report
    echo "$(basename "$OUTPUT_FILE"): $LAST_LINE" >> "$REPORT_FILE"
done

echo "Verification complete - report saved to $REPORT_FILE"
