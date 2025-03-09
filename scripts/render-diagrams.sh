#!/bin/bash

OUTPUT_DIR="../sequence-diagrams-rendered"
mkdir -p "$OUTPUT_DIR"

for file in ../sequence-diagrams/*.txt; do
  OUTPUT_FILE="$OUTPUT_DIR/${file%.txt}.png"
  echo "Rendering $file to $OUTPUT_FILE..."

  curl -X POST "https://www.websequencediagrams.com/index.php" \
    -d "message=$(cat "$file")" \
    -d "style=default" \
    -d "apiVersion=1" \
    -o "$OUTPUT_FILE"
done

echo "✅ All diagrams rendered in $OUTPUT_DIR/"