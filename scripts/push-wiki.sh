#!/bin/bash
# Push wiki content from wiki/ folder to GitHub wiki repo
set -e

REPO="rajamohan1950/CLSplusplus"
WIKI_URL="https://github.com/${REPO}.wiki.git"
WIKI_DIR=".wiki-temp"

cd "$(dirname "$0")/.."

echo "Cloning wiki repo..."
if [ -d "$WIKI_DIR" ]; then
  rm -rf "$WIKI_DIR"
fi
git clone "$WIKI_URL" "$WIKI_DIR" 2>/dev/null || {
  echo "Wiki repo may not exist yet. Create the first wiki page via GitHub UI:"
  echo "  https://github.com/${REPO}/wiki"
  echo "Then run this script again."
  exit 1
}

echo "Copying wiki content..."
cp wiki/*.md "$WIKI_DIR/"

cd "$WIKI_DIR"
git add .
if git diff --staged --quiet; then
  echo "No changes to push."
  cd ..
  rm -rf "$WIKI_DIR"
  exit 0
fi

git commit -m "Update wiki pages"
git push origin master

cd ..
rm -rf "$WIKI_DIR"
echo "Wiki updated successfully: https://github.com/${REPO}/wiki"
