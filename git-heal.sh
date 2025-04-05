#!/bin/bash

echo "Healing your Git repo..."

# Step 1: Remove conflicted refs in local remotes
echo Cleaning conflicted remote refs..."
find .git/refs/remotes/origin -name "*conflicted copy*" -exec rm {} \;

# Step 2: Prune stale remote branches
echo Pruning stale remotes..."
git remote prune origin

# Step 3: Fetch fresh remote info
echo Fetching latest from origin..."
git fetch --all

# Optional: Show status
echo "Done. Your repo is cleaned up."
git status
