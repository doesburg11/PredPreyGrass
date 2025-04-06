#!/bin/bash

echo "Healing your Git repo..."

# Step 1: Remove conflicted remote refs
echo "Cleaning conflicted remote refs..."
find .git/refs/remotes/origin -name "*conflicted copy*" -exec rm {} \;

# Step 2: Prune stale remote refs
echo "Pruning stale remotes..."
git remote prune origin

# Step 3: Check for local changes
echo ""
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "You have local changes:"
  git status -s
  echo ""
  echo "Choose how to handle them:"
  echo "[1] Auto-commit"
  echo "[2] Stash temporarily"
  echo "[3] Discard (reset hard)"
  read -p "Enter option [1-3]: " choice

  case $choice in
    1)
      echo "Committing local changes..."
      git add .
      git commit -m "WIP: auto-commit from git-heal"
      ;;
    2)
      echo "Stashing changes..."
      git stash
      ;;
    3)
      echo "Discarding changes..."
      git reset --hard
      ;;
    *)
      echo "Invalid choice. Exiting."
      exit 1
      ;;
  esac
else
  echo "No local changes to handle."
fi

# Step 4: Fetch and pull
echo ""
echo "Fetching latest..."
git fetch --all

echo "Pulling latest changes..."
git pull

# Step 5: Restore stash if it was used
if [ "$choice" == "2" ]; then
  echo "Restoring stashed changes..."
  git stash pop
fi

echo ""
echo "Done. Repo healed."
git status
