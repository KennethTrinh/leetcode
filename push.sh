#!/bin/bash

# Help menu
if [ "$1" == "-h" ] || [ "$1" == "--help" ]
then
  echo "Usage: ./push.sh <commit message> [--undo]"
  echo "Commits and pushes changes to git. If --undo is supplied, the changes are undone after being pushed."
  exit 0
fi

# Check if commit message was supplied
if [ $# -eq 0 ]
then
    echo "No commit message supplied"
    exit 1
fi

# Check if --undo flag was supplied
if [ "$2" == "--undo" ]
then
    # Save git diff in a variable
    DIFF=$(git diff)

    # Standard git commands
    git add --all
    git commit -m "$1"
    git push

    # Apply the saved diff
    echo "$DIFF" | git apply -R

    # Commit the undo
    git add --all
    git commit -m "Undo: $1"
    git push
else
    # Standard git commands
    git add --all
    git commit -m "$1"
    git push
fi