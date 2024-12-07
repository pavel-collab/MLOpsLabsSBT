#!/bin/bash

# Description: This script automates the check for Git and Git LFS installations, installing them if not already installed.
set -e
# Function to check if Git is installed
is_git_installed() {
    if command -v git &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if Git LFS is installed
is_git_lfs_installed() {
    if command -v git-lfs &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if Git is already installed
if is_git_installed; then
    echo "Git is already installed."
    git --version
    # Check if Git LFS is installed
    if is_git_lfs_installed; then
        echo "Git LFS is already installed."
        git-lfs --version
    else
        echo "Installing Git LFS..."
        apt-get update && apt-get install git-lfs -y || { echo "Failed to install Git LFS"; exit 1; }
        echo "Git LFS has been installed successfully."
    fi
else
    # Update package lists and install Git
    apt-get update && apt-get install git -y || { echo "Failed to install Git"; exit 1; }
    # Install Git LFS
    apt-get install git-lfs -y || { echo "Failed to install Git LFS"; exit 1; }
    echo "Git and Git LFS have been installed successfully."
fi