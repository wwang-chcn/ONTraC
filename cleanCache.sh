#!/bin/bash

# Clean cache directories
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".mypy_cache" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name "venv" -exec rm -rf {} +
find . -type d -name "build" -exec rm -rf {} +

# Clean cache files
find . -type f -name "*.pyc" -delete

echo "Cache cleaned successfully!"