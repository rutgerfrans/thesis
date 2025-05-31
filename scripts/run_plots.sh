#!/usr/bin/env bash
set -e
echo "Starting to run all plotting scripts..."

echo ""
echo "1) Running plotFaultP.py..."
python3 plots/plotFaultP.py

echo ""
echo "2) Running plotHorizontalScaling.py..."
python3 plots/plotHorizontalScaling.py

echo ""
echo "3) Running plotStrongScaling.py..."
python3 plots/plotStrongScaling.py

echo ""
echo "4) Running plotVerticalScaling.py..."
python3 plots/plotVerticalScaling.py

echo ""
echo "5) Running plotWeakScaling.py..."
python3 plots/plotWeakScaling.py

echo ""
echo "All plotting scripts have completed successfully."
