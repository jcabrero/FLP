#!/bin/bash

echo "Launching jupyter notebook on port 10093"
cd ..
jupyter lab --port=10093 --no-browser --ip=0.0.0.0 --allow-root
