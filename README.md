# Statistical Analysis Web Application

# Clone repository
```bash
git clone https://github.com/daffaar0206/Perhitungan-RAL-RAK-
```
# Navigate to directory
```bash
cd app
```
# Install dependencies
```bash
pip install flask pandas scipy numpy
```
# Run application
```bash
python run.py
```
# Access in browser: http://localhost:5000

# API Usage Examples:

# 1. Process Analysis (using curl)
```bash
curl -X POST http://localhost:5000/process \
-H "Content-Type: application/json" \
-d '{
  "design_type": "ral",
  "data": [your_data_here]
}'
```
# 2. Post-hoc Analysis (using curl)
```bash
curl -X POST http://localhost:5000/post_hoc \
-H "Content-Type: application/json" \
-d '{
  "design_type": "ral",
  "post_hoc_type": "duncan",
  "data": [your_data_here]
}'
```
# Features:
# - RAL (Completely Randomized Design)
# - RAK (Randomized Block Design)
# - RAL Factorial Design
# - ANOVA calculations
# - Post-hoc tests (Duncan, BNT/LSD)
# - Descriptive statistics
# - CV calculations

# Dependencies:
# - Flask
# - Pandas
# - SciPy
# - NumPy

# Author: [daffaar0206]

