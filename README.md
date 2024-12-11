# Statistical Analysis Web Application

# Clone repository
git clone <repository-url>

# Navigate to directory
cd app

# Install dependencies
pip install flask pandas scipy numpy

# Run application
python run.py

# Access in browser: http://localhost:5000

# API Usage Examples:

# 1. Process Analysis (using curl)
curl -X POST http://localhost:5000/process \
-H "Content-Type: application/json" \
-d '{
  "design_type": "ral",
  "data": [your_data_here]
}'

# 2. Post-hoc Analysis (using curl)
curl -X POST http://localhost:5000/post_hoc \
-H "Content-Type: application/json" \
-d '{
  "design_type": "ral",
  "post_hoc_type": "duncan",
  "data": [your_data_here]
}'

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
# License: MIT
