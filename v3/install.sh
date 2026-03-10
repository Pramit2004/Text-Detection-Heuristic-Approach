#!/bin/bash
# ============================================================
# Grandpa Text Detector - Installation Script
# ============================================================
set -e

echo "👴 Installing Grandpa Text Detector dependencies..."

pip install -r requirements.txt

# Download NLTK data
python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('✅ NLTK data downloaded.')
"

# Download spaCy model
python3 -m spacy download en_core_web_sm

echo ""
echo "✅ Installation complete!"
echo "🚀 Run: python3 app.py"
