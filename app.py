from flask import Flask, request, jsonify, render_template, flash
import numpy as np
import cv2
import pytesseract
from PIL import Image
import os
from sklearn.ensemble import IsolationForest
import re
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.secret_key = 'UWUVXIW#3S3UG2YGVKI872863VXR25DSW5'  # Required for flashing messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'pdf'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_features(image_path):
    """Extract various features from the bank statement image."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")

        features = {}

        # 1. Basic Image Properties
        features['image_size'] = img.shape
        features['aspect_ratio'] = img.shape[1] / img.shape[0]

        # 2. Font Analysis
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract text using OCR
        text = pytesseract.image_to_string(Image.fromarray(gray))

        # Analyze text properties
        features['text_length'] = len(text)
        features['number_count'] = sum(c.isdigit() for c in text)
        features['uppercase_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0

        # 3. Layout Analysis
        # Find contours to analyze document structure
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)

        # 4. Spacing Analysis
        # Calculate average line spacing using horizontal projection
        h_proj = np.sum(binary, axis=1)
        line_positions = np.where(h_proj > np.mean(h_proj))[0]
        if len(line_positions) > 1:
            features['avg_line_spacing'] = np.mean(np.diff(line_positions))
        else:
            features['avg_line_spacing'] = 0

        # 5. Bank-specific Pattern Detection
        # Look for common banking patterns (account numbers, dates, amounts)
        features['has_account_number'] = bool(re.search(r'\d{8,}', text))
        features['has_date_pattern'] = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
        features['has_amount_pattern'] = bool(re.search(r'[\$£€]\s*\d+[.,]\d{2}', text))

        # 6. Advanced Text Analysis
        # Check for common banking terms
        banking_terms = ['balance', 'withdrawal', 'deposit', 'transaction', 'account']
        features['banking_term_count'] = sum(1 for term in banking_terms if term.lower() in text.lower())

        # 7. Logo Detection (simplified)
        features['has_logo'] = len([c for c in contours if cv2.contourArea(c) > 1000]) > 0

        return features

    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise


def train_anomaly_detector(feature_list):
    """Train an anomaly detection model using Isolation Forest."""
    try:
        if not feature_list:
            # Provide some example features if no training data
            feature_list = [{
                'aspect_ratio': 1.4142,
                'text_length': 1000,
                'number_count': 200,
                'uppercase_ratio': 0.3,
                'contour_count': 50,
                'avg_line_spacing': 30
            }]

        # Convert feature dictionary list to numpy array
        feature_array = np.array([[
            f['aspect_ratio'],
            f['text_length'],
            f['number_count'],
            f['uppercase_ratio'],
            f['contour_count'],
            f['avg_line_spacing']
        ] for f in feature_list])

        # Train isolation forest
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(feature_array)
        return model

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


def analyze_statement(features, model):
    """Analyze a bank statement using the trained model."""
    try:
        feature_array = np.array([[
            features['aspect_ratio'],
            features['text_length'],
            features['number_count'],
            features['uppercase_ratio'],
            features['contour_count'],
            features['avg_line_spacing']
        ]])

        # Get anomaly score (-1 for anomalies, 1 for normal samples)
        score = model.score_samples(feature_array)[0]

        # Additional rule-based checks
        required_patterns = [
            features['has_account_number'],
            features['has_date_pattern'],
            features['has_amount_pattern'],
            features['banking_term_count'] >= 2,
            features['has_logo']
        ]

        pattern_score = sum(required_patterns) / len(required_patterns)

        # Combine model score and pattern checks
        final_score = (score + 1) / 2 * 0.7 + pattern_score * 0.3

        return {
            'authenticity_score': float(final_score),
            'suspicious_patterns': {
                'missing_account_number': not features['has_account_number'],
                'missing_dates': not features['has_date_pattern'],
                'missing_amounts': not features['has_amount_pattern'],
                'missing_banking_terms': features['banking_term_count'] < 2,
                'missing_logo': not features['has_logo']
            },
            'layout_analysis': {
                'aspect_ratio': features['aspect_ratio'],
                'text_density': features['text_length'] / (features['image_size'][0] * features['image_size'][1]),
                'line_spacing_consistency': features['avg_line_spacing'] > 0,
                'structure_complexity': features['contour_count']
            }
        }

    except Exception as e:
        logger.error(f"Error in statement analysis: {str(e)}")
        raise


# Initialize with some example legitimate statements
example_features = []


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Handle document upload and analysis."""
    try:
        if 'file' not in request.files:
            flash('No file provided', 'error')
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return render_template('index.html')

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image file.', 'error')
            return render_template('index.html')

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Extract features from the uploaded document
                features = extract_features(filepath)

                # Train model with example features
                model = train_anomaly_detector(example_features)

                # Analyze the document
                result = analyze_statement(features, model)

                # Clean up
                os.remove(filepath)

                return render_template('result.html', result=result)

            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash('Error processing file. Please try again.', 'error')
                return render_template('index.html')

        flash('Invalid file', 'error')
        return render_template('index.html')

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return render_template('index.html')


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return render_template('index.html'), 413


@app.errorhandler(500)
def server_error(e):
    """Handle internal server error."""
    logger.error(f"Server error: {str(e)}")
    flash('An internal server error occurred. Please try again.', 'error')
    return render_template('index.html'), 500


if __name__ == '__main__':
    app.run(debug=True)