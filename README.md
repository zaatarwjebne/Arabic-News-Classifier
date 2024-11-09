# Arabic News Classifier

## Overview
An advanced machine learning system that classifies Arabic news articles into distinct categories using multiple state-of-the-art algorithms. The project utilizes the SANAD Dataset and implements three different classification approaches, each achieving remarkable accuracy:

- Convolutional Neural Network (CNN): ~97% accuracy
- XGBoost: ~97% accuracy
- K-Nearest Neighbors (KNN): ~95% accuracy

## Features
- Multi-model classification support
- Bilingual interface (Arabic/English)
- Real-time text classification
- User-friendly web interface
- Comprehensive text preprocessing
- Model performance comparison

## Technical Architecture
- **Frontend**: Flask-based web application with responsive design
- **Backend**: Python-based ML models
- **Models**: CNN (PyTorch), XGBoost, and KNN implementations
- **Text Processing**: Advanced Arabic text preprocessing pipeline

## Installation
1. Clone the repository
2. Create conda environment:


conda env create -f environment.yml


conda activate ArabicNewsClassification

2. Access the web interface through your browser
3. Select your preferred model
4. Input Arabic news text
5. Get instant classification results

## Model Performance
- **CNN**: 97.22% accuracy with optimized hyperparameters
- **XGBoost**: 97.08% accuracy after parameter tuning
- **KNN**: 94.48% accuracy with optimal neighbor configuration

## Future Development
- Integration of transformer-based models
- Enhanced multilingual support
- Real-time model performance monitoring
- API endpoint development for external integration
- Continuous model retraining pipeline

## Contributing
Contributions are welcome! Please feel free to submit pull requests.
