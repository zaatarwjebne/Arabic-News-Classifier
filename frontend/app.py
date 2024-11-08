import webbrowser
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from gensim.models import Word2Vec
from Backend.text_processor import preprocess_arabic_text
from Backend.Models.CNN.CNNArch import TextCNN

app = Flask(__name__)

# Load the models
models = {
    'model1': {  # TextCNN model
        'model': torch.load('../Backend/Models/CNN/cnn_model.pth', map_location=torch.device('cpu')),  # Load the saved TextCNN model
        'word2vec': Word2Vec.load('../Backend/Models/CNN/word2vec.model')  # Load the Word2Vec model
    },
    'model2': {  # KNN model
        'model': joblib.load('../Backend/Models/KNN/KNN_model.pkl'),
        'vectorizer': joblib.load('../Backend/Models/KNN/tfidf.pkl')
    },
    'model3': {  # XGBoost model
        'model': joblib.load('../Backend/Models/XGBoost/XGBoost_model.pkl'),
        'vectorizer': joblib.load('../Backend/Models/XGBoost/XGBoost_tfidf_vectorizer.pkl')
    }
}

# Set device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the TextCNN model to the device (for GPU/CPU compatibility)
models['model1']['model'].to(device)

# Helper function to convert words to indices using Word2Vec
def words_to_embeddings(words, word2vec_model):
    embeddings = []
    for word in words:
        # If the word is in the Word2Vec model vocabulary, use its embedding; otherwise, use a zero vector
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])  # Get the word embedding
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))  # Use a zero vector for out-of-vocabulary words
    return np.array(embeddings)

# Helper function for padding sequences using PyTorch
def pad_sequence(sequence, maxlen):
    # Pad sequence using PyTorch's pad function
    pad_length = maxlen - len(sequence)
    if pad_length > 0:
        padded_sequence = F.pad(torch.tensor(sequence, dtype=torch.long), (0, pad_length), "constant", 0)
    else:
        padded_sequence = torch.tensor(sequence[:maxlen], dtype=torch.long)
    return padded_sequence

@app.route('/')
def home():
    return render_template('template.html')  # Render the HTML file

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        # Get the text input and selected model from the request
        text = request.form.get('text')
        model_name = request.form.get('model')

        if not text:
            return jsonify({'error': 'No text provided for classification.'}), 400

        if model_name not in models:
            return jsonify({'error': 'Invalid model selected.'}), 400

        # Preprocess the input text
        preprocessed_text = preprocess_arabic_text(text)

        # Get the selected model and vectorizer
        model_info = models[model_name]
        model = model_info['model']

        if model_name == 'model1':  # TextCNN model
            word2vec_model = model_info['word2vec']

            # Tokenize and generate word indices from the preprocessed text
            words = preprocessed_text.split()
            word_indices = [word2vec_model.wv.key_to_index.get(word, 0) for word in words]  # Get word indices (default to 0 if not in vocab)

            # Pad the word indices sequence to ensure consistent input size
            word_indices_padded = pad_sequence(word_indices, maxlen=100)  # Ensure maxlen is set appropriately

            # Convert to tensor (dtype=torch.long for word indices)
            text_tensor = word_indices_padded.to(device)  # Move to correct device (CPU or GPU)

            # Perform classification using the pre-trained TextCNN model
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                prediction = model(text_tensor.unsqueeze(0))  # Unsqueeze for batch dimension
                prediction = prediction.argmax(dim=1).item()  # Get the predicted class index
                prediction = prediction + 1

                # Process for KNN and XGBoost (Non-PyTorch models)
        else:
            # Use the corresponding vectorizer to transform the text
            vectorizer = model_info.get('vectorizer')
            text_vectorized = vectorizer.transform([preprocessed_text])

            # Use the model to perform classification
            prediction = model.predict(text_vectorized)[0]

        # Map the prediction to categories (if needed)
        category_map = {
            0: ('Culture', 'ثقافة'),
            1: ('Finance', 'اقتصاد'),
            2: ('Medical', 'طبي'),
            3: ('Politics', 'سياسة'),
            4: ('Religion', 'دين'),
            5: ('Sports', 'رياضة'),
            6: ('Tech', 'تكنولوجيا'),
        }

        # Get category in English and Arabic
        category_en, category_ar = category_map.get(prediction, ('Unknown', 'غير معروف'))

        # Return the classification result
        return jsonify({
            'category': category_en,
            'category_ar': category_ar
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5001/')  # Open the browser automatically
    app.run(host='0.0.0.0', port=5001)  # Run the Flask app
