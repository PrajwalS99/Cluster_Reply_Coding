from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import spacy
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Define topics based on your requirements (adjust as needed)
topics = ['Sports', 'Travel', 'Fashion', 'Music', 'Food', 'Technology', 
          'Art', 'Nature', 'Science', 'Politics', 'Business', 'Education'] # Just few examples. We can add more topics as needed.

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Dummy function to extract text from Blob Storage
def parse_url(url):
  container_name, blob_name = url.split('/', 1)
  return container_name, blob_name

def extract_text_from_blob(post_url):
  container_name, blob_name = parse_url(post_url) # Parse URL to get container name and blob name of the instagram post

  # Connect to Blob Storage (replace with your connection string)
  connect_str = "<the_connection_string>"
  blob_service_client = BlobServiceClient.from_connection_string(connect_str)

  # Get blob client
  blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

  # Download text from blob
  text = blob_client.download_blob().readall().decode('utf-8')

  return text

# Function to preprocess text. Basically does the stemming and lemmatization.
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# Class for semi-supervised topic classification
class SemiSupervisedClassifier:
    def __init__(self, topics):
        self.topics = topics
        self.vectorizer = TfidfVectorizer()
        self.clusterer = KMeans(n_clusters=len(topics))
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.labeled_data = []  # List to store labeled data (text, label)

    def train(self, X):
        # Preprocess labeled data
        X_labeled_text = [data["text"] for data in self.labeled_data]
        y_labeled = [data["label"] for data in self.labeled_data]
        # Preprocess unlabeled data
        X_unlabeled_text = [post for post in X if post not in X_labeled_text]
        
        # Train clustering model on unlabeled data
        X_unlabeled_vectorized = self.vectorizer.fit_transform(X_unlabeled_text)
        X_unlabeled_clustered = self.clusterer.fit_predict(X_unlabeled_vectorized)

        # Use manual labeling for a subset of clustered data
        # Here, we label the first 1000 clustered data points as an example
        for i, cluster_label in enumerate(X_unlabeled_clustered[:1000]):
            self.labeled_data.append({"text": X_unlabeled_text[i], "label": topics[cluster_label]})

        # Train classifier on labeled data
        X_labeled_vectorized = self.vectorizer.transform(X_labeled_text)
        self.classifier.fit(X_labeled_vectorized, y_labeled)

    def predict_proba(self, X):
        X_vectorized = self.vectorizer.transform(X)
        probabilities = self.classifier.predict_proba(X_vectorized)
        return probabilities

    
# Instantiate the classifier
classifier = SemiSupervisedClassifier(topics)

@app.route('/classify', methods=['POST'])
def classify():
  # Get text from request body, prioritizing photo URL if available
  post_text = request.json.get('photo_url', '') or request.json.get('text', '')

  if post_text.startswith('http'):  # Assuming a photo URL
    try:
      post_text = extract_text_from_blob(post_text)
    except Exception as e:
      return jsonify({'error': f'Failed to extract text from the image: {str(e)}'}), 500

  preprocessed_text = preprocess(post_text)
  probabilities = classifier.predict_proba([preprocessed_text])[0]
  probabilities_dict = {topic: prob for topic, prob in zip(topics, probabilities)}
  return jsonify(probabilities_dict)

if __name__ == '__main__':
    app.run(debug=True)

