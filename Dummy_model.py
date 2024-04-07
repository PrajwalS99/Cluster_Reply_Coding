from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from azure.storage.blob import BlobServiceClient

# Define topics based on your requirements (adjust as needed)
topics = ['Sports', 'Travel', 'Fashion', 'Music', 'Food', 'Technology', 
          'Art', 'Nature', 'Science', 'Politics', 'Business', 'Education'] # Just few examples. We can add more topics as needed.

# Function to extract text from Blob Storage (replace with your URL parsing logic)
def parse_url(url):
  container_name, blob_name = url.split('/', 3)[-2:]
  return container_name, blob_name

def extract_text_from_blob(post_url):
  """Extracts text from a post stored in Azure Blob Storage."""
  container_name, blob_name = parse_url(post_url)

  # Connect to Blob Storage (replace with your connection string)
  connect_str = "<your_connection_string>"
  blob_service_client = BlobServiceClient.from_connection_string(connect_str)

  # Get blob client
  blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

  # Download text from blob
  text = blob_client.download_blob().readall().decode('utf-8')

  return text

# Class for semi-supervised topic classification
class SemiSupervisedClassifier:
  def __init__(self, topics):
    self.topics = topics
    self.vectorizer = TfidfVectorizer()
    self.lda_model = LatentDirichletAllocation(n_components=len(topics), random_state=0)
    self.classifier = RandomForestClassifier(n_estimators=100)
    self.labeled_data = []  # List to store labeled data (text, label)

  def train(self):
    # Preprocess labeled data
    X_labeled_text = [data["text"] for data in self.labeled_data]
    y_labeled = [data["label"] for data in self.labeled_data]

    # Extract TF-IDF features
    X_labeled_features = self.vectorizer.fit_transform(X_labeled_text)

    # Train the LDA model (unsupervised learning for initial topic representation)
    self.lda_model.fit(X_labeled_features)

    # Extract topic distribution for labeled data using LDA
    X_labeled_topics = self.lda_model.transform(X_labeled_features)

    # Train the classifier (supervised learning on labeled topics)
    self.classifier.fit(X_labeled_topics, y_labeled)

  def predict_proba(self, X):
    # Preprocess unseen data
    X_features = self.vectorizer.transform(X)

    # Extract topic distribution for unseen data using LDA
    X_topics = self.lda_model.transform(X_features)

    # Predict topic probabilities using the trained classifier
    probabilities = self.classifier.predict_proba(X_topics)
    return probabilities

  def add_labeled_data(self, text, label):
    self.labeled_data.append({"text": text, "label": label})

# Instantiate the classifier
classifier = SemiSupervisedClassifier(topics)

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
  # Get text from request body, prioritizing photo URL if available
  post_text = request.json.get('photo_url', '') or request.json.get('text', '')

  if post_text.startswith('http'):  # Assuming a photo URL
    try:
      post_text = extract_text_from_blob(post_text)
    except Exception as e:
      return jsonify({'error': f'Failed to extract text from Blob Storage: {str(e)}'}), 500

  # Call model to get probabilities
  probabilities = classifier.predict_proba([post_text])[0]

  # Convert probabilities to dictionary
  probabilities_dict = {topic: prob for topic, prob in zip(topics, probabilities)}

  # Return probabilities as JSON response
  return jsonify(probabilities_dict)

