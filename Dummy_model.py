from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import spacy
from azure.storage.blob import BlobServiceClient
import pytesseract
from PIL import Image
import io

# Initialize Tesseract OCR to get text from Images
pytesseract.pytesseract.tesseract_cmd = r'<path_to_tesseract_executable>'


# Define topics based on your requirements (adjust as needed)
topics = ['Sports', 'Travel', 'Fashion', 'Music', 'Food', 'Technology', 
          'Art', 'Nature', 'Science', 'Politics', 'Business', 'Education'] # Just few examples. We can add more topics as needed.

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_blob(blob_data):
  # Connect to Blob Storage
  # Use PIL to open image from blob data
  image = Image.open(io.BytesIO(blob_data))
    
  # Perform OCR to extract text from image
  extracted_text = pytesseract.image_to_string(image) # If there is already text extracted, we can use UTF-8 encoding directly.

  return extracted_text

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
        self.clusterer = KMeans(n_clusters=len(topics))  #  For text, use naive bayes, I've resarched that it is usually better for text.
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
        X_labeled_vectorized = self.vectorizer.transform(X_labeled_text) # could also use pipeline here from sklearn.pipeline. 
        self.classifier.fit(X_labeled_vectorized, y_labeled)

    def predict_proba(self, X):
        X_vectorized = self.vectorizer.transform(X)
        probabilities = self.classifier.predict_proba(X_vectorized) # could also pront classification report for better understanding
        return probabilities


def main(blob_data):
    # Instantiate the classifier
    classifier = SemiSupervisedClassifier(topics)

    # Extract text from Blob Storage
    post_text = extract_text_from_blob(blob_data)

    # Preprocess text
    preprocessed_text = preprocess(post_text)

    # Train the classifier
    classifier.train([preprocessed_text])

    # Predict probabilities
    probabilities = classifier.predict_proba([preprocessed_text])[0]
    probabilities_dict = {topic: prob for topic, prob in zip(topics, probabilities)}
    
    return probabilities_dict  # For debugging purposes, to see the probabilities
    # You can return this probabilities_dict as JSON response in your actual API implementation

if __name__ == '__main__':
    main()
 