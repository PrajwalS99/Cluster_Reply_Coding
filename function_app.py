import azure.functions as func
import logging
import json
from Dummy_Model import SemiSupervisedClassifier, extract_text_from_blob, preprocess

dummy_model = SemiSupervisedClassifier(topics)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="clusterreply_final", methods=["POST"])
def clusterreply_final(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        text = req_body.get('text')  # Assuming the text of the Instagram post is provided in the request body asmentioned in the task description
        if not text:
            # If no text is provided, extract text from the blob, for the case when the text is to be extracted from the instagram post
            blob_data = req_body.get('blob_data')
            if not blob_data:
                return func.HttpResponse("Text not found in the request body.", status_code=400)
            
            # Extract text from blob
            text = extract_text_from_blob(blob_data)

        # Preprocess the text
        preprocessed_text = preprocess(text) #do the stemming and lemmatization to inprove performanceof the classifier

        # Classify the preprocessed text using the dummy model
        probabilities = dummy_model.predict_proba([preprocessed_text])[0]

        # Convert probabilities to JSON
        result = json.dumps({topic: prob for topic, prob in zip(topics, probabilities)})

        # Log the result
        logging.info(f"Classified probabilities: {result}")

        # Return the classified probabilities
        return func.HttpResponse(result, mimetype="application/json")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse("An error occurred", status_code=500)
