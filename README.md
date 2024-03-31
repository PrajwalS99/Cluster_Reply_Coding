# Instagram Post Topic Classification Microservice

## Overview

This project implements a microservice for classifying Instagram posts by topic. It receives the text-body of an Instagram post via a REST API and returns a JSON object with the determined probabilities of class membership  as a response.

## Use Case

In this fictitious use case, the goal is to label Instagram posts by topics such as Football, Food types(Countrywise), Stockmarket, and more. The microservice aims to identify at least 20 and a maximum of 100 such topics based on the textual content of the posts.

## Implementation Details

### Dummy Model

The project includes a dummy model that returns probabilities of topics for a given post. This model is simple and does not involve training; it generates dummy probabilities based on input text.

### REST API Endpoint

The microservice exposes a REST API endpoint to handle requests for topic classification. It defines API endpoints and integrates the dummy model to process requests.

### Cloud Deployment

The microservice is deployed on a cloud platform (Azure). Deployment is automated using cloud deployment tools.

## Requirements

- Python >= 3.7
- Azure 
- Git-based source code versioning tool (GitHub, GitLab, etc.)
