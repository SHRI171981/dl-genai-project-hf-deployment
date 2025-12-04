---
title: Genai Sentiment Classifier
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
---

# GenAI Sentiment Classifier

This repository contains a Python-based web application that detects emotions in text. Specifically, it classifies input text into one of five categories: Anger, Fear, Joy, Sadness, or Surprise.

I built the interface using Gradio, which makes it easy to interact with the models via a web browser.

## How It Works

Instead of relying on a single Natural Language Processing model, this application uses an **ensemble approach**. I wanted to improve the reliability of the predictions, so the app loads three separate fine-tuned models at runtime.

When you enter text, the application:
1. Sends the text to all three models simultaneously.
2. Collects the confidence scores for every label from each model.
3. Calculates the mathematical average of these scores to produce a final "weighted" prediction.

This method generally helps smooth out potential biases or errors that might occur if we relied on just one model architecture.

### The Models

I am using the following models hosted on the Hugging Face Hub:
* `shri171981/genai_model_dberta_base`
* `shri171981/genai_model_dberta_large`
* `shri171981/genai_model_roberta_base`

## Project Structure

* **app.py**: This is the main entry point. It handles loading the pipelines, defining the averaging logic, and rendering the Gradio UI.
* **requirements.txt**: Lists the dependencies required to run the app (PyTorch, Transformers, etc.).
* **.github/workflows**: Contains the CI/CD pipeline. I have set this up to automatically sync changes from this GitHub repository to the Hugging Face Space whenever code is pushed to the main branch.

## Setup and Installation

If you want to run this locally on your machine, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/SHRI171981/dl-genai-project-hf-deployment