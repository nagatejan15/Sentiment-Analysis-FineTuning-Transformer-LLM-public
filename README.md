# Sentiment Analysis by Fine-Tuning a Transformer Model

This project demonstrates the process of fine-tuning a pre-trained transformer model for a sentiment analysis task. It uses the `distilbert-base-uncased` model and the IMDB movie review dataset. The project includes scripts for fine-tuning the model and for running inference with the newly trained model.

## Features

* Fine-tunes a pre-trained model from the Hugging Face library.
* Uses the `datasets` library to load and preprocess the IMDB dataset.
* Employs the `Trainer` API from the `transformers` library for efficient training.
* Saves the resulting fine-tuned model and tokenizer for later use.
* Provides a prediction script to perform sentiment analysis on new sentences.

## Model and Dataset

* **Pre-trained Model:** The project utilizes the `distilbert-base-uncased` model. The necessary model files are expected to be stored in the `pre_trained_model/` directory.
    * **Link:** [distilbert-base-uncased on Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased/tree/main)

* **Dataset:** The model is fine-tuned on the IMDB Large Movie Review Dataset. The project expects this dataset to be in the `data/` directory.
    * **Link:** [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model and Dataset**
    * Download the **distilbert-base-uncased** model files from the link above and place them in a `pre_trained_model/` directory.
    * Download the **IMDB dataset** from the link above and place it in a `data/` directory.

5.  **Create Environment File**
    Create a `.env` file in the root directory and define the paths to your data and models. The scripts `finetune.py` and `predict.py` use these variables.

    ```env
    # .env
    DATASET_PATH="data"
    PRETRAINED_MODEL_PATH="pre_trained_model"
    FINETUNED_MODEL_PATH="fine_tuned_model"
    ```

## Usage

### 1. Fine-Tuning the Model

Run the `finetune.py` script to start the training process. The script will:
* Load the dataset and pre-trained model from the paths specified in your `.env` file.
* Tokenize the data.
* Fine-tune the model on the training set and evaluate it on the test set.
* Save the best-performing model and tokenizer to the `FINETUNED_MODEL_PATH` directory.

```bash
python finetune.py 
``` 
### 2. Running Predictions
Once the model has been fine-tuned, you can use predict.py to classify the sentiment of new text.

The script loads the fine-tuned model from FINETUNED_MODEL_PATH.

It then runs inference on a predefined list of test reviews and prints the predicted sentiment (POSITIVE/NEGATIVE) and a confidence score for each.

``` Bash

python predict.py
```