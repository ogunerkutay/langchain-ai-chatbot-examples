# Interacting with AI Models

## Repository Overview

This repository contains three Python scripts that demonstrate how to interact with various AI models using the LangChain library. The scripts utilize different models, including Gemini, Hugging Face, and Mistral AI, to generate responses to user queries. The repository provides examples of how to fine-tune models, create vector stores, and define custom runnables to interact with the models.

## Scripts

- `finetunedGeminiWithRetrievalQA.py`: Utilizes LangChain to fine-tune a Gemini model with retrieval QA capabilities. It sets up a Google Generative AI model and creates a vector store using FAISS. Interact with the model using the custom `GenAIRunnable` class.
  
- `huggingfacemodels.py`: Demonstrates interaction with the Hugging Face API to generate text using a Gemini-7B model. It loads an API token from a `.env` file, sends a request to the Hugging Face API, and prints the generated text or any errors encountered.
  
- `mistralmodel.py`: Utilizes LangChain to interact with a Mistral AI model. It sets up a Mistral AI model using the Hugging Face library and creates a vector store using FAISS. Interact with the model using the custom `MistralAIEmbeddings` class.

## Technical Requirements

- Python 3.7+
- LangChain library
- Hugging Face library
- FAISS library
- Google Generative AI credentials
- Hugging Face API token
- Mistral AI API key

## Usage

1. Clone the repository and navigate to the directory.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Create a `.env` file with the required API keys and tokens.
4. Run the scripts using:
   - `python finetunedGeminiWithRetrievalQA.py`
   - `python huggingfacemodels.py`
   - `python mistralmodel.py`
5. Interact with the models by providing user queries.
