<p align="center">
  <h1>DigH@cktion - Medical Score Assistant</h1>
</p>


<p align="center">
  <img alt="Static Badge" src="https://img.shields.io/badge/Whisper-%20STT-g?style=for-the-badge&logo=openai">
  <img alt="Static Badge" src="https://img.shields.io/badge/medllama2-%20llm-orange?style=for-the-badge&logo=meta">
  <img alt="Static Badge" src="https://img.shields.io/badge/Bark-%20TTS-blue?style=for-the-badge">
  <img alt="Static Badge" src="https://img.shields.io/badge/gTTS-%20TTS-blue?style=for-the-badge&logo=google">
  <img alt="Static Badge" src="https://img.shields.io/badge/Prix-%20Du_publique-gold?style=for-the-badge">
  <img alt="Static Badge" src="https://img.shields.io/badge/Prix-%20Wilco-gold?style=for-the-badge">
  <img alt="Static Badge" src="https://img.shields.io/badge/License-%20MIT-black?style=for-the-badge">
</p>

Welcome to the GitHub repository for the [DigH@cktion](https://www.dighacktion.com/) Hackathon project! 

This project is a **chat/voice assistant** designed to calculate **medical scores**.

Our solution uses advanced AI technologies like **LangChain**, **LLM RAG** with **Vector Search**, **data chunking**, **embeddings** and **Chroma vector database** to provide contextual responses by processing information from **PDFs** and **URLs**.
It also supports **function calling** for automated medical score calculations.

For **speech-to-text**, it uses the model **Whisper** from *OpenAI*, allowing voice interactions with the assistant.

For natural language processing, the project can leverage either an open-source model with **Ollama**, such as **MedLlama2**, or a GPT model with an **OpenAI API Key**. This flexibility allows the project to run **locally** or connect to the cloud with *OpenAI services*.

https://github.com/AndreIglesias/DigHacktion/assets/35022933/949164ef-873f-4e32-a5de-24adea5ed57a

## DigH@cktion Overview

DigH@cktion is the 4th edition of the innovation program dedicated to diseases and cancers of the digestive system.

> This program is supported by 20 scientific societies, national bodies, working groups, and associations in the ecosystem of digestive system diseases and cancers. The goal of DigH@cktion is to encourage the creation of innovative digital solutions that benefit patients and healthcare professionals.

## Achievements

We are proud to announce that our project won **two prizes** at the DigH@cktion Hackathon:

1. **Prix du Public**: Recognizing the project as the best solution by public vote.
3. **Prix Wilco**: Awarded by WILCO, indicating high startup potential and innovation.

These awards demonstrate our commitment to creating an impactful and innovative solution for the medical field.

## Project Description

Our project involves a chat/voice assistant designed to aid healthcare professionals in medical score calculations. Here are some of the key technical components and their functions:

- **LangChain**: Enables structured conversational flows and natural language understanding, allowing the assistant to interact with users in a human-like manner.
- **Chroma Database**: Acts as the vector database to store and retrieve embeddings from various documents, supporting semantic searches across large datasets.
- **LLM RAG**: Stands for "Language Learning Model - Retrieval Augmented Generation," providing context by embedding information from PDFs and URLs, allowing the assistant to offer more accurate and informed responses.
- **Function Calling**: Facilitates the computation of medical scores, allowing the assistant to invoke specific functions based on user requests.
- **Whisper**: An OpenAI speech-to-text model for voice-based interactions.
- **Ollama/MedLlama2** or **OpenAI GPT**: Supports both local processing with open-source models and cloud processing with OpenAI's GPT model.

![rag_llama](https://github.com/AndreIglesias/DigHacktion/assets/35022933/f01e201a-5c08-43aa-a05c-0bacdb5ec36c)

## Getting Started

To set up the project on your local environment, follow these steps:

1. **Clone the Repository**: Download the source code to your local environment.
   ```bash
   git clone https://github.com/AndreIglesias/DigHacktion
   ```
2. **Install the dependencies** with poetry.
   ```bash
   poetry install
   ```
3. **Activate the Virtual Environment**.
   ```bash
   poetry shell
   ```
4. *Optionally* Create an .env file for the `OPENAI_API_KEY`.

---

### License

This project is licensed under the [MIT License](LICENSE).
