# __init__.py

# Common Imports
from rich.console import Console
import sounddevice as sd
from queue import Queue
import numpy as np
import threading
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# ==================================================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ollama RAG
ollama_model = "gpt-4-turbo"

template =  """
Tu es un assistant IA serviable et amical. Tu es poli, respectueux, et tu vises à donner des réponses
concises de moins de 20 mots.

La transcription de la conversation est la suivante :
{history}

Et voici la question de suivi de l'utilisateur : {input}

Ta réponse :
"""

context_template = """
Répondez à la question uniquement en vous basant sur le contexte suivant. Si vous ne pouvez pas
répondre à la question, répondez "Je ne sais pas".

Contexte : {context}

Question : {question}
"""

# TTS
bark_speaker = "v2/fr_speaker_1"
whisper_lang = "french"
gtts_lang = "fr"


# ==================================================================================================


# Shared Rich Console Instance
console = Console()

# Initialize whisper-stt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def create_speech_pipeline():
    model_id = "openai/whisper-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

speech_pipeline = create_speech_pipeline()