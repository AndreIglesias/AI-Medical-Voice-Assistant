from . import gtts_lang
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
from tempfile import NamedTemporaryFile
import os

def speak(text):
    """
    Generates text-to-speech from the given text and plays it using sounddevice.
    """
    tts = gTTS(text=text, lang=gtts_lang, slow=False)

    # Save to a temporary audio file
    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
    
        # Read the audio data with soundfile
        audio_data, sample_rate = sf.read(temp_audio_file.name, dtype='float32')

    # Play the audio with sounddevice
    sd.play(audio_data, sample_rate)
    sd.wait()

    # Clean up the temporary file
    os.remove(temp_audio_file.name)

if __name__ == "__main__":
    text = "Bonjour! Comment puis-je vous aider aujourd'hui?"
    speak(text)
