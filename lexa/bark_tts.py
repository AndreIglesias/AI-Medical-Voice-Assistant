from . import bark_speaker, console, np, sd
import torch
import nltk
from transformers import AutoProcessor, BarkModel
import warnings

# https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c&p=1d4527fc33dc4cf0ac932a1d99fa1019&pm=s

# Download the NLTK punkt tokenizer
nltk.download('punkt')
warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

model_id = "suno/bark-small"

# Class for Text-to-Speech with Bark
class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class with a Bark model.
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = BarkModel.from_pretrained(model_id)
        self.model.enable_cpu_offload()
        # self.model = self.model.to_bettertransformer()
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = bark_speaker):
        """
        Synthesizes audio from the given text using the specified voice preset.
        Returns: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = bark_speaker):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.
        Returns: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)  # Break text into sentences
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sentence in sentences:
            sample_rate, audio_array = self.synthesize(sentence, voice_preset)
            pieces.append(audio_array)
            pieces.append(silence)

        return sample_rate, np.concatenate(pieces)

# Simple function to synthesize and play text-to-speech
def text_to_speech(text):
    tts_service = TextToSpeechService()
    with console.status("[yellow] Sintetizing the text...", spinner="earth"):
        sample_rate, audio_data = tts_service.long_form_synthesize(text)
        # sample_rate, audio_data = tts_service.synthesize(text)
    sd.play(audio_data, sample_rate)
    sd.wait()

# Main function to test the text-to-speech synthesis
if __name__ == "__main__":
    console.print("[cyan]Testing Text-to-Speech Service")

    text = "Bonjour! Comment puis-je vous aider aujourd'hui?"
    text_to_speech(text)
