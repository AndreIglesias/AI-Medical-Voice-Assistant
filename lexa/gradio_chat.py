from . import console, np
import gradio as gr
import scipy.signal
import lexa.whisper_stt as whisper_stt
import lexa.ollama_rag as ollama_rag
import lexa.google_tts as google_tts
import os

import time

def is_pdf(file_path):
    if not file_path.endswith(".pdf"):
        console.print(f"[red]Invalid file type: {file_path}")
        return False
    
    if not os.path.exists(file_path):
        console.print(f"[red]File not found: {file_path}")
        return False
    
    with open(file_path, "rb") as f:
        # Check if the first bytes contain the PDF magic number
        pdf_magic_number = b"%PDF"
        if not f.read(len(pdf_magic_number)) == pdf_magic_number:
            console.print(f"[red]Not a valid PDF magic number: {file_path}")
            return False
    return True


def convert_audio(audio, rate=16000):
    # The audio is in stereo, with sample rate of 44100 Hz
    sr, y = audio

    # Convert stereo to mono (average the two channels)
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.mean(y, axis=1)

    # Calculate the number of samples needed for 16000 Hz
    num_samples = int(len(y) * (rate / sr))

    # Resample the audio to {rate} Hz
    y_rate = scipy.signal.resample(y.astype(np.float32), num_samples)

    # Normalize to avoid clipping issues
    y_rate /= np.max(np.abs(y_rate))

    return y_rate


def transcribe(audio, chat_history):
    if (audio is None) or (len(audio) == 0):
        return gr.MultimodalTextbox(value=None, interactive=False), chat_history

    audio = convert_audio(audio)
    transcription = whisper_stt.transcribe(audio)["text"]
    chat_history.append((transcription, None))
    return gr.MultimodalTextbox(value=None, interactive=False), chat_history

def submit(message, chat_history):
    console.print(f"[cyan]Submit User: {message}")
    for x in message["files"]:
        console.print(f"[cyan] Is it a pdf?: {is_pdf(x)}")
        chat_history.append(((x,), None))
    
    if message["text"] is not None and len(message["text"]) > 0:
        chat_history.append((message["text"], None))

    return gr.MultimodalTextbox(value=None, interactive=False), chat_history

def respond(chat_history):
    if len(chat_history) == 0:
        return gr.MultimodalTextbox(value=None, interactive=False), chat_history
    # If last (user, bot) in chat_history is (None, None), remove it
    if chat_history[-1] == (None, None):
        chat_history.pop()
    # If last (user, bot) in chat_history is (None, response), don't do anything
    if (chat_history[-1][0] == None) and (chat_history[-1][1] != None):
        return gr.MultimodalTextbox(value=None, interactive=False), chat_history
    # If last (user, bot) in chat_history is (response, None), respond to it
    if (chat_history[-1][0] != None) and (chat_history[-1][1] == None):
        console.print(f"[cyan]Respond User: {chat_history[-1][0]}")
        if type(chat_history[-1][0]) == tuple and is_pdf(chat_history[-1][0][0]):
            response = ollama_rag.learn_pdf(chat_history[-1][0][0])
        else:
            response = ollama_rag.get_response(chat_history[-1][0])
        chat_history[-1] = (chat_history[-1][0], response)
        google_tts.speak(response)
    return gr.MultimodalTextbox(value=None, interactive=False), chat_history

def voice_chat():
    with gr.Blocks() as vchat:
        mic = gr.Microphone(label="Record Audio")
        text = gr.Textbox(label="Speech to Text | Whisper")

        # Chatbot Interface
        chatbot = gr.Chatbot()
        msg = gr.MultimodalTextbox(
            show_label=False,
            placeholder="Enter your text or file and press the Enter key, "
                        "or simply speak with the top button.",
            file_types=[".pdf"]
        )
        clear = gr.ClearButton([msg, chatbot])

        chat_msg = msg.submit(submit, [msg, chatbot], [msg, chatbot])
        bot_msg = chat_msg.then(respond, [chatbot], [msg, chatbot], api_name="respond")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
        mic.change(transcribe, [mic, chatbot], [msg, chatbot])

        @gr.on(inputs=[mic], outputs=[text])
        def get_text(audio):
            if (audio is None) or (len(audio) == 0):
                return None
            audio = convert_audio(audio)
            return whisper_stt.transcribe(audio)["text"]
    return vchat

def url_tab():
    rag_url = gr.Interface(fn=ollama_rag.process_urls,
        inputs=[gr.Textbox(label="Enter URLs separated by new lines"), gr.Textbox(label="Question")],
        outputs="text",
        title="Document Query with Ollama",
        description="Enter URLs and a question to query the documents.")
    return rag_url

def pdf_tab():
    rag_pdf = gr.Interface(fn=ollama_rag.process_pdf,
        inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Question")],
        outputs="text",
        title="Document Query with Ollama",
        description="Upload PDFs and a question to query the documents.")
    return rag_pdf

def main():

    vchat = voice_chat()
    rag_url = url_tab()
    rag_pdf = pdf_tab()
    
    demo = gr.TabbedInterface([vchat, rag_url, rag_pdf], ["Voice Chat", "URL RAG Query", "PDF RAG Query"], title="Voice Assistant ✨",)

    with console.status("[yellow]Starting the voice assistant...", spinner="earth"):
        demo.launch()


if __name__ == "__main__":
    main()