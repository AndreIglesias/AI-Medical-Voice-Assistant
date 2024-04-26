from . import speech_pipeline, console, sd, np, Queue, threading
import time

# Helper function to record audio asynchronously
def record_audio(stop_event, data_queue):
    """
    Records audio data and adds it to the queue until the stop_event is set.
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


# Function to transcribe recorded audio
def transcribe(audio_np, sr=16000):
    """
    Transcribes the given audio data using Whisper.
    """
    try:
        result = speech_pipeline(
            {
                "sampling_rate": sr,
                "raw": audio_np
            },
            return_timestamps=True, 
            generate_kwargs={"language": "french"})
        return result
    except EOFError or KeyboardInterrupt:
        return None

def prompt_listening():
    """
    Records audio data until stopped by the user and returns it as text.
    """
    try:
        console.input("Press Enter to start recording, and again to stop...")

        data_queue = Queue()                    # Queue for holding audio data
        stop_event = threading.Event()          # Event to signal recording to stop
        recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
        recording_thread.start()                # Start the recording thread

        console.input("Recording... Press Enter to stop.")
        stop_event.set()                        # Signal recording to stop
        recording_thread.join()                 # Wait for the thread to finish

        audio_data = b"".join(list(data_queue.queue))
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # If there's any recorded audio, transcribe it
        if audio_np.size > 0:
            with console.status("[yellow]Transcribing audio...", spinner="earth"):
                transcribed = transcribe(audio_np)
                transcribed_text = transcribed["text"].strip()

                if transcribed["chunks"][0]["timestamp"][1] == None:
                    console.print("[red]Couldn't understand what you said. Please try again")
                    return None

                console.print(f"[green]Transcription: {transcribed_text}")
                return transcribed_text
        else:
            console.print("[red]No audio recorded. Please try again.")
            return None
                
    except KeyboardInterrupt or EOFError:
        console.print("\n[red]Recording stopped.")
        try:
            stop_event.set()
            recording_thread.join()
        except UnboundLocalError:
            pass


# Main function to record audio on user prompt and transcribe it
if __name__ == "__main__":
    console.print("[cyan]Ready to record audio. Press Enter to start.")
    
    while True:
        try:
            prompt_listening()
        except KeyboardInterrupt or EOFError:
            console.print("[cyan]Exiting...")
            break
            
