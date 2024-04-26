from . import console
import lexa.whisper_stt as whisper_stt
import lexa.ollama_rag as ollama_rag
import lexa.google_tts as google_tts

def main():
    console.print("[cyan]Ready to record audio. Press Enter to start.")
    
    while True:
        try:
            user_text = whisper_stt.prompt_listening()
            if user_text is None:
                continue
            with console.status("[yellow] Processing the user input...", spinner="earth"):
                response = ollama_rag.get_response(user_text)
            console.print(f"[green]Response: {response}")
            with console.status("[yellow] Speaking the response...", spinner="earth"):
                google_tts.speak(response)
            
        except KeyboardInterrupt:
            console.print("\n[cyan]Exiting...")
            break
        except EOFError:
            console.print("\n[cyan]Exiting...")
            break


if __name__ == "__main__":
    main()