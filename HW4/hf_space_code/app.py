import torch
from transformers import pipeline
import gradio as gr


# Check if GPU is available, otherwise default to CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline for automatic speech recognition (ASR)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=device,
)

# Function to handle the audio and transcription
def transcribe_audio(audio):
    prediction = pipe(audio)["text"]
    return prediction

# Gradio interface for recording audio via microphone
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"), 
    outputs=gr.Textbox(label="Transcription"),
    title="Whisper Audio to Text Transcription",
    description="Record audio using your microphone and get a text transcription using the Whisper small model."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()