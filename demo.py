import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS

print("Loading Chatterbox...")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = ChatterboxTTS.from_pretrained(device=device)

text = """
Hello! This is a test of the Chatterbox text to speech system.
The installation is working successfully.
"""

print("Generating speech...")

wav = model.generate(text, exaggeration=0.7)

output_file = "chatterbox_demo.wav"

ta.save(output_file, wav, model.sr)

print(f"Success! Audio saved to: {output_file}")
