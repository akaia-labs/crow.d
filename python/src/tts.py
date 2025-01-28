#!/bin/env python

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

def speak(prompt: str, output_file: str = "parler_tts_out.wav") -> None:
    description = "Laura's voice is monotone, with a very close recording that almost has no background noise."
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    sf.write(output_file, audio_arr, model.config.sampling_rate)

#! For test purposes only
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tts.py 'your text to speak'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    speak(prompt)

if __name__ == "__main__":
    main()
