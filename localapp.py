from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import whisper
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import login
import torch
import os
import uuid
from typing import Generator
from pathlib import Path
import subprocess
import numpy as np
import torchaudio


# Initialize FastAPI app
app = FastAPI()


# Function to convert audio file to WAV if necessary
def convert_to_wav_if_necessary(input_file):
   if input_file.suffix in ['.mp3', '.m4a']:
       output_file = input_file.with_suffix('.wav')
       command = f'ffmpeg -i "{input_file}" "{output_file}"'
       subprocess.run(command, shell=True)
       return output_file
   return input_file


# Function to split audio file into chunks
def split_audio(file_path, chunk_length_ms=30000):  # default chunk length is 30 seconds
   audio = AudioSegment.from_file(file_path)
   chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
   return chunks


# Function to transcribe audio chunk
def transcribe_chunk(chunk_path):
   model = whisper.load_model("base")
   waveform, sample_rate = torchaudio.load(chunk_path)
   audio = waveform.squeeze().numpy()  # Convert to numpy array
   result = model.transcribe(audio)
   return result['text']


# Generator to stream transcriptions
def transcribe_audio_stream(file_path: Path) -> Generator[str, None, None]:
   wav_file_path = convert_to_wav_if_necessary(file_path)
   chunks = split_audio(wav_file_path)
   total_chunks = len(chunks)


   for i, chunk in enumerate(chunks):
       chunk_path = Path(f"/tmp/chunk_{uuid.uuid4()}.wav")
       chunk.export(chunk_path, format="wav")


       transcription = transcribe_chunk(chunk_path)
       yield f"{transcription}\n"
       os.remove(chunk_path)


       progress = (i + 1) / total_chunks * 100
       yield f"progress: {round(progress)}\n"  # rounding to nearest unit


   if wav_file_path != file_path:
       os.remove(wav_file_path)


# Load the T5-base model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Function to summarize text
def generate_summary(text, instructions, max_length=150):
   input_text = f"shorten the following text in three full, grammatically and syntactally correct sentences, with a noun, a verb, etc.: {text}"
   inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
   summary_ids = model.generate(inputs.input_ids, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
   return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@app.get("/", response_class=HTMLResponse)
async def get_form():
   return """
   <!DOCTYPE html>
   <html>
   <head>
       <title>Upload Audio File</title>
       <style>
           body {
               font-family: Arial, sans-serif;
               background-color: #f9f9f9;
               margin: 0;
               padding: 20px;
           }
           h1, h2 {
               color: #333;
           }
           form {
               margin-bottom: 20px;
               background: #fff;
               padding: 20px;
               border-radius: 8px;
               box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
           }
           label {
               display: block;
               margin-bottom: 8px;
               font-weight: bold;
           }
           input[type="text"], input[type="file"], input[type="submit"], input[type="button"], input[type="checkbox"] {
               display: block;
               margin-bottom: 10px;
               padding: 10px;
               width: 100%;
               box-sizing: border-box;
               border-radius: 4px;
               border: 1px solid #ccc;
           }
           input[type="submit"], input[type="button"] {
               background-color: #ADD8E6;
               color: white;
               border: none;
               cursor: pointer;
           }
           input[type="submit"]:hover, input[type="button"]:hover {
               background-color: #87CEEB;
           }
           #transcription, #summary {
               white-space: pre-wrap;
               word-wrap: break-word;
               background: #fff;
               padding: 20px;
               border-radius: 8px;
               box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
               margin-bottom: 20px;
               display: none;
           }
           .progress {
               width: 100%;
               background-color: #f3f3f3;
               border-radius: 4px;
               overflow: hidden;
               margin-bottom: 20px;
           }
           .progress-bar {
               width: 0;
               height: 24px;
               background-color: #ADD8E6;
               text-align: center;
               color: white;
               line-height: 24px;
               transition: width 0.4s;
           }
       </style>
   </head>
   <body>
       <h1>Upload Audio File for Transcription and Summary</h1>
       <form id="uploadForm" enctype="multipart/form-data">
           <input type="file" name="file" id="file" accept=".mp3,.m4a,.wav" required><br><br>
           <label for="displayTranscription">Display Transcription:</label>
           <input type="checkbox" id="displayTranscription" name="displayTranscription"><br><br>
           <input type="button" id="transcribeButton" value="Transcribe">
           <div class="progress">
               <div id="transcribeProgress" class="progress-bar">0%</div>
           </div>
       </form>
       <form id="summaryForm">
           <label for="instructions">Instructions for Summary (e.g., "3 bullet points in French", or "4 sentences in English", etc.):</label>
           <textarea name="instructions" id="instructions" rows="4" required></textarea>
           <input type="button" id="summarizeButton" value="Summarize" disabled>
       </form>
       <h2>Transcription</h2>
       <pre id="transcription"></pre>
       <h2>Summary</h2>
       <div id="summary"></div>
       <script>
           async function updateProgressBar(element, progress) {
               element.style.width = progress + '%';
               element.textContent = progress + '%';
           }


           document.getElementById('transcribeButton').addEventListener('click', async (event) => {
               event.preventDefault();
               const formData = new FormData(document.getElementById('uploadForm'));
               const displayTranscription = document.getElementById('displayTranscription').checked;


               const response = await fetch('/upload/', {
                   method: 'POST',
                   body: formData
               });


               const progressBar = document.getElementById('transcribeProgress');
               if (response.ok) {
                   const reader = response.body.getReader();
                   const decoder = new TextDecoder();
                   let transcription = '';
                   let receivedLength = 0;


                   while (true) {
                       const { done, value } = await reader.read();
                       if (done) break;
                       let text = decoder.decode(value, { stream: true });


                       // Check if the text contains progress update
                       if (text.startsWith("progress: ")) {
                           let progress = parseFloat(text.split("progress: ")[1]);
                           updateProgressBar(progressBar, progress);
                       } else {
                           transcription += text;
                           receivedLength += value.length;
                           if (displayTranscription) {
                               document.getElementById('transcription').style.display = 'block';
                               document.getElementById('transcription').textContent = transcription;
                           }
                       }
                   }
                   document.getElementById('transcription').dataset.transcription = transcription.trim();
                   document.getElementById('summarizeButton').disabled = false;
                   if (!displayTranscription) {
                       document.getElementById('transcription').style.display = 'none';
                       document.getElementById('transcription').textContent = 'Transcription done.';
                   }
               } else {
                   console.error('Failed to upload file');
               }
           });


           document.getElementById('summarizeButton').addEventListener('click', async (event) => {
               event.preventDefault();
               const transcription = document.getElementById('transcription').dataset.transcription;
               const instructions = document.getElementById('instructions').value;


               if (transcription) {
                   const summaryResponse = await fetch('/summarize/', {
                       method: 'POST',
                       headers: { 'Content-Type': 'application/json' },
                       body: JSON.stringify({ transcription, instructions })
                   });


                   const summary = await summaryResponse.json();
                   document.getElementById('summary').style.display = 'block';
                   document.getElementById('summary').textContent = summary.summary;
               } else {
                   console.error('No transcription available for summarization');
               }
           });
       </script>
   </body>
   </html>
   """


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
   if file.filename.endswith(('.mp3', '.m4a', '.wav')):
       file_path = Path(f"/tmp/{uuid.uuid4()}{Path(file.filename).suffix}")
       with open(file_path, "wb") as buffer:
           buffer.write(await file.read())
       return StreamingResponse(transcribe_audio_stream(file_path), media_type="text/event-stream")
   else:
       raise HTTPException(status_code=400, detail="Invalid file type, only .mp3, .m4a, and .wav files are accepted")


@app.post("/summarize/")
async def summarize(transcription: dict):
   transcript = transcription.get("transcription", "")
   instructions = transcription.get("instructions", "")
   summary = generate_summary(transcript, instructions)
   return {"summary": summary}
