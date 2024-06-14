from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pydub import AudioSegment
from pathlib import Path
import subprocess
import uvicorn
import openai
import os
import uuid
from dotenv import load_dotenv
import torch
import whisper
from pyannote.audio import Audio
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
import datetime
import logging


# Initialize the FastAPI application
app = FastAPI()
load_dotenv()


# Load the API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")


# Enable CORS
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # Allow all origins
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


# Define Pydantic models to represent the request body
class ScriptRequest(BaseModel):
   script: str
   instructions: str


class DiarizeRequest(BaseModel):
   transcription: str
   num_speakers: int


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize the embedding model
embedding_model = PretrainedSpeakerEmbedding(
   "speechbrain/spkrec-ecapa-voxceleb",
   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def convert_to_mono(input_path, output_path='mono.wav'):
   """Convert the input audio file to mono using ffmpeg."""
   cmd = f'ffmpeg -i {input_path} -y -ac 1 {output_path}'
   try:
       subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
       logging.info(f'Converted {input_path} to mono.')
   except subprocess.CalledProcessError as e:
       logging.error(f"Error during ffmpeg command: {e.output.decode()}")
       raise


def extract_speakers(model, path, num_speakers=2):
   """Perform diarization with speaker names."""
   mono = 'mono.wav'
   convert_to_mono(path, mono)
  
   result = model.transcribe(mono)
   segments = result["segments"]


   with contextlib.closing(wave.open(mono, 'r')) as f:
       frames = f.getnframes()
       rate = f.getframerate()
       duration = frames / float(rate)


   audio = Audio()


   def segment_embedding(segment):
       start = segment["start"]
       end = min(duration, segment["end"])
       clip = Segment(start, end)
       waveform, sample_rate = audio.crop(mono, clip)
       return embedding_model(waveform[None])


   embeddings = np.zeros((len(segments), 192))
   for i, segment in enumerate(segments):
       embeddings[i] = segment_embedding(segment)
   embeddings = np.nan_to_num(embeddings)


   clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
   labels = clustering.labels_
   for i in range(len(segments)):
       segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
   return segments


def write_segments(segments, outfile):
   """Write out segments to a file."""
   with open(outfile, "w") as f:
       for segment in segments:
           speaker = segment["speaker"]
           text = segment["text"]
           f.write(f"{speaker}: {text}\n")


def read_segments(outfile):
   """Read segments from a file and return formatted text."""
   with open(outfile, "r") as f:
       content = f.read()
   return content


def summarize_text(transcript, instructions):
   client = MistralClient(api_key=mistral_api_key)
   prompt = f""" Please follow these following instructions religiously. they are extremely important and they override everything else: ({instructions}). if the instructions say 'use bullet points', you are obligated to do so. if they specify a number of words or sentences, you are obligated to follow that. using those instructions, which it is very important that you follow religiously, summarize this text: \n\n{transcript}"""
   messages = [
       ChatMessage(role="user", content= prompt )
   ]
   chat_response = client.chat(
       model="mistral-large-latest",
       messages=messages
   )
   return chat_response.choices[0].message.content


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def get_form():
   html_content = """
   <!DOCTYPE html>
   <html>
   <head>
       <title>Scriptifier</title>
       <style>
           body {
               font-family: Arial, sans-serif;
               background-color: #f0f0f0;
               margin: 0;
               padding: 20px;
           }
           h1, h2 {
               color: #333;
           }
           form {
               background-color: #fff;
               padding: 20px;
               border-radius: 8px;
               box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
               margin-bottom: 20px;
           }
           input[type="file"] {
               padding: 10px;
               border: 1px solid #ccc;
               border-radius: 4px;
               margin-bottom: 10px;
           }
           input[type="button"], button {
               background-color: #007bff;
               color: white;
               padding: 10px 20px;
               border: none;
               border-radius: 4px;
               cursor: pointer;
           }
           input[type="button"]:hover, button:hover {
               background-color: #0056b3;
           }
           pre {
               background-color: #fff;
               padding: 10px;
               border-radius: 8px;
               box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
               white-space: pre-wrap;
               word-wrap: break-word;
           }
           textarea {
               width: 100%;
               padding: 10px;
               border: 1px solid #ccc;
               border-radius: 4px;
               margin-top: 10px;
               margin-bottom: 10px;
           }
           #speakers-container button {
               margin: 5px;
           }
           #output {
               background-color: #fff;
               padding: 20px;
               border-radius: 8px;
               box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
           }
       </style>
   </head>
   <body>
       <h1>Upload Audio File for Transcription and Manual Diarization</h1>
       <form id="uploadForm" enctype="multipart/form-data">
           <input type="file" name="file" id="file" accept=".mp3,.m4a,.wav" required><br><br>
           <input type="number" id="speakers" placeholder="Enter the number of speakers" required><br><br>
           <input type="button" id="transcribeButton" value="Transcribe and Diarize" onclick="transcribeAndDiarize()">
       </form>
       <h2>Transcription and Automatic Diarization</h2>
       <pre id="transcription"></pre>
       <h2>Manual Diarization</h2>
       <div id="speakers-container"></div>
       <textarea id="script" placeholder="Script will appear here..." style="display: none;"></textarea>
       <button id="generateButton" style="display: none;" onclick="generateScript()">Generate Final Script</button>
       <button id="undoButton" style="display: none;" onclick="undoLastClassification()">Undo Last Classification</button>
       <div id="output"></div>
       <h2>Summary</h2>
       <textarea id="instructions" placeholder="Enter additional instructions..."></textarea>
       <button onclick="summarize()">Summarize</button>
       <div id="summary"></div>
       <script>
           let speakers = [];
           let currentSpeaker = 0;
           let lastHighlight = null;


           async function transcribeAndDiarize() {
               const fileInput = document.getElementById('file');
               const numSpeakers = document.getElementById('speakers').value;
               const formData = new FormData();
               formData.append('file', fileInput.files[0]);
               formData.append('num_speakers', numSpeakers);


               const response = await fetch('/transcribe_and_diarize', {
                   method: 'POST',
                   body: formData
               });
               const data = await response.json();
               document.getElementById('transcription').textContent = data.transcription;
               document.getElementById('script').value = data.diarized_transcription;
               document.getElementById('script').style.display = 'block';


               loadSpeakerButtons(numSpeakers);
               document.getElementById('generateButton').style.display = 'block';
           }


           function loadSpeakerButtons(numSpeakers) {
               const speakersContainer = document.getElementById("speakers-container");
               speakersContainer.innerHTML = '';
               speakers = [];
               for (let i = 0; i < numSpeakers; i++) {
                   const speakerName = prompt(`Enter name for Speaker ${i + 1}`, `Speaker ${i + 1}`);
                   speakers.push({ id: i, name: speakerName, snippets: [], color: getRandomColor() });
                   const speakerButton = document.createElement("button");
                   speakerButton.innerText = speakerName;
                   speakerButton.style.backgroundColor = speakers[i].color;
                   speakerButton.onclick = () => currentSpeaker = i;
                   speakersContainer.appendChild(speakerButton);
               }
           }


           function getRandomColor() {
               return '#' + Math.floor(Math.random() * 16777215).toString(16);
           }


           document.getElementById("script").addEventListener("mouseup", function () {
               const selection = window.getSelection();
               if (selection.toString()) {
                   highlightSelection(selection, currentSpeaker);
               }
           });


           function highlightSelection(selection, speakerId) {
               const scriptTextarea = document.getElementById("script");
               const scriptText = scriptTextarea.value;
               const start = scriptTextarea.selectionStart;
               const end = scriptTextarea.selectionEnd;


               if (start !== -1 && end !== -1 && start !== end) {
                   const selectedText = scriptText.substring(start, end);
                   // Remove any existing span tags
                   const cleanText = selectedText.replace(/<[^>]*>/g, '');
                   const highlightedText = `[${speakers[speakerId].name} - ${cleanText}]`;
                   speakers[speakerId].snippets.push({ text: cleanText, start: start, end: end, color: speakers[speakerId].color });
                   scriptTextarea.value = scriptText.slice(0, start) + highlightedText + scriptTextarea.value.slice(end);
                   lastHighlight = { speakerId, start, end };
                   document.getElementById('undoButton').style.display = 'inline-block';
               }
           }


           function undoLastClassification() {
               if (lastHighlight) {
                   const { speakerId, start, end } = lastHighlight;
                   const scriptTextarea = document.getElementById("script");
                   const scriptText = scriptTextarea.value;
                   const cleanText = scriptText.substring(start, end).replace(`[${speakers[speakerId].name} - `, '').replace(']', '');
                   scriptTextarea.value = scriptText.slice(0, start) + cleanText + scriptTextarea.value.slice(end);
                   speakers[speakerId].snippets = speakers[speakerId].snippets.filter(snippet => snippet.start !== start && snippet.end !== end);
                   lastHighlight = null;
                   document.getElementById('undoButton').style.display = 'none';
               }
           }


           function generateScript() {
               const scriptTextarea = document.getElementById("script");
               const scriptText = scriptTextarea.value;
               let snippets = [];
               speakers.forEach(speaker => {
                   speaker.snippets.forEach(snippet => {
                       snippets.push({ speakerName: speaker.name, text: snippet.text, start: snippet.start, end: snippet.end, color: snippet.color });
                   });
               });


               // Add any unhighlighted text as "unknown"
               const regex = /\[.*?\]/g;
               let lastIndex = 0;
               let match;
               while ((match = regex.exec(scriptText)) !== null) {
                   if (match.index > lastIndex) {
                       const unknownText = scriptText.substring(lastIndex, match.index).trim();
                       if (unknownText) {
                           snippets.push({ speakerName: "Unknown", text: unknownText, start: lastIndex, end: match.index, color: "#cccccc" });
                       }
                   }
                   lastIndex = match.index + match[0].length;
               }
               if (lastIndex < scriptText.length) {
                   const unknownText = scriptText.substring(lastIndex).trim();
                   if (unknownText) {
                       snippets.push({ speakerName: "Unknown", text: unknownText, start: lastIndex, end: scriptText.length, color: "#cccccc" });
                   }
               }


               snippets.sort((a, b) => a.start - b.start);
               const script = snippets.map(snippet => {
                   return `<span style="background-color: ${snippet.color};"><b>${snippet.speakerName}:</b> ${snippet.text}</span>`;
               }).join("<br>");


               document.getElementById("output").innerHTML = script;
           }


           async function summarize() {
               const scriptTextarea = document.getElementById("script").value;
               const instructions = document.getElementById("instructions").value;
               const response = await fetch('/summarize', {
                   method: 'POST',
                   headers: { 'Content-Type': 'application/json' },
                   body: JSON.stringify({ script: scriptTextarea, instructions: instructions })
               });
               const data = await response.json();
               document.getElementById("summary").textContent = data.summary;
           }
       </script>
   </body>
   </html>
   """
   return HTMLResponse(content=html_content)


@app.post("/transcribe_and_diarize")
async def transcribe_and_diarize(file: UploadFile = File(...), num_speakers: int = 2):
   file_path = Path(f"/tmp/{uuid.uuid4()}{Path(file.filename).suffix}")
   output_file = Path(f"/tmp/{uuid.uuid4()}.txt")
   try:
       with open(file_path, "wb") as buffer:
           buffer.write(await file.read())


       transcription = await transcribe_audio_file(file_path)
       segments = extract_speakers(whisper.load_model("base"), file_path, num_speakers)
       write_segments(segments, output_file)
       diarized_transcription = read_segments(output_file)
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))
   finally:
       if file_path.exists():
           os.remove(file_path)  # Clean up the uploaded file
       if output_file.exists():
           os.remove(output_file)  # Clean up the output file


   return JSONResponse(content={"transcription": transcription, "diarized_transcription": diarized_transcription})


async def transcribe_audio_file(file_path: Path) -> str:
   try:
       wav_file_path = convert_to_wav_if_necessary(file_path)
       transcription_text = await transcribe_chunks(wav_file_path)
   finally:
       if wav_file_path.exists():
           os.remove(wav_file_path)  # Clean up the WAV file
   return transcription_text


def convert_to_wav_if_necessary(input_file: Path) -> Path:
   if (input_file.suffix.lower() in ['.mp3', '.m4a']):
       output_file = input_file.with_suffix('.wav')
       command = ['ffmpeg', '-i', str(input_file), str(output_file)]
       subprocess.run(command, check=True)
       return output_file
   return input_file


async def transcribe_chunks(file_path: Path, chunk_length_ms=60000) -> str:
   audio = AudioSegment.from_file(file_path)
   chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
   all_results = []
   for chunk in chunks:
       chunk_path = Path(f"/tmp/{uuid.uuid4()}.wav")
       chunk.export(chunk_path, format="wav")
       try:
           with open(chunk_path, "rb") as audio_file:
               response = openai.audio.transcriptions.create(
                   model="whisper-1",
                   file=audio_file,
               )
           all_results.append(response.text)
       finally:
           if (chunk_path.exists()):
               os.remove(chunk_path)  # Clean up the audio chunk
   return ' '.join(all_results)


@app.post("/generate")
async def generate_script(request: ScriptRequest):
   script = request.script
   # Process script to remove speaker tags and timestamps
   import re
   processed_script = re.sub(r'\[.*? - ', '', script).replace(']', '')
   # Apply bold formatting to speaker names and replace newline characters with <br> for HTML rendering
   processed_script = processed_script.replace('\\n', '<br>')
   return {"generated_script": processed_script}


@app.post("/summarize")
async def summarize_script(request: ScriptRequest):
   summary = summarize_text(request.script, request.instructions)
   return {"summary": summary}


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)



