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


# Define a Pydantic model to represent the request body
class ScriptRequest(BaseModel):
   script: str
   instructions: str


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
           <input type="button" id="transcribeButton" value="Transcribe" onclick="transcribeAudio()">
       </form>
       <h2>Transcription</h2>
       <pre id="transcription"></pre>
       <h2>Diarization</h2>
       <input type="number" id="speakers" placeholder="Enter the number of speakers">
       <button onclick="handleSpeakers()">Load Speakers</button>
       <div id="speakers-container"></div>
       <textarea id="script" placeholder="Script will appear here..." style="display: none;"></textarea>
       <button onclick="generateScript()">Generate</button>
       <div id="output"></div>
       <h2>Summary</h2>
       <textarea id="instructions" placeholder="Enter additional instructions..."></textarea>
       <button onclick="summarize()">Summarize</button>
       <div id="summary"></div>
       <script>
           let speakers = [];
           let currentSpeaker = 0;


           async function transcribeAudio() {
               const fileInput = document.getElementById('file');
               const formData = new FormData();
               formData.append('file', fileInput.files[0]);


               const response = await fetch('/transcribe', {
                   method: 'POST',
                   body: formData
               });
               const data = await response.json();
               document.getElementById('transcription').textContent = data.transcription;
               document.getElementById('script').value = data.transcription;
               document.getElementById('script').style.display = 'block';
           }


           function handleSpeakers() {
               const numSpeakers = parseInt(document.getElementById("speakers").value);
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
               document.getElementById('generate').style.display = 'block';
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


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
   file_path = Path(f"/tmp/{uuid.uuid4()}{Path(file.filename).suffix}")
   try:
       with open(file_path, "wb") as buffer:
           buffer.write(await file.read())


       transcription = await transcribe_audio_file(file_path)
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))
   finally:
       if file_path.exists():
           os.remove(file_path)  # Clean up the uploaded file


   return JSONResponse(content={"transcription": transcription})


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
   # Process script to remove brackets and replace hyphen with colon
   processed_script = script.replace('[', '').replace(']', '').replace(' - ', ': ')
   # Apply bold formatting to speaker names and replace newline characters with <br> for HTML rendering
   processed_script = processed_script.replace('\\n', '<br>')
   return {"generated_script": processed_script}


@app.post("/summarize")
async def summarize_script(request: ScriptRequest):
   summary = summarize_text(request.script, request.instructions)
   return {"summary": summary}


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)
