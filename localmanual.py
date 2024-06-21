from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import uuid
from typing import Generator
from pathlib import Path
import subprocess
import numpy as np
import torchaudio
import uvicorn  # Importing uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    result = model.transcribe(str(chunk_path))
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
    input_text = f"shorten the following text: {text}. Make sure to use the following instructions: {instructions}"
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
            input[type="text"], input[type="file"], input[type="submit"], input[type="button"], input[type="checkbox"], input[type="number"], textarea, button {
                display: block;
                margin-bottom: 10px;
                padding: 10px;
                width: 100%;
                box-sizing: border-box;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            input[type="submit"], input[type="button"], button {
                background-color: #ADD8E6;
                color: white;
                border: none;
                cursor: pointer;
            }
            input[type="submit"]:hover, input[type="button"]:hover, button:hover {
                background-color: #87CEEB;
            }
            #transcription, #summary, #script, #output, #speakers-container {
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
        <h2>Diarization</h2>
        <div>
            <label for="speakers">Number of Speakers:</label>
            <input type="number" id="speakers" min="1">
            <button onclick="handleSpeakers()">Load Speakers</button>
        </div>
        <div id="speakers-container"></div>
        <textarea id="script" placeholder="Script will appear here..." style="display: none;"></textarea>
        <button id="generate" style="display: none;" onclick="generateScript()">Generate</button>
        <div id="output"></div>
        <h2>Summary</h2>
        <div id="summary"></div>
        <script>
            let speakers = [];
            let currentSpeaker = 0;

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
                    document.getElementById('script').style.display = 'block';
                    document.getElementById('script').value = transcription.trim();
                    document.getElementById('generate').style.display = 'block';
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
                    speakerButton.onclick = () => {
                        currentSpeaker = i;
                        highlightButton(speakerButton);
                    };
                    speakersContainer.appendChild(speakerButton);
                }
                document.getElementById('speakers-container').style.display = 'block';
            }

            function highlightButton(button) {
                const buttons = document.querySelectorAll("#speakers-container button");
                buttons.forEach(btn => btn.style.outline = "none");
                button.style.outline = "2px solid black";
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
                document.getElementById("output").style.display = 'block';  // Display the diarized script
            }
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

# Define a Pydantic model to represent the request body for script generation
class ScriptRequest(BaseModel):
    script: str
    instructions: str

@app.post("/generate")
async def generate_script(request: ScriptRequest):
    script = request.script
    # Process script to remove brackets and replace hyphen with colon
    processed_script = script.replace('[', '').replace(']', '').replace(' - ', ': ')
    # Apply bold formatting to speaker names and replace newline characters with <br> for HTML rendering
    processed_script = processed_script.replace('\\n', '<br>')
    print(processed_script)
    return {"generated_script": processed_script}

@app.post("/summarize_script")
async def summarize_script(request: ScriptRequest):
    summary = generate_summary(request.script, request.instructions)
    return {"summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
