from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import openai
from pydub import AudioSegment
from pathlib import Path
import subprocess
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import os
import uuid
from typing import Generator

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
# i am getting errors regarding the path, as in windows i am getting paths with backslashes but in mac-os i am getting paths with /. how do I fix this and fix this error? 
# Function to convert audio file to WAV if necessary
def convert_to_wav_if_necessary(input_file):
    if (input_file.suffix in ['.mp3', '.m4a']):
        output_file = input_file.with_suffix('.wav')
        command = f'ffmpeg -i "{input_file}" "{output_file}"'
        subprocess.run(command, shell=True)
        print(f"""temp file path: {output_file}""")
        return output_file
    print(f"""temp file path not used, perm is: {input_file}""")
    return input_file

# Function to split audio file into chunks
def split_audio(file_path, chunk_length_ms=30000):  # default chunk length is 60 seconds
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

# Function to transcribe audio chunk
def transcribe_chunk(chunk_path):
    with open(chunk_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    print(transcription.text)
    return transcription.text

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

# Function to summarize text
def summarize_text(transcript, instructions):
    client = MistralClient(api_key=mistral_api_key)
    prompt = f"""Do not output a word count. Only use information from your summary, avoiding consultation of outside sources. If the user instructions ask for a piece of information not contained in the transcription, do not browse for that piece of information; do not answer the question in your answer at all, not even as a note at the end - rather, please say something along the lines of "this piece of information is not contained in the audio without outputting the actual answer". For example - if I ask “give me a three bullet sentence summary” of an audio that does not mention to current capital of mongolia, you should output: “Genghis Khan, born Temüjin around 1162, was the founder and first Khan of the Mongol Empire, ruling from 1206 until his death in 1227. After uniting the Mongol tribes, he conquered large parts of China and Central Asia. His empire later became the largest contiguous empire in history. Temüjin's early life was marked by hardship and poverty after his father's death and his tribe's abandonment. As he grew older, he gained followers, made alliances, and overcame numerous battles to become the sole ruler in the Mongolian steppe.”, rather than “Genghis Khan, born Temüjin around 1162, was the founder and first Khan of the Mongol Empire, ruling from 1206 until his death in 1227. After uniting the Mongol tribes, he conquered large parts of China and Central Asia. His empire later became the largest contiguous empire in history. Temüjin's early life was marked by hardship and poverty after his father's death and his tribe's abandonment. As he grew older, he gained followers, made alliances, and overcame numerous battles to become the sole ruler in the Mongolian steppe. Note: The current capital of Mongolia is Ulaanbaatar, but this information is not contained in the audio.” Please follow these following instructions religiously. they are extremely important and the override everything else: ({instructions}). if the instructions say 'use bullet points', you are obligated to do so. if they specify a number of words or sentences, you are obligated to follow that. using those isntructions, which it is very important that you follow religiously, summarize this text: \n\n{transcript}"""
    messages = [
        ChatMessage(role="user", content= prompt )
    ]
    print(prompt)
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    return chat_response.choices[0].message.content





def separate(transcript):
    client = MistralClient(api_key=mistral_api_key)
    prompt = f""" The speakers in the following transcript involves a professor and several students. As much as possible and based on context, separate it into "professor", "student 1", "student 2", etc. - so that it feels like a srcipt. this is the transcript for you to scrip-ify:  {transcript}"""
    messages = [
        ChatMessage(role="user", content= prompt )
    ]
    print(prompt)
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    script = chat_response.choices[0].message.content
    print(script)
    return script

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
                background-color: #ADD8E6;  # changing color to light blue
                color: white;
                border: none;
                cursor: pointer;
            }
            input[type="submit"]:hover, input[type="button"]:hover {
                background-color: #87CEEB;  # changing color to light blue
            }
            #transcription, #summary {
                white-space: pre-wrap;
                word-wrap: break-word;
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                display: none;  # Initially hide the transcription and summary
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
                background-color: #ADD8E6;  # changing color to light blue
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

                    const progressBar = document.getElementById('summarizeProgress');
                    if (summaryResponse.ok) {
                        const reader = summaryResponse.body.getReader();
                        const decoder = new TextDecoder();
                        let summary = '';
                        let receivedLength = 0;

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            summary += decoder.decode(value, { stream: true });
                            receivedLength += value.length;
                            let progress = Math.round((receivedLength / transcription.length) * 100);
                            updateProgressBar(progressBar, progress);
                        }

                        const result = JSON.parse(summary);
                        document.getElementById('summary').style.display = 'block';
                        document.getElementById('summary').textContent = result.summary;
                        updateProgressBar(progressBar, 100);
                    } else {
                        console.error('Failed to summarize transcription');
                    }
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
    summary = summarize_text(transcript, instructions)
    script = separate(transcript)
    return {"summary": summary}
