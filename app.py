from flask import Flask, request, jsonify, render_template_string
from azure.cognitiveservices.speech import AudioConfig, SpeechConfig, SpeechRecognizer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
import threading
import json

with open('keys.json') as f:
    keys = json.load(f)

# Set your Azure and Mistral API keys and region
azure_speech_key = keys['azure_speech_key']
azure_service_region = keys['azure_service_region']
mistral_api_key = keys['mistral_api_key']
# Initialize Flask app
app = Flask(__name__)

def transcribe_audio(file_path):
    import azure.cognitiveservices.speech as speechsdk
    import threading

    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_service_region)
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.AudioConfig(filename=file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    done = threading.Event()
    
    def stop_cb(evt):
        """Callback to stop continuous recognition upon receiving a completed event"""
        done.set()

    all_results = []

    def handle_final_result(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            all_results.append(evt.result.text)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(evt.result.no_match_details))

    speech_recognizer.recognized.connect(handle_final_result)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition()
    done.wait()  # Waits until continuous recognition is stopped

    speech_recognizer.stop_continuous_recognition()

    result = ' '.join(all_results)
    print(result)
    return result

def summarize_text(text):
    client = MistralClient(api_key=mistral_api_key)
    messages = [ChatMessage(role="user", content=text)]
    chat_response = client.chat(
        model="mistral-medium",
        messages=messages
    )
    return chat_response.choices[0].message.content

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload WAV File</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <h1>Upload WAV File for Transcription and Summary</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file" accept=".wav" required>
            <input type="submit" value="Upload">
        </form>
        <h2>Transcription</h2>
        <p id="transcription"></p>
        <h2>Summary</h2>
        <p id="summary"></p>

        <script>
            $(document).ready(function (e) {
                $("#uploadForm").on('submit', function (e) {
                    e.preventDefault();
                    var formData = new FormData(this);
                    $.ajax({
                        type: 'POST',
                        url: '/upload',
                        data: formData,
                        contentType: false,
                        processData: false,
                        success: function (response) {
                            $('#transcription').text(response.transcription);
                            $('#summary').html(response.summary);
                        },
                        error: function (response) {
                            alert('Error: ' + response.responseJSON.error);
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.wav'):
        file_path = os.path.join("/tmp", file.filename)
        file.save(file_path)
        
        # Transcribe the audio file
        transcription = transcribe_audio(file_path)
        
        # Summarize the transcription
        message = f"Please summarize the following in two sentences in markdown format:\n{transcription}"
        summary = summarize_text(message)
        
        return jsonify({"transcription": transcription, "summary": summary})
    else:
        return jsonify({"error": "Invalid file type, only .wav files are accepted"}), 400

if __name__ == '__main__':
    app.run(debug=True)