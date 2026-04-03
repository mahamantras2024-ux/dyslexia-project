"""
Created on Fri Apr 18 21:56:05 2025

@author: Atharva Berde, Rajeev Raghuram
"""

from io import BytesIO
from flask import Flask, render_template
import os
import signal
from flask import Flask, request, jsonify, send_file
import openai
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import requests
from PIL import Image
import base64
import pytesseract
import io
import tensorflow as tf
from keras.models import load_model
import numpy as np
from dotenv import load_dotenv
import openai
from flask_cors import CORS
load_dotenv()
agent_id = os.getenv("")
api_key = os.getenv("")
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key= "AIzaSyD6kMAT2p-u6IkT9aIusodwV9y9F_6jlMk")
VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


prompt_reading = (
    "You are a kind and patient reading tutor helping a student who is learning to read. The student may be young or may struggle with dyslexia. Respond in short, clear sentences using a warm and encouraging tone. Always write as if you're speaking to a student directly through text. Break big ideas into small, simple pieces. If a word is tricky, show how to sound it out and give them a very clear and easy to understand meaning. Avoid long paragraphs—use brief, bite-sized responses that are easy to read. Make reading feel like a fun and safe activity, never stressful or overwhelming."
)

prompt_reading_coach = "You are a warm, friendly, and patient reading tutor guiding a student through a reading comprehension session. There are 10 passages total, starting with simple and short texts, and gradually increasing in difficulty. For each passage, follow this process: 1) Share the short passage clearly. 2) Ask the student to summarize it in their own words. 3) If the student struggles or gets it wrong, break the passage into smaller parts and explain gently. 4) Then ask 1–2 short comprehension questions (like who, what, where, or why). 5) Give positive feedback after each attempt, even if it's not perfect — use phrases like 'Nice try!' or 'You're doing great!' 6) Ask if the student is ready for the next passage. Keep your responses short, friendly, and easy to understand. Break big ideas into small pieces. Make the session fun, gentle, and encouraging."

model = genai.GenerativeModel("gemini-1.5-pro")
chat = model.start_chat(history=[
    {"role": "user", "parts": [prompt_reading]}
])

chat_coach = model.start_chat(history=[
    {"role": "user", "parts": [prompt_reading_coach]}
])

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/help.html")
def help_page():
    return render_template("help.html")

@app.route("/speech.html", methods = ['GET', 'POST'])
def speech_page():
    return render_template("speech.html")

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with your desired voice
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    return send_file(BytesIO(audio), mimetype="audio/mpeg", as_attachment=False)



    return jsonify(transcription)

@app.route("/reading.html", methods=["GET", "POST"])
def reading_page():
    return render_template("reading.html")

@app.route("/readChat", methods = ["POST", "GET"])
def readChat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        response = chat.send_message(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reading_coach.html", methods=["GET", "POST"])
def reading_coach_page():
    return render_template("reading_coach.html")

@app.route("/readCoach", methods = ["POST", "GET"])
def readCoach():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        response = chat_coach.send_message(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def vectara_query(user_query):
    url = "https://api.vectara.io/v2/query"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VECTARA_API_KEY}"
    }

    payload = {
        "query": [
            {
                "query": user_query,
                "start": 0,
                "numResults": 3,
                "corpusKey": [
                    {
                        "customerId": VECTARA_CUSTOMER_ID,
                        "corpusId": VECTARA_CORPUS_ID
                    }
                ]
            }
        ]
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        responses = data["responseSet"][0]["response"]
        return [r["text"] for r in responses]
    else:
        raise Exception(f"Vectara Error: {response.status_code} - {response.text}")
def generate_response_with_gpt(query, vectara_contexts):
    context = "\n\n".join(vectara_contexts)
    prompt = (
        f"Use the following information to answer the parent’s question clearly and helpfully:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful educational assistant for parents of children with dyslexia."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

@app.route("/QandApage.html", methods=["GET", "POST"])
def QandA():
    return render_template("QandApage.html")

@app.route("/parent-help", methods=["POST"])
def parent_help():
    data = request.get_json()
    question = data.get("question", "")

    try:
        context_docs = vectara_query(question)
        #final_answer = generate_response_with_gpt(question, context_docs)
        combined = "\n\n".join(context_docs)
        return jsonify({"answer": combined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500







if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
