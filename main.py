import os
import sys
import base64
import threading
import time
import queue
import subprocess
import json
from datetime import datetime

import cv2
import numpy as np
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import face_recognition

load_dotenv(override=True)

client = OpenAI()
perplexity_client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Text-to-Speech — dedicated thread so it never blocks anything
# ---------------------------------------------------------------------------
speech_queue = queue.Queue()

def tts_worker():
    """Dedicated thread that processes speech requests one by one."""
    while True:
        text = speech_queue.get()
        print(f"[SPEAKING] {text}")
        safe = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-Command",
             f"Add-Type -AssemblyName System.Speech; "
             f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.SelectVoice('Microsoft Zira Desktop'); "
             f"$s.Rate = 1; $s.Speak('{safe}')"],
            creationflags=0x08000000
        )

def speak(text: str):
    """Queue text to be spoken (safe to call from any thread)."""
    speech_queue.put(text)

# ---------------------------------------------------------------------------
# RAG Memory — ChromaDB for semantic search over past interactions
# ---------------------------------------------------------------------------
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
memory_collection = chroma_client.get_or_create_collection(
    name="vera_memory",
    metadata={"hnsw:space": "cosine"},
)

def get_embedding(text: str) -> list[float]:
    """Get OpenAI embedding for a text string."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def save_to_memory(command: str, response: str, image_path: str = ""):
    """Save an interaction to ChromaDB with its embedding."""
    doc_text = f"User asked: {command}\nAssistant replied: {response}"
    doc_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp = datetime.now().isoformat()

    try:
        embedding = get_embedding(doc_text)
        memory_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "command": command,
                "response": response,
                "timestamp": timestamp,
                "image_path": image_path,
            }],
        )
        print(f"[RAG] Saved to memory (ID: {doc_id})")
    except Exception as e:
        print(f"[RAG ERROR] Failed to save: {e}")

def search_memory(query: str, n_results: int = 3) -> str:
    """Search past interactions for relevant context."""
    if memory_collection.count() == 0:
        return ""

    try:
        query_embedding = get_embedding(query)
        results = memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, memory_collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            timestamp = metadata.get("timestamp", "unknown time")
            context_parts.append(f"[{timestamp}] {doc}")

        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[RAG ERROR] Search failed: {e}")
        return ""

def save_snapshot(frame) -> str:
    """Save a webcam frame to disk and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath

# ---------------------------------------------------------------------------
# Face Recognition — local face database using OpenCV
# ---------------------------------------------------------------------------
FACES_DIR = os.path.join(BASE_DIR, "known_faces")
FACES_DB_FILE = os.path.join(BASE_DIR, "faces_db.json")
os.makedirs(FACES_DIR, exist_ok=True)

ENCODINGS_FILE = os.path.join(BASE_DIR, "face_encodings.json")

def load_faces_db() -> dict:
    if os.path.exists(FACES_DB_FILE):
        try:
            with open(FACES_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_faces_db(db: dict):
    with open(FACES_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def load_encodings_db() -> dict:
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_encodings_db(db: dict):
    with open(ENCODINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def detect_faces(frame):
    """Detect face locations using dlib (via face_recognition)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    return locations

def register_face(name: str, frame):
    """Register a face using dlib's 128-dimensional face encoding."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) == 0:
        return False

    # Pick the largest face
    largest = max(locations, key=lambda r: (r[2] - r[0]) * (r[1] - r[3]))
    encodings = face_recognition.face_encodings(rgb, [largest])
    if len(encodings) == 0:
        return False

    encoding = encodings[0]

    # Save face image
    top, right, bottom, left = largest
    pad = int(0.2 * (bottom - top))
    y1 = max(0, top - pad)
    y2 = min(frame.shape[0], bottom + pad)
    x1 = max(0, left - pad)
    x2 = min(frame.shape[1], right + pad)
    face_img = frame[y1:y2, x1:x2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(FACES_DIR, filename)
    cv2.imwrite(filepath, face_img)

    # Save image path
    db = load_faces_db()
    if name not in db:
        db[name] = []
    db[name].append(filepath)
    save_faces_db(db)

    # Save 128-d encoding
    enc_db = load_encodings_db()
    if name not in enc_db:
        enc_db[name] = []
    enc_db[name].append(encoding.tolist())
    save_encodings_db(enc_db)

    # Save face registration to RAG memory
    save_to_memory(f"Register face: {name}", f"Learned to recognize {name}", filepath)

    print(f"[FACE] Registered '{name}' with dlib encoding ({filename})")
    return True

def recognize_faces(frame) -> list[str]:
    """Recognize faces using dlib's 128-d encodings (compares facial geometry)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) == 0:
        return []

    enc_db = load_encodings_db()
    if not enc_db:
        return []

    # Build arrays of known encodings and names
    known_encodings = []
    known_names = []
    for name, encs in enc_db.items():
        for enc in encs:
            known_encodings.append(np.array(enc))
            known_names.append(name)

    if not known_encodings:
        return []

    # Get encodings for all faces in the current frame
    frame_encodings = face_recognition.face_encodings(rgb, locations)

    recognized = []
    for encoding in frame_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = np.argmin(distances)
        if distances[best_idx] < 0.6:  # threshold (lower = stricter)
            recognized.append(known_names[best_idx])

    return recognized

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------
camera = cv2.VideoCapture(1)  # external USB webcam
frame_lock = threading.Lock()
latest_frame = None

def update_frame():
    global latest_frame
    ret, frame = camera.read()
    if ret:
        with frame_lock:
            latest_frame = frame.copy()
    return ret

def get_latest_frame():
    with frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()

def capture_frame_b64() -> str | None:
    with frame_lock:
        if latest_frame is None:
            return None
        _, buffer = cv2.imencode(".jpg", latest_frame)
    return base64.b64encode(buffer).decode("utf-8")

# ---------------------------------------------------------------------------
# Vision analysis via GPT-4o + RAG context
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Vera, an AI assistant for a blind user. You receive an image from "
    "their webcam along with a spoken request. Respond in 1-3 short, clear "
    "sentences that are easy to understand when read aloud. Be specific about "
    "details like numbers, colors, text content, and currency denominations. "
    "If asked to read text, read ALL the text you can see exactly as written. "
    "When told who people are, use their names naturally in your response. "
    "You MUST describe people freely — their clothing, posture, actions, "
    "appearance, expressions, and any visible details. The user is blind and "
    "relies on you to describe everything including people. Never refuse to "
    "describe a person. If a person's name is provided in the context, use it. "
    "You have access to past conversation memory — use it to give contextual answers."
)

def analyze_image(command: str) -> str:
    image_b64 = capture_frame_b64()
    if image_b64 is None:
        return "Sorry, I could not access the camera."

    # Try to recognize faces in the current frame
    frame = get_latest_frame()
    recognized = recognize_faces(frame) if frame is not None else []

    prompt = command
    if recognized:
        names = ", ".join(recognized)
        prompt += f"\n\n[The following people have been identified in the image: {names}]"

    # RAG: search past memory for relevant context
    rag_context = search_memory(command)
    system = SYSTEM_PROMPT
    if rag_context:
        system += "\n\nRelevant past interactions:\n" + rag_context

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=500,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    result = response.choices[0].message.content

    # Save snapshot + interaction to RAG memory
    image_path = ""
    if frame is not None:
        image_path = save_snapshot(frame)
    save_to_memory(command, result, image_path)

    return result

# ---------------------------------------------------------------------------
# Voice listener (background thread)
# ---------------------------------------------------------------------------
WAKE_PHRASE = "hey vera"
is_active = False
stay_active = False

recognizer = sr.Recognizer()
recognizer.energy_threshold = 1000
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.8

def handle_command(command: str):
    """Process a voice command — face registration, people questions, or image analysis."""
    lower = command.lower()

    # Face registration: "this is John" or "that is Sarah" or "remember John"
    # Search anywhere in the sentence, not just at the start
    register_phrases = ["this is ", "that is ", "remember ", "i am ", "i'm ", "my name is ", "call me "]
    for phrase in register_phrases:
        idx = lower.find(phrase)
        if idx != -1:
            name = command[idx + len(phrase):].strip().title()
            # Remove trailing filler words
            for filler in [" right", " okay", " please", " now"]:
                if name.lower().endswith(filler):
                    name = name[:len(name) - len(filler)].strip()
            if not name:
                speak("I didn't catch the name. Please try again.")
                return

            frame = get_latest_frame()
            if frame is None:
                speak("I can't see anything right now.")
                return

            if register_face(name, frame):
                speak(f"Got it! I'll remember {name}.")
            else:
                speak("I can't see a face clearly. Please face the camera and try again.")
            return

    # People-related questions — handle locally with face recognition
    # "who is this" / "who am i" = face recognition
    # "who won the super bowl" = general question → Perplexity
    face_phrases = ["who is this", "who is that", "who am i", "who are they", "who are you looking at",
                    "who do you see", "who's this", "who's that", "whose face"]
    people_keywords = ["person", "people", "someone", "somebody", "met", "faces"]
    recall_keywords = ["met", "seen", "talked", "earlier", "before", "today", "yesterday", "remember", "recall", "past", "how many"]
    live_keywords = ["this", "that", "here", "front", "looking at", "see right now"]

    is_face_question = any(phrase in lower for phrase in face_phrases)
    is_people_question = is_face_question or any(kw in lower for kw in people_keywords)
    is_recall_question = any(kw in lower for kw in recall_keywords)
    is_live_question = any(kw in lower for kw in live_keywords)

    # "have you met this person" = live check, "how many people have you met" = recall
    if is_people_question and is_recall_question and not is_live_question:
        # General recall — answer from the faces database, filter by date
        from datetime import timedelta
        db = load_faces_db()

        # Determine which date to filter by
        filter_date = None
        time_label = ""
        if "today" in lower:
            filter_date = datetime.now().strftime("%Y%m%d")
            time_label = " today"
        elif "yesterday" in lower:
            filter_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            time_label = " yesterday"

        names = []
        for name, paths in db.items():
            if filter_date:
                if any(filter_date in os.path.basename(p) for p in paths):
                    names.append(name)
            else:
                names.append(name)

        if names:
            count = len(names)
            names_str = ", ".join(names[:-1]) + " and " + names[-1] if count > 1 else names[0]
            speak(f"I've met {count} {'person' if count == 1 else 'people'}{time_label}: {names_str}.")
        else:
            speak(f"I haven't met anyone{time_label}.")
        return

    if is_people_question:
        # Live question — check camera for faces
        frame = get_latest_frame()
        if frame is not None:
            recognized = recognize_faces(frame)
            if recognized:
                names = " and ".join(recognized)
                if is_recall_question:
                    speak(f"Yes, I know {'them' if len(recognized) > 1 else 'this person'}. That's {names}.")
                elif len(recognized) == 1:
                    speak(f"That's {names}.")
                else:
                    speak(f"I can see {names}.")
                save_to_memory(command, f"Identified: {names}")
            else:
                faces = detect_faces(frame)
                if len(faces) > 0:
                    if is_recall_question:
                        speak(f"I can see someone, but I don't recognize them. Say 'this is' followed by their name to teach me.")
                    else:
                        speak(f"I can see {len(faces)} person{'s' if len(faces) > 1 else ''}, but I don't recognize them. Say 'this is' followed by their name to teach me.")
                else:
                    speak("I don't see anyone right now.")
        else:
            speak("I can't see anything right now.")
        return

    # Decide if this is a vision command (needs camera) or a general question
    vision_keywords = ["see", "look", "read", "show", "describe", "what's in front",
                       "holding", "wearing", "color", "text", "document", "price",
                       "money", "currency", "bill", "label", "sign", "screen",
                       "bottle", "food", "object", "item", "brand", "product"]
    needs_vision = any(kw in lower for kw in vision_keywords)

    if needs_vision:
        speak("Let me take a look.")
        result = analyze_image(command)
    else:
        # General question — use Perplexity for web-powered answers
        speak("Let me look that up.")
        try:
            rag_context = search_memory(command)
            system = (
                "You are Vera, an AI assistant for a blind user. "
                "Answer in 1-3 short, clear sentences that are easy to understand when read aloud. "
                "Be concise and conversational, like a smart assistant."
            )
            if rag_context:
                system += "\n\nRelevant past interactions:\n" + rag_context

            response = perplexity_client.chat.completions.create(
                model="sonar",
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": command},
                ],
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"[PERPLEXITY ERROR] {e}")
            # Fallback to GPT-4o if Perplexity fails
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=300,
                messages=[
                    {"role": "system", "content": "You are Vera, an AI assistant for a blind user. Answer in 1-3 short sentences."},
                    {"role": "user", "content": command},
                ],
            )
            result = response.choices[0].message.content
        save_to_memory(command, result)

    speak(result)

def listen_thread():
    while True:
        mic = None
        try:
            mic = sr.Microphone()
            with mic as source:
                print("[CALIBRATING] Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=2)
            print('[READY] Say "Hey Vera" to activate, "Bye" to sleep.')
            print()
            print("  Voice commands (after activation):")
            print('    "Read this document"')
            print('    "What do you see?"')
            print('    "This is John"          — register a face')
            print('    "Who is this?"           — identify a person')
            print('    "What did I look at earlier?" — RAG memory recall')
            print('    "Bye"                    — go to sleep')
            print()

            while True:
                try:
                    global is_active, stay_active
                    with mic as source:
                        audio = recognizer.listen(source, phrase_time_limit=6)

                    text = recognizer.recognize_google(audio).lower().strip()
                    status = "ACTIVE" if is_active else "SLEEPING"
                    print(f"[HEARD ({status})] {text}")

                    # Ignore noise
                    words = text.split()
                    alpha_words = [w for w in words if w.isalpha()]
                    # Skip if no real words at all
                    if len(alpha_words) == 0 and "hey" not in text and "bye" not in text:
                        print("[IGNORED] No real words, likely noise")
                        continue
                    # When active, require at least 2 words to avoid processing background chatter
                    if is_active and len(words) < 2 and text not in ["bye", "bye vera", "goodbye"]:
                        print("[IGNORED] Single word, likely background noise")
                        continue

                    # Check for wake phrase "hey vera" (with common misrecognitions)
                    wake_variants = ["hey vera", "hey vira", "hey veera", "a vera", "hey vero", "heyfera", "avera", "hey vert", "hey verra", "hey bear", "hey vara", "hey fara", "hey verda", "hey bird", "hey bureau", "ever a", "hey there"]
                    wake_found = None
                    for variant in wake_variants:
                        if variant in text:
                            wake_found = variant
                            break

                    if wake_found:
                        command = text.split(wake_found, 1)[1].strip().lstrip(",").strip()

                        if command:
                            print("[ONE-SHOT] Processing single command...")
                            is_active = True
                            stay_active = False
                            handle_command(command)
                            is_active = False
                            print("[SLEEPING] Done. Waiting for Hey Vera...")
                        else:
                            is_active = True
                            stay_active = True
                            speak("Hi! I'm Vera. How can I help?")
                            print("[ACTIVE] Vera is in ongoing session. Say 'bye' to stop.")
                        continue

                    if not is_active:
                        continue

                    if text.strip() in ["bye", "bye vera", "goodbye", "goodbye vera", "bye bye"]:
                        is_active = False
                        stay_active = False
                        speak("Goodbye! Say Hey Vera when you need me.")
                        print("[SLEEPING] Vera is now asleep.")
                        continue

                    handle_command(text)

                except sr.UnknownValueError:
                    pass
                except sr.WaitTimeoutError:
                    pass
                except sr.RequestError as e:
                    print(f"[ERROR] Speech recognition service error: {e}")
                    time.sleep(2)

        except Exception as e:
            print(f"[ERROR] Listener crashed: {e}")
            print("[RESTARTING] Restarting listener in 3 seconds...")
            time.sleep(3)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  Vera — Vision Assistant (RAG-powered)")
    print("=" * 50)
    print()

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-api-key-here":
        print("[ERROR] Set your OPENAI_API_KEY in the .env file first!")
        sys.exit(1)

    if not camera.isOpened():
        print("[ERROR] Cannot open webcam. Check that it is connected.")
        sys.exit(1)

    print(f"[RAG] Memory has {memory_collection.count()} stored interactions.")

    # Start TTS worker thread
    tts = threading.Thread(target=tts_worker, daemon=True)
    tts.start()

    speak("Vera is ready. Say Hey Vera to start.")

    # Start voice listener thread
    voice = threading.Thread(target=listen_thread, daemon=True)
    voice.start()

    # Main thread: webcam preview
    print("[PREVIEW] Webcam window open. Press 'q' in the window to quit.")
    try:
        while True:
            update_frame()
            with frame_lock:
                if latest_frame is not None:
                    cv2.imshow("Vision Assistant - Webcam", latest_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[EXIT] Goodbye!")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[CLEANUP] Camera released.")

if __name__ == "__main__":
    main()
