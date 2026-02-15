import os
import sys
import platform
import signal
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
from pyzbar.pyzbar import decode as decode_barcodes

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
current_tts_process = None
tts_process_lock = threading.Lock()
TEMP_AUDIO = os.path.join(BASE_DIR, "_tts_temp.mp3")

def tts_worker():
    """Dedicated thread that processes speech requests using OpenAI TTS."""
    global current_tts_process
    current_os = platform.system()

    while True:
        text = speech_queue.get()
        if text is None:
            continue
        print(f"[SPEAKING] {text}")
        try:
            # Generate speech with OpenAI TTS — natural, expressive voice
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=text,
                speed=1.0,
            ) as response:
                response.stream_to_file(TEMP_AUDIO)

            # Play MP3 using OS-appropriate method
            if current_os == "Darwin":  # macOS
                # Use afplay (built-in on macOS)
                proc = subprocess.Popen(["afplay", TEMP_AUDIO])
            elif current_os == "Windows":
                # Use PowerShell MediaPlayer (supports mp3 natively)
                audio_path = TEMP_AUDIO.replace("'", "''")
                proc = subprocess.Popen(
                    ["powershell", "-Command",
                     f"Add-Type -AssemblyName PresentationCore; "
                     f"$p = New-Object System.Windows.Media.MediaPlayer; "
                     f"$p.Open([uri]'{audio_path}'); "
                     f"$p.Play(); "
                     f"Start-Sleep -Milliseconds 500; "
                     f"while($p.Position -lt $p.NaturalDuration.TimeSpan) {{ Start-Sleep -Milliseconds 100 }}; "
                     f"$p.Close()"],
                    creationflags=0x08000000
                )
            else:  # Linux or other
                # Try mpg123 or ffplay
                try:
                    proc = subprocess.Popen(["mpg123", "-q", TEMP_AUDIO])
                except FileNotFoundError:
                    proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", TEMP_AUDIO])

            with tts_process_lock:
                current_tts_process = proc
            proc.wait()
            with tts_process_lock:
                current_tts_process = None
        except Exception as e:
            print(f"[TTS ERROR] {e} — falling back to system TTS")
            # Fallback to system TTS if OpenAI fails
            if current_os == "Darwin":  # macOS
                safe = text.replace('"', '\\"')
                proc = subprocess.Popen(["say", text])
            elif current_os == "Windows":
                safe = text.replace("'", "''")
                proc = subprocess.Popen(
                    ["powershell", "-Command",
                     f"Add-Type -AssemblyName System.Speech; "
                     f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                     f"$s.Rate = 2; $s.Speak('{safe}')"],
                    creationflags=0x08000000
                )
            else:  # Linux
                proc = subprocess.Popen(["espeak", text])

            with tts_process_lock:
                current_tts_process = proc
            proc.wait()
            with tts_process_lock:
                current_tts_process = None

def speak(text: str):
    """Queue text to be spoken (safe to call from any thread)."""
    speech_queue.put(text)

def stop_speaking():
    """Kill current TTS and clear the queue."""
    global current_tts_process
    # Clear all pending speech atomically
    with speech_queue.mutex:
        speech_queue.queue.clear()
    # Kill current TTS process
    with tts_process_lock:
        if current_tts_process and current_tts_process.poll() is None:
            current_tts_process.terminate()
            current_tts_process = None
    print("[STOPPED] Speech interrupted.")

# ---------------------------------------------------------------------------
# Conversation History — enables follow-up questions
# ---------------------------------------------------------------------------
conversation_history = []  # list of {"role": "user"|"assistant", "content": str}
MAX_HISTORY = 10  # keep last 10 exchanges

def add_to_history(role: str, content: str):
    """Add a message to conversation history."""
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY * 2:
        del conversation_history[:2]  # drop oldest pair

def get_history_messages() -> list:
    """Return conversation history formatted for API calls."""
    return list(conversation_history)

# ---------------------------------------------------------------------------
# Deep Research — background Perplexity queries with sonar-deep-research
# ---------------------------------------------------------------------------
research_results = {}  # {question: {"status": "pending"|"done", "answer": str}}
research_lock = threading.Lock()

def deep_research_worker(question: str):
    """Run a deep research query in the background via Perplexity."""
    print(f"[RESEARCH] Starting deep research: {question}")
    try:
        response = perplexity_client.chat.completions.create(
            model="sonar-deep-research",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": (
                    "You are a thorough research assistant. Provide a comprehensive, "
                    "well-structured answer with key facts and details. "
                    "Be detailed but clear enough to be read aloud to a blind user."
                )},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message.content
        print(f"[RESEARCH] Completed: {question[:50]}...")
    except Exception as e:
        print(f"[RESEARCH ERROR] {e}")
        answer = f"Sorry, the research failed: {e}"

    with research_lock:
        research_results[question] = {"status": "done", "answer": answer}
    save_to_memory(f"Deep research: {question}", answer)
    speak(f"Your research on '{question[:50]}' is ready. Ask me for the results when you're ready.")

def start_deep_research(question: str):
    """Kick off a background research thread."""
    with research_lock:
        research_results[question] = {"status": "pending", "answer": ""}
    thread = threading.Thread(target=deep_research_worker, args=(question,), daemon=True)
    thread.start()

def get_research_results() -> str:
    """Get all completed research results."""
    with research_lock:
        done = {q: r for q, r in research_results.items() if r["status"] == "done"}
        pending = {q: r for q, r in research_results.items() if r["status"] == "pending"}

    if not done and not pending:
        return "You haven't asked me to research anything yet."

    parts = []
    if done:
        for question, result in done.items():
            parts.append(f"Research on '{question}': {result['answer']}")
    if pending:
        pending_list = ", ".join(f"'{q[:40]}'" for q in pending)
        parts.append(f"Still researching: {pending_list}")

    return "\n\n".join(parts)

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
# Barcode / QR Code Scanner
# ---------------------------------------------------------------------------
def scan_barcodes(frame) -> list[dict]:
    """Scan frame for barcodes and QR codes, return decoded data."""
    results = []
    try:
        barcodes = decode_barcodes(frame)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            results.append({"data": data, "type": barcode_type})
    except Exception as e:
        print(f"[BARCODE ERROR] {e}")
    return results

def lookup_barcode_product(barcode_data: str) -> str:
    """Look up a barcode using Perplexity to find product info."""
    try:
        response = perplexity_client.chat.completions.create(
            model="sonar",
            max_tokens=300,
            messages=[
                {"role": "system", "content": (
                    "You are helping a blind user identify a product from its barcode. "
                    "Give the product name, brand, key ingredients or allergens, and any "
                    "important details. Be concise — 2-3 sentences max."
                )},
                {"role": "user", "content": f"What product has the barcode/UPC: {barcode_data}"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[BARCODE LOOKUP ERROR] {e}")
        return f"I found a barcode: {barcode_data}, but couldn't look up the product."

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
    "You are Vera, an AI vision assistant built into smart glasses worn by a blind user. "
    "You see exactly what the user sees through their glasses camera. When the user says "
    "'what am I looking at' or 'what do I see', you describe what is in front of THEM. "
    "Always respond from the user's perspective — say 'you're looking at' or 'in front of you' "
    "instead of 'I see in the image'. Respond in 1-3 short, clear sentences that are easy "
    "to understand when spoken aloud. Be specific about details like numbers, colors, text, "
    "and currency. If asked to read text, read ALL visible text exactly as written. "
    "When told who people are, use their names naturally. You MUST describe people freely — "
    "clothing, posture, actions, appearance, expressions. The user is blind and relies on you "
    "to describe everything including people. Never refuse to describe a person. "
    "If a person's name is provided in context, use it. "
    "You have access to past conversation memory — use it for contextual answers."
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
        model="gpt-4o-mini",
        max_tokens=150,
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
                            "detail": "low",
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
recognizer.energy_threshold = 1500
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.8

def handle_command(command: str):
    """Process a voice command — face registration, people questions, or image analysis."""
    lower = command.lower()

    # Deep research: "research X" or "deep dive into X"
    research_triggers = ["research ", "deep dive ", "look into ", "investigate "]
    for trigger in research_triggers:
        if lower.startswith(trigger):
            question = command[len(trigger):].strip()
            if question:
                speak(f"I'll research that in the background. Ask me for results when you're ready.")
                start_deep_research(question)
            else:
                speak("What would you like me to research?")
            return

    # Check for research results: "what did you find", "research results", etc.
    results_keywords = ["what did you find", "research results", "what did you research",
                        "give me the results", "research update", "what did you learn",
                        "did you find", "the results", "find results", "research ready",
                        "what are the results", "get the results"]
    if any(kw in lower for kw in results_keywords):
        results = get_research_results()
        # Summarize if too long for speech
        if len(results) > 500:
            try:
                summary = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": "Summarize this research into 2-4 clear sentences for a blind user listening via text-to-speech. Be specific with key facts."},
                        {"role": "user", "content": results},
                    ],
                )
                results = summary.choices[0].message.content
            except Exception:
                results = results[:500]
        add_to_history("user", command)
        add_to_history("assistant", results)
        speak(results)
        return

    # Barcode/QR scanning: "scan barcode", "scan this", "what product is this"
    barcode_keywords = ["barcode", "qr code", "scan", "upc", "product code"]
    if any(kw in lower for kw in barcode_keywords):
        frame = get_latest_frame()
        if frame is None:
            speak("I can't get a clear view right now.")
            return
        barcodes = scan_barcodes(frame)
        if barcodes:
            for bc in barcodes:
                speak(f"I see a {bc['type']} code. Let me look up the product.")
                product_info = lookup_barcode_product(bc["data"])
                speak(product_info)
                save_to_memory(f"Scanned barcode: {bc['data']}", product_info)
                add_to_history("user", command)
                add_to_history("assistant", product_info)
        else:
            # No barcode detected by pyzbar — try GPT-4o vision as fallback
            speak("I don't see a barcode. Let me try reading what's in front of you.")
            result = analyze_image("Scan and read any barcode, QR code, or product code visible in this image. Tell me the product name, brand, and key details.")
            speak(result)
            add_to_history("user", command)
            add_to_history("assistant", result)
        return

    # Expiration date: "is this expired", "expiration date", "when does this expire"
    expiry_keywords = ["expir", "expire", "best by", "best before", "use by", "sell by", "freshness"]
    if any(kw in lower for kw in expiry_keywords):
        speak("Let me check that for you.")
        result = analyze_image(
            "Look carefully for any expiration date, best-by date, use-by date, or sell-by date on this product. "
            "Read the exact date. Then tell the user if the product is still good or expired based on today's date. "
            f"Today's date is {datetime.now().strftime('%B %d, %Y')}."
        )
        speak(result)
        add_to_history("user", command)
        add_to_history("assistant", result)
        return

    # Medication identification: "what pill is this", "identify this medication"
    # Only trigger camera-based ID when asking about a physical pill (not general health questions)
    medication_keywords = ["pill", "medication", "medicine", "tablet", "capsule", "drug", "prescription", "vitamin"]
    medication_visual_cues = ["this", "that", "identify", "what is", "scan", "check this", "look at"]
    has_medication_word = any(kw in lower for kw in medication_keywords)
    is_physical_med = has_medication_word and any(cue in lower for cue in medication_visual_cues)
    if is_physical_med:
        speak("Let me take a closer look at that.")
        # Step 1: Identify what's in the user's hand via camera
        visual_result = analyze_image(
            "Identify this medication, pill, or medicine package. Read the name, brand, and any visible "
            "dosage information on the label or imprints. Be specific about what you can read."
        )
        speak(visual_result)
        # Step 2: Search for detailed dosage and safety info via Perplexity
        try:
            dosage_response = perplexity_client.chat.completions.create(
                model="sonar",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": (
                        "You are a medication assistant for a blind user. Given a medication identification, "
                        "provide the standard adult dosage, how often to take it, key warnings (like drowsiness "
                        "or interactions), and what it's used for. Keep it to 2-3 sentences, spoken aloud."
                    )},
                    {"role": "user", "content": f"I'm holding this medication: {visual_result}. "
                     f"What's the dosage and important info I should know?"},
                ],
            )
            dosage_info = dosage_response.choices[0].message.content
            speak(dosage_info)
            result = visual_result + " " + dosage_info
        except Exception as e:
            print(f"[PERPLEXITY ERROR] {e}")
            result = visual_result
        add_to_history("user", command)
        add_to_history("assistant", result)
        save_to_memory(command, result)
        return

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
                speak("I can't get a clear view right now.")
                return

            if register_face(name, frame):
                speak(f"Got it! I'll remember {name}.")
            else:
                speak("I can't see a face clearly. Make sure they're in front of you and try again.")
            return

    # People-related questions — handle locally with face recognition
    # "who is this" / "who am i" = face recognition
    # "who won the super bowl" = general question → Perplexity
    face_phrases = [
        # First-person / glasses perspective
        "who am i looking at", "who am i talking to", "who am i with",
        "who is in front of me", "who's in front of me", "who is near me",
        "who is with me", "who's with me", "who is around me",
        "who do i see", "who am i facing", "who is standing",
        "who is sitting", "who is next to me", "who's next to me",
        # Classic triggers
        "who is this", "who is that", "who am i", "who are they",
        "who's this", "who's that", "whose face",
        "do you recognize", "do you know who", "do you know this",
        "who can you see", "who do you see",
    ]
    people_keywords = ["person", "people", "someone", "somebody", "met", "talked", "faces"]
    recall_keywords = [
        "met", "seen", "talked to", "spoken to", "earlier", "before",
        "today", "yesterday", "remember", "recall", "past", "how many",
        "who did i", "who have i", "who've i",
    ]
    live_keywords = ["this", "that", "here", "front of me", "looking at", "see right now",
                     "near me", "with me", "next to me", "facing", "around me"]

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
            speak(f"You've met {count} {'person' if count == 1 else 'people'}{time_label}: {names_str}.")
        else:
            speak(f"You haven't met anyone{time_label}.")
        return

    if is_people_question:
        # Live question — check camera for faces
        frame = get_latest_frame()
        if frame is not None:
            recognized = recognize_faces(frame)
            if recognized:
                names = " and ".join(recognized)
                if is_recall_question:
                    speak(f"Yes, you know {'them' if len(recognized) > 1 else 'this person'}. That's {names}.")
                elif len(recognized) == 1:
                    speak(f"You're looking at {names}.")
                else:
                    speak(f"You're with {names}.")
                save_to_memory(command, f"Identified: {names}")
            else:
                faces = detect_faces(frame)
                if len(faces) > 0:
                    if is_recall_question:
                        speak(f"There's someone in front of you, but I don't recognize them. Say 'this is' followed by their name to introduce me.")
                    else:
                        speak(f"There {'is' if len(faces) == 1 else 'are'} {len(faces)} {'person' if len(faces) == 1 else 'people'} in front of you, but I don't recognize them. Say 'this is' followed by their name to introduce me.")
                else:
                    speak("I don't see anyone in front of you right now.")
        else:
            speak("I can't get a clear view right now.")
        return

    # Decide if this is a vision command (needs camera) or a general question
    vision_keywords = [
        # First-person / glasses perspective
        "what am i looking at", "what do i see", "what's in front of me",
        "what am i holding", "what is in my hand", "what's around me",
        "what am i wearing", "what color is", "what does this say",
        # General vision triggers
        "see", "look", "read", "show", "describe",
        "holding", "wearing", "color", "text", "document", "price",
        "money", "currency", "bill", "label", "sign", "screen",
        "bottle", "food", "object", "item", "brand", "product",
    ]
    needs_vision = any(kw in lower for kw in vision_keywords)

    if needs_vision:
        speak("Let me see.")
        result = analyze_image(command)
    else:
        # General question — use Perplexity for web-powered answers with conversation history
        speak("One sec.")
        try:
            rag_context = search_memory(command)
            system = (
                "You are Vera, an AI assistant built into smart glasses worn by a blind user. "
                "Answer in 1-3 short, clear sentences that are easy to understand when spoken aloud. "
                "Be concise and natural, like a helpful friend. Speak from the user's perspective. "
                "If the user asks a follow-up question, use conversation history for context."
            )
            if rag_context:
                system += "\n\nRelevant past interactions:\n" + rag_context

            # Build messages with conversation history for follow-ups
            messages = [{"role": "system", "content": system}]
            messages.extend(get_history_messages())
            messages.append({"role": "user", "content": command})

            response = perplexity_client.chat.completions.create(
                model="sonar",
                max_tokens=150,
                messages=messages,
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"[PERPLEXITY ERROR] {e}")
            # Fallback to GPT-4o-mini with conversation history
            messages = [
                {"role": "system", "content": "You are Vera, an AI assistant built into smart glasses for a blind user. Answer in 1-3 short sentences. Speak from the user's perspective."},
            ]
            messages.extend(get_history_messages())
            messages.append({"role": "user", "content": command})
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=150,
                messages=messages,
            )
            result = response.choices[0].message.content
        save_to_memory(command, result)

    # Track conversation for follow-ups
    add_to_history("user", command)
    add_to_history("assistant", result)
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
                    # But always allow stop/bye commands
                    stop_words = ["stop", "quiet", "enough", "bye", "goodbye", "buy", "by",
                                  "done", "sleep", "nevermind"]
                    if is_active and len(words) < 2 and text.strip() not in stop_words:
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
                            speak("I'm here. What do you need?")
                            print("[ACTIVE] Vera is in ongoing session. Say 'bye' to stop.")
                        continue

                    # Stop works even when not active
                    if text.strip() in ["stop", "stop talking", "shut up", "quiet", "enough", "stop vera", "stop it"]:
                        stop_speaking()
                        continue

                    if not is_active:
                        continue

                    # Stop command — interrupt speech immediately
                    if text.strip() in ["stop", "stop talking", "shut up", "quiet", "enough", "stop vera", "stop it"]:
                        stop_speaking()
                        speak("Okay.")
                        continue

                    sleep_phrases = [
                        "bye", "bye vera", "goodbye", "goodbye vera", "bye bye",
                        "go to sleep", "go sleep", "sleep", "sleep vera",
                        "i'm done", "im done", "done", "that's all", "thats all",
                        "thanks bye", "thank you bye", "see you", "see ya",
                        "nevermind", "never mind", "good night", "goodnight",
                        "buy", "buy vera", "by", "by vera",  # common misrecognitions
                        "i'm good", "im good", "all good", "that's it", "thats it",
                    ]
                    if text.strip() in sleep_phrases or any(p in text.strip() for p in ["go to sleep", "i'm done", "that's all", "that's it"]):
                        stop_speaking()
                        is_active = False
                        stay_active = False
                        speak("Going to sleep. Say Hey Vera when you need me.")
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

    speak("Vera is ready. Just say Hey Vera whenever you need me.")

    # Start voice listener thread
    voice = threading.Thread(target=listen_thread, daemon=True)
    voice.start()

    # Main thread: webcam preview
    print("[PREVIEW] Webcam window open. Press 'q' in the window to quit.")

    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n[SHUTDOWN] Stopping Vera...")
        camera.release()
        cv2.destroyAllWindows()
        # Delete temp audio file
        if os.path.exists(TEMP_AUDIO):
            try:
                os.remove(TEMP_AUDIO)
            except:
                pass
        print("[CLEANUP] Done. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

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
        # Delete temp audio file
        if os.path.exists(TEMP_AUDIO):
            try:
                os.remove(TEMP_AUDIO)
            except:
                pass
        print("[CLEANUP] Camera released.")

if __name__ == "__main__":
    main()
