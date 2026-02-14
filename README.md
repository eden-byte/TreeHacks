# Vision Assistant

Accessibility tool for visually impaired users. Uses voice commands to trigger webcam analysis via OpenAI GPT-4o, then reads the result aloud.

## Setup

```bash
pip install -r requirements.txt
```

Put your OpenAI API key in `.env`:
```
OPENAI_API_KEY=sk-...
```

## Run

```bash
python main.py
```

## Usage

Say **"Hey"** followed by a command:

| Voice Command | What It Does |
|---|---|
| "Hey, read this document" | OCR — reads text from a document |
| "Hey, what currency am I holding?" | Identifies bills/coins |
| "Hey, what is the price?" | Reads price tags |
| "Hey, describe what you see" | General scene description |

Press `Ctrl+C` to quit.
