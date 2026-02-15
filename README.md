# AI-Native Assistive Vision System

An AI-powered navigation and assistance system for people who are blind or visually impaired.

Built at TreeHacks 2026

## Overview

285 million people worldwide are blind or visually impaired. Current assistive technologies are either expensive ($30,000 for a guide dog) or limited in capability (traditional white canes). We built a comprehensive AI-powered system that combines real-time vision understanding, haptic feedback, and conversational AI to provide independence and safety.

## What We Built

An AI-native assistive vision system that provides:

- **AI Vision Intelligence**: Identifies currency, colors, text, and obstacles in real-time
- **Multi-Zone Haptic Feedback**: 5-direction vibration necklace for spatial awareness
- **Conversational AI Assistant**: Voice interface with scene understanding
- **Medical Integration**: Medication safety checking and vitals monitoring
- **Cloud-Native Scale**: Scalable architecture using Modal sandboxes

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   USER HARDWARE                         │
│  ┌──────────────────┐      ┌────────────────────────┐  │
│  │  Meta Glasses    │      │  Vibration Necklace    │  │
│  │  (Camera Input)  │      │  (5-Zone Haptic Output)│  │
│  └────────┬─────────┘      └──────────┬─────────────┘  │
└───────────┼────────────────────────────┼────────────────┘
            │                            │
            ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│              JETSON ORIN NANO (Edge AI)                 │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Obstacle     │  │ Object       │  │ Haptic       │ │
│  │ Detection    │  │ Recognition  │  │ Controller   │ │
│  │ (Real-time)  │  │ (YOLOv5)     │  │ (GPIO)       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│           MODAL SANDBOXES (Cloud Intelligence)          │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Per-User Isolated Sandboxes (GPU-Accelerated)│    │
│  │                                                │    │
│  │  ┌──────────────┐  ┌────────────────────────┐ │    │
│  │  │ Vision AI    │  │ Conversational Agent   │ │    │
│  │  │ (GPT-4V)     │  │ (ChatGPT + RAG)        │ │    │
│  │  └──────────────┘  └────────────────────────┘ │    │
│  │                                                │    │
│  │  ┌──────────────┐  ┌────────────────────────┐ │    │
│  │  │ Medical AI   │  │ Emergency Monitor      │ │    │
│  │  │ (Drug Check) │  │ (Fall Detection)       │ │    │
│  │  └──────────────┘  └────────────────────────┘ │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  Auto-scales: 1 user → 50,000 users                    │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**Edge Computing:**
- NVIDIA Jetson Orin Nano (Edge AI processor)
- YOLOv5 (Real-time object detection)
- OpenCV (Computer vision pipeline)
- Custom GPIO haptic controller

**AI & Cloud:**
- OpenAI GPT-4 Vision (Scene understanding)
- ChatGPT API (Conversational intelligence)
- Modal (Serverless GPU sandboxes)
- Text-to-Speech (Audio feedback)

**Hardware:**
- Meta Ray-Ban glasses (Camera + audio)
- Custom 5-zone vibration necklace
- Arduino/ESP32 (Haptic motor control)

## Key Capabilities

### AI Vision Intelligence

Real-time object and scene understanding using computer vision and AI.

```python
# User holds up object to glasses camera
→ AI Vision Processing
→ "This is a twenty dollar bill"
→ "Prescription bottle: Ibuprofen 200mg"
→ "Red shirt, medium size"
```

### Multi-Zone Haptic Navigation

5-zone vibration necklace provides directional obstacle awareness.

```
Vibration Zones:
[Left] [Center-Left] [Center] [Center-Right] [Right]

Obstacle at 2 o'clock → Center-Right motor vibrates
Person approaching left → Left motor pulses
Clear path ahead → All motors off
```

### Conversational AI Assistant

Natural voice interaction for questions and guidance.

```
User: "What am I looking at?"
AI: "You're in front of a CVS pharmacy. The entrance is 15 feet ahead."

User: "What color is this?"
AI: "Navy blue"

User: "Can I take this medicine?"
AI: "This is ibuprofen. Checking your medical history... 
     WARNING: You have a documented allergy to NSAIDs."
```

### Medical Integration

Continuous health monitoring and medication safety.

- Medication Safety: OCR prescription labels with drug interaction checking
- Vitals Monitoring: Heart rate and movement patterns from necklace sensors
- Fall Detection: Automatic emergency alerts
- Provider Dashboard: Healthcare professionals can view safety events and location logs

### Scalable Cloud Architecture

Modal sandboxes provide personalized AI for each user.

```python
# Each user gets isolated AI sandbox
@app.function(gpu="T4", secrets=[...])
def user_vision_sandbox(user_id: str, image: bytes):
    """
    Personalized AI processing:
    - Learns user preferences
    - Stores medical context
    - Adapts to environment
    """
    return personalized_results
```

## Impact

This system addresses the needs of 285 million visually impaired people worldwide by providing:

- Independent navigation in unfamiliar environments
- Real-time object and obstacle identification
- Medication safety and health monitoring
- Emergency detection and response
- Affordable, scalable access to AI assistance ($5-10/month vs $30,000 for guide dogs)

## Getting Started

### Prerequisites

Hardware:
- NVIDIA Jetson Orin Nano
- Meta Ray-Ban Smart Glasses (or USB webcam)
- 5x vibration motors + motor driver
- Arduino/ESP32 for haptic control

Software:
- Python 3.8+
- PyTorch
- OpenCV
- Modal account (for cloud deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sunu-ai-native.git
cd sunu-ai-native

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Configure environment
cp .env.example .env
# Add your API keys: OPENAI_KEY, MODAL_TOKEN

# Run on Jetson
python main.py

# Deploy to Modal (optional)
modal deploy cloud_deploy.py
```

### Quick Test

```bash
# Test camera + object detection
python test_vision.py

# Test haptic feedback
python test_haptics.py

# Test full system
python main.py --demo
```

## License

MIT License - see LICENSE file for details
