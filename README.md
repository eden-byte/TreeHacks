# AI-Native Assistive Vision System

An AI-powered navigation and assistance system for people who are blind or visually impaired.

Built at TreeHacks 2026

## Overview

285 million people worldwide are blind or visually impaired. Current assistive technologies are either expensive ($30,000 for a guide dog) or limited in capability (traditional white canes). We built a comprehensive AI-powered system that combines real-time vision understanding, haptic feedback, and conversational AI to provide independence and safety.

## What We Built

An AI-native assistive vision system for the visually impaired:

- **AI Vision Intelligence**: Identifies currency, colors, text, and obstacles (not just proximity)
- **Multi-Zone Haptic Feedback**: 5-direction vibration necklace for spatial awareness
- **Conversational AI Assistant**: Voice interface with real-time scene understanding
- **Medical Integration**: Vitals monitoring, medication checking, emergency detection
- **Cloud-Native Scale**: Modal sandboxes enable massive concurrent usage

### Sunu (2017) → Our AI-Native Version (2026)

| Feature | Sunu (2017) | Sunu 2.0 (2026) |
|---------|-------------|-----------------|
| **Detection** | Sonar proximity only | AI vision: objects, text, colors, obstacles |
| **Feedback** | Single wrist vibration | 5-zone directional haptic necklace |
| **Interface** | Hardware-only | Voice + haptics + vision |
| **Intelligence** | Distance measurement | Scene understanding + medical context |
| **Architecture** | $299 standalone device | $300 glasses + $5/month cloud AI |
| **Scalability** | One device per user | Modal sandboxes → unlimited users |
| **Personalization** | One-size-fits-all | AI learns each user's needs |

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

## Key Features

### AI Vision Intelligence

Sunu's limitation: Could only detect that something was there.
Our solution: Full scene understanding.

```python
# User holds up object to glasses camera
→ AI Vision Processing
→ "This is a twenty dollar bill"
→ "Prescription bottle: Ibuprofen 200mg - WARNING: You're allergic to NSAIDs"
→ "Red shirt, medium size"
```

### Multi-Zone Haptic Navigation

Sunu's limitation: Single wrist vibration point.
Our solution: 5-zone spatial mapping.

```
Vibration Zones:
[Left] [Center-Left] [Center] [Center-Right] [Right]

Obstacle at 2 o'clock → Center-Right motor vibrates
Person approaching left → Left motor pulses
Clear path ahead → All motors off
```

### Conversational AI Assistant

Sunu's limitation: No ability to answer questions.
Our solution: Natural voice interaction.

```
User: "What am I looking at?"
AI: "You're in front of a CVS pharmacy. The entrance is 15 feet ahead."

User: "What color is this?"
AI: "Navy blue"

User: "Can I take this medicine?"
AI: "This is ibuprofen. Checking your medical history... 
     WARNING: You have a documented allergy to NSAIDs. Do not take this."
```

### Medical Integration

Sunu's limitation: No health monitoring.
Our solution: Continuous health and safety.

- Medication Safety: OCR prescription labels + drug interaction checking
- Vitals Monitoring: Heart rate, movement patterns from necklace sensors
- Fall Detection: Automatic emergency alerts
- Provider Dashboard: Healthcare workers see near-miss events, location logs

### Cloud-Native Scalability

Sunu's limitation: One device per user, hardware scaling.
Our solution: Modal sandboxes for unlimited users.

```python
# Each user gets isolated AI sandbox
@app.function(gpu="T4", secrets=[...])
def user_vision_sandbox(user_id: str, image: bytes):
    """
    Personalized AI processing per user:
    - Learns user preferences (pharmacy = CVS on 34th St)
    - Stores medical context (allergies, prescriptions)
    - Adapts to environment (NYC subway vs rural roads)
    """
    return personalized_results

# Modal auto-scales: 1 user or 50,000 users
```

## The Transformation

### Business Model Evolution

| Metric | Sunu (2017) | Sunu 2.0 (2026) |
|--------|-------------|-----------------|
| **Hardware Cost** | $299 per user | $300 glasses (reusable) |
| **Revenue Model** | One-time purchase | $5-10/month subscription |
| **Total Cost (5yr)** | $299 | $300 + $300-600 = $600-900 |
| **Gross Margin** | ~40% (hardware) | ~80% (software) |
| **Scalability** | Linear (1 device = 1 user) | Exponential (cloud serves all) |
| **Updates** | Requires new hardware | OTA software updates |
| **Market Reach** | Thousands | Millions |

### Impact at Scale

**Sunu's Achievement (2017-2024):**
- ~5,000 users in 50+ countries
- $937K revenue since launch
- 90% reduction in accidents for users

**Our Potential (2026+):**
- 285 million visually impaired people worldwide
- $5/month × 1M users = $60M ARR
- TAM: $17B+ (285M × $5/month × 12 months)

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

## Team

Built at TreeHacks 2026 by [Your Team Name]

## License

MIT License - see LICENSE file for details
