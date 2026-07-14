# Audience Lab 🎭

**Your digital audience. Before the world hears it.**

Audience Lab is an AI-powered platform that simulates how a real-world audience would react to your song *before* you release it. Upload a track, generate a digital twin audience of AI agents with unique demographics and psychological profiles, and get data-driven recommendations to optimize your song for streaming platforms like Spotify.

---

## How It Works

1. **Upload your track** through the web interface (`index.html`).
2. **Choose your audience size** (number of AI agents) and **target genres**.
3. The backend (`main.py`) analyzes the song's audio "DNA": BPM, key, energy, loudness (LUFS), spectral brightness, and lyrics (via transcription).
4. **IBM Watsonx** generates a population of unique synthetic listener agents — each with a name, age, location, Big Five personality traits, favorite genres, reference artists, mood, and listening context.
5. Each agent's reaction is simulated: skip probability, replay probability, share probability, playlist fit, and a written comment.
6. The song is **benchmarked against real Spotify data** for the target genre(s), producing a similarity score and actionable AI recommendations (e.g., target BPM, target energy).
7. Optionally, `audio_optimizer.py` applies those recommendations directly to the audio file using professional DSP techniques, producing an optimized preview.
8. Results are displayed in a live analytics dashboard: Track DNA, Reaction Breakdown, Emotional Map, Audience Verdict, and Spotify Benchmarking with Top Similar Tracks.

---

## Architecture

```
index.html  →  main.py (FastAPI)  →  audio_optimizer.py
   ↑                  ↓                      ↓
   └──────── JSON response with results ─────┘
```

- **`index.html`** — Frontend UI: file upload, agent/genre configuration, live simulation view, and analytics dashboard.
- **`main.py`** — FastAPI backend. Handles audio feature extraction, transcription (Whisper), agent generation (IBM Watsonx), reaction simulation, and Spotify benchmarking.
- **`audio_optimizer.py`** — DSP module that applies AI recommendations to the actual audio file (BPM adjustment, loudness normalization, brightness EQ, energy/compression), using only free open-source libraries (`librosa`, `scipy`, `numpy`, `soundfile`).

---

## Core Features

- **Audio DNA Analysis** — BPM (with tempo-doubling correction), musical key detection (Krumhansl-Schmuckler algorithm), RMS energy, LUFS loudness estimate, spectral brightness, and catchiness score.
- **AI-Generated Digital Twin Audience** — Synthetic listener personas built with IBM Watsonx Foundation Models, each with a full psychological (Big Five) profile.
- **Reaction Simulation** — Predicts skip/replay/share probabilities and generates individual agent comments per song.
- **Spotify Benchmarking** — Compares your track against real genre-specific playlists and top similar tracks using a weighted similarity vector (BPM, energy, loudness, brightness, catchiness).
- **AI Recommendations Engine** — Generates specific, data-backed suggestions (target BPM, target energy, target loudness) with priority levels and technical notes.
- **Audio Optimizer** — Applies the above recommendations directly to the audio: time-stretching for BPM, gain + soft-clipping for loudness, high-shelf EQ for brightness, and dynamic range compression for energy.
- **Live Analytics Dashboard** — Visual breakdown of agent reactions, emotional map, and engagement/viral potential scores.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| AI Agents | IBM Watsonx Foundation Models |
| Audio Analysis | librosa, numpy, scipy |
| Transcription | OpenAI Whisper |
| Audio Optimization | librosa, scipy, soundfile |
| Market Data | Spotify Web API |

---

## Getting Started

### Prerequisites

- Python 3.10+
- IBM Watsonx API credentials
- Spotify Developer credentials (Client ID & Secret)

### Installation

```bash
pip install fastapi uvicorn librosa openai-whisper numpy scipy soundfile httpx python-dotenv ibm-watsonx-ai pydantic
```

### Environment Variables

Create a `.env` file in the project root:

```
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
```

### Running the App

```bash
python main.py
```

Then open `index.html` in your browser (or serve it through a local web server) and start uploading tracks.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Uploads and analyzes an audio file, returns audio DNA + transcript |
| `/generate-agents` | POST | Generates a batch of synthetic listener agents |
| `/simulate-reactions` | POST | Simulates agent reactions to a given song |
| `/benchmark` | POST | Benchmarks song against Spotify genre data and returns AI recommendations |

---

## Project Structure

```
.
├── main.py              # FastAPI backend (core logic)
├── audio_optimizer.py   # DSP audio optimization module
├── index.html           # Frontend application
└── README.md            # This file
```

---

## Notes

- Built as part of the **AI Builders Challenge with IBM Bob** (IBM SkillsBuild x BeMyApp).
- `audio_optimizer.py` uses only free, open-source DSP libraries — no paid audio processing tools required.
- Spotify benchmarking gracefully falls back to empirical genre profiles when live API data is unavailable.

---

*Made with IBM Bob.*
