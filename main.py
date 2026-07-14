import os
import tempfile
import numpy as np
import asyncio
import traceback
import json
import re
import time
import base64
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import librosa
import whisper
import uvicorn
import httpx
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import audio_optimizer

load_dotenv()

app = FastAPI(title="AudienceMind Audio Analysis API", version="2.0.0")
executor = ThreadPoolExecutor(max_workers=2)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
spotify_token_cache = {"token": None, "expires_at": 0.0}

GENRE_SEARCH_QUERIES = {
    "trap": {"query": "genre:trap", "name": "Trap Scene"},
    "hip-hop": {"query": "genre:hip-hop", "name": "Hip-Hop Scene"},
    "r&b": {"query": "genre:r-n-b", "name": "R&B Scene"},
    "electronic": {"query": "genre:electronic", "name": "Electronic Scene"},
    "reggaeton": {"query": "reggaeton", "name": "Reggaeton Scene"},
    "indie": {"query": "genre:indie", "name": "Indie Scene"},
    "pop": {"query": "genre:pop", "name": "Pop Scene"},
}

GENRE_ALIASES = {
    "rnb": "r&b",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "hiphop": "hip-hop",
    "hip hop": "hip-hop",
}

GENRE_BPM_PROFILES = {
    "trap": {"avg_bpm": 140, "bpm_std": 12, "avg_energy": 0.72, "avg_danceability": 0.70, "avg_valence": 0.40, "avg_loudness": -5.5},
    "hip-hop": {"avg_bpm": 95, "bpm_std": 15, "avg_energy": 0.65, "avg_danceability": 0.75, "avg_valence": 0.50, "avg_loudness": -6.0},
    "r&b": {"avg_bpm": 95, "bpm_std": 18, "avg_energy": 0.55, "avg_danceability": 0.72, "avg_valence": 0.58, "avg_loudness": -7.0},
    "electronic": {"avg_bpm": 128, "bpm_std": 14, "avg_energy": 0.82, "avg_danceability": 0.68, "avg_valence": 0.45, "avg_loudness": -5.0},
    "reggaeton": {"avg_bpm": 100, "bpm_std": 8, "avg_energy": 0.78, "avg_danceability": 0.88, "avg_valence": 0.65, "avg_loudness": -5.5},
    "indie": {"avg_bpm": 120, "bpm_std": 20, "avg_energy": 0.60, "avg_danceability": 0.58, "avg_valence": 0.55, "avg_loudness": -8.0},
    "pop": {"avg_bpm": 118, "bpm_std": 22, "avg_energy": 0.65, "avg_danceability": 0.72, "avg_valence": 0.60, "avg_loudness": -6.5},
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper model loaded successfully")


class AudioAnalysisResponse(BaseModel):
    title: str
    bpm: float
    key: str
    energy: float
    catchiness_score: float
    duration_seconds: float
    loudness_db: float
    spectral_brightness: float
    transcript: str


class GenerateAgentsRequest(BaseModel):
    num_agents: int
    music_genre: str


class Big5(BaseModel):
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float


class Agent(BaseModel):
    name: str
    age: int
    city: str
    country: str
    big5: Big5
    favorite_genres: list[str]
    reference_artists: list[str]
    current_mood: str
    listening_context: str
    demographic_weight: float


class GenerateAgentsResponse(BaseModel):
    agents: list[Agent]
    total: int


class SongAnalysis(BaseModel):
    bpm: float
    key: str
    energy: float
    catchiness_score: float
    transcript: str = ""
    loudness_db: float = -8.0
    duration_seconds: float = 180.0
    spectral_brightness: float = 0.5


class AgentReaction(BaseModel):
    agent_name: str
    emotional_response: str
    would_skip: bool
    skip_probability: float
    replay_probability: float
    share_probability: float
    playlist_fit: str
    comment: str
    streaming_prediction: str


class ReactionSummary(BaseModel):
    avg_skip_probability: float
    avg_replay_probability: float
    avg_share_probability: float
    top_emotional_responses: list[str]
    predicted_audience_fit: str


class SimulateReactionsRequest(BaseModel):
    song_analysis: SongAnalysis
    agents: list[Agent]
    song_title: str = "Untitled Song"


class SimulateReactionsResponse(BaseModel):
    song_title: str
    total_agents: int
    reactions: list[AgentReaction]
    summary: ReactionSummary


class BenchmarkRequest(BaseModel):
    song_analysis: dict
    genres: list[str] = Field(default_factory=list)


class PlaylistBenchmark(BaseModel):
    playlist_name: str
    genre: str
    avg_bpm: float
    avg_energy: float
    avg_danceability: float
    avg_valence: float
    avg_loudness: float
    avg_popularity: float
    sample_size: int
    fit_label: str
    similarity_score: float
    your_song_bpm: float
    your_song_energy: float
    market_source: str


class SimilarTrack(BaseModel):
    name: str
    artist: str
    similarity: float
    popularity: float = 0.0
    spotify_url: Optional[str] = None


class AIRecommendation(BaseModel):
    metric: str
    current: float | str
    target: float | str
    difference: float
    impact_score: float
    priority: str
    action: str
    technical_note: str


class BenchmarkResponse(BaseModel):
    similarity_score: float
    playlist_benchmarks: list[PlaylistBenchmark]
    top_similar_tracks: list[SimilarTrack]
    recommendation: str
    ai_recommendations: list[AIRecommendation] = []


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_genre(genre: str) -> str:
    g = (genre or "").strip().lower()
    return GENRE_ALIASES.get(g, g)


def stable_rng_for_genre(genre: str) -> np.random.Generator:
    seed = abs(hash(genre)) % (2**32)
    return np.random.default_rng(seed)


def mean_or_default(values: list[float], default: float) -> float:
    return round(float(sum(values) / len(values)), 3) if values else round(float(default), 3)


def extract_audio_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    
    # FIXED: Better BPM doubling detection (librosa often detects half-tempo)
    # If BPM is suspiciously low for most genres, double it
    if bpm < 90:
        bpm = bpm * 2
    elif 90 <= bpm < 140 and bpm < 100:
        # Edge case: 90-100 BPM might be half of 180-200 (fast genres)
        # Check if doubling would be more reasonable
        doubled = bpm * 2
        if 140 <= doubled <= 200:  # Common range for fast genres
            bpm = doubled

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    max_corr = -1
    best_key = "C major"
    for i in range(12):
        rotated_major = np.roll(major_profile, i)
        rotated_minor = np.roll(minor_profile, i)
        major_corr = np.corrcoef(chroma_mean, rotated_major)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, rotated_minor)[0, 1]
        if major_corr > max_corr:
            max_corr = major_corr
            best_key = f"{keys[i]} major"
        if minor_corr > max_corr:
            max_corr = minor_corr
            best_key = f"{keys[i]} minor"

    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))
    energy_normalized = min(energy / 0.3, 1.0)
    duration_seconds = float(librosa.get_duration(y=y, sr=sr))
    
    # FIXED: Proper loudness calculation using peak-normalized RMS
    # Calculate RMS in dB relative to full scale
    rms_db = float(20 * np.log10(np.mean(rms) + 1e-10))
    # Convert to approximate LUFS (typical music RMS is around -20 to -10 dBFS)
    # LUFS is typically 3-5 dB lower than RMS dBFS for music
    loudness_db = float(rms_db - 3.0)  # More accurate offset
    # Clamp to realistic range for music (-20 to -3 LUFS)
    loudness_db = max(-20.0, min(-3.0, loudness_db))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(spectral_centroid))
    spectral_brightness = min(centroid_mean / 8000, 1.0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_variance = np.var(onset_env)
    tempo_stability = 1 / (1 + tempo_variance)

    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_density = len(onsets) / max(duration_seconds, 1e-6)
    onset_density_normalized = min(onset_density / 10, 1)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = float(np.mean(spectral_contrast))
    contrast_normalized = min(contrast_mean / 50, 1)

    catchiness_score = (tempo_stability * 0.3 + onset_density_normalized * 0.4 + contrast_normalized * 0.3) * 100

    return {
        "bpm": round(bpm, 2),
        "key": best_key,
        "energy": round(energy_normalized, 4),
        "catchiness_score": round(catchiness_score, 2),
        "duration_seconds": round(duration_seconds, 2),
        "loudness_db": round(loudness_db, 2),
        "spectral_brightness": round(spectral_brightness, 4),
    }


def transcribe_audio(audio_path: str) -> str:
    y, sr = librosa.load(audio_path, sr=16000)
    result = whisper_model.transcribe(y, fp16=False)
    text = result["text"]
    if isinstance(text, list):
        return " ".join(str(item) for item in text)
    return str(text)


@app.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...), title: str = Form(None)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".mp3", ".mp4", ".wav", ".m4a", ".flac"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload MP3, MP4, WAV, M4A, or FLAC file")

    song_title = title if title else os.path.splitext(file.filename)[0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    try:
        print(f"Analyzing audio file: {file.filename}")
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(executor, extract_audio_features, temp_path)
        print("Transcribing audio...")
        transcript = await loop.run_in_executor(executor, transcribe_audio, temp_path)
        result = {"title": song_title, **features, "transcript": transcript}
        print("Analysis complete")
        return result
    except Exception:
        error_detail = traceback.format_exc()
        print(f"FULL ERROR:\n{error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def generate_agents_batch(batch_size: int, music_genre: str, watsonx_model: Model) -> list[dict]:
    prompt = f"""Generate {batch_size} diverse music listener personas for the genre "{music_genre}".
For each persona, return ONLY a valid JSON object with these exact fields:
- name: string (realistic full name)
- age: integer between 16 and 55
- city: string
- country: string
- big5: object with openness, conscientiousness, extraversion, agreeableness, neuroticism (all floats between 0 and 1)
- favorite_genres: array of strings (2-4 music genres)
- reference_artists: array of strings (2-4 artist names)
- current_mood: string
- listening_context: string
- demographic_weight: float

Return ONLY a JSON array of {batch_size} objects, no additional text or explanation."""

    response_text = ""
    try:
        response = watsonx_model.generate_text(prompt=prompt)
        if isinstance(response, str):
            response_text = response.strip()
        elif isinstance(response, list):
            response_text = str(response[0]) if response else ""
        elif isinstance(response, dict):
            response_text = json.dumps(response)
        else:
            response_text = str(response)

        # Find the FIRST complete JSON array (ignore any text after it)
        start_idx = response_text.find("[")
        if start_idx == -1:
            return []
        
        # Find matching closing bracket using bracket counting
        bracket_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(response_text)):
            char = response_text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
        
        if end_idx == -1:
            return []
        
        json_str = response_text[start_idx:end_idx]

        try:
            agents = json.loads(json_str)
            if isinstance(agents, list):
                # Print each agent for visibility
                for agent in agents:
                    print(f"  ✓ Generated: {agent.get('name', 'Unknown')} ({agent.get('age', '?')}y, {agent.get('city', '?')}, {agent.get('country', '?')})")
                return agents
            return []
        except json.JSONDecodeError:
            last_complete = json_str.rfind("},")
            if last_complete != -1:
                try:
                    agents = json.loads(json_str[:last_complete + 1] + "]")
                    if isinstance(agents, list):
                        for agent in agents:
                            print(f"  ✓ Generated: {agent.get('name', 'Unknown')} ({agent.get('age', '?')}y, {agent.get('city', '?')}, {agent.get('country', '?')})")
                        return agents
                    return []
                except json.JSONDecodeError:
                    return []
            return []
    except Exception as e:
        print(f"Error generating batch: {e}")
        return []


def simulate_reactions_batch(agents_batch: list[dict], song_analysis: dict, song_title: str, model_inference: ModelInference) -> list[dict]:
    agent_summaries = []
    for agent in agents_batch:
        big5 = agent.get("big5", {})
        if isinstance(big5, dict):
            big5_str = f"O:{big5.get('openness',0):.1f}, C:{big5.get('conscientiousness',0):.1f}, E:{big5.get('extraversion',0):.1f}, A:{big5.get('agreeableness',0):.1f}, N:{big5.get('neuroticism',0):.1f}"
        else:
            big5_str = "O:0.5, C:0.5, E:0.5, A:0.5, N:0.5"

        summary = (
            f"- {agent.get('name','Unknown')} (age {agent.get('age',25)}, {agent.get('city','')}, {agent.get('country','')}): "
            f"Big5({big5_str}), Genres: {', '.join(agent.get('favorite_genres', []))}, "
            f"Mood: {agent.get('current_mood', 'neutral')}, Context: {agent.get('listening_context', 'casual')}"
        )
        agent_summaries.append(summary)

    agents_text = "\n".join(agent_summaries)
    prompt = f"""You are simulating how different music listeners would react to a song based on their personality and preferences.

Song: "{song_title}"
Audio Features:
- BPM: {song_analysis["bpm"]}
- Key: {song_analysis["key"]}
- Energy: {song_analysis["energy"]}
- Catchiness Score: {song_analysis["catchiness_score"]}
- Lyrics: {song_analysis.get("transcript", "N/A")[:200]}

Listeners:
{agents_text}

For EACH listener above, predict their psychological reaction. Return ONLY a JSON array with one object per listener.
Each object must have these exact fields:
- agent_name: string
- emotional_response: string (one word ONLY from: "hyped", "bored", "nostalgic", "neutral", "annoyed", "relaxed", "energized", "melancholic")
- would_skip: boolean
- skip_probability: float 0-1
- replay_probability: float 0-1
- share_probability: float 0-1
- playlist_fit: string (ONLY one of: "perfect_match", "good_fit", "neutral", "poor_fit", "terrible_fit")
- comment: string
- streaming_prediction: string (ONLY one of: "skip", "listen_once", "add_to_playlist", "share")

Return ONLY the JSON array, no markdown, no explanation."""

    response_text = ""
    try:
        response = model_inference.generate_text(prompt=prompt)
        if isinstance(response, str):
            response_text = response.strip()
        elif isinstance(response, list):
            response_text = str(response[0]) if response else ""
        elif isinstance(response, dict):
            response_text = json.dumps(response)
        else:
            response_text = str(response)

        response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Find the FIRST complete JSON array (ignore any text after it)
        start_idx = response_text.find("[")
        if start_idx == -1:
            return []
        
        # Find matching closing bracket using bracket counting
        bracket_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(response_text)):
            char = response_text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
        
        if end_idx == -1:
            return []
        
        json_str = response_text[start_idx:end_idx]

        try:
            reactions = json.loads(json_str)
            if isinstance(reactions, list):
                print(f"  ✓ Generated {len(reactions)} reactions for batch")
                return reactions
            else:
                print(f"  ✗ Invalid response format (not a list)")
                return []
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON decode error: {str(e)[:100]}")
            # Try to recover by finding last complete object
            last_complete = json_str.rfind("},")
            if last_complete != -1:
                try:
                    reactions = json.loads(json_str[:last_complete + 1] + "]")
                    if isinstance(reactions, list):
                        print(f"  ✓ Recovered {len(reactions)} reactions from partial JSON")
                        return reactions
                    return []
                except json.JSONDecodeError:
                    print(f"  ✗ Could not recover from partial JSON")
                    return []
            return []
    except Exception as e:
        print(f"  ✗ Error simulating reactions batch: {e}")
        traceback.print_exc()
        return []


def _sanitize_reactions(all_reactions: list) -> list:
    valid_emotions = ["hyped", "bored", "nostalgic", "neutral", "annoyed", "relaxed", "energized", "melancholic"]
    valid_fits = ["perfect_match", "good_fit", "neutral", "poor_fit", "terrible_fit"]
    valid_predictions = ["skip", "listen_once", "add_to_playlist", "share"]
    sanitized = []

    for r in all_reactions:
        if not isinstance(r, dict):
            continue

        agent_name = r.get("agent_name") or r.get("name") or r.get("listener") or "Unknown Agent"
        emotion = r.get("emotional_response", "neutral")
        if emotion not in valid_emotions:
            emotion = "neutral"

        skip_prob = float(r.get("skip_probability") or 0.3)
        replay_prob = float(r.get("replay_probability") or 0.4)
        share_prob = float(r.get("share_probability") or 0.2)
        would_skip = bool(r.get("would_skip", skip_prob > 0.5))

        playlist_fit = r.get("playlist_fit", "neutral")
        if playlist_fit not in valid_fits:
            playlist_fit = "neutral"

        comment = r.get("comment") or "No comment provided."
        prediction = r.get("streaming_prediction", "listen_once")
        if prediction not in valid_predictions:
            prediction = "listen_once"

        sanitized.append({
            "agent_name": str(agent_name),
            "emotional_response": emotion,
            "would_skip": would_skip,
            "skip_probability": round(clamp(skip_prob, 0.0, 1.0), 3),
            "replay_probability": round(clamp(replay_prob, 0.0, 1.0), 3),
            "share_probability": round(clamp(share_prob, 0.0, 1.0), 3),
            "playlist_fit": playlist_fit,
            "comment": str(comment),
            "streaming_prediction": prediction,
        })

    return sanitized


@app.post("/generate-agents", response_model=GenerateAgentsResponse)
async def generate_agents(request: GenerateAgentsRequest):
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not api_key or not project_id:
        raise HTTPException(status_code=500, detail="Missing IBM Watsonx credentials.")

    credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": api_key}
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_P: 1,
        GenParams.TOP_K: 50,
    }

    try:
        watsonx_model = Model(
            model_id="meta-llama/llama-3-3-70b-instruct",
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize IBM Watsonx model: {str(e)}")

    batch_size = 5
    num_batches = (request.num_agents + batch_size - 1) // batch_size
    all_agents = []

    print(f"Generating {request.num_agents} agents in {num_batches} batches (sequential processing)...")

    for batch_num in range(num_batches):
        current_batch_size = min(batch_size, request.num_agents - len(all_agents))
        print(f"Processing batch {batch_num + 1}/{num_batches} ({current_batch_size} agents)...")
        
        # Retry logic for failed batches
        max_retries = 3
        batch_agents = []
        for retry in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                batch_agents = await loop.run_in_executor(executor, generate_agents_batch, current_batch_size, request.music_genre, watsonx_model)
                if batch_agents and len(batch_agents) > 0:
                    break  # Success, exit retry loop
                elif retry < max_retries - 1:
                    print(f"Batch returned empty, retrying ({retry + 1}/{max_retries})...")
                    await asyncio.sleep(2)
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Batch failed: {str(e)}, retrying ({retry + 1}/{max_retries})...")
                    await asyncio.sleep(2)
                else:
                    print(f"Batch failed after {max_retries} retries: {str(e)}")
        
        all_agents.extend(batch_agents)
        
        # Add delay between batches to respect rate limits (except after last batch)
        if batch_num < num_batches - 1:
            print(f"Waiting 3 seconds before next batch...")
            await asyncio.sleep(3)

    print(f"\n{'='*60}")
    print(f"AGENT GENERATION COMPLETE")
    print(f"Requested agents: {request.num_agents}")
    print(f"Generated agents: {len(all_agents)}")
    if len(all_agents) < request.num_agents:
        print(f"⚠ WARNING: Missing {request.num_agents - len(all_agents)} agents!")
    else:
        print(f"✓ SUCCESS: All agents generated")
    print(f"{'='*60}\n")
    
    return {"agents": all_agents, "total": len(all_agents)}


@app.post("/simulate-reactions", response_model=SimulateReactionsResponse)
async def simulate_reactions(request: SimulateReactionsRequest):
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not api_key or not project_id:
        raise HTTPException(status_code=500, detail="Missing IBM Watsonx credentials.")

    credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": api_key}
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_P: 1,
        GenParams.TOP_K: 50,
    }

    try:
        model_inference = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize IBM Watsonx ModelInference: {str(e)}")

    song_analysis_dict = request.song_analysis.model_dump()
    agents_dicts = [agent.model_dump() for agent in request.agents]
    batch_size = 3
    num_batches = (len(agents_dicts) + batch_size - 1) // batch_size
    all_reactions = []

    print(f"Simulating reactions for {len(agents_dicts)} agents in {num_batches} batches (sequential processing)...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(agents_dicts))
        batch_agents = agents_dicts[start_idx:end_idx]
        print(f"Processing reaction batch {i + 1}/{num_batches} ({len(batch_agents)} agents)...")
        
        # Retry logic for failed batches
        max_retries = 3
        batch_reactions = []
        for retry in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                batch_reactions = await loop.run_in_executor(
                    executor,
                    simulate_reactions_batch,
                    batch_agents,
                    song_analysis_dict,
                    request.song_title,
                    model_inference
                )
                if batch_reactions and len(batch_reactions) >= len(batch_agents):
                    break  # Success - got reactions for all agents in batch
                elif batch_reactions and len(batch_reactions) > 0:
                    print(f"  ⚠ Only got {len(batch_reactions)}/{len(batch_agents)} reactions, accepting partial")
                    break  # Accept partial results
                elif retry < max_retries - 1:
                    print(f"  ⚠ Batch returned empty, retrying ({retry + 1}/{max_retries})...")
                    await asyncio.sleep(2)
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  ✗ Batch failed: {str(e)[:100]}, retrying ({retry + 1}/{max_retries})...")
                    await asyncio.sleep(2)
                else:
                    print(f"  ✗ Batch failed after {max_retries} retries")
        
        all_reactions.extend(batch_reactions)
        
        # Add delay between batches to respect rate limits (except after last batch)
        if i < num_batches - 1:
            print(f"Waiting 3 seconds before next batch...")
            await asyncio.sleep(3)

    all_reactions = _sanitize_reactions(all_reactions)
    
    print(f"\n{'='*60}")
    print(f"REACTION SIMULATION COMPLETE")
    print(f"Input agents: {len(agents_dicts)}")
    print(f"Output reactions: {len(all_reactions)}")
    if len(all_reactions) < len(agents_dicts):
        print(f"⚠ WARNING: Missing {len(agents_dicts) - len(all_reactions)} reactions!")
    print(f"{'='*60}\n")

    if all_reactions:
        avg_skip = sum(r["skip_probability"] for r in all_reactions) / len(all_reactions)
        avg_replay = sum(r["replay_probability"] for r in all_reactions) / len(all_reactions)
        avg_share = sum(r["share_probability"] for r in all_reactions) / len(all_reactions)

        emotion_counts = {}
        for r in all_reactions:
            e = r["emotional_response"]
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        if avg_skip < 0.3:
            audience_fit = "Excellent - Strong audience engagement expected"
        elif avg_skip < 0.5:
            audience_fit = "Good - Moderate audience engagement expected"
        elif avg_skip < 0.7:
            audience_fit = "Fair - Mixed audience reception expected"
        else:
            audience_fit = "Poor - Low audience engagement expected"

        summary = {
            "avg_skip_probability": round(avg_skip, 3),
            "avg_replay_probability": round(avg_replay, 3),
            "avg_share_probability": round(avg_share, 3),
            "top_emotional_responses": [e for e, _ in top_emotions],
            "predicted_audience_fit": audience_fit,
        }
    else:
        summary = {
            "avg_skip_probability": 0.0,
            "avg_replay_probability": 0.0,
            "avg_share_probability": 0.0,
            "top_emotional_responses": [],
            "predicted_audience_fit": "Unable to determine",
        }

    return {
        "song_title": request.song_title,
        "total_agents": len(agents_dicts),
        "reactions": all_reactions,
        "summary": summary
    }


async def get_spotify_token() -> str:
    global spotify_token_cache

    if spotify_token_cache["token"] and time.time() < spotify_token_cache["expires_at"]:
        return str(spotify_token_cache["token"])

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Missing Spotify credentials in .env")

    auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://accounts.spotify.com/api/token",
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data={"grant_type": "client_credentials"},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Spotify auth failed: {resp.text}")

    data = resp.json()
    spotify_token_cache["token"] = data["access_token"]
    spotify_token_cache["expires_at"] = time.time() + float(data["expires_in"]) - 60
    print("Spotify token obtained successfully")
    return str(data["access_token"])


def _fallback_tracks(genre: str, n: int = 20) -> list[dict]:
    profile = GENRE_BPM_PROFILES.get(genre, GENRE_BPM_PROFILES["pop"])
    rng = stable_rng_for_genre(genre)
    tracks = []

    for i in range(n):
        popularity = clamp(float(rng.normal(62, 15)), 20, 95)
        duration_ms = int(clamp(float(rng.normal(210000, 30000)), 120000, 300000))
        tracks.append({
            "id": f"fallback-{genre}-{i+1}",
            "name": f"{genre.title()} Reference Track {i+1}",
            "artist": f"{genre.title()} Artist {i+1}",
            "tempo": round(float(rng.normal(profile["avg_bpm"], profile["bpm_std"])), 1),
            "energy": round(float(clamp(rng.normal(profile["avg_energy"], 0.08), 0, 1)), 3),
            "danceability": round(float(clamp(rng.normal(profile["avg_danceability"], 0.08), 0, 1)), 3),
            "valence": round(float(clamp(rng.normal(profile["avg_valence"], 0.10), 0, 1)), 3),
            "loudness": round(float(rng.normal(profile["avg_loudness"], 2.0)), 2),
            "popularity": round(popularity, 1),
            "duration_ms": duration_ms,
            "spotify_url": None,
            "source": "empirical_fallback",
        })
    return tracks


def _estimate_features_from_metadata(track: dict, genre: str) -> dict:
    profile = GENRE_BPM_PROFILES.get(genre, GENRE_BPM_PROFILES["pop"])
    popularity = float(track.get("popularity") or 50)
    duration_ms = int(track.get("duration_ms") or 210000)

    duration_factor = (210000 - duration_ms) / 210000
    popularity_factor = (popularity - 50) / 50

    tempo = profile["avg_bpm"] + duration_factor * profile["bpm_std"] * 0.6 + popularity_factor * 2.5
    energy = profile["avg_energy"] + duration_factor * 0.05 + popularity_factor * 0.03
    danceability = profile["avg_danceability"] + popularity_factor * 0.03
    valence = profile["avg_valence"] + popularity_factor * 0.04
    loudness = profile["avg_loudness"] + popularity_factor * 0.8

    return {
        "tempo": round(float(tempo), 1),
        "energy": round(float(clamp(energy, 0, 1)), 3),
        "danceability": round(float(clamp(danceability, 0, 1)), 3),
        "valence": round(float(clamp(valence, 0, 1)), 3),
        "loudness": round(float(loudness), 2),
    }


async def _spotify_search_tracks_for_genre(genre: str, token: str, limit: int = 20) -> list[dict]:
    genre_info = GENRE_SEARCH_QUERIES.get(genre)
    if not genre_info:
        return []

    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": genre_info["query"], "type": "track", "limit": min(limit, 50), "market": "US"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get("https://api.spotify.com/v1/search", headers=headers, params=params)

    print(f"Spotify search [{genre}] status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"  ✗ Spotify API error: {resp.text[:200]}")
        return []

    response_data = resp.json()
    tracks_data = response_data.get("tracks", {})
    items = tracks_data.get("items", [])
    
    print(f"  → Found {len(items)} tracks for query: '{genre_info['query']}'")
    
    if len(items) == 0:
        print(f"  ✗ No tracks returned by Spotify API")
        print(f"  → Response keys: {list(response_data.keys())}")
        if "tracks" in response_data:
            print(f"  → Tracks keys: {list(tracks_data.keys())}")
    
    results = []

    for item in items:
        artists = ", ".join(a.get("name", "") for a in item.get("artists", []))
        spotify_url = item.get("external_urls", {}).get("spotify")
        popularity = float(item.get("popularity") or 0)
        duration_ms = int(item.get("duration_ms") or 210000)

        estimated = _estimate_features_from_metadata(item, genre)
        results.append({
            "id": item.get("id"),
            "name": item.get("name", "Unknown"),
            "artist": artists or "Unknown Artist",
            "tempo": estimated["tempo"],
            "energy": estimated["energy"],
            "danceability": estimated["danceability"],
            "valence": estimated["valence"],
            "loudness": estimated["loudness"],
            "popularity": popularity,
            "duration_ms": duration_ms,
            "spotify_url": spotify_url,
            "source": "spotify_search_estimated",
        })

    return results


async def _try_audio_features_enrichment(tracks: list[dict], token: str) -> list[dict]:
    track_ids = [t["id"] for t in tracks if t.get("id")]
    if not track_ids:
        return tracks

    headers = {"Authorization": f"Bearer {token}"}
    params = {"ids": ",".join(track_ids[:100])}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get("https://api.spotify.com/v1/audio-features", headers=headers, params=params)

        print(f"Spotify audio-features status: {resp.status_code}")

        if resp.status_code != 200:
            return tracks

        features = resp.json().get("audio_features", [])
        feature_map = {}
        for f in features:
            if f and f.get("id"):
                feature_map[f["id"]] = f

        enriched = []
        for track in tracks:
            f = feature_map.get(track.get("id"))
            if f:
                enriched.append({
                    **track,
                    "tempo": float(f.get("tempo") or track["tempo"]),
                    "energy": float(f.get("energy") or track["energy"]),
                    "danceability": float(f.get("danceability") or track["danceability"]),
                    "valence": float(f.get("valence") or track["valence"]),
                    "loudness": float(f.get("loudness") or track["loudness"]),
                    "source": "spotify_audio_features",
                })
            else:
                enriched.append(track)
        return enriched
    except Exception as e:
        print(f"Audio-features enrichment failed: {e}")
        return tracks


async def _search_and_get_benchmark_tracks(genre: str, token: str, target_size: int = 20) -> list[dict]:
    """
    Get benchmark tracks using Spotify search + local estimation.
    NO LONGER USES audio-features endpoint (403 error).
    """
    spotify_tracks = await _spotify_search_tracks_for_genre(genre, token, limit=target_size)
    if not spotify_tracks:
        print(f"No Spotify tracks found for genre '{genre}', using fallback")
        return _fallback_tracks(genre, target_size)

    # Spotify tracks already have estimated features from _spotify_search_tracks_for_genre
    # No need to call audio-features endpoint (which returns 403)
    print(f"Using {len(spotify_tracks)} Spotify tracks with estimated features for '{genre}'")

    if len(spotify_tracks) < target_size:
        fallback_needed = target_size - len(spotify_tracks)
        spotify_tracks.extend(_fallback_tracks(genre, fallback_needed))

    return spotify_tracks[:target_size]


def _normalized_feature_vector(song: dict) -> dict:
    bpm = float(song.get("bpm", 120.0))
    energy = float(song.get("energy", 0.5))
    loudness_db = float(song.get("loudness_db", -8.0))
    brightness = float(song.get("spectral_brightness", 0.5))
    catchiness = float(song.get("catchiness_score", 50.0)) / 100.0

    return {
        "bpm": clamp((bpm - 60) / 140, 0.0, 1.5),
        "energy": clamp(energy, 0.0, 1.0),
        "loudness": clamp((loudness_db + 60) / 60, 0.0, 1.0),
        "brightness": clamp(brightness, 0.0, 1.0),
        "catchiness": clamp(catchiness, 0.0, 1.0),
    }


def _track_vector(track: dict) -> dict:
    return {
        "bpm": clamp((float(track.get("tempo", 120.0)) - 60) / 140, 0.0, 1.5),
        "energy": clamp(float(track.get("energy", 0.5)), 0.0, 1.0),
        "loudness": clamp((float(track.get("loudness", -8.0)) + 60) / 60, 0.0, 1.0),
        "brightness": 0.5,
        "catchiness": clamp((float(track.get("danceability", 0.5)) * 0.6 + float(track.get("popularity", 50.0)) / 100.0 * 0.4), 0.0, 1.0),
    }


def _similarity_from_vectors(song_vec: dict, ref_vec: dict) -> float:
    distance = np.sqrt(
        (song_vec["bpm"] - ref_vec["bpm"]) ** 2 * 0.32 +
        (song_vec["energy"] - ref_vec["energy"]) ** 2 * 0.28 +
        (song_vec["loudness"] - ref_vec["loudness"]) ** 2 * 0.18 +
        (song_vec["brightness"] - ref_vec["brightness"]) ** 2 * 0.10 +
        (song_vec["catchiness"] - ref_vec["catchiness"]) ** 2 * 0.12
    )
    return round(clamp(100.0 - distance * 100.0, 0.0, 100.0), 1)


def _fit_label(similarity: float) -> str:
    if similarity >= 80:
        return "Strong fit 🔥"
    if similarity >= 60:
        return "Moderate fit ✅"
    if similarity >= 40:
        return "Weak fit ⚠️"
    return "Poor fit ❌"


def _market_source_for_tracks(tracks: list[dict]) -> str:
    real_count = sum(1 for t in tracks if t.get("source") in {"spotify_search_estimated", "spotify_audio_features"})
    feature_count = sum(1 for t in tracks if t.get("source") == "spotify_audio_features")
    total = len(tracks)

    if total == 0:
        return "no_data"
    if feature_count == total:
        return "spotify_search_plus_audio_features"
    if real_count > 0:
        return "spotify_search_with_empirical_backfill"
    return "empirical_fallback"


def _generate_ai_recommendations(song: dict, viral_tracks: list[dict], primary_benchmark: dict) -> list[dict]:
    """
    Generate precise, data-driven recommendations based on mathematical analysis.
    Each recommendation includes the metric, current value, target value, impact score, and actionable advice.
    """
    recommendations = []
    
    if not viral_tracks:
        return recommendations
    
    # Calculate viral averages
    viral_bpm_avg = np.mean([float(t.get("tempo", 120)) for t in viral_tracks])
    viral_energy_avg = np.mean([float(t.get("energy", 0.5)) for t in viral_tracks])
    viral_loudness_avg = np.mean([float(t.get("loudness", -8)) for t in viral_tracks])
    viral_danceability_avg = np.mean([float(t.get("danceability", 0.5)) for t in viral_tracks])
    viral_popularity_avg = np.mean([float(t.get("popularity", 50)) for t in viral_tracks])
    
    # Current song values
    current_bpm = float(song.get("bpm", 120))
    current_energy = float(song.get("energy", 0.5))
    current_loudness = float(song.get("loudness_db", -8))
    current_catchiness = float(song.get("catchiness_score", 50)) / 100.0
    
    # BPM Analysis (weight: 0.32 in similarity calculation)
    bpm_diff = abs(current_bpm - viral_bpm_avg)
    if bpm_diff > 10:
        direction = "increase" if current_bpm < viral_bpm_avg else "decrease"
        impact = min(100, (bpm_diff / viral_bpm_avg) * 100 * 0.32)
        recommendations.append({
            "metric": "BPM (Tempo)",
            "current": round(current_bpm, 1),
            "target": round(viral_bpm_avg, 1),
            "difference": round(bpm_diff, 1),
            "impact_score": round(impact, 1),
            "priority": "high" if impact > 15 else "medium",
            "action": f"{direction.capitalize()} tempo to {round(viral_bpm_avg, 0)} BPM for better viral alignment. Viral tracks average {round(viral_bpm_avg, 0)} BPM.",
            "technical_note": f"Adjust DAW tempo or use time-stretching. Current deviation: {round(bpm_diff, 0)} BPM ({round((bpm_diff/viral_bpm_avg)*100, 1)}%)"
        })
    
    # Energy Analysis (weight: 0.28)
    energy_diff = abs(current_energy - viral_energy_avg)
    if energy_diff > 0.1:
        direction = "increase" if current_energy < viral_energy_avg else "decrease"
        impact = min(100, (energy_diff / 1.0) * 100 * 0.28)
        recommendations.append({
            "metric": "Energy Level",
            "current": round(current_energy * 100, 1),
            "target": round(viral_energy_avg * 100, 1),
            "difference": round(energy_diff * 100, 1),
            "impact_score": round(impact, 1),
            "priority": "high" if impact > 12 else "medium",
            "action": f"{direction.capitalize()} energy to {round(viral_energy_avg * 100, 0)}%. Viral tracks are {'more intense' if viral_energy_avg > current_energy else 'more relaxed'}.",
            "technical_note": f"{'Add compression, saturation, or louder drums' if direction == 'increase' else 'Reduce compression, lower drum levels, add space'}. Target: {round(viral_energy_avg * 100, 0)}%"
        })
    
    # Loudness Analysis (weight: 0.18)
    loudness_diff = abs(current_loudness - viral_loudness_avg)
    if loudness_diff > 2:
        direction = "increase" if current_loudness < viral_loudness_avg else "decrease"
        impact = min(100, (loudness_diff / 60) * 100 * 0.18)
        recommendations.append({
            "metric": "Loudness",
            "current": round(current_loudness, 1),
            "target": round(viral_loudness_avg, 1),
            "difference": round(loudness_diff, 1),
            "impact_score": round(impact, 1),
            "priority": "medium" if impact > 8 else "low",
            "action": f"{direction.capitalize()} loudness to {round(viral_loudness_avg, 1)} dB LUFS. Viral tracks average {round(viral_loudness_avg, 1)} dB.",
            "technical_note": f"Use mastering limiter to target {round(viral_loudness_avg, 1)} dB LUFS. Current: {round(current_loudness, 1)} dB"
        })
    
    # Catchiness/Danceability Analysis (weight: 0.12)
    catchiness_diff = abs(current_catchiness - viral_danceability_avg)
    if catchiness_diff > 0.15:
        direction = "increase" if current_catchiness < viral_danceability_avg else "decrease"
        impact = min(100, (catchiness_diff / 1.0) * 100 * 0.12)
        recommendations.append({
            "metric": "Catchiness/Danceability",
            "current": round(current_catchiness * 100, 1),
            "target": round(viral_danceability_avg * 100, 1),
            "difference": round(catchiness_diff * 100, 1),
            "impact_score": round(impact, 1),
            "priority": "high" if impact > 10 else "medium",
            "action": f"{direction.capitalize()} danceability to {round(viral_danceability_avg * 100, 0)}%. {'Add stronger rhythmic elements, clearer beat' if direction == 'increase' else 'Simplify rhythm, reduce groove complexity'}.",
            "technical_note": f"{'Emphasize kick/snare pattern, add rhythmic hooks' if direction == 'increase' else 'Reduce percussive elements, add melodic focus'}. Target: {round(viral_danceability_avg * 100, 0)}%"
        })
    
    # Popularity-based insight (FIXED: show actual similarity score as current)
    if viral_popularity_avg >= 70:
        # Calculate average similarity with viral tracks as "current market position"
        avg_similarity_with_viral = np.mean([t.get("similarity", 0) for t in viral_tracks]) if viral_tracks else 0
        recommendations.append({
            "metric": "Market Positioning",
            "current": round(avg_similarity_with_viral, 1),
            "target": round(viral_popularity_avg, 0),
            "difference": round(abs(viral_popularity_avg - avg_similarity_with_viral), 1),
            "impact_score": 100,
            "priority": "critical",
            "action": f"Your song has {round(avg_similarity_with_viral, 1)}% similarity with tracks averaging {round(viral_popularity_avg, 0)}/100 popularity. Focus on the recommendations above to align with proven viral formulas.",
            "technical_note": f"High-performing reference tracks suggest strong commercial potential if you match their production characteristics."
        })
    
    # Sort by impact score (highest first)
    recommendations.sort(key=lambda x: x["impact_score"], reverse=True)
    
    return recommendations


def _build_recommendation(overall: float, primary_benchmark: dict, song: dict, top_similar: list[dict]) -> str:
    bpm_diff = float(song["bpm"]) - float(primary_benchmark["avg_bpm"])
    energy_diff = float(song["energy"]) - float(primary_benchmark["avg_energy"])

    bpm_direction = "higher" if bpm_diff > 0 else "lower" if bpm_diff < 0 else "equal to"
    energy_direction = "above" if energy_diff > 0.03 else "below" if energy_diff < -0.03 else "close to"

    if overall >= 80:
        verdict = "Excellent market fit. Your track already sits very close to the current scene."
    elif overall >= 60:
        verdict = "Good market fit. You are in-range, but a targeted production tweak could improve positioning."
    elif overall >= 40:
        verdict = "Mixed market fit. The song shares some scene traits, but it diverges in ways that may reduce immediate playlist alignment."
    else:
        verdict = "Low market fit. The song is currently far from the center of the selected scene, which makes it either risky or intentionally distinctive."

    refs = ", ".join([f"{t['name']} by {t['artist']}" for t in top_similar[:3]]) if top_similar else "no strong Spotify references found"
    return (
        f"{verdict} Your BPM is {abs(bpm_diff):.0f} points {bpm_direction} than the {primary_benchmark['genre']} scene average "
        f"({primary_benchmark['avg_bpm']:.1f} BPM). Your energy is {energy_direction} that scene average "
        f"({primary_benchmark['avg_energy']:.2f}). Top reference tracks: {refs}."
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_song(request: BenchmarkRequest):
    try:
        token = await get_spotify_token()
        song = dict(request.song_analysis or {})
        genres = [normalize_genre(g) for g in request.genres if str(g).strip()]
        genres = [g for g in genres if g in GENRE_SEARCH_QUERIES][:2]

        if not genres:
            raise HTTPException(
                status_code=400,
                detail=f"No supported genres provided. Supported genres: {', '.join(GENRE_SEARCH_QUERIES.keys())}"
            )

        required_fields = ["bpm", "energy"]
        missing_fields = [f for f in required_fields if f not in song]
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Missing song_analysis fields: {', '.join(missing_fields)}")

        song.setdefault("loudness_db", -8.0)
        song.setdefault("spectral_brightness", 0.5)
        song.setdefault("catchiness_score", 50.0)

        song_vec = _normalized_feature_vector(song)
        playlist_benchmarks = []
        all_tracks = []

        for genre in genres:
            tracks = await _search_and_get_benchmark_tracks(genre, token, target_size=20)
            if not tracks:
                continue

            avg_bpm = mean_or_default([float(t["tempo"]) for t in tracks], GENRE_BPM_PROFILES[genre]["avg_bpm"])
            avg_energy = mean_or_default([float(t["energy"]) for t in tracks], GENRE_BPM_PROFILES[genre]["avg_energy"])
            avg_danceability = mean_or_default([float(t["danceability"]) for t in tracks], GENRE_BPM_PROFILES[genre]["avg_danceability"])
            avg_valence = mean_or_default([float(t["valence"]) for t in tracks], GENRE_BPM_PROFILES[genre]["avg_valence"])
            avg_loudness = mean_or_default([float(t["loudness"]) for t in tracks], GENRE_BPM_PROFILES[genre]["avg_loudness"])
            avg_popularity = mean_or_default([float(t.get("popularity", 0.0)) for t in tracks], 50.0)

            scene_ref = {
                "bpm": clamp((avg_bpm - 60) / 140, 0.0, 1.5),
                "energy": clamp(avg_energy, 0.0, 1.0),
                "loudness": clamp((avg_loudness + 60) / 60, 0.0, 1.0),
                "brightness": 0.5,
                "catchiness": clamp(avg_danceability * 0.65 + avg_popularity / 100.0 * 0.35, 0.0, 1.0),
            }
            similarity = _similarity_from_vectors(song_vec, scene_ref)

            playlist_benchmarks.append({
                "playlist_name": GENRE_SEARCH_QUERIES[genre]["name"],
                "genre": genre,
                "avg_bpm": round(avg_bpm, 1),
                "avg_energy": round(avg_energy, 3),
                "avg_danceability": round(avg_danceability, 3),
                "avg_valence": round(avg_valence, 3),
                "avg_loudness": round(avg_loudness, 2),
                "avg_popularity": round(avg_popularity, 1),
                "sample_size": len(tracks),
                "fit_label": _fit_label(similarity),
                "similarity_score": similarity,
                "your_song_bpm": round(float(song["bpm"]), 2),
                "your_song_energy": round(float(song["energy"]), 3),
                "market_source": _market_source_for_tracks(tracks),
            })

            for track in tracks:
                track_similarity = _similarity_from_vectors(song_vec, _track_vector(track))
                all_tracks.append({
                    "name": track["name"],
                    "artist": track["artist"],
                    "similarity": track_similarity,
                    "popularity": round(float(track.get("popularity", 0.0)), 1),
                    "spotify_url": track.get("spotify_url"),
                })

        if not playlist_benchmarks:
            return {
                "similarity_score": 0.0,
                "playlist_benchmarks": [],
                "top_similar_tracks": [],
                "recommendation": "Unable to build benchmark for the requested genres.",
                "ai_recommendations": []
            }

        dedup = {}
        for t in all_tracks:
            key = (t["name"].strip().lower(), t["artist"].strip().lower())
            if key not in dedup or t["similarity"] > dedup[key]["similarity"]:
                dedup[key] = t

        # FIXED: Filter only viral tracks (popularity >= 60) for benchmark
        viral_tracks = [t for t in dedup.values() if t["popularity"] >= 60]
        
        # If we have viral tracks, use them; otherwise fall back to all tracks
        tracks_to_use = viral_tracks if viral_tracks else list(dedup.values())
        top_similar = sorted(tracks_to_use, key=lambda x: (x["similarity"], x["popularity"]), reverse=True)[:10]
        
        overall = round(sum(p["similarity_score"] for p in playlist_benchmarks) / len(playlist_benchmarks), 1)

        primary_benchmark = max(playlist_benchmarks, key=lambda x: x["similarity_score"])
        recommendation = _build_recommendation(overall, primary_benchmark, song, top_similar)
        
        # Generate AI recommendations based on viral tracks analysis
        ai_recommendations = _generate_ai_recommendations(song, all_tracks, primary_benchmark) if viral_tracks else []

        return {
            "similarity_score": overall,
            "playlist_benchmarks": playlist_benchmarks,
            "top_similar_tracks": top_similar,
            "recommendation": recommendation,
            "ai_recommendations": ai_recommendations,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Benchmark error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "AudienceMind Audio Analysis API",
        "endpoints": {
            "/analyze": "POST - Upload audio file for analysis",
            "/generate-agents": "POST - Generate synthetic music listener agents",
            "/simulate-reactions": "POST - Simulate agent reactions to a song",
            "/benchmark": "POST - Benchmark song against Spotify catalog by genre",
        },
        "spotify_note": "Benchmark uses Spotify search as the primary verified source and gracefully degrades when restricted endpoints are unavailable.",
    }


@app.post("/optimize-audio")
async def optimize_audio(
    file: UploadFile = File(...),
    recommendations: str = Form(...)
):
    """
    Optimize audio file based on AI recommendations.
    Returns optimized audio file for download.
    """
    try:
        # Parse recommendations JSON
        print(f"Received recommendations: {recommendations}")
        recs = json.loads(recommendations)
        print(f"Parsed recommendations: {recs}")
        
        # Transform recommendations list to dict format expected by optimizer
        recs_dict = {}
        for rec in recs:
            metric = rec['metric'].lower()
            if metric == 'bpm':
                recs_dict['bpm'] = {
                    'current': rec['current'],
                    'target': rec['target']
                }
            elif metric == 'loudness':
                recs_dict['loudness'] = {
                    'current': rec['current'],
                    'target': rec['target']
                }
            elif metric == 'brightness' or metric == 'spectral brightness':
                recs_dict['brightness'] = {
                    'increase': rec['target'] > rec['current']
                }
            elif metric == 'energy':
                recs_dict['energy'] = {
                    'increase': rec['target'] > rec['current']
                }
            elif metric == 'catchiness':
                recs_dict['catchiness'] = {
                    'current': rec['current'],
                    'target': rec['target']
                }
        
        print(f"Transformed recommendations: {recs_dict}")
        
        # Save uploaded file temporarily
        temp_input = tempfile.mktemp(suffix='.mp3')
        with open(temp_input, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Create output path
        temp_output = tempfile.mktemp(suffix='_optimized.wav')
        
        # Optimize audio
        result = audio_optimizer.optimize_audio_file(
            temp_input,
            temp_output,
            recs_dict
        )
        
        # Clean up input file
        if os.path.exists(temp_input):
            os.remove(temp_input)
        
        if result['status'] == 'success':
            # Convert WAV to MP3 for smaller file size
            temp_mp3 = tempfile.mktemp(suffix='_optimized.mp3')
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(temp_output)
                audio.export(temp_mp3, format='mp3', bitrate='320k')
                
                # Clean up WAV file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                
                # Return optimized MP3 file
                original_name = file.filename.rsplit('.', 1)[0] if file.filename else "audio"
                return FileResponse(
                    temp_mp3,
                    media_type='audio/mpeg',
                    filename=f"optimized_{original_name}.mp3",
                    headers={
                        "X-Applied-Changes": json.dumps(result['applied_changes'])
                    }
                )
            except Exception as e:
                print(f"MP3 conversion failed, returning WAV: {str(e)}")
                # Fallback to WAV if MP3 conversion fails
                original_name = file.filename.rsplit('.', 1)[0] if file.filename else "audio"
                return FileResponse(
                    temp_output,
                    media_type='audio/wav',
                    filename=f"optimized_{original_name}.wav",
                    headers={
                        "X-Applied-Changes": json.dumps(result['applied_changes'])
                    }
                )
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Optimization failed'))
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print(f"Received data: {recommendations}")
        raise HTTPException(status_code=400, detail=f"Invalid recommendations format: {str(e)}")
    except Exception as e:
        print(f"Optimization error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)