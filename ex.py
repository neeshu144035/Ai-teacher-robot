import os
import re
import json
import faiss
import numpy as np
import torch
import requests
import random
import yt_dlp
from sentence_transformers import SentenceTransformer, util

# --- NEW IMPORTS for Voice Agent ---
import speech_recognition as sr
from gtts import gTTS
import time # For potential delays or filename uniqueness

import pygame # For controllable audio playback and display
import keyboard # For push-to-talk interrupt detection
import threading # To run keyboard listener concurrently
import queue # For communication between threads
import nltk # For sentence splitting
import subprocess # To run Rhubarb
from pydub import AudioSegment # For MP3 to WAV conversion
import io # To handle audio data in memory potentially

# -------------------------
# General Debugging Utility
# -------------------------
debug_mode = True  # Enable debugging
# You might want to set a global debug level
# current_debug_level = 3

def debug_print(message, level=1):
    # Use a global debug level variable if you want to control verbosity
    # if level <= current_debug_level:
    if debug_mode:
        prefix = "  " * level
        print(f"{prefix}🔹 {message}")

# -------------------------------------
# --- Configuration & Data Loading ---
# -------------------------------------

# --- !!! USER CONFIGURATION REQUIRED !!! ---
LLM_API_KEY = "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" # Replace with your actual API Key
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions" # Or your LLM endpoint

# Path to your Rhubarb executable
# Example Windows: RHUBARB_EXECUTABLE = r"C:\Program Files\Rhubarb-Lip-Sync-1.13.0-Windows\rhubarb.exe"
# Example macOS/Linux (if in PATH): RHUBARB_EXECUTABLE = "rhubarb"
# Example macOS (if downloaded): RHUBARB_EXECUTABLE = r"/path/to/Rhubarb-Lip-Sync-1.13.0-macOS/rhubarb"
# Example Linux (if downloaded): RHUBARB_EXECUTABLE = r"/path/to/Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb"
RHUBARB_EXECUTABLE = r"C:\Rhubarb-Lip-Sync-1.14.0-Windows\rhubarb.exe" # Adjust this path!

# Temporary file prefix and base name for TTS audio
TTS_TEMP_PREFIX = "temp_ai_speech_"


# Paths for existing data files
IMAGE_DIR = "images"
FIGURES_JSON = "output.json" # Assuming this maps figure references to descriptions and subchapters

# Data for textual content (Ensure these JSON files exist and are correctly formatted)
try:
    with open("knowledgebase.json", "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
except FileNotFoundError as e:
    print(f"❌ Critical Error: Data file not found - {e}. Make sure knowledgebase.json and metadata.json are in the correct directory.")
    exit()
except json.JSONDecodeError as e:
    print(f"❌ Critical Error: Could not decode data file - {e}. Check JSON file formatting.")
    exit()


# Normalize function
def normalize_title(title):
    """Normalizes titles for comparison."""
    return title.strip().lower()

# Create normalized KB lookup for faster access
normalized_kb = {}
for chapter, topics in kb_data.items():
    for title, content in topics.items():
        norm_key = (chapter, normalize_title(title))
        normalized_kb[norm_key] = content
debug_print(f"Loaded Knowledge Base with {len(normalized_kb)} entries.", 2)


# Initialize embedding model (for text search)
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    debug_print(f"SentenceTransformer model loaded to {device}.", 2)
except Exception as e:
    print(f"❌ Critical Error: Could not load SentenceTransformer model: {e}")
    # Depending on criticality, you might exit or run in a limited mode
    exit()


# Load FAISS index (text) (Ensure this file exists)
try:
    index = faiss.read_index("textbook_faiss.index")
    debug_print("FAISS text index loaded.", 2)
except FileNotFoundError:
    print("❌ Critical Error: FAISS text index 'textbook_faiss.index' not found. Run the indexing script first.")
    exit()
except Exception as e:
    print(f"❌ Critical Error: Could not load FAISS text index: {e}")
    exit()


# search function (for text content)
def search(query, top_k=5, similarity_threshold=0.98, mode="hybrid"):
    """
    Searches the knowledge base using exact matching and/or semantic search (FAISS).
    Returns relevant content entries.
    """
    norm_query = normalize_title(query)
    results = []
    seen_embeddings = []
    seen_titles = set() # Use set for faster lookups

    debug_print(f"Searching KB for query: '{query}' (Mode: {mode}, TopK: {top_k})", 2)

    def get_exact_matches():
        exact_results = []
        for item in metadata: # Iterate through metadata which is smaller
            title = item["title"]
            chapter = item["chapter"]
            norm_title = normalize_title(title)
            # Simple substring match in normalized title
            if norm_query in norm_title:
                 norm_key = (chapter, norm_title)
                 content = normalized_kb.get(norm_key)
                 if content:
                     # Check if this exact content (by key) has already been added
                     if norm_key not in seen_titles:
                         debug_print(f"Found exact/substring match: {title}", 3)
                         seen_titles.add(norm_key)
                         exact_results.append({
                             "title_key": title,
                             "chapter": chapter,
                             "score": 0.0, # Assign 0 score for exact match
                             "content": content
                         })
                         # For exact matches, sometimes you only want the very first one
                         # If you only want ONE exact match, uncomment the next line:
                         # return exact_results
        return exact_results


    def get_semantic_matches():
        semantic_results = []
        try:
            query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
            # FAISS returns L2 distance, convert to similarity if needed, but distance is fine for thresholding
            distances, indices = index.search(query_embedding, top_k)

            for i in range(len(indices[0])):
                idx = indices[0][i]
                # Skip invalid indices if top_k is larger than index size
                if idx == -1: continue

                raw_title = metadata[idx]["title"]
                chapter = metadata[idx]["chapter"]
                norm_key = (chapter, normalize_title(raw_title))
                content = normalized_kb.get(norm_key)
                score = distances[0][i] # This is the L2 distance

                # Only consider if content exists in KB and hasn't been added by exact match
                if content and norm_key not in seen_titles:
                    # Optional: Semantic de-duplication based on content embedding similarity
                    # (This part from original code was complex; simplified here)
                    # A more robust approach would require indexing chunks of content
                    # For simplicity with title-level index, we rely on score and seen_titles
                    debug_print(f"Found semantic match candidate: {raw_title} (Score: {score:.4f})", 3)
                    seen_titles.add(norm_key) # Mark as seen to avoid adding same title multiple times
                    semantic_results.append({
                        "title_key": raw_title,
                        "chapter": chapter,
                        "score": score,
                        "content": content
                    })
        except Exception as e:
             print(f"❌ Error during semantic search: {e}")
             # Return empty semantic results if an error occurs
             return []

        return semantic_results

    # --- MODE HANDLING ---
    if mode == "exact":
        results = get_exact_matches()
    elif mode == "semantic":
        results = get_semantic_matches()
    else:  # hybrid
        # Prioritize exact matches
        results = get_exact_matches()
        # If no exact matches found, or if you want to blend, add semantic matches
        if not results or mode == "hybrid_blend": # Added hybrid_blend idea
             semantic_results = get_semantic_matches()
             # Merge and sort if needed, or just append if exact matches are prioritized
             if not results: # Only use semantic if no exact matches were found
                 results = semantic_results
             # If blending is desired, you'd merge results here, remove duplicates, and sort
             # For simplicity, we'll stick to 'exact first, then semantic if no exact'
             pass # Current logic already does this

    debug_print(f"Search returned {len(results)} results.", 2)
    return results

# Function to generate a brief, general explanation (no KB search)
# This is used for out-of-syllabus topics determined by is_in_syllabus
def generate_general_explanation(topic):
    """Generates a brief, general explanation for a topic using the LLM (no KB search)."""
    if not LLM_API_KEY or LLM_API_KEY == "YOUR_GROQ_API_KEY":
         print("❌ LLM_API_KEY is not set. Cannot generate general explanation.")
         return f"I can't give a general explanation of {topic} right now because my external knowledge service is not configured."

    debug_print(f"Generating general explanation for: {topic}", 2)
    prompt = f"""
You are an AI science teacher providing a brief overview of a topic that is outside the current syllabus.
Provide a clear, concise explanation of "{topic}" suitable for an 8th grader.
Keep it brief, ideally 3-5 sentences.
DO NOT mention specific textbook chapters, sections, or figures.
DO NOT ask questions back.
Output plain text only.

Your General Explanation:
"""
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192", # Or your preferred model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200, # Limit tokens for brevity
                "temperature": 0.7 # A bit creative but still factual
            }
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        if "choices" in result and result["choices"]:
            explanation_text = result["choices"][0]["message"]["content"].strip()
            debug_print("LLM generated general explanation.", 3)
            return explanation_text
        else:
            debug_print(f"LLM response format issue for general explanation: {result}", 3)
            return f"I can't give a detailed explanation of {topic} right now."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLM for general explanation: {e}")
        return "Sorry, I encountered an error while trying to explain."
    except Exception as e:
        print(f"❌ Unexpected error during general explanation generation: {e}")
        return "An unexpected error occurred."

# Check if topic is "in syllabus" (based on search results and relevance threshold)
def is_in_syllabus(query):
    """Checks if the query yields sufficiently relevant results from the knowledge base."""
    debug_print(f"Checking if '{query}' is in syllabus (with threshold)...", 2)

    # You will need to tune this threshold value based on your data.
    # A lower distance_threshold means the match must be more similar (closer).
    # This threshold applies to the L2 distance returned by FAISS for semantic search.
    # Exact matches (score 0.0) will always be considered in syllabus.
    distance_threshold = 0.8 # Example distance threshold - lower is better. Needs tuning.

    # Call the search function to get the best match and its score (distance)
    # Use top_k=1 and hybrid mode to get the single best match (exact or semantic)
    search_results = search(query, mode="hybrid", top_k=1)

    if search_results: # Check if search returned any result
        best_match = search_results[0]
        best_score = best_match["score"] # This is the distance (0.0 for exact match)
        syllabus_content = best_match["content"]
        cleaned_title = re.sub(r"^\d+(\.\d+)\s", "", best_match["title_key"]).strip()

        debug_print(f"Best match found: '{best_match['title_key']}' (Score: {best_score:.4f}) (Threshold: {distance_threshold})", 2)

        # Check if it's an exact match (score is 0.0) or the distance is below the threshold
        if best_score == 0.0 or best_score < distance_threshold:
            debug_print("Found sufficiently relevant syllabus content.", 2)
            # Return True, the content, and the cleaned topic title
            return True, syllabus_content, cleaned_title
        else:
            debug_print("Closest match score is above the distance threshold. Not considered in syllabus.", 2)
            return False, None, None
    else:
        debug_print("No results found by search function.", 2) # Should rarely happen with top_k=1 unless index is empty
        return False, None, None


# Groq LLM API configuration (already done above with user config)
# LLM_API_KEY = "..."
# LLM_API_URL = "..."

# Load figures data (Keep)
try:
    figures_data = json.load(open(FIGURES_JSON, "r", encoding="utf-8"))
    debug_print(f"Loaded Figures data with {len(figures_data)} entries.", 2)
except FileNotFoundError:
    debug_print(f"Warning: Figures JSON file '{FIGURES_JSON}' not found. Figure fetching will not work.", 2)
    figures_data = [] # Set to empty list if file is missing
except json.JSONDecodeError as e:
    print(f"❌ Error loading figures JSON '{FIGURES_JSON}': {e}. Figure fetching may not work.")
    figures_data = []


# Initialize image model & FAISS (figures) - Keep if still needed for image search
# Note: The current code doesn't seem to *use* the image FAISS index directly in the voice flow,
# but keeping the loading logic if it's part of the larger project.
try:
    # Reuse the existing model if it's the same type, otherwise load a new one
    # Assuming image_model is same as model for simplicity based on original code
    image_model = model # Reuse the already loaded text model
    debug_print("Reusing text SentenceTransformer model for image search (assuming same).", 2)
except Exception as e:
    print(f"❌ Critical Error: Could not prepare image model: {e}")
    # Decide if this is critical or if the voice agent can run without image search
    # For this voice-only context, maybe not critical.

FAISS_INDEX_FILE_FIGURES = "subchapter_faiss.index" # Renamed to avoid conflict
METADATA_FILE_FIGURES = "subchapter_metadata.json" # Renamed to avoid conflict

# Load figure metadata and index (Keep if still needed)
index_figures = None
metadata_figures = {}
try:
    if os.path.exists(FAISS_INDEX_FILE_FIGURES) and os.path.exists(METADATA_FILE_FIGURES):
        index_figures = faiss.read_index(FAISS_INDEX_FILE_FIGURES)
        with open(METADATA_FILE_FIGURES, "r", encoding="utf-8") as f:
            metadata_figures = json.load(f)
        debug_print("FAISS figures index and metadata loaded.", 2)
    else:
        debug_print(f"Warning: Figures index '{FAISS_INDEX_FILE_FIGURES}' or metadata '{METADATA_FILE_FIGURES}' not found. Figure search will not work.", 2)
except Exception as e:
    print(f"❌ Error loading figure FAISS index or metadata: {e}. Figure search may not work.")
    index_figures = None # Ensure index is None if loading fails


# search_exact_subchapter function (Keep if needed, though role in voice flow unclear)
def search_exact_subchapter(query, top_k=1):
    """Find the most relevant subchapter using FAISS (for figures)."""
    if index_figures is None or image_model is None:
        debug_print("Figure search index or model not available.", 2)
        return None

    debug_print(f"Searching for exact subchapter match (for figures): {query}", 2)
    try:
        query_embedding = image_model.encode([query], convert_to_numpy=True).astype('float32')
        # Reshape for a single query
        distances, indices = index_figures.search(query_embedding.reshape(1, -1), top_k)
        # Pick only the closest match
        best_match_index_str = str(indices[0][0]) # Index from FAISS is int, metadata key is string
        best_subchapter = metadata_figures.get(best_match_index_str, None)
        debug_print(f"Best match subchapter for figures: {best_subchapter}", 3)
        return best_subchapter
    except Exception as e:
        print(f"❌ Error during figure subchapter search: {e}")
        return None


# get_image_path function (Keep if needed)
def get_image_path(figure_ref):
    """Find image path with multiple fallback patterns."""
    if not IMAGE_DIR or not os.path.exists(IMAGE_DIR):
        debug_print(f"Image directory '{IMAGE_DIR}' not found. Cannot get image path.", 2)
        return None

    debug_print(f"Locating image for: {figure_ref}", 2)
    base_name = figure_ref.replace(" ", "_").replace(":", "_").replace("/", "_") # Sanitize figure_ref for filenames
    attempts = [
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"figure_{base_name}.png",
        f"figure_{base_name}.jpg"
    ]
    for attempt in attempts:
        test_path = os.path.join(IMAGE_DIR, attempt)
        if os.path.exists(test_path):
            debug_print(f"✅ Found image at: {test_path}", 3)
            return test_path
    debug_print("❌ No valid image path found", 3)
    return None


# fetch_figures_only function (Keep if needed)
def fetch_figures_only(subchapter_name): # Changed parameter name to be more explicit
    """Retrieve only figures (images + raw descriptions) for a given subchapter."""
    if not figures_data:
         debug_print("Figures data not loaded. Cannot retrieve figures.", 2)
         return "Figure data is not available."

    debug_print(f"Retrieving figures for subchapter: {subchapter_name}", 2)
    figures = [fig for fig in figures_data if fig.get("subchapter") == subchapter_name] # Use .get for safety
    if not figures:
        debug_print(f"No relevant figures found for subchapter: {subchapter_name}", 2)
        return "No relevant figures found."

    figure_blocks = []
    for fig in figures:
        fig_path = get_image_path(fig.get('figure')) # Use .get for safety
        if fig_path:
            figure_blocks.append({
                "name": fig.get('figure', 'N/A'), # Use .get for safety
                "path": fig_path,
                "desc": fig.get('description', '') # Use .get for safety, default to empty string
            })
    debug_print(f"Found {len(figure_blocks)} figures with paths.", 2)
    return figure_blocks

# retrieve_and_expand_figures function (generates HTML, less relevant for voice - Keep but likely not used in voice loop)
def retrieve_and_expand_figures(query):
    """
    Retrieve figures related to the query by using the title of the
    most relevant text content and generate HTML to display them.
    """
    debug_print(f"Retrieving and formatting figures for query: {query}", 2)
    # Find the most relevant text content first to get a potential subchapter name
    search_results = search(query, mode="hybrid", top_k=1)
    if not search_results:
        debug_print("No relevant text found for image retrieval.", 2)
        return "<p>No relevant text found for image retrieval.</p>"

    best_text_match = search_results[0]
    subchapter_name = best_text_match["title_key"] # Use the title_key as the subchapter name

    blocks = fetch_figures_only(subchapter_name)

    if isinstance(blocks, str):
        # An error message was returned by fetch_figures_only
        debug_print(f"fetch_figures_only returned an error: {blocks}", 2)
        return f"<p>{blocks}</p>"

    if not blocks:
         debug_print("No figure blocks returned after fetch.", 2)
         return "<p>No relevant figures found.</p>"

    figure_html = "<div style='margin-top: 20px;'><h3>📊 Visual Aids</h3>"
    # Limit to 3 figures for display in a web/HTML context
    for fig in blocks[:3]:
        # Optional: you might want to generate a short spoken description here for the voice agent
        clean_desc = fig['desc'] # Optionally, you can process the description further for HTML
        figure_html += f"""
        <div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>
            <img src='{fig['path']}' style='max-width: 100%; height: auto; display: block; margin: 0 auto;'>
            <p style='text-align: center; font-style: italic;'>{clean_desc or 'Visual demonstration'}</p>
        </div>
        """
    figure_html += "</div>"
    debug_print("Generated figure HTML.", 2)
    return figure_html

# fetch_animated_videos function (keep for potential future use in voice flow)
def fetch_animated_videos(topic, num_videos=1):
    """Searches YouTube for short educational animations related to the topic."""
    search_query = f"ytsearch{num_videos}:{topic} animation explained in english"
    debug_print(f"Searching YouTube for: {search_query}", 2)

    ydl_opts = {
        "quiet": True,
        "extract_flat": True, # Get minimal info quickly
        "force_generic_extractor": True, # Avoids issues with some URLs
        "skip_download": True, # Never download
        "geo_bypass": True, # Attempt to bypass geo-restrictions
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)

        if info and "entries" in info and len(info["entries"]) > 0:
             # Filter for short videos
            short_videos = [
                video for video in info["entries"]
                if video.get("duration", 301) is not None and video.get("duration", 301) <= 300 # Check if duration exists and is <= 300s
            ]
            if short_videos:
                video = short_videos[0] # Pick the first short one
                debug_print(f"Found relevant short video: {video.get('title', 'Untitled')}", 2)
                return {
                    "title": video.get("title", "Untitled Video"),
                    "url": video.get("url", f"https://www.youtube.com/watch?v={video.get('id')}"), # Construct URL if missing
                    "id": video.get("id")
                }
            else:
                 debug_print("No short videos found matching the search criteria.", 2)

        debug_print("No video entries found.", 2)
        return None

    except Exception as e:
        print(f"❌ Error fetching animated videos: {e}")
        return None


# generate_topic_hook function (Keep for potential use in spoken explanation intro)
def generate_topic_hook(topic):
    """Generate a short, engaging hook for the topic using the LLM."""
    if not LLM_API_KEY or LLM_API_KEY == "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":
         print("❌ LLM_API_KEY is not set. Cannot generate topic hook.")
         return f"Let's dive into {topic}!" # Simple fallback

    debug_print(f"Generating topic hook for: {topic}", 2)
    prompt = f"""
You are a science educator. Create a SHORT (1-2 sentences), engaging hook for the topic *{topic}* for 8th-grade students using one of these techniques:
- A surprising fact/question
- A relatable analogy/metaphor
- A real-world application
- A mini thought experiment

Return ONLY the hook.
"""
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192", # Or your preferred model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100, # Sufficient for 1-2 sentences
                "temperature": 0.9 # High temp for creativity
            }
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"]:
            hook = result["choices"][0]["message"]["content"].strip()
            debug_print("LLM generated topic hook.", 3)
            return hook
        else:
             debug_print(f"LLM response format issue for hook: {result}", 3)
             return f"Let's learn about {topic}!" # Simple fallback
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLM for topic hook: {e}")
        return f"Ready to explore {topic}?" # Error fallback
    except Exception as e:
        print(f"❌ Unexpected error during hook generation: {e}")
        return f"Discover {topic} with me!" # Error fallback

# generate_funny_intro function (Keep for potential use)
def generate_funny_intro(topic):
    """Generate an introduction that begins with a funny story or meme about the topic using the LLM."""
    if not LLM_API_KEY or LLM_API_KEY == "YOUR_GROQ_API_KEY":
         print("❌ LLM_API_KEY is not set. Cannot generate funny intro.")
         return "Let's start with a little something about science." # Simple fallback

    debug_print(f"Generating funny intro for: {topic}", 2)
    prompt = f"""
You are a creative and humorous science educator. Tell a short, funny story or describe a relatable meme about *{topic}* to engage 8th-grade students. Avoid using video introductions. Return ONLY the story.
"""
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192", # Or your preferred model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200, # Allow a short story
                "temperature": 0.9 # High temp for humor
            }
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"]:
            funny_intro = result["choices"][0]["message"]["content"].strip()
            debug_print("LLM generated funny intro.", 3)
            return funny_intro
        else:
             debug_print(f"LLM response format issue for funny intro: {result}", 3)
             return "Here's a fun fact to get started." # Simple fallback
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLM for funny intro: {e}")
        return "Let's kick off our lesson with something fun!" # Error fallback
    except Exception as e:
        print(f"❌ Unexpected error during funny intro generation: {e}")
        return "Prepare for some science fun!" # Error fallback


# generate_dynamic_intro function (generates HTML, less relevant for voice - Keep but likely not used)
def generate_dynamic_intro(topic):
    """Generate an introductory paragraph with a funny story or meme (for HTML display)."""
    funny_intro = generate_funny_intro(topic)
    hook = generate_topic_hook(topic)
    return f"""
<p>{funny_intro}</p>
<p>{hook}</p>
<p>Today, we're exploring the fascinating world of <strong>{topic}</strong>! 🔍<br>
Quick prediction: What do you think happens when...? Let's find out in our lesson!</p>
"""

# --- NEW VOICE AGENT CORE FUNCTIONS ---

# 1. Text-to-Speech (TTS) Function (Handled within speak now)

# 2. Speech-to-Text (ASR) Function
def listen_to_student(timeout_seconds=8, phrase_time_limit_seconds=5): # Adjusted timeout slightly
    """Listens for student's speech using the microphone."""
    r = sr.Recognizer()
    r.energy_threshold = 4000 # Adjust if needed, higher means less sensitive to quiet sounds
    r.dynamic_energy_threshold = True # Recommended by speech_recognition docs
    # r.pause_threshold = 0.8 # Seconds of non-speaking audio before a phrase is considered complete
    # r.operation_timeout = 5 # Max seconds the recognizer will wait for a phrase to start

    with sr.Microphone() as source:
        # Adjust for ambient noise - Do this once at the start of the application if possible
        # or if the environment changes significantly. Doing it repeatedly can cause issues.
        # r.adjust_for_ambient_noise(source, duration=1)
        # debug_print(f"Adjusted for ambient noise. Energy threshold: {r.energy_threshold}", 3)

        debug_print("Listening for student...", 2)
        # Ensure mouth is in listening/idle pose
        set_mouth_pose("X") # Set mouth to Rest/Idle

        print("🎤 (Speak now)") # User feedback
        try:
            # Listen with timeout and phrase time limit
            audio = r.listen(source, timeout=timeout_seconds, phrase_time_limit=phrase_time_limit_seconds)
            debug_print("Processing speech...", 2)
            # Ensure mouth goes back to idle/processing pose while thinking
            set_mouth_pose("X") # Set mouth to Rest/Idle while processing

            # Recognize speech using Google Web Speech API
            text = r.recognize_google(audio)
            debug_print(f"Student said: {text}", 2)
            print(f"👂 You said: {text}") # User feedback
            return text.lower() # Return lowercase for easier processing
        except sr.WaitTimeoutError:
            debug_print("No speech detected within timeout.", 2)
            print("묵 No speech detected.")
            return None
        except sr.UnknownValueError:
            debug_print("Google Speech Recognition could not understand audio.", 2)
            print("❓ Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            debug_print(f"Could not request results from Google Speech Recognition service; {e}", 2)
            print(f"🚫 Could not connect to speech service: {e}")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred during listening: {e}")
            return None
        finally:
             # Ensure mouth is reset after listening attempt finishes
             set_mouth_pose("X")


# 3. Natural Language Understanding (NLU) Function (using LLM)
def understand_intent_and_topic(text):
    """Uses the LLM to determine intent and extract the topic from student input."""
    if not text:
        debug_print("NLU received empty text.", 2)
        return "NO_INPUT", None # Explicitly handle empty input

    if not LLM_API_KEY or LLM_API_KEY == "YOUR_GROQ_API_KEY":
         print("❌ LLM_API_KEY is not set. Cannot perform NLU.")
         return "ERROR", None

    debug_print(f"Sending NLU request to LLM for text: '{text}'", 2)
    prompt = f"""
Analyze the following student input to identify the primary intent and the specific topic or question.
Input: "{text}"

Possible Intents:
- REQUEST_EXPLANATION (e.g., "tell me about X", "explain Y")
- ASK_QUESTION (e.g., "what is X?", "how does Y work?", "why is Z like that?")
- GREETING (e.g., "hello", "hi")
- FAREWELL (e.g., "goodbye", "bye")
- RESUME (e.g., "resume", "continue", "go on")
- OTHER (If none of the above fit)
- REPHRASE (e.g., "say that again", "repeat") # Added REPHRASE intent

Identify the main intent and extract the core topic or question discussed.

Return the result ONLY as a JSON object with keys "intent" and "topic_or_question".
If no specific topic is mentioned for explanation or question, set "topic_or_question" to null.
If the intent is GREETING, FAREWELL, RESUME, REPHRASE, or OTHER, set "topic_or_question" to null.

Example 1:
Input: "Explain magnetic fields"
Output: {{"intent": "REQUEST_EXPLANATION", "topic_or_question": "magnetic fields"}}

Example 2:
Input: "hi there"
Output: {{"intent": "GREETING", "topic_or_question": null}}

Example 3:
Input: "why is the sky blue?"
Output: {{"intent": "ASK_QUESTION", "topic_or_question": "why is the sky blue?"}}

Example 4:
Input: "tell me more"
Output: {{"intent": "REQUEST_EXPLANATION", "topic_or_question": null}} # No specific topic

Example 5:
Input: "resume"
Output: {{"intent": "RESUME", "topic_or_question": null}}

Example 6:
Input: "repeat that"
Output: {{"intent": "REPHRASE", "topic_or_question": null}}

Input to analyze: "{text}"
Output:
"""
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192", # Or your preferred model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100, # Should be plenty for the JSON
                "temperature": 0.0, # Use 0.0 for deterministic classification
                "response_format": {"type": "json_object"} # Request JSON output if API supports
            }
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        # Extract JSON content - handle potential variations in LLM output
        json_string = result["choices"][0]["message"]["content"].strip()
        # Sometimes LLMs wrap JSON in backticks or ```json ... ```
        json_string = re.sub(r"^```json\s*", "", json_string)
        json_string = re.sub(r"\s*```$", "", json_string)

        debug_print(f"NLU Raw Response: {json_string}", 3)

        parsed_result = json.loads(json_string)
        intent = parsed_result.get("intent")
        topic = parsed_result.get("topic_or_question")
        debug_print(f"NLU Parsed: Intent={intent}, Topic={topic}", 2)
        return intent, topic
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLM for NLU: {e}")
        return "ERROR", None
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding NLU JSON response: {e}")
        print(f"Raw response was: {json_string}") # Log the problematic response
        return "ERROR", None
    except Exception as e:
        print(f"❌ Unexpected error during NLU: {e}")
        return "ERROR", None


# 4. Function to Generate Spoken Explanation (Adapting existing logic)
# This function decides *what content* to generate based on syllabus check
def generate_spoken_explanation(topic_query):
     """
     Generates the text for a spoken explanation based on syllabus check.
     Decides between detailed KB-based explanation or general LLM explanation.
     Returns the explanation text and the cleaned topic title (if found in syllabus).
     """
     debug_print(f"Preparing spoken explanation for topic query: {topic_query}", 2)

     # Check if the topic is in the syllabus using the refined function
     in_syllabus, syllabus_content, cleaned_title = is_in_syllabus(topic_query)

     if not in_syllabus:
         debug_print(f"Topic '{topic_query}' not found in syllabus based on threshold.", 2)
         # Generate a general explanation using the original query
         explanation_text = generate_general_explanation(topic_query)
         # We don't have a syllabus title for out-of-syllabus topics
         return explanation_text, None, False
     else:
         debug_print(f"Topic '{topic_query}' found in syllabus as '{cleaned_title}'.", 2)
         # Generate the full, rich lesson text based on the syllabus content
         # The generate_full_lesson_text function handles adding intro/hook etc.
         explanation_text = generate_full_lesson_text(topic_query) # Pass the original query or cleaned_title? Let's pass original query, generate_full_lesson_text will re-search/confirm
         # Pass the cleaned title back for context saving
         return explanation_text, cleaned_title, True


# Function to generate the full, rich lesson text for voice (from previous turn)
# This function is now called *after* is_in_syllabus check confirms content exists
def generate_full_lesson_text(topic_query):
    """
    Retrieves relevant content (already confirmed by is_in_syllabus),
    generates intro/hook, and uses LLM to create a rich, expanded explanation
    based on textbook content for voice output.
    """
    if not LLM_API_KEY or LLM_API_KEY == "YOUR_GROQ_API_KEY":
         print("❌ LLM_API_KEY is not set. Cannot generate full lesson text.")
         return "I cannot generate a detailed lesson right now."


    debug_print(f"Generating full lesson text for syllabus topic: {topic_query}", 2)

    # Re-run search *just to get the content and cleaned title reliably* now that we know it's in syllabus
    # This avoids passing large content strings around unnecessarily if called from generate_spoken_explanation
    search_results = search(topic_query, mode="hybrid", top_k=1)
    if not search_results:
         # This case should theoretically not happen if called after a successful is_in_syllabus
         debug_print("Error: generate_full_lesson_text called but search found no content.", 1)
         return "Sorry, I had trouble retrieving the lesson details."

    best_match = search_results[0]
    retrieved_content = best_match["content"]
    cleaned_title = re.sub(r"^\d+(\.\d+)\s", "", best_match["title_key"]).strip()
    debug_print(f"Using retrieved content for lesson '{cleaned_title}'", 2)

    # 2. Generate Intro and Hook (re-using existing functions)
    funny_intro = generate_funny_intro(cleaned_title)
    hook = generate_topic_hook(cleaned_title)

    # 3. Combine Intro, Hook, and Textbook Content for LLM Prompt
    # Create a prompt that guides the LLM to act as a teacher and expand
    prompt_text = f"""
You are an engaging, fun-loving, and knowledgeable 8th-grade science teacher speaking directly to a student.

Start with an engaging introduction similar to this (do not include explicit labels like 'Funny Story' or 'Hook'):
"{funny_intro}"
"{hook}"
"Today, we're exploring the fascinating world of {cleaned_title}!"

Then, based only on the following textbook content, provide a detailed, smooth, and engaging explanation for voice:
- Expand each idea from the textbook content with real-life analogies, fun facts, surprising trivia, and interesting stories kids can relate to.
- Break down complex terms into simple, visual language.
- Ensure smooth transitions between different points.
- DO NOT include HTML tags. Output plain text only.
- DO NOT include explicit section headers like "Introduction", "Explanation", etc. just make it flow naturally.
- DO NOT include greetings or sign-offs at the very beginning or end of the overall response.

Textbook Content:
"{retrieved_content}"

Your Spoken Explanation:
"""
    debug_print("Sending LLM request for full lesson explanation...", 2)
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192", # Use a capable model
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 1500, # Allow for a longer explanation
                "temperature": 0.8 # Use a higher temp for more creativity/engagement
            }
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"]:
            explanation_text = result["choices"][0]["message"]["content"].strip()
            debug_print("LLM generated full lesson text.", 3)
            return explanation_text
        else:
            debug_print(f"LLM response format issue for lesson: {result}", 3)
            return f"I found information on {cleaned_title}, but had trouble creating a detailed lesson right now."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLM for lesson generation: {e}")
        return "Sorry, I encountered an error while preparing the lesson explanation."
    except Exception as e:
        print(f"❌ Unexpected error during lesson generation: {e}")
        return "An unexpected error occurred while creating the lesson."


# 5. Function to Generate Spoken Answer (for QA)
def generate_spoken_answer(question, context_content=None):
    """
    Generates a concise spoken answer (ideally 2-4 sentences) to a student's question
    using the LLM, optionally using retrieved context.
    """
    if not LLM_API_KEY or LLM_API_KEY == "YOUR_GROQ_API_KEY":
         print("❌ LLM_API_KEY is not set. Cannot generate spoken answer.")
         return "I cannot answer questions right now."

    debug_print(f"Generating concise spoken answer for question: '{question}'", 2)

    prompt = f"""
You are an AI science teacher answering a student's question. Provide a clear, concise answer suitable for an 8th grader.
The answer MUST be brief, ideally *2 to 4 sentences long*.
DO NOT ask questions back.
Output plain text only.

Student's Question: "{question}"

Context from knowledge base (if available and deemed relevant by search):
"{context_content if context_content else 'No specific syllabus context found for this question. Answer from general knowledge.'}"

Your Concise Spoken Answer (2-4 sentences):
"""
    debug_print("Sending LLM request for concise answer...", 2)
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192", # Or your preferred model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150, # Limit tokens to encourage brevity (adjust if needed)
                "temperature": 0.6 # Keep it factual
            }
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        if "choices" in result and result["choices"]:
            answer_text = result["choices"][0]["message"]["content"].strip()
            # Simple check to try and enforce sentence count (LLM might ignore) - This is a fallback
            # Use NLTK for more robust sentence splitting
            sentences = split_into_sentences(answer_text)
            if len(sentences) > 4:
                debug_print(f"Answer text generated {len(sentences)} sentences, truncating to 4.", 3)
                answer_text = ". ".join(sentences[:4]).strip() + ("..." if len(sentences) > 4 else "") # Join first 4 and add ellipsis if more existed
            else:
                answer_text = ". ".join(sentences).strip() # Join back just in case splitting changed anything

            debug_print(f"LLM generated spoken answer ({len(split_into_sentences(answer_text))} sentences approx).", 3)

            return answer_text
        else:
            debug_print(f"LLM response format issue for answer: {result}", 3)
            return "I'm having a bit of trouble formulating an answer right now."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLM for answer generation: {e}")
        return "Sorry, I encountered an error while trying to answer."
    except Exception as e:
        print(f"❌ Unexpected error during answer generation: {e}")
        return "An unexpected error occurred while answering."
STATE_IDLE = "IDLE" # Waiting for initial input, or after completing a task
STATE_SPEAKING_EXPLANATION = "SPEAKING_EXPLANATION" # AI is speaking a full explanation
STATE_SPEAKING_ANSWER = "SPEAKING_ANSWER"           # AI is speaking a concise answer
STATE_SPEAKING_GREETING = "SPEAKING_GREETING"       # AI is speaking a greeting/farewell
STATE_SPEAKING_TRANSITION = "SPEAKING_TRANSITION" # AI is speaking a transition phrase ("Okay, let's learn...")
STATE_LISTENING = "LISTENING" # AI is actively listening for student input
# STATE_PROCESSING = "PROCESSING" # AI is processing input (NLU, search, LLM calls) - no audio/listening
STATE_HANDLING_INTERRUPTION = "HANDLING_INTERRUPTION" # AI detected interruption, asking "Yes?" or processing input
STATE_CHECK_RESUME = "CHECK_RESUME" # AI finished handling interrupt, checking if explanation should resume
# -- Global State Variables --
current_state = STATE_IDLE # Starting state
interrupt_flag = threading.Event() # Event to signal interruption across threads
audio_thread = None # Not strictly needed with current pygame approach, but could be useful later
interrupt_queue = queue.Queue() # Queue for communication from keyboard listener to main loop
tts_finished_event = threading.Event() # Event signaled by speak() when audio finishes normally

# --- Variables for Storing Interruption Context ---
interrupted_context = {
    "type": None, # 'explanation'
    "topic": None, # Topic of the interrupted explanation
    "full_text": None, # Full text of the interrupted explanation
    "sentences": [], # Explanation text split into sentences
    "resume_index": 0, # Index of the sentence to resume from
    "saved": False # Flag to indicate if context is actively saved for resume
}

# -- Pygame Mixer Initialization (Keep as is) --
# Initialized later in __main__

# --- NEW: Pygame Display Setup ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
# screen = None # Initialized later in __main__
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# font = None # Initialized later in __main__

# --- NEW: Load Mouth Shape Images ---
MOUTH_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50) # Position for the mouth
mouth_images = {}
# Map Rhubarb codes to your image files (adjust filenames as needed)
# This mapping ensures you only load images for the shapes you actually use.
# 'A', 'D' might map to 'B' (open), 'I', 'L', 'N', 'R', 'S', 'Th', 'U', 'W', 'Y', 'Z' might map to 'E' (mid open) or 'X'
rhubarb_map = {
    "A": "B", # Rhubarb's A often maps to an open mouth (like B)
    "B": "B", # Open like "ah"
    "C": "C", # Wide like "ee"
    "D": "E", # Less open variation, map to E
    "E": "E", # Less open like "uh"
    "F": "F", # Teeth-Lip like "f", "v"
    "G": "G", # Puckered like "oo"
    "H": "H", # Lips together like "m", "b", "p"
    "X": "X", # Rest/Closed

}

# Placeholder for missing image (a black rectangle)
missing_mouth_image = None
MOUTH_IMAGE_FOLDER = "RoboMouths_PNGs"
def load_mouth_images():
    """Loads mouth shape images into the mouth_images dictionary from a specified folder."""
    global mouth_images, missing_mouth_image # Declare as global to modify

    debug_print(f"Loading mouth images from folder: {MOUTH_IMAGE_FOLDER}...", 2)
    mouth_images = {} # Clear any previous loads

    # Ensure the mouth image folder exists
    if not os.path.exists(MOUTH_IMAGE_FOLDER):
        print(f"❌ Critical Error: Mouth image folder '{MOUTH_IMAGE_FOLDER}' not found.")
        # Create a fallback image and return only that if the folder is missing
        missing_mouth_image = pygame.Surface((int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.05)))
        missing_mouth_image.fill(BLACK)
        mouth_images["X"] = missing_mouth_image # Ensure at least a rest pose fallback
        print("    Using fallback image for all mouth shapes.")
        return mouth_images.get("X") # Return the fallback image

    # Ensure a fallback image exists before trying to load others
    missing_mouth_image = pygame.Surface((int(SCREEN_WIDTH * 0.15), int(SCREEN_HEIGHT * 0.05))) # Example size
    missing_mouth_image.fill(BLACK)

    try:
        # Load images based on the *mapped* codes we'll use from the rhubarb_map values
        mouth_shapes_to_load = set(rhubarb_map.values())

        for shape_code in mouth_shapes_to_load:
            # --- Construct the full file path using os.path.join ---
            filename = f"mouth_{shape_code}.png"
            full_file_path = os.path.join(MOUTH_IMAGE_FOLDER, filename)

            try:
                # Check if the file exists at the full path
                if not os.path.exists(full_file_path):
                    debug_print(f"Mouth image file not found: {full_file_path}. Using fallback.", 2)
                    mouth_images[shape_code] = missing_mouth_image # Use fallback if file is missing
                    continue # Skip to next shape code

                img = pygame.image.load(full_file_path).convert_alpha()
                # Optional: Scale images if needed to fit your design better
                # img = pygame.transform.scale(img, (120, 60)) # Example scale

                mouth_images[shape_code] = img
                debug_print(f"Loaded mouth image: {full_file_path}", 3)
            except pygame.error as e:
                # Catch specific pygame loading errors
                print(f"⚠️ Warning: Could not load mouth image {full_file_path} using Pygame: {e}. Using fallback.")
                mouth_images[shape_code] = missing_mouth_image # Use fallback on pygame error
            except Exception as e:
                # Catch any other unexpected errors during loading loop
                print(f"⚠️ Warning: Unexpected error loading mouth image {full_file_path}: {e}. Using fallback.")
                mouth_images[shape_code] = missing_mouth_image # Use fallback on other errors


        # --- Ensure 'X' (Rest) and 'B' (Open) have images, even if fallbacks ---
        # If 'X' somehow wasn't in rhubarb_map values, make sure it's added
        if "X" not in mouth_images:
            debug_print("Adding fallback for 'X' (Rest) shape.", 2)
            mouth_images["X"] = missing_mouth_image
        # If 'B' somehow wasn't in rhubarb_map values, make sure it's added (useful for 'A' mapping)
        if "B" not in mouth_images:
            debug_print("Adding fallback for 'B' (Open) shape.", 2)
            mouth_images["B"] = missing_mouth_image


    except Exception as e:
        # Catch errors outside the inner loop (less likely)
        print(f"❌ Error during initial mouth image loading loop setup: {e}")
        # Ensure at least a rest pose fallback exists
        if "X" not in mouth_images:
            mouth_images["X"] = missing_mouth_image
            print("    Assigned fallback for Rest pose.")

    debug_print(f"Finished mouth image loading. Loaded {len(mouth_images)} shapes.", 2)
    # Store the initial mouth image to display (default to Rest)
    return mouth_images.get("X") # Return the initial image



# -- Keyboard Listener Function (Keep as is) --
INTERRUPT_KEY = 'space'
def keyboard_listener():
    """Listens for the interrupt key press and signals interruption."""
    # This function runs in a separate thread
    debug_print(f"Keyboard listener thread started. Listening for '{INTERRUPT_KEY.upper()}'...", 2)
    try:
        def on_key_press(event):
            # Only trigger on key *down* event to avoid multiple triggers while held
            if event.name == INTERRUPT_KEY and event.event_type == keyboard.KEY_DOWN:
                # Check if an interruption is not already being actively handled
                # This prevents spamming the queue/flag
                if not interrupt_flag.is_set():
                    debug_print(f"Interrupt key '{INTERRUPT_KEY}' pressed! Signaling interruption.", 1)
                    # Use a queue to safely communicate with the main thread
                    interrupt_queue.put("INTERRUPT")
                    # Set the flag. The main thread clears it after handling.
                    interrupt_flag.set()

        # Hook the keyboard events globally
        keyboard.hook(on_key_press)
        # Keep the thread alive
        keyboard.wait() # This blocks the thread indefinitely until keyboard.unhook_all() or program exit

    except Exception as e:
        # Handle potential errors in the keyboard hook
        print(f"❌ Error in keyboard listener thread: {e}")
    finally:
        debug_print("Keyboard listener thread exiting.", 2)


# -- Modified stop_speech Function (Keep as is) --
def stop_speech():
    """Stops audio playback via pygame mixer and unloads."""
    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
        debug_print("Stopping speech via pygame mixer...", 2)
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload() # Unload the audio file
            debug_print("Speech stopped and audio unloaded.", 2)
        except pygame.error as e:
            print(f"⚠️ Warning: Pygame error during stop_speech: {e}")
        except Exception as e:
            print(f"❌ Unexpected error during stop_speech: {e}")
    else:
        debug_print("Pygame mixer not busy or not initialized, nothing to stop.", 2)

# -- Sentence Splitting Function --
def split_into_sentences(text):
    """Splits text into sentences using NLTK."""
    if not text:
        return []

    try:
        # Ensure punkt tokenizer is downloaded - this check is better done once at startup
        # try:
        #     nltk.data.find('tokenizers/punkt')
        # except nltk.downloader.DownloadError:
        #     debug_print("NLTK 'punkt' tokenizer not found. Downloading...", 2)
        #     nltk.download('punkt')
        #     debug_print("NLTK 'punkt' tokenizer downloaded.", 2)

        # Simple replacements to help NLTK, especially after lists, etc.
        # Normalize line breaks and spacing around periods
        text = text.replace('\r\n', ' ').replace('\n', ' ') # Replace different line breaks with space
        text = re.sub(r'\s+', ' ', text).strip() # Collapse multiple spaces to one, trim whitespace
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text) # Ensure space after sentence-ending punctuation
        text = re.sub(r'\.\s*(\.)', r'..', text) # Prevent splitting on ellipsis

        sentences = nltk.sent_tokenize(text)
        # Filter out potentially empty or very short sentences from artifacts
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1 or s.strip() in ".!?"] # Keep single punctuation marks if they somehow occur

        debug_print(f"Split text into {len(sentences)} sentences.", 3)
        # debug_print(f"Sentences: {sentences}", 4)
        return sentences
    except Exception as e:
        print(f"❌ Error splitting text into sentences: {e}. Returning text as a single sentence.")
        # Fallback: split by double newline or just return the whole text as one "sentence"
        fallback_sentences = [s.strip() for s in text.split('\n\n') if s.strip()]
        if fallback_sentences:
            
            return fallback_sentences
        else:
            return [text.strip()] if text.strip() else []


# --- Temp File Cleanup Function (Combined Version) ---
def cleanup_temp_files(extensions=('.mp3', '.wav', '.json', '.tsv', '.txt')):
    """
    Deletes temporary files starting with 'temp_ai_speech_' and
    ending with specified extensions.

    It iterates through files in the script's directory and attempts
    to delete each matching file using a retry mechanism, similar to
    the original second function, but applied to all identified temp files.
    """
    debug_print("Starting temp file cleanup...", 3)
    try:
        # --- Get the directory where the script is located ---
        try:
            # Use os.path.dirname(os.path.abspath(__file__)) for script's directory
            temp_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback if __file__ is not defined (e.g., interactive session)
            temp_dir = os.getcwd()
            debug_print("Warning: __file__ not defined, using os.getcwd() for temp file cleanup directory.", 2)

        debug_print(f"Checking for temp files to clean in: {temp_dir}", 3)

        # --- Iterate through all files in the directory ---
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)

            # Check if it matches the temporary file pattern and one of the extensions
            # Also ensure it's actually a file, not a directory
            if (filename.startswith(TTS_TEMP_PREFIX) and
                filename.lower().endswith(tuple(ext.lower() for ext in extensions)) and # Case-insensitive extension check
                os.path.isfile(file_path)):

                debug_print(f"Found potential temp file: {filename}", 3)

                # --- Apply the retry logic to THIS specific file ---
                successfully_removed = False
                debug_print(f"Attempting to remove {filename}...", 3)
                for attempt in range(5): # Retry up to 5 times
                    try:
                        debug_print(f"Attempt {attempt+1}/5 to remove {filename}...", 4)
                        os.remove(file_path)
                        debug_print(f"Cleaned up temp file: {filename}", 3)
                        successfully_removed = True
                        break # Success for *this* file, move to the next file in the outer loop
                    except OSError as e:
                        # This often happens if the file is in use by another process
                        debug_print(f"Attempt {attempt+1} failed for {filename} (OSError): {e} (File likely in use)", 3)
                        time.sleep(0.1) # Wait a bit before retrying
                    except Exception as e:
                        # Catch any other unexpected errors during removal (e.g., permissions, FileNotFoundError if timing is weird)
                        debug_print(f"Attempt {attempt+1} failed for {filename} (unexpected error): {e}", 3)
                        time.sleep(0.1) # Wait a bit before retrying

                if not successfully_removed:
                    debug_print(f"Failed to remove temp file {filename} after multiple attempts.", 3)
                # --- End of retry logic for the current file ---

            # The outer loop continues to the next file regardless of
            # whether the current one was deleted or failed after retries.

    except Exception as e:
        # Catch errors that happen during directory listing, path joining, etc.
        debug_print(f"Error during temp file cleanup process: {e}", 2)
    debug_print("Temp file cleanup finished.", 3)
    
# --- NEW: Function to Run Rhubarb Lip Sync ---
def get_lip_sync_data(wav_filepath, text_content):
    """Runs Rhubarb Lip Sync and returns the parsed timing data."""
    if not os.path.exists(RHUBARB_EXECUTABLE):
        print(f"❌ Rhubarb executable not found at '{RHUBARB_EXECUTABLE}'. Lip sync data cannot be generated.")
        return None

    debug_print(f"Running Rhubarb on: {os.path.basename(wav_filepath)}", 2)
    output_format = "json"
    # Use temp file prefix for Rhubarb output file name
    output_filepath = wav_filepath.replace(".wav", f".{output_format}")
    # Use temp file prefix for dialog file name
    dialog_filepath = wav_filepath.replace(".wav", ".txt")

    # Create dialog file - Rhubarb needs this for context/timing accuracy
    try:
        with open(dialog_filepath, "w", encoding="utf-8") as f:
            f.write(text_content)
        debug_print(f"Created dialog file: {os.path.basename(dialog_filepath)}", 3)
    except Exception as e:
        print(f"❌ Error writing dialog file {dialog_filepath}: {e}")
        return None

    # Construct Rhubarb command
    # Adjust recognizer based on language if needed: -r phonetic or -r standard
    # --extendedShapes GHX is often useful for the Blair set
    command = [
        RHUBARB_EXECUTABLE,
        "-f", output_format,
        "-o", output_filepath,
        "--dialogFile", dialog_filepath,
        "--extendedShapes", "GHX", # Ensure cues for G, H, X are generated
        # "--quiet", # Uncomment to suppress Rhubarb's own output
        wav_filepath # The audio file to process
    ]
    debug_print(f"Rhubarb command: {' '.join(command)}", 3)

    rhubarb_data = None
    try:
        # Run Rhubarb as subprocess
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30) # Added timeout
        debug_print("Rhubarb executed successfully.", 3)
        # Optional verbose debug of Rhubarb's own output
        # if process.stdout: debug_print(f"Rhubarb stdout: {process.stdout}", 4)
        # if process.stderr: debug_print(f"Rhubarb stderr: {process.stderr}", 4)

        # Read the output JSON file
        if os.path.exists(output_filepath):
            with open(output_filepath, "r", encoding="utf-8") as f:
                rhubarb_data = json.load(f)
            debug_print(f"Loaded Rhubarb output from {os.path.basename(output_filepath)}", 3)
        else:
            print(f"❌ Rhubarb output file not created: {output_filepath}")
            # Print stderr/stdout from process if available to diagnose
            if process.stdout: print(f"Rhubarb stdout: {process.stdout}")
            if process.stderr: print(f"Rhubarb stderr: {process.stderr}")


    except FileNotFoundError:
        # This specific error is caught at the top check, but defensive programming
        print(f"❌ Error: '{RHUBARB_EXECUTABLE}' command not found. Is Rhubarb installed and in PATH?")
        rhubarb_data = None
    except subprocess.TimeoutExpired:
        print(f"❌ Error: Rhubarb command timed out after 30 seconds for {os.path.basename(wav_filepath)}.")
        rhubarb_data = None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Rhubarb (CalledProcessError): {e}")
        print(f"    Rhubarb stdout: {e.stdout}")
        print(f"    Rhubarb stderr: {e.stderr}")
        rhubarb_data = None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Rhubarb output file {output_filepath}: {e}")
        rhubarb_data = None
    except Exception as e:
        print(f"❌ Unexpected error getting lip sync data: {e}")
        rhubarb_data = None
    finally:
        # Clean up dialog file and Rhubarb output file immediately
        try:
            if os.path.exists(dialog_filepath): os.remove(dialog_filepath)
            debug_print(f"Cleaned up dialog file {os.path.basename(dialog_filepath)}", 4)
        except Exception as e: debug_print(f"Could not clean up dialog file {os.path.basename(dialog_filepath)}: {e}", 4)
        # Only clean up output file here if it was successfully loaded, otherwise leave it for debugging
        # if rhubarb_data is not None and os.path.exists(output_filepath):
        #     try: os.remove(output_filepath)
        #     except Exception as e: debug_print(f"Could not clean up rhubarb output file {os.path.basename(output_filepath)}: {e}", 4)
        # DECISION: Cleanup temp files at the end of speak, not here, to use the robust cleanup function

    # Return the list of cues or None if data was not successfully loaded
    return rhubarb_data.get("mouthCues") if rhubarb_data else None


# --- MODIFIED speak Function (Integrates Lip Sync) ---
def speak(text_or_sentences, speech_type="generic", topic=None, start_sentence_index=0):
    """
    Generates TTS (WAV), runs Rhubarb, plays audio sentence by sentence,
    and updates global mouth shape based on timings. Handles interruption
    and cleans up temporary audio files for each sentence after playback.
    """
    global current_state, interrupted_context, current_mouth_image, current_mouth_rect

    if not pygame.mixer.get_init():
        print("❌ Pygame mixer not initialized, cannot speak.")
        print(f"(Fallback) AI says: {text_or_sentences}")
        current_state = STATE_IDLE
        return

    # Clean up temp files from *previous* speak calls *before* starting new audio
    # This attempts to clean up files that might have been left locked by the previous speak
    # Adding a small delay here might help if files are lingering after the last unload
    # time.sleep(0.05) # Optional small delay before cleanup
    cleanup_temp_files() # Clean up files from *previous* speak calls

    is_list = isinstance(text_or_sentences, list)
    sentences = text_or_sentences if is_list else split_into_sentences(text_or_sentences)
    full_original_text = " ".join(sentences)

    if not sentences:
        debug_print("No sentences to speak.", 2)
        current_state = STATE_IDLE
        return

    # Set State based on Type
    original_state_before_speaking = current_state # Store state before speaking starts
    if speech_type == "explanation": current_state = STATE_SPEAKING_EXPLANATION
    elif speech_type == "answer": current_state = STATE_SPEAKING_ANSWER
    elif speech_type == "transition": current_state = STATE_SPEAKING_TRANSITION
    elif speech_type == "greeting" or speech_type == "farewell": current_state = STATE_SPEAKING_GREETING
    else: current_state = STATE_SPEAKING_GREETING # Default for generic/other types

    debug_print(f"Starting to speak ({current_state}). Total sentences: {len(sentences)}. Starting index: {start_sentence_index}", 2)

    interrupt_flag.clear()
    tts_finished_event.clear()
    interrupted_mid_speech = False
    # start_index is passed as argument now
    # start_index = 0 # Ensure it starts from 0 unless resuming


    # --- Sentence Iteration Loop ---
    for i in range(start_sentence_index, len(sentences)):
        sentence = sentences[i]
        if not sentence.strip(): # Skip empty sentences
            continue

        # --- Generate Unique Filenames (MP3, WAV, JSON) for this sentence ---
        base_filename = f"{TTS_TEMP_PREFIX}{i}_{int(time.time())}" # Add timestamp for more uniqueness
        mp3_filepath = f"{base_filename}.mp3"
        wav_filepath = f"{base_filename}.wav"

        debug_print(f"Processing sentence {i+1}/{len(sentences)}: '{sentence[:70]}...'", 2)

        lip_sync_cues = None
        audio_generated = False
        try:
            # --- Generate TTS MP3 using gTTS ---
            debug_print(f"Generating TTS for sentence {i+1}...", 3)
            tts = gTTS(text=sentence, lang='en', slow=False)
            tts.save(mp3_filepath)
            debug_print(f"TTS MP3 saved to {os.path.basename(mp3_filepath)}", 3)

            # --- Convert MP3 to WAV using Pydub ---
            debug_print(f"Converting {os.path.basename(mp3_filepath)} to {os.path.basename(wav_filepath)}...", 3)
            audio = AudioSegment.from_mp3(mp3_filepath)
            audio.export(wav_filepath, format="wav")
            debug_print(f"WAV file created: {os.path.basename(wav_filepath)}", 3)
            audio_generated = True # Set flag if WAV is successfully created

            # --- Get Lip Sync Data from Rhubarb ---
            debug_print(f"Getting lip sync data for {os.path.basename(wav_filepath)}...", 3)
            lip_sync_cues = get_lip_sync_data(wav_filepath, sentence)

            if lip_sync_cues is None:
                print(f"⚠️ Warning: Could not get lip sync cues for sentence {i+1}. Lip sync will be static during this sentence.")

        except FileNotFoundError as e:
            print(f"❌ Error: FFmpeg/Libav not found or required file missing? ({e}). Cannot generate/convert audio to WAV for sentence {i+1}.")
            print("    Please ensure FFmpeg or Libav is installed and accessible in your system's PATH.")
            interrupted_mid_speech = True; # Flag error, but don't break yet, proceed to cleanup attempt
        except Exception as e:
            print(f"❌ Error during audio generation/conversion/rhubarb for sentence {i+1}: {e}")
            interrupted_mid_speech = True; # Flag error, but don't break yet


        # --- Playback and Lip Sync Animation ---
        # Only attempt playback if WAV file was successfully created AND no interrupt was signaled during generation
        if audio_generated and not interrupt_flag.is_set():
            try:
                debug_print(f"Loading and playing {os.path.basename(wav_filepath)}...", 3)
                pygame.mixer.music.load(wav_filepath)
                pygame.mixer.music.play()

                # --- Playback and Lip Sync Monitor Loop ---
                set_mouth_pose("X") # Start with Rest before sound begins

                while pygame.mixer.music.get_busy() and not interrupt_flag.is_set():
                    playback_time_sec = pygame.mixer.music.get_pos() / 1000.0

                    # --- Lip Sync Logic ---
                    active_cue_found = False
                    if lip_sync_cues:
                        for cue in lip_sync_cues:
                            if cue["start"] <= playback_time_sec < cue["end"]:
                                mapped_code = rhubarb_map.get(cue["value"], "X")
                                set_mouth_pose(mapped_code)
                                active_cue_found = True
                                break
                    if not active_cue_found:
                        set_mouth_pose("X") # Default to Rest

                    # --- Interrupt Check --- (Remains the same)
                    try:
                        interrupt_signal = interrupt_queue.get_nowait()
                        if interrupt_signal == "INTERRUPT": interrupt_flag.set()
                    except queue.Empty: pass

                    # --- Pygame Display Update ---
                    update_display() # Update display while speaking

                    # --- Handle Pygame Events (Crucial) ---
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("QUIT event detected in speak() playback loop. Signaling exit.")
                            global running
                            if 'running' in globals(): running = False
                            interrupt_flag.set()
                            stop_speech() # This also unloads the audio
                            interrupted_mid_speech = True
                            break
                    if interrupted_mid_speech: break # Exit playback loop if QUIT

                # --- End of Playback/Monitor Loop ---

                # --- Post Sentence Playback ---
                # Ensure audio is stopped and unloaded after playback finishes naturally
                # If interrupted, stop_speech() handles this. If finished naturally, do it explicitly.
                if not pygame.mixer.music.get_busy(): # Check if music finished playing naturally
                    debug_print(f"Sentence {i+1} playback finished naturally.", 3)
                    try:
                        pygame.mixer.music.stop() # Ensure stopped
                        pygame.mixer.music.unload() # Explicitly unload the audio file
                        debug_print(f"Unloaded {os.path.basename(wav_filepath)} after natural finish.", 3)
                    except pygame.error as e:
                        print(f"⚠️ Warning: Pygame error unloading {os.path.basename(wav_filepath)}: {e}")
                    except Exception as e:
                        print(f"❌ Unexpected error unloading {os.path.basename(wav_filepath)}: {e}")

                # Reset mouth to Rest when sentence audio finishes or is interrupted
                set_mouth_pose("X")
                update_display() # Draw final Rest pose

                # If loop exited because of interruption flag being set
                if interrupt_flag.is_set():
                    debug_print(f"Interrupt detected while speaking sentence {i+1}.", 1)
                    # stop_speech() was already called by the interrupt logic or QUIT handler
                    interrupted_mid_speech = True # Confirm interruption occurred
                    break # Exit the sentence loop

            except pygame.error as e:
                print(f"❌ Error during pygame playback/drawing for sentence {i+1}: {e}")
                interrupted_mid_speech = True
                break
            except Exception as e:
                print(f"❌ Unexpected error during sentence playback/drawing loop {i+1}: {e}")
                interrupted_mid_speech = True
                break

        # --- Clean up temp audio files for this specific sentence ---
        # Attempt cleanup *after* playback loop AND unloading
        # This is where the WinError 32 likely occurs if unload didn't work
        try:
            if os.path.exists(mp3_filepath): os.remove(mp3_filepath)
            debug_print(f"Cleaned up {os.path.basename(mp3_filepath)}", 4)
        except Exception as e: debug_print(f"Error cleaning up mp3 {os.path.basename(mp3_filepath)}: {e}", 4)

        try:
            # This is the WAV file that might be locked
            if os.path.exists(wav_filepath): os.remove(wav_filepath)
            debug_print(f"Cleaned up {os.path.basename(wav_filepath)}", 4)
        except Exception as e:
            # Catch and report the error but continue
            debug_print(f"⚠️ Error cleaning up wav {os.path.basename(wav_filepath)}: {e}", 3)
            # The file might be cleaned up by a subsequent call to cleanup_temp_files if it gets unlocked later

        # Rhubarb output file (.json) and dialog file (.txt) cleanup handled in get_lip_sync_data finally block

        # If interrupted while processing or speaking this sentence (or error occurred)
        if interrupted_mid_speech:
            # --- SAVE CONTEXT IF IT WAS AN EXPLANATION ---
            if current_state == STATE_SPEAKING_EXPLANATION:
                debug_print("Saving explanation context for resume...", 2)
                interrupted_context["type"] = "explanation"
                interrupted_context["topic"] = topic # Save the original topic query passed to speak
                interrupted_context["full_text"] = full_original_text # Save the full text
                interrupted_context["sentences"] = sentences # Save the list of sentences
                # Save the index of the *next* sentence to speak
                interrupted_context["resume_index"] = i + 1
                interrupted_context["saved"] = True
                debug_print(f"Explanation context saved. Resume index: {interrupted_context['resume_index']}", 2)
            else:
                # Clear context if interrupted during non-explanation speech or on error
                debug_print(f"Interrupted/Error during non-explanation speech ({current_state}). Clearing context.", 2)
                interrupted_context.clear()
                interrupted_context["saved"] = False
            break # Exit the main sentence iteration loop as we've handled the interruption/error


    # --- Post-Loop Handling ---
    if not interrupted_mid_speech:
        debug_print("Finished speaking all sentences naturally.", 2)
        tts_finished_event.set() # Signal that TTS is finished normally
        # Clear context after successfully completing an explanation
        if current_state == STATE_SPEAKING_EXPLANATION:
            debug_print("Explanation completed naturally. Clearing context.", 2)
            interrupted_context.clear()
            interrupted_context["saved"] = False
        # Transition to IDLE state after finishing speaking normally
        # The state is set *after* the speak call returns in the main loop logic flow
        # For now, let speak handle setting IDLE if it finishes normally
        # current_state = STATE_IDLE # Let the logic *after* the speak call set the next state

    else:
        debug_print(f"Exited speak function due to interruption/error (Flag: {interrupt_flag.is_set()}).", 2)
        # If interrupted, the state was set by the interrupt handler or QUIT event handler
        # The main loop logic after the speak call will handle the state transition based on interrupt_flag.

# --- Placeholder for transition phrase generation ---
def generate_transition_phrase(topic=None):
    """Generates a simple transition phrase."""
    phrases = [
        "Okay, getting back to it.",
        "Resuming our discussion.",
        "Let's continue.",
        "Picking up where we left off."
    ]
    if topic:
        phrases.extend([
            f"Okay, getting back to {topic}.",
            f"Let's continue our lesson on {topic}."
            ])
    return random.choice(phrases)

# --- Pygame Display Update Function ---
def update_display():
    """Clears the screen and draws the current visual elements."""
    global screen, current_mouth_image, current_mouth_rect # Access global display variables

    if screen is None:
        # Screen hasn't been initialized yet
        return

    # Check for QUIT event during drawing update loop - redundant if main loop checks, but safer
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global running # Access main loop flag
            if 'running' in globals():
                running = False # Signal main loop to exit
            pygame.quit() # Quit pygame subsystems immediately
            # This will likely cause errors later if the loop continues,
            # but ensures the window closes. Main loop should handle graceful exit.
            return # Stop drawing update if QUIT

    screen.fill(WHITE) # Clear screen each frame

    # Draw optional head image here (if you have one loaded)
    # if 'head_image' in globals() and 'head_rect' in globals() and head_image and head_rect:
    #      screen.blit(head_image, head_rect)

    # Draw the current mouth image (updated by set_mouth_pose)
    if current_mouth_image and current_mouth_rect:
        # Center the mouth image on the screen at MOUTH_POS
        # current_mouth_rect = current_mouth_image.get_rect(center=MOUTH_POS) # Update rect center each time? Or once on image change? Doing it once on change is better.
        screen.blit(current_mouth_image, current_mouth_rect)

    # Draw status text (optional)
    # status_text_surface = font.render(f"State: {current_state}", True, BLACK)
    # screen.blit(status_text_surface, (10, 10))

    pygame.display.flip() # Update the entire screen


# --- Function to set the current mouth pose ---
def set_mouth_pose(code):
    """Sets the global current_mouth_image based on a mapped code."""
    global current_mouth_image, current_mouth_rect, mouth_images, missing_mouth_image # Access global variables

    # Get the corresponding image, default to Rest ('X') or the fallback if the code or 'X' image is missing
    img_to_set = mouth_images.get(code, mouth_images.get("X", missing_mouth_image))

    if current_mouth_image != img_to_set:
        # Only update if the image is actually changing
        current_mouth_image = img_to_set
        # Update the rectangle's position to keep it centered at MOUTH_POS
        if current_mouth_image:
            current_mouth_rect = current_mouth_image.get_rect(center=MOUTH_POS)



# --- The Main Application Loop ---
def run_voice_teacher():
    """
    Runs the main state-driven interaction loop for the AI voice teacher.
    Handles user input, processing, speaking, interruptions, and display updates.
    """
    global current_state, interrupt_flag, interrupted_context, running
    global screen, font, WHITE, BLACK # Access pygame variables

    # --- Initial Setup (called from __main__) ---
    # Pygame initialized in __main__
    # Keyboard listener started in __main__
    # Mouth images loaded in __main__
    # Initial state is STATE_IDLE

    debug_print("Starting AI Teacher main loop...", 1)

    interrupted_context = {"saved": False} # Ensure context is initially clear
    running = True # Flag to control the main loop


    # --- Initial Greeting ---
    # The speak function sets the state and handles its own duration/interrupts
    speak("Hello! Press SPACE to interrupt. What topic today?", speech_type="greeting")

    # After speak() returns, check the interrupt flag to determine the next state
    if interrupt_flag.is_set():
        debug_print("Initial greeting interrupted.", 1)
        # If interrupted, clear flag and go straight to handling interruption
        interrupt_flag.clear()
        current_state = STATE_HANDLING_INTERRUPTION # Go to interrupt handling state
    else:
        debug_print("Initial greeting finished normally.", 1)
        current_state = STATE_IDLE # Return to idle after greeting

    # Prompt the user after the greeting
    print("\nAI is ready. What science topic would you like to learn about or ask a question?")
    # Initial state after greeting is IDLE or HANDLING_INTERRUPTION


    # --- Main State-Driven Loop ---
    while running:
        # --- Pygame Event Handling ---
        # Handle events like closing the window - CRUCIAL for responsiveness
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("QUIT event detected in main loop. Signaling exit.")
                running = False # Signal loop to stop
                interrupt_flag.set() # Signal any active speak() or listen() to stop
                stop_speech() # Ensure audio is stopped
                break # Exit event loop
            # You can add other event handling here (e.g., mouse clicks, other key presses)
            # elif event.type == pygame.KEYDOWN:
            #     debug_print(f"Key pressed: {event.key}", 4)
            #     pass # Handle other key presses if needed

        if not running: break # Exit main loop if QUIT detected

        # --- Update Pygame Display ---
        # This handles drawing when speak() is NOT running (speak draws while it's active)
        # It also handles drawing the final pose after speak() finishes.
        update_display()


        # --- Check for Interruptions ---
        # This block is triggered if the interrupt key was pressed *while not already in HANDLING_INTERRUPTION*
        if interrupt_flag.is_set() and current_state != STATE_HANDLING_INTERRUPTION:
            debug_print("Main loop detected interrupt flag set. Transitioning to HANDLING_INTERRUPTION.", 1)
            stop_speech() # Ensure audio stopped if speak was running
            interrupt_flag.clear() # Clear the flag as we are starting to handle it
            current_state = STATE_HANDLING_INTERRUPTION # Transition to the handling state
            # The logic for *what happens in* HANDLING_INTERRUPTION is below


        # --- Handle States ---
        # --- STATE: IDLE ---
        if current_state == STATE_IDLE:
            debug_print("State: IDLE. Transitioning to LISTENING.", 2)
            # In IDLE, we are waiting for the user to start a new interaction.
            # Immediately transition to LISTENING to get user input.
            # The prompt message was printed earlier, and will be printed again if we return to IDLE
            current_state = STATE_LISTENING


        # --- STATE: LISTENING ---
        elif current_state == STATE_LISTENING:
            debug_print("State: LISTENING. Waiting for input.", 2)
            # Ensure previous context is cleared if we're starting a new conversation phase
            if interrupted_context.get("saved"):
                debug_print("Clearing old interruption context before listening for new command.", 2)
                interrupted_context.clear()
                interrupted_context["saved"] = False

            # Only listen if the mixer is NOT busy (shouldn't be in LISTENING, but defensive)
            if not pygame.mixer.music.get_busy():
                # Call the listen function - it blocks until speech or timeout
                student_input = listen_to_student() # This function sets mouth pose

                # --- After listen_to_student returns, process the input ---
                if student_input is not None: # Check if speech was recognized (not None)
                    debug_print(f"Student input received: '{student_input}'. Understanding intent...", 2)
                    # Process the input and understand intent
                    intent, topic_or_question = understand_intent_and_topic(student_input)

                    # --- Handle Input Intents (No Saved Context) ---
                    # This block is for entirely new requests when not in an interrupted state
                    debug_print(f"Handling new request intent: {intent}, topic: {topic_or_question}", 1)

                    # Reset interrupt flag and tts_finished_event before starting a new speaking task
                    interrupt_flag.clear()
                    tts_finished_event.clear()

                    if intent == "REQUEST_EXPLANATION":
                        if topic_or_question:
                            debug_print(f"Handling new REQUEST_EXPLANATION for topic: {topic_or_question}", 1)
                            # Decide between in-syllabus detailed or general explanation
                            explanation_text, cleaned_topic, in_syllabus = generate_spoken_explanation(topic_or_question)

                            if not in_syllabus:
                                speak(f"That topic doesn't seem to be in our current syllabus, but here's a general overview.", speech_type="transition")
                                # After speak returns, check interrupt flag
                                if interrupt_flag.is_set(): debug_print("Out-of-syllabus transition interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue # Go to interrupt handling
                                else: debug_print("Out-of-syllabus transition finished.", 1) # State remains LISTENING until speak is called next

                                # Now speak the explanation text (pass original query as topic for context saving)
                                speak(explanation_text, speech_type="explanation", topic=topic_query, start_sentence_index=0)
                            else: # In syllabus
                                speak(f"Okay, let's learn about {cleaned_topic}.", speech_type="transition")
                                # After speak returns, check interrupt flag
                                if interrupt_flag.is_set(): debug_print("In-syllabus transition interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                                else: debug_print("In-syllabus transition finished.", 1)

                                # Now speak the explanation text (generate_full_lesson_text includes intro/hook)
                                speak(explanation_text, speech_type="explanation", topic=cleaned_topic, start_sentence_index=0) # Pass cleaned topic


                            # After speaking the explanation (either general or detailed), check the flag
                            if interrupt_flag.is_set():
                                debug_print("Explanation spoken after new request was interrupted.", 1)
                                # Context is saved inside speak(). State is now SPEAKING_EXPLANATION (correct).
                                # Main loop will transition to HANDLING_INTERRUPTION because flag is set.
                                pass # Handled by the interrupt check at top of loop
                            else:
                                debug_print("Explanation spoken after new request finished normally.", 1)
                                # Explanation finished, clear context as it wasn't interrupted mid-speak
                                interrupted_context.clear()
                                interrupted_context["saved"] = False
                                # Transition to IDLE
                                current_state = STATE_IDLE
                                debug_print("Transitioning to IDLE after explanation.", 2)

                        else: # REQUEST_EXPLANATION but no topic extracted
                            debug_print("REQUEST_EXPLANATION intent with no topic extracted.", 1)
                            speak("Please tell me what topic you'd like to learn about.", speech_type="answer")
                            if interrupt_flag.is_set(): debug_print("Prompt interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                            else: debug_print("Prompt finished.", 1); current_state = STATE_IDLE


                    elif intent == "ASK_QUESTION":
                        if topic_or_question:
                            debug_print(f"Handling new ASK_QUESTION: {topic_or_question}", 1)
                            speak("Let me see...", speech_type="transition")
                            if interrupt_flag.is_set(): debug_print("Answer transition interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                            else: debug_print("Answer transition finished.", 1)

                            # Generate and speak the answer (no specific context needed for a new question)
                            # Check syllabus relevance even for questions if you want to use KB context
                            in_syllabus, syllabus_content, _ = is_in_syllabus(topic_or_question)
                            answer_context = syllabus_content if in_syllabus else None

                            answer = generate_spoken_answer(topic_or_question, answer_context)
                            speak(answer, speech_type="answer")

                            # After speaking the answer, check the flag
                            if interrupt_flag.is_set():
                                debug_print("Answer spoken after new request was interrupted.", 1)
                                # Context is NOT saved for answers. State is SPEAKING_ANSWER (correct).
                                # Main loop will transition to HANDLING_INTERRUPTION because flag is set.
                                pass # Handled by the interrupt check at top of loop
                            else:
                                debug_print("Answer spoken after new request finished normally.", 1)
                                # Transition to IDLE
                                current_state = STATE_IDLE
                                debug_print("Transitioning to IDLE after answer.", 2)

                        else: # ASK_QUESTION but no topic extracted
                            debug_print("ASK_QUESTION intent with no topic extracted.", 1)
                            speak("Please ask me a specific question.", speech_type="answer")
                            if interrupt_flag.is_set(): debug_print("Prompt interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                            else: debug_print("Prompt finished.", 1); current_state = STATE_IDLE


                    elif intent == "GREETING":
                        debug_print("Handling GREETING intent.", 1)
                        speak(random.choice(["Hello!", "Hi there!", "Greetings!"]), speech_type="greeting")
                        if interrupt_flag.is_set(): debug_print("Greeting interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                        else: debug_print("Greeting finished.", 1); current_state = STATE_IDLE

                    elif intent == "FAREWELL":
                        debug_print("Handling FAREWELL intent. Exiting.", 1)
                        speak(random.choice(["Goodbye!", "See you later!", "Farewell!"]), speech_type="farewell")
                        # The speak function will handle the final audio. After speak returns, exit.
                        running = False # Signal main loop to exit

                    elif intent == "RESUME":
                        debug_print("Handling RESUME intent with no saved context.", 1)
                        # Resume command when no context exists (user error)
                        speak("There is no explanation saved to resume.", speech_type="answer")
                        if interrupt_flag.is_set(): debug_print("Resume failed message interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                        else: debug_print("Resume failed message finished.", 1); current_state = STATE_IDLE

                    elif intent == "REPHRASE":
                        debug_print("Handling REPHRASE intent.", 1)
                        # Rephrase command when no context exists (user error or feature not implemented)
                        speak("Sorry, I don't have the previous phrase saved to rephrase.", speech_type="answer") # Or implement rephrase logic
                        if interrupt_flag.is_set(): debug_print("Rephrase failed message interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                        else: debug_print("Rephrase failed message finished.", 1); current_state = STATE_IDLE


                    elif intent == "OTHER":
                        debug_print("Handling OTHER intent.", 1)
                        speak("Hmm, I'm not sure how to respond to that. Could you try rephrasing?", speech_type="answer")
                        if interrupt_flag.is_set(): debug_print("Generic response interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                        else: debug_print("Generic response finished.", 1); current_state = STATE_IDLE

                    elif intent == "ERROR" or intent == "NO_INPUT": # NLU failed or listen timed out/failed
                        debug_print(f"Handling NLU/Listen ERROR/NO_INPUT intent: {intent}", 1)
                        speak("Sorry, I had trouble understanding that. Could you repeat or rephrase?", speech_type="answer")
                        if interrupt_flag.is_set(): debug_print("Error response interrupted.", 1); current_state = STATE_HANDLING_INTERRUPTION; continue
                        else: debug_print("Error response finished.", 1); current_state = STATE_IDLE


            time.sleep(0.05)


        # --- STATE: HANDLING_INTERRUPTION ---
        elif current_state == STATE_HANDLING_INTERRUPTION:
            debug_print("State: HANDLING_INTERRUPTION.", 2)
            # This state is entered when an interrupt flag is set in other states.
            # We need to listen for the student's command after they interrupted.

            # Check if we were interrupted *while* handling the interruption (e.g., while speaking "Yes?")
            # If the flag is set upon entering this state, it means the previous action was interrupted.
            if interrupt_flag.is_set():
                debug_print("Interrupt flag set upon entering HANDLING_INTERRUPTION. Clearing and re-processing.", 2)
                stop_speech() # Ensure audio is stopped again
                interrupt_flag.clear() # Clear flag as we are starting to handle it
                # Proceed to listen for the command immediately below in this state's logic
                pass # Continue

            else:
                # This is the standard flow: interrupt detected, speak was stopped,
                # flag was cleared *before* entering this state. Now prompt the user.
                debug_print("Prompting student after interruption.", 2)
                speak("Yes?", speech_type="transition")

                # After speak("Yes?") returns, check the flag again
                if interrupt_flag.is_set():
                    debug_print("'Yes?' response was interrupted.", 1)
                    # If 'Yes?' was interrupted, stay in HANDLING_INTERRUPTION.
                    # Flag is already set by keyboard listener. Next loop checks flag.
                    interrupt_flag.clear() # Clear flag for next handling cycle
                    # State remains HANDLING_INTERRUPTION
                    debug_print("Staying in HANDLING_INTERRUPTION.", 2)
                    pass # Continue to the listen logic below

                else:
                    # If 'Yes?' finished normally, listen for the student's command
                    debug_print("'Yes?' response finished normally. Listening for interruption command.", 2)
                    print(f"\n🎤 (Listening for your command after interruption)") # User feedback
                    interruption_input = listen_to_student() # This blocks until speech or timeout

                    # --- Process the interruption input ---
                    if interruption_input is not None:
                        debug_print(f"Interruption input received: '{interruption_input}'. Understanding command.", 2)
                        intent, topic_or_question = understand_intent_and_topic(interruption_input)

                        # Reset interrupt flag and tts_finished_event before starting a new speaking task
                        interrupt_flag.clear()
                        tts_finished_event.clear()

                        # --- Handle Interruption Command Intents (With Saved Context Check) ---
                        # This logic is adapted from the main_interaction_loop provided previously

                        # Handle input when SAVED context exists (most common after interrupting explanation)
                        if interrupted_context.get("saved"):
                            debug_print("Saved context found. Evaluating interruption input.", 1)

                            if intent == "ASK_QUESTION" and topic_or_question:
                                debug_print("Interruption input is a question. Answering and preparing to resume.", 1)
                                # Handle the question - Integrate syllabus check and concise answer
                                # Use the *interrupted explanation topic* as context for the answer if available
                                context_for_answer = interrupted_context.get("topic")
                                answer = generate_spoken_answer(topic_or_question, context_for_answer)
                                speak(answer, speech_type="answer")
                                # After speaking the answer, transition to CHECK_RESUME (unless interrupted)
                                # The transition happens after the speak call returns.

                            elif intent == "RESUME":
                                debug_print("Interruption input is an explicit resume command. Proceeding to resume.", 1)
                                # No speech needed here, just transition directly
                                current_state = STATE_CHECK_RESUME
                                # Skip remaining logic in this block and go straight to CHECK_RESUME state in next loop
                                continue # Skip to the next iteration

                            elif intent == "REQUEST_EXPLANATION" and topic_or_question:
                                debug_print(f"Interruption input is request for NEW explanation topic ('{topic_or_question}'). Acknowledging and will then resume original.", 1)
                                speak(f"Okay, I can make a note about {topic_or_question}. Let's get back to...", speech_type="transition")
                                # Saved context is NOT cleared here.
                                # Transition to CHECK_RESUME after speak returns.

                            elif intent in ["GREETING", "FAREWELL", "REPHRASE", "OTHER", "ERROR", "NO_INPUT"]:
                                # Handle greetings, farewells, inability to understand, etc. when context is saved.
                                # Provide a response but preserve the context and then transition to CHECK_RESUME (unless FAREWELL).
                                if intent == "GREETING":
                                        speak(random.choice(["Hello again!", "Hi!", "Greetings!"]), speech_type="greeting")
                                elif intent == "FAREWELL":
                                        speak(random.choice(["Okay, goodbye!", "See you later!"]), speech_type="farewell")
                                        running = False # Signal exit
                                elif intent == "REPHRASE":
                                        speak("Could you please repeat what you said before?", speech_type="answer")
                                elif intent == "ERROR" or intent == "NO_INPUT":
                                        speak("Sorry, I didn't get that. Could you please repeat your command?", speech_type="answer")
                                elif intent == "OTHER":
                                        speak("Hmm, I'm not sure about that command right now. Were you trying to ask a question or resume?", speech_type="answer")

                                # For most of these (except FAREWELL), transition to CHECK_RESUME after speaking.
                                if intent != "FAREWELL":
                                        # Transition happens after speak call returns.
                                        pass # Handled by the state check after speak

                            # --- State Transition after handling interruption input (when saved context) ---
                            # After processing the input and speaking a response (if any),
                            # check if the *response speech* was interrupted.
                            if running and intent != "FAREWELL": # Only transition if not exiting
                                if interrupt_flag.is_set():
                                        debug_print("Speech following interruption input was interrupted. Staying in HANDLING_INTERRUPTION.", 1)
                                        # Flag is already set. Next loop will re-enter HANDLING_INTERRUPTION.
                                        stop_speech() # Ensure stopped
                                        interrupt_flag.clear() # Clear flag for the next cycle
                                        # State remains HANDLING_INTERRUPTION
                                        debug_print("Staying in HANDLING_INTERRUPTION.", 2)
                                        pass # Continue to the start of HANDLING_INTERRUPTION logic in next loop
                                else:
                                        debug_print("Speech following interruption input finished normally. Transitioning to CHECK_RESUME.", 1)
                                        # If response speech finished, proceed to check if resume is needed
                                        current_state = STATE_CHECK_RESUME # Go to check resume state


                        # Handle interruption input when NO saved context exists
                        else:
                            debug_print("No saved context found. Handling interruption input as a new command.", 1)
                            # This is similar to the main IDLE -> LISTENING -> PROCESSING flow, but triggered by interrupt

                            if intent == "REQUEST_EXPLANATION" and topic_or_question:
                                debug_print(f"Handling new REQUEST_EXPLANATION from interruption: {topic_or_question}", 1)
                                explanation_text, cleaned_topic, in_syllabus = generate_spoken_explanation(topic_or_question)

                                if not in_syllabus:
                                    speak(f"That topic doesn't seem to be in our current syllabus, but here's a general overview.", speech_type="transition")
                                    if interrupt_flag.is_set(): debug_print("Out-of-syllabus transition interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                    else: debug_print("Out-of-syllabus transition finished.", 1)
                                    speak(explanation_text, speech_type="explanation", topic=topic_query, start_sentence_index=0)
                                else:
                                    speak(f"Okay, let's learn about {cleaned_topic}.", speech_type="transition")
                                    if interrupt_flag.is_set(): debug_print("In-syllabus transition interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                    else: debug_print("In-syllabus transition finished.", 1)
                                    speak(explanation_text, speech_type="explanation", topic=cleaned_topic, start_sentence_index=0)

                                # After speaking the explanation, check the flag
                                    if interrupt_flag.is_set():
                                        debug_print("Explanation spoken after interruption was interrupted.", 1)
                                        # Context is saved inside speak(). State is now SPEAKING_EXPLANATION (correct).
                                        # Main loop will transition to HANDLING_INTERRUPTION.
                                        stop_speech()
                                        interrupt_flag.clear() # Clear flag
                                        debug_print("Staying in HANDLING_INTERRUPTION after interrupted explanation.", 2)
                                        pass
                                    else:
                                        debug_print("Explanation spoken after interruption finished normally.", 1)
                                        # Explanation finished, clear context as it wasn't interrupted mid-speak
                                        interrupted_context.clear()
                                        interrupted_context["saved"] = False
                                        # Transition to IDLE
                                        current_state = STATE_IDLE
                                        debug_print("Transitioning to IDLE after explanation.", 2)

                            elif intent == "ASK_QUESTION" and topic_or_question:
                                debug_print("Handling new ASK_QUESTION from interruption.", 1)
                                speak("Let me see...", speech_type="transition")
                                if interrupt_flag.is_set(): debug_print("Answer transition interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                else: debug_print("Answer transition finished.", 1)

                                in_syllabus, syllabus_content, _ = is_in_syllabus(topic_or_question)
                                answer_context = syllabus_content if in_syllabus else None
                                answer = generate_spoken_answer(topic_or_question, answer_context)
                                speak(answer, speech_type="answer")

                                if interrupt_flag.is_set():
                                        debug_print("Answer spoken after interruption was interrupted.", 1)
                                        stop_speech()
                                        interrupt_flag.clear()
                                        current_state = STATE_HANDLING_INTERRUPTION
                                        debug_print("Staying in HANDLING_INTERRUPTION after interrupted answer.", 2)
                                        pass
                                else:
                                        debug_print("Answer spoken after interruption finished normally.", 1)
                                        current_state = STATE_IDLE
                                        debug_print("Transitioning to IDLE after answer.", 2)

                            elif intent == "GREETING":
                                debug_print("Handling GREETING from interruption.", 1)
                                speak(random.choice(["Hello!", "Hi there!", "Greetings!"]), speech_type="greeting")
                                if interrupt_flag.is_set(): debug_print("Greeting interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                else: debug_print("Greeting finished.", 1); current_state = STATE_IDLE

                            elif intent == "FAREWELL":
                                debug_print("Handling FAREWELL from interruption. Exiting.", 1)
                                speak(random.choice(["Goodbye!", "See you later!", "Farewell!"]), speech_type="farewell")
                                running = False

                            elif intent == "RESUME":
                                debug_print("Handling RESUME from interruption with no saved context.", 1)
                                speak("There is no explanation saved to resume.", speech_type="answer")
                                if interrupt_flag.is_set(): debug_print("Resume failed message interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                else: debug_print("Resume failed message finished.", 1); current_state = STATE_IDLE

                            elif intent == "REPHRASE":
                                debug_print("Handling REPHRASE from interruption.", 1)
                                speak("Sorry, I don't have the previous phrase saved to rephrase.", speech_type="answer")
                                if interrupt_flag.is_set(): debug_print("Rephrase failed message interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                else: debug_print("Rephrase failed message finished.", 1); current_state = STATE_IDLE

                            elif intent in ["OTHER", "ERROR", "NO_INPUT"]:
                                debug_print(f"Handling OTHER/ERROR/NO_INPUT from interruption: {intent}", 1)
                                speak("Sorry, I didn't get that. Could you try rephrasing?", speech_type="answer")
                                if interrupt_flag.is_set(): debug_print("Generic response interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                                else: debug_print("Generic response finished.", 1); current_state = STATE_IDLE

                        # End of handling interruption input

                    else: # listen_to_student returned None after interruption prompt ("Yes?")
                        debug_print("Listen failed after interruption prompt ('Yes?'). Asking for clarification or attempting resume if context saved.", 2)
                        # If listening fails after 'Yes?', ask again or attempt resume if context exists
                        if interrupted_context.get("saved"):
                            # Assume they might have meant to resume if they interrupted
                            debug_print("Context saved. Assuming attempt to resume. Transitioning to CHECK_RESUME.", 2)
                            # No speak needed here, transition directly
                            current_state = STATE_CHECK_RESUME # Go check resume state
                        else:
                            # No context, listen failed, just go back to IDLE to prompt again
                            debug_print("No context saved. Listen failed. Transitioning to IDLE.", 2)
                            speak("I didn't catch your command.", speech_type="answer")
                            if interrupt_flag.is_set(): debug_print("Didn't catch message interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                            else: debug_print("Didn't catch message finished.", 1); current_state = STATE_IDLE


        # --- STATE: CHECK_RESUME ---
        elif current_state == STATE_CHECK_RESUME:
            debug_print("State: CHECK_RESUME.", 2)
            # This state is entered after successfully handling an interruption command
            # and the system needs to decide if it should resume the previous explanation.

            if interrupted_context.get("saved"):
                debug_print(f"Saved context found. Resuming explanation from index: {interrupted_context.get('resume_index')}", 1)
                # Add a transition phrase before resuming
                transition_text = generate_transition_phrase(interrupted_context.get("topic"))
                speak(transition_text, speech_type="transition")

                # After transition speak returns, check interrupt flag
                if interrupt_flag.is_set():
                    debug_print("Resume transition interrupted.", 1)
                    stop_speech()
                    interrupt_flag.clear()
                    current_state = STATE_HANDLING_INTERRUPTION
                    continue # Go to next loop iteration to handle interruption
                else:
                    debug_print("Resume transition finished normally. Proceeding with resume.", 1)
                    # State remains CHECK_RESUME until resume speak starts

                # Speak the saved sentences, starting from the resume index
                # The speak function will handle state transition to SPEAKING_EXPLANATION internally
                # It will also update the resume_index within interrupted_context if interrupted mid-speak
                speak(interrupted_context["sentences"], speech_type="explanation", topic=interrupted_context.get("topic"), start_sentence_index=interrupted_context.get("resume_index", 0))

                # After speak returns, check interrupt flag
                if interrupt_flag.is_set():
                    debug_print("Resumed explanation was interrupted.", 1)
                    # Context is already saved inside speak(). State is now SPEAKING_EXPLANATION (correct).
                    # Main loop will transition to HANDLING_INTERRUPTION because flag is set.
                    stop_speech()
                    interrupt_flag.clear() # Clear flag
                    debug_print("Staying in HANDLING_INTERRUPTION after interrupted resume.", 2)
                    pass
                else:
                    debug_print("Resumed explanation finished normally.", 1)
                    debug_print("Explanation resume completed.", 1)
                    # Clear context after successful resume completion
                    interrupted_context.clear()
                    interrupted_context["saved"] = False # Explicitly mark as not saved
                    # Transition to IDLE
                    current_state = STATE_IDLE
                    debug_print("Transitioning to IDLE after successful resume.", 2)

            else:
                # This state should theoretically only be reached if saved context is expected but missing (logic error)
                debug_print("Error: Entered CHECK_RESUME state but no saved context found.", 1)
                # Respond that there's nothing to resume and go to IDLE
                speak("It seems there was no explanation saved to resume.", speech_type="answer")
                if interrupt_flag.is_set(): debug_print("No resume message interrupted.", 1); stop_speech(); interrupt_flag.clear(); current_state = STATE_HANDLING_INTERRUPTION; continue
                else: debug_print("No resume message finished normally.", 1); current_state = STATE_IDLE


        # --- STATES: SPEAKING ---
        # These states mean the speak() function is currently active.
        # The main loop should just breathe and let speak() handle its internal loop
        # (audio playback, lip-sync, event handling, interruption checking while speaking).
        # Transitions from these states happen *inside* or *immediately after* the speak() call returns
        # (to IDLE if finishes normally, to HANDLING_INTERRUPTION if interrupted).
        elif current_state in [STATE_SPEAKING_EXPLANATION, STATE_SPEAKING_ANSWER, STATE_SPEAKING_GREETING, STATE_SPEAKING_TRANSITION]:
            # debug_print(f"State: {current_state}. Speak function is active.", 4) # Very verbose
            # Add a small sleep to prevent the loop from consuming too much CPU
            time.sleep(0.01) # Main loop sleep

        else:
            # Should not happen, handle unexpected states
            print(f"❗ Unexpected state: {current_state}. Resetting to IDLE.")
            current_state = STATE_IDLE
            time.sleep(1) # Prevent rapid looping on error


    # --- End of Main Loop (when running is False) ---
    debug_print("Exiting main application loop.", 1)
    # Final cleanup handled in the finally block outside this function



# --- Start the AI Teacher ---
if __name__ == "__main__":
    # --- Global Pygame Display Variables ---
    # Initialize these here so they are accessible to functions like update_display
    screen = None
    font = None
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    # current_mouth_image and current_mouth_rect are global and set during loading/speaking

    print("Initializing AI Teacher...")

    # --- NLTK Download Check ---
    # Ensure punkt tokenizer is downloaded before splitting sentences
    try:
        nltk.data.find('tokenizers/punkt')
        debug_print("NLTK 'punkt' tokenizer found.", 2)
    except nltk.downloader.DownloadError:
        debug_print("NLTK 'punkt' tokenizer not found. Downloading...", 2)
        try:
            nltk.download('punkt')
            debug_print("NLTK 'punkt' tokenizer downloaded successfully.", 2)
        except Exception as e:
            print(f"❌ Error downloading NLTK 'punkt' tokenizer: {e}. Sentence splitting may not work correctly.")
    except Exception as e:
        print(f"❌ Unexpected error during NLTK check: {e}. Sentence splitting may not work correctly.")


    # --- Pygame Initialization ---
    try:
        pygame.init() # Initialize all pygame modules (mixer, display, font, etc.)
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Teacher Voice Agent")
        font = pygame.font.Font(None, 28) # Initialize default font
        pygame.mixer.init() # Explicitly initialize mixer again for clarity
        debug_print("Pygame initialized successfully (Display, Mixer, Font).", 2)
    except pygame.error as e:
        print(f"❌ Critical Error: Failed to initialize pygame: {e}. Cannot run visual or audio components.")
        # Exit if pygame fails to initialize
        exit()


    # --- Load Mouth Images ---
    # This sets the global mouth_images dictionary and the initial current_mouth_image
    initial_mouth_image = load_mouth_images()
    if initial_mouth_image:
        current_mouth_image = initial_mouth_image
        current_mouth_rect = current_mouth_image.get_rect(center=MOUTH_POS)
        debug_print("Mouth images loaded and initial pose set.", 2)
    else:
        print("❌ Critical Error: Could not load any mouth images. Cannot run visual component.")
        # Decide if you want to exit or run voice-only. Let's exit for now.
        # exit() # Exiting might be too harsh, the app might still work voice-only.
        # Let's just ensure current_mouth_image/rect are None if loading failed critically.
        current_mouth_image = None
        current_mouth_rect = None


    # --- Initial cleanup of temp files on startup ---
    cleanup_temp_files()
    debug_print("Initial temp file cleanup completed.", 2)

    # --- Start keyboard listener in a separate thread ---
    # Only start the thread ONCE in the main execution block
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    debug_print("Keyboard listener thread started.", 2)


    # --- Start the main AI Teacher interaction loop ---
    try:
        # Call the main function that contains the state loop
        run_voice_teacher()
    except KeyboardInterrupt:
        # Handle Ctrl+C from the console
        print("\nExiting AI Teacher (Ctrl+C detected)...")
    except Exception as e:
        # Catch any unhandled exception in the main loop
        print(f"\n❌ An unexpected error occurred in the main loop: {e}")

    finally:
        # --- Final Cleanup ---
        debug_print("Starting final cleanup...", 1)
        stop_speech() # Ensure audio is stopped just in case
        # Clean up all temp files one last time
        cleanup_temp_files(extensions=('.mp3', '.wav', '.json', '.tsv', '.txt')) # Add txt for dialog file cleanup
        debug_print("Final temp file cleanup completed.", 2)

        # Quit Pygame subsystems
        if pygame.get_init(): # Check if pygame was initialized before quitting
            pygame.quit()
            print("Pygame quit.")
        else:
            debug_print("Pygame was not initialized, skipping pygame.quit().", 2)

        print("AI Teacher session ended.")