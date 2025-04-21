from flask import Flask, request, render_template
from markupsafe import Markup
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
from IPython.display import display as ipython_display, HTML as ipython_HTML, Image as ipython_Image

app = Flask(__name__)

# -------------------------
# General Debugging Utility
# -------------------------
debug_mode = False  # Keep debugging off for the web app by default
def debug_print(message, level=1):
    if debug_mode:
        prefix = "  " * level
        print(f"{prefix}🔹 {message}")

# -------------------------
# Configuration & Data Loading
# -------------------------
IMAGE_DIR = "images"
FIGURES_JSON = "output.json"
KNOWLEDGEBASE_JSON = "knowledgebase.json"
METADATA_JSON = "metadata.json"
FAISS_TEXT_INDEX = "textbook_faiss.index"
FAISS_FIGURES_INDEX = "subchapter_faiss.index"
METADATA_FIGURES_JSON = "subchapter_metadata.json"

# Data for textual content
with open(KNOWLEDGEBASE_JSON, "r", encoding="utf-8") as f:
    kb_data = json.load(f)
with open(METADATA_JSON, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Normalize function for matching
def normalize_title(title):
    return title.strip().lower()

# Create normalized KB lookup
normalized_kb = {}
for chapter, topics in kb_data.items():
    for title, content in topics.items():
        norm_key = (chapter, normalize_title(title))
        normalized_kb[norm_key] = content

# Initialize embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load FAISS index
index = faiss.read_index(FAISS_TEXT_INDEX)

# -------------------------
# Search Function (as is)
# -------------------------
def search(query, top_k=5, similarity_threshold=0.98, mode="hybrid"):
    norm_query = normalize_title(query)
    results = []
    seen_embeddings = []
    seen_titles = set()

    def get_exact_matches():
        for item in metadata:
            title = item["title"]
            chapter = item["chapter"]
            norm_title = normalize_title(title)
            if norm_query in norm_title:
                norm_key = (chapter, norm_title)
                content = normalized_kb.get(norm_key)
                if content:
                    seen_titles.add(norm_key)
                    return [{
                        "title_key": title,
                        "chapter": chapter,
                        "score": 0.0,
                        "content": content
                    }]
        return []

    def get_semantic_matches():
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        semantic_results = []

        for i in range(len(indices[0])):
            idx = indices[0][i]
            raw_title = metadata[idx]["title"]
            chapter = metadata[idx]["chapter"]
            norm_key = (chapter, normalize_title(raw_title))
            content = normalized_kb.get(norm_key)

            if content and norm_key not in seen_titles:
                content_embedding = model.encode(content, convert_to_tensor=True)

                # Check for semantic duplication
                is_duplicate = False
                for prev_emb in seen_embeddings:
                    if util.cos_sim(content_embedding, prev_emb).item() >= similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    seen_embeddings.append(content_embedding)
                    seen_titles.add(norm_key)
                    semantic_results.append({
                        "title_key": raw_title,
                        "chapter": chapter,
                        "score": distances[0][i],
                        "content": content
                    })
        return semantic_results

    # MODE HANDLING
    if mode == "exact":
        results = get_exact_matches()
    elif mode == "semantic":
        results = get_semantic_matches()
    else:  # hybrid
        results = get_exact_matches()
        if not results:
            results = get_semantic_matches()

    return results

# -------------------------
# Image Fetching Code (as is, with adjustments for Flask)
# -------------------------
# Data & FAISS index for figure retrieval
def load_figures():
    with open(FIGURES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

figures_data = load_figures()

# Initialize FAISS and embedding model for image retrieval (reusing the same model for simplicity)
index_figures = faiss.read_index(FAISS_FIGURES_INDEX)

# Load metadata mapping (Index → Subchapter)
with open(METADATA_FIGURES_JSON, "r", encoding="utf-8") as f:
    metadata_figures = json.load(f)

def search_exact_subchapter(query, top_k=1):
    """Find the most relevant subchapter using FAISS."""
    debug_print(f"Searching for exact subchapter match: {query}")
    query_embedding = image_model.encode([query], convert_to_numpy=True).astype('float32')
    _, indices = index_figures.search(query_embedding.reshape(1, -1), top_k)
    # Pick only the closest match
    best_match_index = str(indices[0][0])
    best_subchapter = metadata_figures.get(best_match_index, None)
    debug_print(f"Best match subchapter: {best_subchapter}", 2)
    return best_subchapter

def get_image_path(figure_ref):
    """Find image path with multiple fallback patterns."""
    debug_print(f"Locating image for: {figure_ref}", 2)
    base_name = figure_ref.replace(" ", "_")
    attempts = [
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"figure_{base_name}.png"
    ]
    for attempt in attempts:
        test_path = os.path.join(IMAGE_DIR, attempt)
        if os.path.exists(test_path):
            debug_print(f"✅ Found image at: {test_path}", 3)
            return os.path.join(IMAGE_DIR, attempt) # Return the full path for Flask
    debug_print("❌ No valid image path found", 3)
    return None

def fetch_figures_only(subchapter_name): # Changed parameter name to be more explicit
    """Retrieve only figures (images + raw descriptions) for a given subchapter."""
    debug_print(f"Retrieving figures for subchapter: {subchapter_name}")
    figures = [fig for fig in figures_data if fig["subchapter"] == subchapter_name]
    if not figures:
        debug_print(f"No relevant figures found for subchapter: {subchapter_name}")
        return "No relevant figures found."
    figure_blocks = []
    for fig in figures:
        fig_path = get_image_path(fig['figure'])
        if fig_path:
            figure_blocks.append({
                "name": fig['figure'],
                "path": fig_path,
                "desc": fig['description']
            })
    return figure_blocks

# -------------------------
# Revised Figure Retrieval for Lesson Multimedia Integration (Adjusted for Flask)
# -------------------------
def retrieve_and_expand_figures(query):
    """
    Retrieve figures related to the query and generate HTML to display them.
    """
    search_results = search(query, mode="hybrid", top_k=1)
    if not search_results:
        return "<p>No relevant text found for image retrieval.</p>"

    best_text_match = search_results[0]
    subchapter_name = best_text_match["title_key"] # Use the title_key as the subchapter name

    blocks = fetch_figures_only(subchapter_name)
    if isinstance(blocks, str):
        # An error message was returned
        return f"<p>{blocks}</p>"

    figure_html = "<div style='margin-top: 20px;'><h3>📊 Visual Aids</h3>"
    # Limit to 3 figures
    for fig in blocks[:3]:
        clean_desc = fig['desc']  # Optionally, you can process the description further
        figure_html += f"""
        <div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>
            <img src='/{fig['path']}' style='max-width: 100%; height: auto; display: block; margin: 0 auto;'>
            <p style='text-align: center; font-style: italic;'>{clean_desc or 'Visual demonstration'}</p>
        </div>
        """
    figure_html += "</div>"
    return figure_html

# -------------------------
# Functions for Video & Lesson Generation (Adjusted for Flask)
# -------------------------
def fetch_animated_videos(topic, num_videos=1):
    search_query = f"ytsearch{num_videos}:{topic} animation explained in english"
    print(f"Searching for: {search_query}")

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)

            if "entries" in info and len(info["entries"]) > 0:
                video = info["entries"][0]
                print(f"Found video: {video['title']}")
                if video.get("duration", 301) <= 300:
                    return {
                        "title": video["title"],
                        "url": video["url"],
                        "id": video["id"]
                    }
        except yt_dlp.DownloadError as e:
            print(f"Error fetching video: {e}")
            return None
    return None

def generate_topic_hook(topic):
    """Generate a short, engaging hook for the topic using the LLM."""
    LLM_API_KEY = "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" # Ensure this is secure in a real app
    LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
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
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.9
            }
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        hook = response.json()["choices"][0]["message"]["content"].strip()
        return hook
    except requests.exceptions.RequestException as e:
        print(f"Error generating topic hook: {e}")
        return "Let's explore this exciting topic!" # Fallback

def generate_funny_intro(topic):
    """Generate an introduction that begins with a funny story or meme about the topic."""
    LLM_API_KEY = "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" # Ensure this is secure
    LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"""
You are a creative and humorous science educator. Tell a short, funny story or describe a relatable meme about *{topic}* to engage 8th-grade students. Avoid using video introductions. Return ONLY the story.
"""
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.9
            }
        )
        response.raise_for_status()
        funny_intro = response.json()["choices"][0]["message"]["content"].strip()
        return funny_intro
    except requests.exceptions.RequestException as e:
        print(f"Error generating funny intro: {e}")
        return f"Get ready for some fun as we dive into {topic}!" # Fallback

def generate_dynamic_intro(topic):
    """Generate an introductory paragraph with a funny story or meme."""
    funny_intro = generate_funny_intro(topic)
    hook = generate_topic_hook(topic)
    return f"""
<p>{funny_intro}</p>
<p>{hook}</p>
<p>Today, we're exploring the fascinating world of <strong>{topic}</strong>! 🔍<br>
Quick prediction: What do you think happens when...? Let's find out in our lesson!</p>
"""

def generate_text_lesson(query):
    """Generate a dynamic lesson using the FAISS-retrieved textbook content."""
    debug_print(f"Searching for relevant text using hybrid search: {query}")
    search_results = search(query, mode="hybrid", top_k=1) # Adjust top_k as needed
    if not search_results:
        return "<p>No relevant information found.</p>"
    # Use the top result from the search
    best_match = search_results[0]
    retrieved_content = best_match["content"]
    cleaned_title = re.sub(r"^\d+(\.\d+)*\s*", "", best_match["title_key"]).strip()
    # Build the introductory section using a funny story/meme.
    introduction = generate_dynamic_intro(cleaned_title)
    # Enhanced Explanation Generation from Textbook Content via LLM
    LLM_API_KEY = "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" # Ensure this is secure
    LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"""
You are an engaging, fun-loving, and knowledgeable 8th-grade science teacher.
Below is the textbook content for the topic titled '{cleaned_title}'.
Your task is to generate a richly detailed, smooth, and engaging explanation that:
- Uses every sentence from the textbook content as a base.
- Expands each idea with real-life analogies, fun facts, surprising trivia, and interesting stories kids can relate to.
- Breaks down complex terms into simple, visual language.
- Feels like a passionate teacher telling a story, not reading a script.
- Uses HTML with <h2>, <h3>, <p>, and <ul><li> where helpful.
- Ensures smooth transitions between paragraphs.
Textbook Content:
"{retrieved_content}"
"""
    debug_print("Sending LLM request with enhanced textbook expansion prompt...", 2)
    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.8
            }
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result:
            ai_explanation = result["choices"][0]["message"]["content"]
        else:
            ai_explanation = f"<p>Error: {result}</p>"
    except requests.exceptions.RequestException as e:
        ai_explanation = f"<p>Error generating explanation: {e}</p>"

    # Multimedia Integration: Figures & Video
    multimedia_html = retrieve_and_expand_figures(query)
    video = fetch_animated_videos(cleaned_title)
    if video:
        video_html = f"""
        <div style='margin: 20px 0;'>
            <h3>🎥 Video Explanation</h3>
            <p>Watch this short animation about {cleaned_title}:</p>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/{video['id']}"
                        frameborder="0" allowfullscreen style='max-width: 100%;'></iframe>
                        <p><em>{video['title']}</em></p>
        </div>
        """
        multimedia_html += video_html
    # Build the final lesson HTML
    text_lesson_html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h2>🌟 Introduction</h2>
        {introduction}
        <h2>📚 Deep Dive Explanation</h2>
        <div>{ai_explanation}</div>
        {multimedia_html}
        <h2>🎓 Key Takeaways</h2>
        <p><strong>Summary:</strong> We covered all the key points in detail with stories,
        analogies, and visuals that make learning fun and meaningful. Keep exploring!</p>
    </div>
    """
    return text_lesson_html

# -------------------------
# Final Integration: AI Teacher Lesson with Multimedia (Adjusted for Flask)
# -------------------------
def generate_ai_teacher_lesson(query):
    debug_print("Generating AI Teacher Lesson...")
    text_lesson_html = generate_text_lesson(query)
    final_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            h2 {{ color: #3498db; margin-top: 25px; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            h3 {{ color: #27ae60; margin-top: 20px; }}
            p {{ margin-bottom: 15px; }}
            .lesson-container {{ background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }}
            .video-container {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }}
            .video-container iframe {{ width: 100%; display: block; }}
            .figure-container {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; }}
            .figure-container img {{ max-width: 100%; height: auto; display: block; margin: 0 auto 10px auto; }}
            .figure-container p {{ font-style: italic; color: #777; }}
        </style>
    </head>
    <body>
        <div class="lesson-container">
            <h1>AI Teacher Lesson</h1>
            {text_lesson_html}
        </div>
    </body>
    </html>
    """
    return final_html

# -------------------------
# Flask Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/lesson", methods=["POST"])
def generate_lesson():
    query = request.form["query"]
    lesson_html = generate_ai_teacher_lesson(query)
    return render_template("lesson.html", lesson=Markup(lesson_html))

@app.route('/<path:filename>')
def send_image(filename):
    return send_from_directory('.', filename)

if __name__ == "__main__":
    from flask import send_from_directory
    app.run(debug=False) # Set debug to False for production