import os
import time
import tempfile

import streamlit as st
import yt_dlp
from google import genai
from google.genai import types

# ---------- SETTINGS (you can change these) ----------
MODEL = "gemini-3.8-flash"   # if you get a "model not found" error, change this name
FREE_ANALYSES = 2            # free tries per visit
RETRY_WAITS = [4, 8, 15]     # seconds to wait before retrying when Gemini is busy
APP_NAME = "Viral Video Analyzer"
# -----------------------------------------------------

st.set_page_config(page_title=APP_NAME, page_icon="🔥")


def get_secret(name, default=""):
    """Read a secret from Streamlit secrets or from environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


API_KEY = get_secret("GEMINI_API_KEY")
PAY_LINK = get_secret("PAY_LINK")
ACCESS_CODES = [c.strip() for c in get_secret("ACCESS_CODES").split(",") if c.strip()]

PROMPT = """You are an expert short-form video strategist (TikTok, Reels, YouTube Shorts).
Watch this video carefully, including the audio, on-screen text and editing.

Creator's niche (may be empty): {niche}

Write a clear report with EXACTLY these sections:

## 1. The Hook (first 3 seconds)
What happens, what is said or shown, and why it stops the scroll.

## 2. Why This Video Works
Explain the psychology: emotion, curiosity, relatability, trend, controversy, humor, etc.
Be specific to THIS video. Do not guess view counts or claim certainty.

## 3. Structure Breakdown
A timeline with approximate timestamps: what happens in each part and its job.

## 4. Editing, Sound and Text
Pacing, cuts, music, voice, captions, text on screen, visual style.

## 5. Step-by-Step: How To Recreate It
A numbered list a beginner can follow with just a phone. Include what to film,
what to say, how to edit it, and what to write in the caption.

## 6. 3 Original Ideas For Your Niche
Three fresh video ideas using the same viral pattern, each with a hook line.

## 7. Viral Pattern Score
Give a score out of 10 for: Hook, Emotion, Shareability, Rewatch value.
Add one sentence for each.
"""


def is_youtube(link):
    return "youtube.com" in link or "youtu.be" in link


def download_video(url, folder):
    """Download a video link with yt-dlp and return the file path."""
    options = {
        "outtmpl": os.path.join(folder, "video.%(ext)s"),
        "format": "best[height<=720]/best",
        "noplaylist": True,
        "quiet": True,
        "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    for name in os.listdir(folder):
        if name.startswith("video."):
            return os.path.join(folder, name)
    raise RuntimeError("Could not download that video.")


def is_busy_error(e):
    """True when Google's AI is overloaded (temporary, not your fault)."""
    text = str(e).lower()
    return any(w in text for w in ["503", "unavailable", "overloaded", "high demand"])


def generate(client, contents):
    """Ask Gemini. If it is busy, wait a little and try again automatically."""
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return client.models.generate_content(model=MODEL, contents=contents).text
        except Exception as e:
            if is_busy_error(e) and attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            raise


def analyze(client, niche, url=None, file_path=None):
    """Send the video to Gemini and return the written report."""
    prompt = PROMPT.format(niche=niche or "not given")

    # YouTube links can be given to Gemini directly (most reliable)
    if url and is_youtube(url):
        contents = [types.Part(file_data=types.FileData(file_uri=url)), prompt]
        return generate(client, contents)

    # Everything else: upload the file to Gemini first
    video = client.files.upload(file=file_path)
    while video.state.name == "PROCESSING":
        time.sleep(3)
        video = client.files.get(name=video.name)
    if video.state.name == "FAILED":
        raise RuntimeError("Gemini could not process this video.")
    return generate(client, [video, prompt])


def show_error(e):
    """Show a clear message that says WHAT kind of problem happened."""
    text = str(e).lower()

    if is_busy_error(e):
        st.error("🚦 Google's AI is very busy right now (high demand). This is "
                 "temporary and not your fault. Wait 1-2 minutes and try again.")
    elif any(w in text for w in ["getaddrinfo", "11001", "connection", "timed out",
                               "timeout", "network", "name resolution",
                               "unreachable", "max retries"]):
        st.error("📡 Internet problem. Check your connection, turn off any VPN, "
                 "and try again.")
    elif "404" in text or "not_found" in text:
        st.error("⚙️ The AI model name is outdated. Change MODEL at the top of app.py.")
    elif "429" in text or "quota" in text or "resource_exhausted" in text:
        st.error("⏳ Too many requests, or the free limit is used up. "
                 "Wait a minute and try again.")
    elif "api key" in text or "api_key" in text or "401" in text or "403" in text \
            or "permission" in text:
        st.error("🔑 Problem with the API key. Check GEMINI_API_KEY in your secrets.")
    elif "unsupported url" in text or "unable to download" in text or "yt-dlp" in text \
            or "http error" in text or "could not download" in text:
        st.error("🔗 Could not download that link. Download the video yourself "
                 "and use the upload option instead.")
    else:
        st.error("Something went wrong. If this was a TikTok/Instagram link, "
                 "download the video and use the upload option instead.")

    st.caption(str(e))  # the real technical reason, in small grey text


# ---------- PAGE ----------
st.title(f"🔥 {APP_NAME}")
st.write("Paste a video link or upload a video. Get a breakdown of why it works "
         "and a step-by-step plan to make your own.")

if "used" not in st.session_state:
    st.session_state.used = 0

if not API_KEY:
    st.error("🔑 Missing GEMINI_API_KEY. Add it in your secrets (see the guide).")
    st.stop()

code = st.sidebar.text_input("Access code (if you paid)", type="password")
is_paid = code in ACCESS_CODES and code != ""
if is_paid:
    st.sidebar.success("Unlimited access active ✅")
else:
    left = max(FREE_ANALYSES - st.session_state.used, 0)
    st.sidebar.info(f"Free analyses left: {left}")

niche = st.text_input("Your niche (optional)", placeholder="e.g. fitness, comedy, cooking")
url = st.text_input("Video link (YouTube, TikTok, Instagram...)")
upload = st.file_uploader("...or upload a video file", type=["mp4", "mov", "webm"])

if st.button("Analyze 🚀", type="primary"):
    if not url and not upload:
        st.warning("Paste a link or upload a video first.")
        st.stop()

    if not is_paid and st.session_state.used >= FREE_ANALYSES:
        st.warning("You used your free analyses.")
        if PAY_LINK:
            st.markdown(f"👉 [Get unlimited access here]({PAY_LINK})")
        st.stop()

    try:
        client = genai.Client(api_key=API_KEY)
        with st.spinner("Watching the video... this can take about a minute"):
            with tempfile.TemporaryDirectory() as folder:
                if upload:
                    path = os.path.join(folder, "video_" + upload.name)
                    with open(path, "wb") as f:
                        f.write(upload.getbuffer())
                    result = analyze(client, niche, file_path=path)
                elif is_youtube(url):
                    result = analyze(client, niche, url=url)
                else:
                    path = download_video(url, folder)
                    result = analyze(client, niche, url=url, file_path=path)
        st.session_state.used += 1
        st.markdown(result)
    except Exception as e:
        show_error(e)

