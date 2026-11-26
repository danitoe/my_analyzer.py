"""
MY OWN VIDEO ANALYZER  -  accounts, analysis points, history and AI ideas.

How it works (3 steps):
  1. MEASURE: ffmpeg counts cuts, checks loudness and silence.
     (Optional) Whisper, a free open-source model, writes down what is said.
  2. SCORE: simple rules turn the numbers into scores out of 10.
  3. REPORT: the numbers + scores become a written breakdown.

Every rule is in plain Python below, so you can read it, change it, and
later replace the rules with a model you train yourself.
"""
import base64
import csv
import datetime
import hashlib
import hmac
import io
import json
import os
import random
import re
import subprocess
import tempfile
import uuid

MAX_SECONDS = 180          # only analyze the first 3 minutes (keeps it fast)

# ---- PLANS (change these numbers to change your plans) ----
POINTS_PER_ANALYSIS = 10       # one video analysis costs 10 points
FREE_AI_PER_DAY = 40           # safety cap: free analyses per day for ALL free users together
HISTORY_STORE_MAX = 50         # how many analyses are kept per user
PLANS = {
    "free":    {"label": "Free",    "points": 20,  "days": 10.5, "history": 2,
                "ideas": 1, "price": "$0"},
    "creator": {"label": "Creator", "points": 150, "days": 30,   "history": 30,
                "ideas": 3, "price": "$9 / month"},
    "pro":     {"label": "Pro",     "points": 500, "days": 30,   "history": 50,
                "ideas": 3, "price": "$19 / month"},
}
CUT_SENSITIVITY = 0.30     # lower = more cuts detected, higher = fewer
HOOK_WORDS = ["why", "how", "never", "stop", "wait", "secret", "nobody",
              "mistake", "truth", "watch", "free", "worst", "best", "what if",
              "don't", "dont", "you"]


# ---------------------------------------------------------------
#  STEP 1: MEASURE
# ---------------------------------------------------------------
def run_ffmpeg(args):
    """Run ffmpeg and return everything it printed (its log)."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats"] + args,
                            capture_output=True, text=True)
    return result.stderr


def probe(path):
    """Ask ffprobe for duration, size, frame rate and whether there is audio."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True).stdout
    data = json.loads(out or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    fps = 0.0
    try:
        a, b = video.get("r_frame_rate", "0/1").split("/")
        fps = float(a) / float(b) if float(b) else 0.0
    except Exception:
        pass
    return {
        "duration": duration,
        "width": video.get("width", 0),
        "height": video.get("height", 0),
        "fps": round(fps, 1),
        "has_audio": has_audio,
    }


def find_cuts(path):
    """Return the times (in seconds) where the picture changes sharply."""
    log = run_ffmpeg(["-i", path, "-t", str(MAX_SECONDS), "-an",
                      "-vf", f"select='gt(scene,{CUT_SENSITIVITY})',showinfo",
                      "-f", "null", "-"])
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", log)]


def audio_stats(path):
    """Return average volume, peak volume and how much of the video is silent."""
    log = run_ffmpeg(["-i", path, "-t", str(MAX_SECONDS), "-vn",
                      "-af", "volumedetect,silencedetect=noise=-35dB:d=0.4",
                      "-f", "null", "-"])
    mean = re.search(r"mean_volume:\s*(-?[0-9.]+)", log)
    peak = re.search(r"max_volume:\s*(-?[0-9.]+)", log)
    silent = sum(float(x) for x in re.findall(r"silence_duration:\s*([0-9.]+)", log))
    return {
        "mean_volume": float(mean.group(1)) if mean else None,
        "peak_volume": float(peak.group(1)) if peak else None,
        "silence_seconds": silent,
    }


_whisper = None


def transcribe(path):
    """Write down the speech with Whisper (free, open source, no key).
    Returns a list of (start, end, text), or None if Whisper is not installed."""
    global _whisper
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    try:
        if _whisper is None:
            _whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = _whisper.transcribe(path, vad_filter=True)
        return [(s.start, s.end, s.text.strip()) for s in segments if s.start < MAX_SECONDS]
    except Exception:
        return None


def measure(path):
    """Do all the measuring and return one dictionary of facts."""
    info = probe(path)
    length = min(info["duration"], MAX_SECONDS) or 1
    cuts = find_cuts(path)
    facts = dict(info)
    facts["analyzed_seconds"] = round(length, 1)
    facts["cuts"] = len(cuts)
    facts["cuts_first_3s"] = sum(1 for t in cuts if t <= 3)
    facts["avg_shot_seconds"] = round(length / (len(cuts) + 1), 2)

    if info["has_audio"]:
        facts.update(audio_stats(path))
        facts["silence_ratio"] = round(facts["silence_seconds"] / length, 2)
    else:
        facts.update({"mean_volume": None, "peak_volume": None,
                      "silence_seconds": 0, "silence_ratio": 0})

    segs = transcribe(path) if info["has_audio"] else None
    facts["has_transcript"] = segs is not None
    facts["transcript"] = " ".join(s[2] for s in segs) if segs else ""
    facts["first_speech_at"] = round(segs[0][0], 1) if segs else None
    first_8s = " ".join(s[2] for s in segs if s[0] <= 8).lower() if segs else ""
    facts["hook_text"] = first_8s
    facts["hook_words_found"] = [w for w in HOOK_WORDS
                                 if re.search(r"\b" + re.escape(w) + r"\b", first_8s)]
    facts["hook_has_question"] = "?" in first_8s
    words = len(facts["transcript"].split())
    facts["words_per_minute"] = round(words / (length / 60)) if segs and words else None
    return facts


# ---------------------------------------------------------------
#  STEP 2: SCORE  (these are RULES OF THUMB you can change)
# ---------------------------------------------------------------
def score(f):
    d = f["duration"]
    if 15 <= d <= 45:
        length_score = 10
    elif 10 <= d < 15 or 45 < d <= 60:
        length_score = 8
    elif 60 < d <= 90:
        length_score = 6
    else:
        length_score = 4

    s = f["avg_shot_seconds"]
    if 1 <= s <= 3.5:
        pace_score = 9
    elif 0.6 <= s < 1:
        pace_score = 8
    elif 3.5 < s <= 6:
        pace_score = 6
    else:
        pace_score = 4

    hook = 4
    if f["cuts_first_3s"] >= 1:
        hook += 2
    if f["has_transcript"]:
        if f["first_speech_at"] is not None and f["first_speech_at"] <= 1.0:
            hook += 2
        if f["hook_words_found"] or f["hook_has_question"]:
            hook += 2
    hook = min(hook, 10)

    if not f["has_audio"]:
        audio = 3
    else:
        m = f["mean_volume"]
        if m is None:
            audio = 5
        elif -24 <= m <= -10:
            audio = 9
        elif m < -30:
            audio = 3
        elif m < -24:
            audio = 6
        else:
            audio = 6
        if f["silence_ratio"] > 0.35:
            audio = max(audio - 2, 1)

    return {"Hook": hook, "Pacing": pace_score, "Audio": audio, "Length": length_score}


# ---------------------------------------------------------------
#  STEP 3: REPORT
# ---------------------------------------------------------------
def build_report(f, scores, stats=None):
    lines = []
    lines.append("## 1. Measured facts")
    lines.append(f"- Length: **{f['duration']:.0f} s**, {f['width']}x{f['height']}, {f['fps']} fps")
    lines.append(f"- Cuts: **{f['cuts']}** (about one every **{f['avg_shot_seconds']} s**)")
    lines.append(f"- Cuts in the first 3 seconds: **{f['cuts_first_3s']}**")
    if f["has_audio"] and f["mean_volume"] is not None:
        lines.append(f"- Average volume: **{f['mean_volume']} dB**, silence: "
                     f"**{int(f['silence_ratio'] * 100)}%** of the video")
    elif not f["has_audio"]:
        lines.append("- This video has **no audio**")
    if f["has_transcript"]:
        if f["first_speech_at"] is not None:
            lines.append(f"- First words spoken at **{f['first_speech_at']} s**")
        if f["words_per_minute"]:
            lines.append(f"- Speaking speed: **{f['words_per_minute']} words/min**")
    else:
        lines.append("- Speech was not transcribed (Whisper is not installed here)")

    if stats:
        lines.append("")
        lines.append("## 2. Public numbers")
        views = stats.get("view_count") or 0
        likes = stats.get("like_count") or 0
        comments = stats.get("comment_count") or 0
        lines.append(f"- Views: **{views:,}**, likes: **{likes:,}**, comments: **{comments:,}**")
        if views:
            lines.append(f"- Like rate: **{likes / views * 100:.1f}%**, "
                         f"comment rate: **{comments / views * 100:.2f}%**")
            lines.append("- Rule of thumb: a like rate above about 4% means viewers "
                         "really liked it, not just watched.")

    lines.append("")
    lines.append("## 3. Scores (out of 10)")
    for name, value in scores.items():
        lines.append(f"- **{name}: {value}/10**")

    lines.append("")
    lines.append("## 4. What the numbers suggest")
    notes = []
    if f["cuts_first_3s"] >= 1:
        notes.append("The picture changes in the first 3 seconds, which helps stop the scroll.")
    else:
        notes.append("Nothing changes in the first 3 seconds. Add a cut, zoom or text right at the start.")
    if f["avg_shot_seconds"] > 6:
        notes.append("Shots are long (slow pacing). Fast pacing usually holds attention better.")
    elif f["avg_shot_seconds"] < 0.6:
        notes.append("Cuts are extremely fast. Make sure viewers can still follow the story.")
    else:
        notes.append("The cutting speed is in a healthy range for short videos.")
    if f["has_transcript"]:
        if f["hook_words_found"]:
            notes.append("The opening words use attention words: " + ", ".join(f["hook_words_found"]) + ".")
        else:
            notes.append("The opening words use no obvious attention words (why, how, never, stop...).")
        if f["hook_has_question"]:
            notes.append("The video opens with a question, which creates curiosity.")
    if f["has_audio"] and f["silence_ratio"] > 0.35:
        notes.append("A lot of the video is silent. Dead air loses viewers.")
    if f["duration"] > 60:
        notes.append("Long for a short-form video. Under 45 seconds often performs better.")
    for n in notes:
        lines.append(f"- {n}")

    target_len = "15-45 seconds" if f["duration"] > 45 else f"about {max(int(f['duration']), 10)} seconds"
    lines.append("")
    lines.append("## 5. How to make something similar")
    lines.append(f"1. Aim for a video of **{target_len}**.")
    lines.append("2. Start with a hook in the **first 3 seconds**: a question, a bold claim, or a surprising picture.")
    lines.append(f"3. Cut or change the picture about every **{min(max(f['avg_shot_seconds'], 1), 3.5):.1f} seconds**.")
    lines.append("4. Start speaking within the first second. No slow intro.")
    lines.append("5. Keep your voice and music loud and clear, with no long silences.")
    lines.append("6. End with one clear action: follow, comment, or watch the next video.")

    lines.append("")
    lines.append("*These are measured facts and simple rules of thumb. This tool does not "
                 "understand the story or the joke, and it cannot promise a video will go viral.*")
    return "\n".join(lines)


# The numbers your own model learns from (must match train_my_model.py)
FEATURES = ["duration", "cuts", "cuts_first_3s", "avg_shot_seconds",
            "silence_ratio", "mean_volume"]
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


def model_prediction(f):
    """If you trained a model (model.joblib in this folder), return the chance
    (0 to 1) that this video looks like a 'winner'. Otherwise return None."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        import joblib
        model = joblib.load(MODEL_FILE)
        row = [[f.get(k) if f.get(k) is not None else -30 for k in FEATURES]]
        return float(model.predict_proba(row)[0][1])
    except Exception:
        return None


def facts_to_csv(f, scores, stats=None, extra=None):
    """One row of data. Collect many rows, then you can TRAIN your own model."""
    row = {k: v for k, v in f.items()
           if k not in ("transcript", "hook_text", "hook_words_found")}
    row.update({f"score_{k.lower()}": v for k, v in scores.items()})
    if stats:
        for k in ("view_count", "like_count", "comment_count"):
            row[k] = stats.get(k)
    if extra:
        row.update(extra)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------
#  DOWNLOAD A LINK (needs yt-dlp)
# ---------------------------------------------------------------
def fetch_link(url, folder):
    import yt_dlp
    options = {
        "outtmpl": os.path.join(folder, "video.%(ext)s"),
        # YouTube sends picture and sound separately, so ask for both and join them
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    path = next((os.path.join(folder, n) for n in os.listdir(folder)
                 if n.startswith("video.")), None)
    if not path:
        raise RuntimeError("Could not download that video.")
    stats = {k: info.get(k) for k in ("view_count", "like_count", "comment_count", "title")}
    return path, stats


def friendly_error(e):
    text = str(e).lower()
    if any(w in text for w in ["getaddrinfo", "11001", "connection", "timed out",
                               "timeout", "network", "name resolution"]):
        return "📡 Internet problem. Check your connection and try again."
    if any(w in text for w in ["unsupported url", "unable to download", "http error",
                               "could not download", "sign in", "403", "429",
                               "requested format", "not available", "bot"]):
        return ("🔗 Could not download that link. Download the video yourself "
                "and use the upload option instead.")
    if "no such file" in text and "ffmpeg" in text:
        return "🛠️ ffmpeg is not installed on this machine."
    return "😕 Something went wrong."




# ---------------------------------------------------------------
#  OPTIONAL: AI COACH  (uses free AI keys if you add them)
#  The app tries each AI in order. If one is busy, empty or over its
#  limit, it moves on to the next. With NO keys the app still works.
# ---------------------------------------------------------------
# If you see "model not found", copy the current model name from that
# provider's model list and paste it in "model" below.
PROVIDERS = [
    {"name": "Groq", "key": "GROQ_API_KEY",
     "url": "https://api.groq.com/openai/v1/chat/completions",
     "model": "meta-llama/llama-4-scout-17b-16e-instruct", "max_images": 4},
    {"name": "OpenRouter", "key": "OPENROUTER_API_KEY",
     "url": "https://openrouter.ai/api/v1/chat/completions",
     "model": "nvidia/nemotron-nano-12b-v2-vl:free", "max_images": 4},
    {"name": "Gemini", "key": "GEMINI_API_KEY",
     "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
     "model": "gemini-3.8-flash", "max_images": 6},
]


def get_secret(name):
    """Read a key from Streamlit secrets or from environment variables."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def ai_keys_present():
    return any(get_secret(p["key"]) for p in PROVIDERS)


def grab_frames(path, duration, n=6):
    """Take n small snapshots spread across the video (as base64 text)."""
    frames = []
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            t = max(duration, 1) * (i + 0.5) / n
            out = os.path.join(d, f"f{i}.jpg")
            run_ffmpeg(["-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                        "-vf", "scale=512:-2", "-q:v", "6", out])
            if os.path.exists(out):
                with open(out, "rb") as fh:
                    frames.append(base64.b64encode(fh.read()).decode())
    return frames


def facts_lines(f):
    """Turn the measured numbers into short lines of text for the AI."""
    lines = [
        f"Length: {f['duration']:.0f} s",
        f"Cuts: {f['cuts']} (one every {f['avg_shot_seconds']} s), "
        f"{f['cuts_first_3s']} in the first 3 s",
    ]
    if f["has_audio"] and f["mean_volume"] is not None:
        lines.append(f"Average volume {f['mean_volume']} dB, silence {int(f['silence_ratio'] * 100)}%")
    if f["has_transcript"]:
        lines.append("Spoken words: " + f["transcript"][:1500])
    return lines


def build_json_prompt(f, scores, niche):
    return (
        "You are a short-form video coach (TikTok, Reels, Shorts).\n"
        "You get a few snapshots from one video (in time order) plus measurements made by software.\n"
        "You did NOT hear the audio or see motion, so be honest about that. "
        "Never claim certainty about why a video went viral.\n\n"
        f"Creator niche: {niche}\n"
        "Measurements:\n- " + "\n- ".join(facts_lines(f)) + "\n"
        "Scores (rules of thumb): " + ", ".join(f"{k} {v}/10" for k, v in scores.items()) + "\n\n"
        "Reply with ONLY valid JSON. No markdown fences, no extra words. Use exactly this shape:\n"
        "{\n"
        '  "summary": "2 short sentences about what the video is",\n'
        '  "hook": "what the opening shows and whether it grabs attention",\n'
        '  "why_it_works": "likely reasons it holds attention (say which are guesses)",\n'
        '  "structure": "setup, middle, payoff in simple words",\n'
        '  "recreate_steps": ["step 1", "step 2", "step 3", "step 4", "step 5"],\n'
        '  "ideas": [\n'
        "    {\n"
        '      "title": "a video title for the creator\'s niche",\n'
        '      "viral_rating": 70,\n'
        '      "hook_line": "the first sentence to say or show",\n'
        '      "what_to_film": "exactly what video to make",\n'
        '      "why_it_could_go_viral": "the reasons this idea can spread",\n'
        '      "steps": ["step 1", "step 2", "step 3", "step 4"],\n'
        '      "caption": "a caption with 3 hashtags"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Give EXACTLY 3 different ideas in the ideas list, all fitting the niche.\n"
        "viral_rating is your honest estimate from 1 to 99. Most ideas deserve 30 to 80. "
        "Never give a rating above 90.\n"
        "Write in simple, clear English a beginner understands."
    )


def _text(value):
    return str(value).strip() if value is not None else ""


def _steps(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_text(s) for s in value if _text(s)][:8]


def clean_ai_data(data):
    """Check the AI's answer has the shape we need. Returns clean data or None."""
    if not isinstance(data, dict):
        return None
    ideas = data.get("ideas")
    if not isinstance(ideas, list) or len(ideas) < 3:
        return None
    clean_ideas = []
    for item in ideas[:3]:
        if not isinstance(item, dict) or not _text(item.get("title")):
            return None
        try:
            rating = int(float(item.get("viral_rating", 50)))
        except (TypeError, ValueError):
            rating = 50
        clean_ideas.append({
            "title": _text(item["title"])[:120],
            "viral_rating": max(1, min(rating, 99)),
            "hook_line": _text(item.get("hook_line")),
            "what_to_film": _text(item.get("what_to_film")),
            "why_it_could_go_viral": _text(item.get("why_it_could_go_viral")),
            "steps": _steps(item.get("steps")),
            "caption": _text(item.get("caption")),
        })
    clean = {
        "summary": _text(data.get("summary")),
        "hook": _text(data.get("hook")),
        "why_it_works": _text(data.get("why_it_works")),
        "structure": _text(data.get("structure")),
        "recreate_steps": _steps(data.get("recreate_steps")),
        "ideas": clean_ideas,
    }
    if not (clean["why_it_works"] or clean["summary"]):
        return None
    return clean


def parse_ai_json(text):
    """Pull the JSON out of the AI's reply (it sometimes adds extra words)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(t[start:end + 1])
    except Exception:
        return None
    return clean_ai_data(data)


def ask_ai_json(f, scores, niche, frames):
    """Try each AI with a key, in order, until one gives a usable answer.
    Returns (name, data, list_of_errors). data is None if all failed."""
    import requests
    errors = []
    prompt = build_json_prompt(f, scores, niche)
    for p in PROVIDERS:
        key = get_secret(p["key"])
        if not key:
            continue
        content = [{"type": "text", "text": prompt}]
        for b64 in frames[:p["max_images"]]:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + b64}})
        try:
            r = requests.post(
                p["url"],
                headers={"Authorization": "Bearer " + key},
                json={"model": p["model"], "max_tokens": 3000,
                      "messages": [{"role": "user", "content": content}]},
                timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {r.text[:160]}")
            text = r.json()["choices"][0]["message"]["content"]
            data = parse_ai_json(text)
            if data is None:
                raise RuntimeError("the answer was not in the expected format")
            return p["name"], data, errors
        except Exception as e:
            errors.append(f"{p['name']}: {e}")
    return None, None, errors


# ---------------------------------------------------------------
#  DATABASE  (remembers accounts, points and history)
#  Everything is saved as small JSON "documents" under a key like
#  "user:someone@email.com". Two versions of the same idea:
#    SupabaseStore = a real free online database (use this)
#    LocalStore    = a file next to the app (for testing; it is erased
#                    when free hosting restarts, so do NOT rely on it)
# ---------------------------------------------------------------
class LocalStore:
    def __init__(self, path=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.path = path or os.path.join(base, "local_data.json")

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def get(self, key):
        return self._load().get(key)

    def set(self, key, value):
        data = self._load()
        data[key] = value
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)


class SupabaseStore:
    def __init__(self, url, key):
        self.url = url.rstrip("/") + "/rest/v1/app_data"
        self.headers = {"apikey": key, "Authorization": "Bearer " + key,
                        "Content-Type": "application/json"}

    def get(self, key):
        import requests
        r = requests.get(self.url, headers=self.headers, timeout=20,
                         params={"key": "eq." + key, "select": "value"})
        r.raise_for_status()
        rows = r.json()
        return rows[0]["value"] if rows else None

    def set(self, key, value):
        import requests
        headers = dict(self.headers)
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
        r = requests.post(self.url, headers=headers, timeout=20,
                          json={"key": key, "value": value})
        r.raise_for_status()


def make_store():
    url, key = get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY")
    if url and key:
        return SupabaseStore(url, key)
    return LocalStore()


# ---------------------------------------------------------------
#  ACCOUNTS, POINTS, HISTORY  (plain functions, easy to test)
# ---------------------------------------------------------------
def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def to_iso(dt):
    return dt.isoformat()


def from_iso(text):
    return datetime.datetime.fromisoformat(text)


def normalize_email(email):
    return (email or "").strip().lower()


def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), digest.hex()


def user_key(email):
    return "user:" + email


def new_user(email, password, now):
    salt, pw_hash = hash_password(password)
    return {"email": email, "salt": salt, "hash": pw_hash, "plan": "free",
            "points": PLANS["free"]["points"], "reset_at": None,
            "plan_expires": None, "history": [], "created": to_iso(now)}


def signup(store, email, password, now):
    """Returns (user, error_text)."""
    email = normalize_email(email)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return None, "Please enter a valid email."
    if len(password or "") < 6:
        return None, "Your password needs at least 6 characters."
    if store.get(user_key(email)):
        return None, "This email already has an account. Use Log in."
    user = new_user(email, password, now)
    store.set(user_key(email), user)
    return user, None


def login(store, email, password):
    """Returns (user, error_text)."""
    email = normalize_email(email)
    user = store.get(user_key(email)) if email else None
    if not user:
        return None, "Wrong email or password."
    _, pw_hash = hash_password(password or "", user["salt"])
    if not hmac.compare_digest(pw_hash, user["hash"]):
        return None, "Wrong email or password."
    return user, None


def refresh_user(user, now):
    """Apply plan expiry and the free points reset. Returns True if something changed."""
    changed = False
    if user["plan"] != "free":
        expires = user.get("plan_expires")
        if not expires or now >= from_iso(expires):
            user.update(plan="free", points=PLANS["free"]["points"],
                        reset_at=None, plan_expires=None)
            changed = True
    else:
        reset_at = user.get("reset_at")
        if reset_at and now >= from_iso(reset_at):
            # Always back to exactly the free amount, never more.
            user["points"] = PLANS["free"]["points"]
            user["reset_at"] = None
            changed = True
    return changed


def can_afford(user):
    return user["points"] >= POINTS_PER_ANALYSIS


def spend_points(user, now):
    user["points"] -= POINTS_PER_ANALYSIS
    if user["plan"] == "free" and not user.get("reset_at"):
        # The waiting clock starts the first time points are spent.
        user["reset_at"] = to_iso(now + datetime.timedelta(days=PLANS["free"]["days"]))


def time_left_text(iso_time, now):
    seconds = int((from_iso(iso_time) - now).total_seconds())
    if seconds <= 0:
        return "a moment"
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days} day{'s' if days != 1 else ''} {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    return f"{max(minutes, 1)} min"


def activate_plan(user, plan, now):
    info = PLANS[plan]
    start = now
    expires = user.get("plan_expires")
    if user["plan"] == plan and expires and from_iso(expires) > now:
        start = from_iso(expires)          # renewing: add time on top
    end = start + datetime.timedelta(days=info["days"])
    user.update(plan=plan, points=info["points"],
                plan_expires=to_iso(end), reset_at=to_iso(end))


def redeem_code(store, user, code, now):
    """Turn a code you sold into a plan. Returns (ok, message)."""
    code = (code or "").strip()
    if not code:
        return False, "Type your code first."
    plan = None
    for name, secret_name in (("creator", "CREATOR_CODES"), ("pro", "PRO_CODES")):
        valid = [c.strip() for c in get_secret(secret_name).split(",") if c.strip()]
        if code in valid:
            plan = name
    if not plan:
        return False, "That code is not valid."
    if store.get("usedcode:" + code):
        return False, "That code was already used."
    activate_plan(user, plan, now)
    store.set("usedcode:" + code, {"by": user["email"], "at": to_iso(now), "plan": plan})
    store.set(user_key(user["email"]), user)
    return True, f"{PLANS[plan]['label']} plan is active for {PLANS[plan]['days']:.0f} days."


def make_record(niche, source, facts, scores, report_md, ai, who, chance, now):
    return {
        "id": uuid.uuid4().hex[:10],
        "time": to_iso(now),
        "niche": niche.strip(),
        "source": source[:200],
        "scores": scores,
        "report_md": report_md,
        "ai": ai,
        "answered_by": who,
        "chance": chance,
        "free_idx": random.randrange(3),   # the one idea a free user may open
    }


def add_history(user, record):
    user["history"].insert(0, record)
    del user["history"][HISTORY_STORE_MAX:]


def visible_history(user):
    """Returns (records the plan may see, how many are hidden)."""
    allowed = PLANS[user["plan"]]["history"]
    return user["history"][:allowed], max(len(user["history"]) - allowed, 0)


def unlocked_ideas(record, plan):
    if PLANS[plan]["ideas"] >= 3:
        return {0, 1, 2}
    return {record.get("free_idx", 0)}


# ---------------------------------------------------------------
#  THE WEB PAGES
# ---------------------------------------------------------------
def render_ideas(st, rec, user, kp):
    unlocked = unlocked_ideas(rec, user["plan"])
    ideas = rec["ai"]["ideas"]
    st.subheader("💡 Video ideas")
    st.caption("🔥 = viral potential, an AI estimate (not a promise). Tap an idea to open it.")
    sel_key = f"sel_{kp}_{rec['id']}"
    for i, idea in enumerate(ideas):
        lock = "🔒 " if i not in unlocked else ""
        label = f"{lock}{idea['title']}   🔥 {idea['viral_rating']}%"
        if st.button(label, key=f"{kp}_{rec['id']}_{i}", use_container_width=True):
            st.session_state[sel_key] = i
    chosen = st.session_state.get(sel_key)
    if chosen is not None:
        idea = ideas[chosen]
        if chosen in unlocked:
            st.markdown(f"### {idea['title']}")
            st.markdown(f"**Viral potential:** {idea['viral_rating']}% (AI estimate)")
            if idea["hook_line"]:
                st.markdown(f"**🎣 Start with:** {idea['hook_line']}")
            if idea["what_to_film"]:
                st.markdown(f"**🎬 What to film:** {idea['what_to_film']}")
            if idea["steps"]:
                st.markdown("**Steps:**")
                st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(idea["steps"], 1)))
            if idea["why_it_could_go_viral"]:
                st.markdown(f"**🚀 Why it could go viral:** {idea['why_it_could_go_viral']}")
            if idea["caption"]:
                st.markdown(f"**📝 Caption:** {idea['caption']}")
        else:
            st.warning("🔒 This idea is locked on the Free plan. "
                       "Open the 💎 Plans page to unlock all 3 ideas.")
    if len(unlocked) < len(ideas):
        st.info("Free plan: 1 of 3 ideas is unlocked for each analysis. "
                "Upgrade in 💎 Plans to open them all.")


def render_analysis(st, rec, user, kp):
    ai = rec["ai"]
    st.divider()
    when = from_iso(rec["time"]).strftime("%d %b %Y %H:%M")
    st.subheader(f"📊 {rec['niche']}")
    st.caption(f"{when} · {rec['source']}")
    cols = st.columns(len(rec["scores"]))
    for col, (name, value) in zip(cols, rec["scores"].items()):
        col.metric(name, f"{value}/10")
    if rec.get("chance") is not None:
        st.info(f"🧠 Your trained model: this video looks "
                f"**{rec['chance'] * 100:.0f}%** like a winner in your niche.")

    st.subheader(f"🤖 Why it works (answered by {rec['answered_by']})")
    if ai["summary"]:
        st.markdown(ai["summary"])
    for title, field in (("🎣 The hook", "hook"), ("🧠 Why it works", "why_it_works"),
                         ("🧱 Structure", "structure")):
        if ai[field]:
            st.markdown(f"**{title}**\n\n{ai[field]}")
    if ai["recreate_steps"]:
        st.markdown("**🛠️ How to make something similar**")
        st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(ai["recreate_steps"], 1)))
    st.caption("The AI saw a few snapshots and the measured numbers, not the full video or sound.")
    with st.expander("📏 Measured numbers"):
        st.markdown(rec["report_md"])
    render_ideas(st, rec, user, kp)


def page_analyze(st, store, user, now, usage):
    st.title("🔥 My Video Analyzer")
    niche = st.text_input("Your niche *", placeholder="e.g. comedy, gaming, cooking")
    url = st.text_input("Video link (YouTube, TikTok, Instagram...)")
    upload = st.file_uploader("...or upload a video file", type=["mp4", "mov", "webm"])

    owner_email = normalize_email(get_secret("OWNER_EMAIL"))
    is_owner = bool(owner_email) and user["email"] == owner_email
    views_in = usual_in = 0
    if is_owner:
        with st.expander("Owner tools (training data)"):
            views_in = st.number_input("This video's views", min_value=0, step=1, value=0)
            usual_in = st.number_input("Channel's usual views per video", min_value=0, step=1, value=0)

    if st.button(f"Analyze 🚀  ({POINTS_PER_ANALYSIS} points)", type="primary"):
        if not niche.strip():
            st.error("Please fill out the niche type")
            st.stop()
        if not url and not upload:
            st.warning("Paste a link or upload a video first.")
            st.stop()
        if not can_afford(user):
            when = time_left_text(user["reset_at"], now) if user.get("reset_at") else "a while"
            st.error(f"You have {user['points']} points. One analysis needs "
                     f"{POINTS_PER_ANALYSIS}. Your points refresh in {when}.")
            st.info("Or open 💎 Plans to get more points now.")
            st.stop()
        if not ai_keys_present():
            st.error("🤖 The AI is not set up yet. The owner must add an AI key in Secrets.")
            st.stop()
        if user["plan"] == "free" and usage["count"] >= FREE_AI_PER_DAY:
            st.warning("The free analyses are full for today (shared by everyone). "
                       "Please come back tomorrow, or open 💎 Plans.")
            st.stop()

        try:
            with st.spinner("Measuring the video..."):
                with tempfile.TemporaryDirectory() as folder:
                    stats = None
                    if upload:
                        path = os.path.join(folder, "video_" + upload.name)
                        with open(path, "wb") as fh:
                            fh.write(upload.getbuffer())
                        source = upload.name
                    else:
                        path, stats = fetch_link(url, folder)
                        source = url
                    facts = measure(path)
                    frames = grab_frames(path, facts["duration"])
            scores = score(facts)
            with st.spinner("AI coach is thinking... (about a minute)"):
                who, ai, errs = ask_ai_json(facts, scores, niche.strip(), frames)
        except Exception as e:
            st.error(friendly_error(e))
            st.caption(str(e))
            st.stop()

        if ai is None:
            st.warning("🤖 The AI is busy or over its limit right now. "
                       "**No points were used.** Please try again in a minute.")
            for e in errs:
                st.caption(e)
            st.stop()

        report_md = build_report(facts, scores, stats)
        record = make_record(niche, source, facts, scores, report_md, ai, who,
                             model_prediction(facts), now)
        spend_points(user, now)
        add_history(user, record)
        try:
            store.set(user_key(user["email"]), user)
        except Exception as e:
            st.error("Could not save your result. No points were used. Please try again.")
            st.caption(str(e))
            st.stop()
        if user["plan"] == "free":
            usage["count"] += 1
        st.session_state.current = record
        if is_owner:
            extra = {"views": views_in or (stats or {}).get("view_count") or 0,
                     "usual_views": usual_in}
            st.session_state.owner_csv = facts_to_csv(facts, scores, stats, extra)
        st.rerun()

    rec = st.session_state.get("current")
    if rec:
        render_analysis(st, rec, user, "a")
        if is_owner and st.session_state.get("owner_csv"):
            st.download_button("⬇️ Save training data (CSV)", st.session_state.owner_csv,
                               file_name="video_data.csv", mime="text/csv")


def page_history(st, user):
    st.title("🕘 History")
    plan = PLANS[user["plan"]]
    if not user["history"]:
        st.info("No analyses yet. Run your first one in 🔍 Analyze.")
    else:
        shown, hidden = visible_history(user)
        labels = [f"{from_iso(r['time']).strftime('%d %b %H:%M')} · {r['niche']} · {r['source'][:40]}"
                  for r in shown]
        pick = st.selectbox("Pick an analysis", range(len(shown)), format_func=lambda i: labels[i])
        render_analysis(st, shown[pick], user, "h")
        if hidden:
            st.warning(f"🔒 {hidden} older analysis{'es' if hidden != 1 else ''} hidden. "
                       "Upgrade in 💎 Plans to see them.")
    if user["plan"] == "free":
        st.info(f"Free plan keeps your last {plan['history']} analyses. "
                "Upgrade in 💎 Plans to get a longer history "
                f"(Creator {PLANS['creator']['history']}, Pro {PLANS['pro']['history']}).")


def page_plans(st, store, user, now):
    st.title("💎 Plans")
    st.write(f"Your plan: **{PLANS[user['plan']]['label']}**  ·  "
             f"**{user['points']}** points  ·  one analysis = {POINTS_PER_ANALYSIS} points")
    for key in ("free", "creator", "pro"):
        p = PLANS[key]
        every = "every 10.5 days" if key == "free" else "per 30 days"
        ideas = "1 of 3 ideas open" if p["ideas"] < 3 else "all 3 ideas open"
        st.markdown(
            f"### {p['label']} — {p['price']}\n"
            f"- **{p['points']} points** {every} ({p['points'] // POINTS_PER_ANALYSIS} analyses)\n"
            f"- History: last **{p['history']}** analyses\n"
            f"- Ideas: {ideas}")
        link = get_secret({"creator": "PAY_LINK_CREATOR", "pro": "PAY_LINK_PRO"}.get(key, ""))
        if link and key != "free":
            st.markdown(f"👉 [Buy {p['label']}]({link})")
    st.divider()
    st.subheader("Have a code?")
    code = st.text_input("Enter the code you received after paying", key="redeem_code")
    if st.button("Activate code"):
        try:
            ok, msg = redeem_code(store, user, code, now)
        except Exception as e:
            ok, msg = False, "Could not reach the database. Try again."
            st.caption(str(e))
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()


def login_screen(st, store):
    st.title("🔥 My Video Analyzer")
    st.write(f"Create a free account and get **{PLANS['free']['points']} analysis points** "
             f"(2 video analyses).")
    tab_in, tab_up = st.tabs(["Log in", "Sign up"])
    with tab_in:
        email = st.text_input("Email", key="li_email")
        password = st.text_input("Password", type="password", key="li_pw")
        if st.button("Log in", key="li_btn"):
            try:
                user, err = login(store, email, password)
            except Exception as e:
                user, err = None, "Could not reach the database. Try again."
                st.caption(str(e))
            if err:
                st.error(err)
            else:
                st.session_state.email = user["email"]
                st.rerun()
    with tab_up:
        email = st.text_input("Email", key="su_email")
        password = st.text_input("Choose a password (6+ characters)", type="password", key="su_pw")
        if st.button("Create my account", key="su_btn"):
            try:
                user, err = signup(store, email, password, utcnow())
            except Exception as e:
                user, err = None, "Could not reach the database. Try again."
                st.caption(str(e))
            if err:
                st.error(err)
            else:
                st.session_state.email = user["email"]
                st.rerun()


def main():
    import streamlit as st

    st.set_page_config(page_title="My Video Analyzer", page_icon="🔥")

    @st.cache_resource
    def get_store():
        return make_store()

    @st.cache_resource
    def get_usage():
        """Counts free analyses today for everyone (a safety cap for your wallet)."""
        return {"day": "", "count": 0}

    store = get_store()
    usage = get_usage()
    today = datetime.date.today().isoformat()
    if usage["day"] != today:
        usage["day"], usage["count"] = today, 0

    if isinstance(store, LocalStore):
        st.warning("⚠️ Database not connected. Accounts are TEMPORARY. "
                   "Add SUPABASE_URL and SUPABASE_KEY in Secrets.")

    if "email" not in st.session_state:
        login_screen(st, store)
        st.stop()

    now = utcnow()
    try:
        user = store.get(user_key(st.session_state.email))
    except Exception as e:
        st.error("Could not reach the database. Please try again in a minute.")
        st.caption(str(e))
        st.stop()
    if not user:
        del st.session_state["email"]
        st.rerun()
    if refresh_user(user, now):
        store.set(user_key(user["email"]), user)

    # ----- points dashboard (always visible) -----
    plan = PLANS[user["plan"]]
    st.sidebar.markdown(f"### 👤 {user['email']}")
    st.sidebar.metric("Analysis points", f"{user['points']} / {plan['points']}")
    st.sidebar.progress(min(max(user["points"] / plan["points"], 0.0), 1.0))
    st.sidebar.caption(f"{plan['label']} plan · "
                       f"{user['points'] // POINTS_PER_ANALYSIS} analyses left")
    if user.get("reset_at"):
        word = "Points refresh in" if user["plan"] == "free" else "Plan ends in"
        st.sidebar.caption(f"⏳ {word} {time_left_text(user['reset_at'], now)}")
    elif user["plan"] == "free":
        st.sidebar.caption("Your points refresh 10.5 days after you first use them.")

    page = st.sidebar.radio("Menu", ["🔍 Analyze", "🕘 History", "💎 Plans"])
    if st.sidebar.button("Log out"):
        for k in ("email", "current", "owner_csv"):
            st.session_state.pop(k, None)
        st.rerun()

    if page == "🔍 Analyze":
        page_analyze(st, store, user, now, usage)
    elif page == "🕘 History":
        page_history(st, user)
    else:
        page_plans(st, store, user, now)


def running_in_streamlit():
    """True when this file is being run as a website by Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:
        return False


if running_in_streamlit():
    main()
elif __name__ == "__main__":
    # Quick test without a website:  python my_analyzer.py yourvideo.mp4
    import sys
    if len(sys.argv) > 1:
        facts = measure(sys.argv[1])
        print(build_report(facts, score(facts)))
    else:
        print("Usage: python my_analyzer.py yourvideo.mp4")
 
