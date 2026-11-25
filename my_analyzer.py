"""
MY OWN VIDEO ANALYZER  -  no API key, no cloud AI.

How it works (3 steps):
  1. MEASURE: ffmpeg counts cuts, checks loudness and silence.
     (Optional) Whisper, a free open-source model, writes down what is said.
  2. SCORE: simple rules turn the numbers into scores out of 10.
  3. REPORT: the numbers + scores become a written breakdown.

Every rule is in plain Python below, so you can read it, change it, and
later replace the rules with a model you train yourself.
"""
import csv
import io
import json
import os
import re
import subprocess
import tempfile

MAX_SECONDS = 180          # only analyze the first 3 minutes (keeps it fast)
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


def facts_to_csv(f, scores, stats=None):
    """One row of data. Collect many rows, then you can TRAIN your own model."""
    row = {k: v for k, v in f.items()
           if k not in ("transcript", "hook_text", "hook_words_found")}
    row.update({f"score_{k.lower()}": v for k, v in scores.items()})
    if stats:
        for k in ("view_count", "like_count", "comment_count"):
            row[k] = stats.get(k)
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
        "format": "best[height<=720]/best",
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
                               "could not download", "sign in", "403", "429"]):
        return ("🔗 Could not download that link. Download the video yourself "
                "and use the upload option instead.")
    if "no such file" in text and "ffmpeg" in text:
        return "🛠️ ffmpeg is not installed on this machine."
    return "😕 Something went wrong."


# ---------------------------------------------------------------
#  THE WEB PAGE
# ---------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="My Video Analyzer", page_icon="🔥")
    st.title("🔥 My Video Analyzer")
    st.write("No API key. No cloud AI. Everything is measured by my own code.")

    url = st.text_input("Video link (YouTube, TikTok, Instagram...)")
    upload = st.file_uploader("...or upload a video file", type=["mp4", "mov", "webm"])

    if st.button("Analyze 🚀", type="primary"):
        if not url and not upload:
            st.warning("Paste a link or upload a video first.")
            st.stop()
        try:
            with st.spinner("Measuring the video..."):
                with tempfile.TemporaryDirectory() as folder:
                    stats = None
                    if upload:
                        path = os.path.join(folder, "video_" + upload.name)
                        with open(path, "wb") as f:
                            f.write(upload.getbuffer())
                    else:
                        path, stats = fetch_link(url, folder)
                    facts = measure(path)
            scores = score(facts)
            cols = st.columns(len(scores))
            for col, (name, value) in zip(cols, scores.items()):
                col.metric(name, f"{value}/10")
            st.markdown(build_report(facts, scores, stats))
            st.download_button("⬇️ Save this data (CSV)", facts_to_csv(facts, scores, stats),
                               file_name="video_data.csv", mime="text/csv")
        except Exception as e:
            st.error(friendly_error(e))
            st.caption(str(e))


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
