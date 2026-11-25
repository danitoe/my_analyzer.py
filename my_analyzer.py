import os
import time

import streamlit as st
from google import genai
from google.genai import types

# =====================================================
#  1. CHANGE THESE THINGS TO MAKE THE AI YOURS
# =====================================================
AI_NAME = "My AI"
MODEL = "gemini-3.8-flash"     # if you see "model not found", change this name
MAX_QUESTIONS = 10             # questions each visitor can ask per visit
RETRY_WAITS = [3, 6, 12]       # seconds to wait if Google's AI is busy

# The personality: tell the AI who it is and how to behave.
PERSONALITY = """
You are a friendly, helpful AI assistant called My AI.
You answer clearly and simply, in short paragraphs.
If you do not know something, say so honestly. Never invent facts.
"""

# Your own knowledge: paste anything the AI should know
# (your business, prices, services, FAQs, contact details...).
KNOWLEDGE = """
About us:
- (Write facts about you or your business here.)
- (Example: We help creators go viral with AI tools.)
"""
# =====================================================


st.set_page_config(page_title=AI_NAME, page_icon="🤖")


def get_secret(name, default=""):
    """Read a secret from Streamlit secrets or environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


API_KEY = get_secret("GEMINI_API_KEY")


def is_busy_error(e):
    text = str(e).lower()
    return any(w in text for w in ["503", "unavailable", "overloaded", "high demand"])


def ask_ai(client, history):
    """Send the whole conversation to Gemini and return its answer."""
    contents = []
    for m in history[-20:]:  # remember the last 20 messages
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    config = types.GenerateContentConfig(
        system_instruction=PERSONALITY + "\n" + KNOWLEDGE
    )

    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            reply = client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
            return reply.text
        except Exception as e:
            if is_busy_error(e) and attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            raise


def friendly_error(e):
    """Turn a technical error into a message a visitor can understand."""
    text = str(e).lower()
    if is_busy_error(e):
        return "🚦 The AI is very busy right now. Please try again in a minute."
    if any(w in text for w in ["getaddrinfo", "11001", "connection", "timed out",
                               "timeout", "network", "name resolution"]):
        return "📡 Internet problem. Check your connection and try again."
    if "404" in text or "not_found" in text:
        return "⚙️ The AI model name is outdated. The owner needs to update MODEL."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "⏳ Too many requests right now. Please wait a minute and try again."
    if "api key" in text or "api_key" in text or "401" in text or "403" in text:
        return "🔑 The owner's API key has a problem."
    return "😕 Something went wrong. Please try again."


# ---------- PAGE ----------
st.title(f"🤖 {AI_NAME}")

if not API_KEY:
    st.error("🔑 Missing GEMINI_API_KEY. Add it in your secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "asked" not in st.session_state:
    st.session_state.asked = 0

# Show the conversation so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask me anything...")

if question:
    if st.session_state.asked >= MAX_QUESTIONS:
        st.warning("You reached the question limit for this visit. Come back later!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = genai.Client(api_key=API_KEY)
                answer = ask_ai(client, st.session_state.messages)
            except Exception as e:
                answer = friendly_error(e)
                st.caption(str(e))  # real technical reason (you can delete this line later)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.asked += 1
