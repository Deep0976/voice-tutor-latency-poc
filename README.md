# Voice Tutor POC

A browser-based **spoken tutor** for 11th/12th grade **Maths, Physics and
Chemistry** doubts in **Telugu, Tamil, Hindi and Kannada**.

You talk to it in your browser; it talks back.

**Stack**
- [Pipecat](https://github.com/pipecat-ai/pipecat) 1.4.0 — voice-agent framework
- [Sarvam AI](https://dashboard.sarvam.ai) — speech-to-text (Indian languages)
- OpenAI **GPT-4o** — the tutor's reasoning
- [ElevenLabs](https://elevenlabs.io) — text-to-speech (the voice)
- Self-hosted **WebRTC** — browser ↔ server audio (no third-party call service)

How a single turn flows:

```
Your voice ─▶ Sarvam STT ─▶ GPT-4o ─▶ ElevenLabs TTS ─▶ Bot's voice
            (speech→text)   (brain)    (text→speech)
```

---

## Quick start

Everything below is run from inside this folder. The `.venv` folder is a
private box holding this project's Python packages.

**1. Activate the virtual environment** (you'll do this each new terminal):

```bash
source .venv/bin/activate
```

**2. Add your API keys** — copy the template and paste your keys into `.env`:

```bash
cp .env.example .env
```

You need three keys plus an ElevenLabs voice ID:
- `SARVAM_API_KEY` — from https://dashboard.sarvam.ai → API Keys
- `OPENAI_API_KEY` — from https://platform.openai.com/api-keys
- `ELEVENLABS_API_KEY` — from https://elevenlabs.io → Profile → API Keys
- `ELEVENLABS_VOICE_ID` — from https://elevenlabs.io/app/voices → pick a voice → copy its Voice ID

**3. Run the bot:**

```bash
python bot.py
```

**4. Open it in Chrome:** http://localhost:7860

Click **Connect**, allow the microphone when asked, and speak a doubt out loud.

To stop the bot, press `Ctrl+C` in the terminal.

---

## Choosing the language

This POC speaks **one language per session**. Set it in `.env`:

```
TUTOR_LANGUAGE=te   # te=Telugu  ta=Tamil  hi=Hindi  kn=Kannada
```

Change the line, save, and restart `python bot.py` to switch.

---

See **GUIDE.md** for a plain-English walkthrough, how to read the latency
(TTFB) numbers, and troubleshooting.
