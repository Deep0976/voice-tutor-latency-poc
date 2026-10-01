# Plain-English Guide

This guide assumes you're new to voice AI and Python. It explains what each
piece does, how to run it, and how to read the speed numbers.

## What is this thing?

It's a small web server that runs on your own Mac. When you open it in Chrome
and click **Connect**, your browser streams your microphone to the server. The
server:

1. **Listens** and converts your speech to text (**Sarvam STT**).
2. **Thinks** about the doubt and writes an answer (**GPT-4o**).
3. **Speaks** the answer back as audio (**ElevenLabs TTS**).

Pipecat is the "glue" that wires these three together and handles the tricky
real-time audio plumbing (knowing when you've stopped talking, letting you
interrupt, etc.).

## The files

| File | What it is |
|------|------------|
| `bot.py` | The whole tutor. Defines the pipeline and the tutor's instructions. |
| `requirements.txt` | The list of Python packages to install. |
| `.env.example` | A template for your secret keys. You copy it to `.env`. |
| `.env` | **Your** real keys live here. Never share or commit this. |
| `.venv/` | The private box of installed packages. |

## Running it, step by step

1. **Open Terminal** and move into this folder. (In Finder you can drag the
   folder onto the Terminal icon, or `cd` to it.)

2. **Activate the environment.** This tells your terminal to use this project's
   private Python:
   ```bash
   source .venv/bin/activate
   ```
   Your prompt will show `(.venv)` at the start. Do this once per terminal window.

3. **Run the bot:**
   ```bash
   python bot.py
   ```
   When it's ready you'll see a line mentioning **http://localhost:7860**.

4. **Open http://localhost:7860 in Chrome.** Click **Connect**, and when Chrome
   asks, click **Allow** for the microphone. Then just talk — ask a Maths,
   Physics or Chemistry doubt out loud. The tutor will greet you first.

5. **Stop the bot** with `Ctrl+C` in the terminal.

## Reading the speed numbers (TTFB / latency)

"TTFB" = **Time To First Byte** = how long a stage took to start producing its
output. It's the key responsiveness metric for a voice bot.

Because `bot.py` turns on metrics (`enable_metrics=True`), Pipecat prints a
**TTFB** line for each stage in the terminal every time you speak. Watch the
terminal while you talk — after each thing you say you'll see lines like:

```
... SarvamSTTService    TTFB: 0.42  ...
... OpenAILLMService    TTFB: 0.88  ...
... ElevenLabsTTSService TTFB: 0.31  ...
```

(Exact wording/format depends on the version, but each AI service reports its
own TTFB.) The **total delay** before you hear a reply is roughly the **sum** of
the three, plus a little network time.

**Finding your bottleneck:** the stage with the **biggest** TTFB is what's
slowing you down. In a setup like this it's almost always the **LLM
(OpenAILLMService)** — GPT-4o has to "think" before the first word comes out,
while Sarvam STT and TTS are typically fast. So if replies feel slow, the LLM is
usually the place to optimise first (e.g. a faster model, or shorter answers).
STT is the next most likely culprit; TTS is usually the quickest.

## Switching language

`.env` has `TUTOR_LANGUAGE=te`. Change `te` to `ta`, `hi`, or `kn` and restart
`python bot.py`. (Telugu, Tamil, Hindi, Kannada.) One language per run — this
keeps the POC simple.

## Troubleshooting

- **"command not found: python"** — you didn't activate the environment. Run
  `source .venv/bin/activate` first.
- **An "authentication" / 401 error** in the terminal — a key in `.env` is
  missing or wrong. Re-check `SARVAM_API_KEY` and `OPENAI_API_KEY`.
- **The page loads but Connect does nothing** — make sure the terminal still
  shows the bot running, and that you allowed the microphone in Chrome.
- **No sound back** — check your Mac's output volume and that the browser tab
  isn't muted.
- **It answers in the wrong language** — set `TUTOR_LANGUAGE` in `.env` and
  restart.
