"""
Voice Tutor POC — a browser-based spoken tutor for 11th/12th grade
Maths, Physics and Chemistry doubts in Indian languages.

Pipeline (a "cascade" bot):

    Your voice ──▶ Sarvam STT ──▶ GPT-4o ──▶ Sarvam TTS ──▶ Bot's voice
    (speech to text)   (the "brain")   (text to speech)

Built for Pipecat 1.4.0. Run it with:

    python bot.py

Then open the URL it prints (http://localhost:7860) in Chrome.
"""

import os
import re
import time
import asyncio

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

# --- Pipecat core: the pieces that move audio through the pipeline ---
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.frames.frames import (
    LLMRunFrame,
    LLMContextFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.utils.text.base_text_filter import BaseTextFilter

# --- Turn-taking + conversation memory ---
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# --- The browser <-> server audio connection (self-hosted WebRTC) ---
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

# --- The AI services ---
from pipecat.transcriptions.language import Language
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.stt import SarvamSTTService          # ears (speech -> text)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService  # voice (text -> speech)

# Read API keys and settings from the .env file in this folder.
# override=True means the .env file wins over any stale shell variables.
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------
# TUTOR_LANGUAGE in .env sets the STARTING language. During a call the student
# can just say "talk in Hindi" and the bot switches its ears (STT) and voice
# (TTS) live — see the switch_language tool in run_bot().
# Sarvam's models expect the region-specific code (e.g. "te-IN"), so we use the
# Pipecat Language *_IN enum members, which Sarvam maps correctly.
# Each language carries TWO codes because two different services need different
# forms: Sarvam STT wants the region code (e.g. Language.HI_IN -> "hi-IN"), while
# ElevenLabs TTS wants the base code (e.g. Language.HI -> "hi").
#   value = (display name, Sarvam-STT enum, ElevenLabs-TTS enum)
SUPPORTED_LANGUAGES = {
    "te": ("Telugu", Language.TE_IN, Language.TE),
    "ta": ("Tamil", Language.TA_IN, Language.TA),
    "hi": ("Hindi", Language.HI_IN, Language.HI),
    "kn": ("Kannada", Language.KN_IN, Language.KN),
}

# Same languages keyed by spoken name, for the switch_language tool.
LANGUAGE_BY_NAME = {
    name.lower(): (name, stt, tts) for name, stt, tts in SUPPORTED_LANGUAGES.values()
}

_lang_code = os.getenv("TUTOR_LANGUAGE", "te").strip().lower()
if _lang_code not in SUPPORTED_LANGUAGES:
    logger.warning(
        f"TUTOR_LANGUAGE='{_lang_code}' is not one of {list(SUPPORTED_LANGUAGES)}; "
        "falling back to Telugu ('te')."
    )
    _lang_code = "te"

LANGUAGE_NAME, STT_LANGUAGE, TTS_LANGUAGE = SUPPORTED_LANGUAGES[_lang_code]


# The "personality" and rules for the tutor. Because every reply is read out
# loud by the text-to-speech engine, we tell the model to write the way a
# patient teacher *speaks*, not the way a textbook is printed.
SYSTEM_PROMPT = f"""You are a warm, fun, friendly study buddy for Indian students in \
classes 11 and 12 (CBSE/State boards) — like a cool elder brother or sister who is \
great at studies. You help with doubts in Mathematics, Physics and Chemistry only.

TONE — talk like a friend, NOT like a textbook
- Be upbeat and a little energetic — bring positive, encouraging energy, like a \
friend who's genuinely excited to help. Keep it natural, not over-the-top.
- ALWAYS start your very first greeting with a quick, casual "Hey" — say it \
briefly and naturally like a friend saying hi (NO exclamation mark, not stretched \
or drawn out), then flow straight into the sentence. E.g. "Hey, ..." not "Heyyy!".
- Be casual, warm and encouraging, the way friends actually talk. Use a friendly \
informal register (in Hindi, everyday "tum"-style bol-chaal, NOT formal shuddh/ \
literary Hindi). Use the everyday words people really use in daily conversation \
(for example प्रयोग करो → इस्तेमाल करो, सहायता → मदद).
- Sound human and relaxed: short reactions like "हाँ बिल्कुल", "अरे easy है", \
"चल देख", "समझ आया?", a little encouragement when they get it right.
- Avoid heavy, formal or bookish vocabulary. If a simple everyday word exists, use it.

LANGUAGE — write in the NATIVE SCRIPT, mix English only for technical terms
- Write your reply in {LANGUAGE_NAME} using ITS OWN script — Hindi in Devanagari, \
Telugu in Telugu script, Tamil in Tamil script, Kannada in Kannada script. This is \
CRITICAL: the voice mispronounces {LANGUAGE_NAME} words when they are written in \
English/Roman letters. So write "मदद", "कैसे", "समझ", "निकालो" — NEVER "madad", \
"kaise", "samajh", "nikaalo".
- Mix in English ONLY for genuine technical/maths/science terms, written in normal \
English letters — e.g. derivative, integration, velocity, acceleration, force, \
equation, mole, reaction, slope, function, graph. Do not translate those.
- ALWAYS say the subject names in English: "maths", "physics", "chemistry" — never \
their {LANGUAGE_NAME} translations (not गणित, भौतिकी, रसायन; say maths, physics, \
chemistry).
- Example (Hindi): "इसका derivative निकालो, फिर slope मिल जाएगा। समझ आया?"
- Example (Hindi): "कोई बात नहीं, physics का ये doubt हम अभी clear करते हैं।"
- If the student asks you to talk in another language (Telugu, Tamil, Hindi or \
Kannada) — for example "talk in Hindi" — you MUST call the switch_language tool \
with that language, and then continue the whole conversation in the new language \
(again, in that language's native script). Do not just translate one reply; switch.

HOW TO TEACH
- You are a tutor, not an answer key. Guide the student step by step and, where \
useful, ask a short question to check their thinking before revealing the next step.
- Keep each spoken reply short and chatty — a couple of sentences. The student can \
always ask for more.
- Encourage warmly when they're on the right track ("shabaash", "exactly!").

SPEAKING STYLE (very important — your words are READ ALOUD by a voice)
- Do NOT use markdown, bullet points, asterisks, code blocks, or LaTeX.

MATHS & FORMULAS — spell EVERYTHING out in spoken English words
Your output is spoken aloud, so it must contain NO mathematical symbols, NO \
superscripts/subscripts, NO LaTeX, NO slashes — every expression must be written \
exactly as a person would SAY it, in English words. Convert before you speak:
- Powers: write "x squared" (not x^2), "x cubed", "x to the power n", \
"ten to the power five" (not 10^5).
- Roots: "the square root of two" (not √2), "the cube root of x".
- Fractions: "a over b" or "a divided by b" (not a/b); "one half", "three by four".
- Operators: say "plus, minus, times (or into), divided by, equals" — never + - * / =.
- SINGLE-LETTER VARIABLES: just write them as plain single English letters \
(a, b, c, x, y, n ...). Do NOT spell them out — a separate step automatically gives \
each letter its correct spoken sound. So write "a squared plus b", "x squared", \
"a x plus b". (The student will hear them as the letter names "ay", "bee", "ex".)
- Subscripts: "x one", "a sub n", "v naught" (not x_1, a_n, v0).
- Calculus: "d y by d x" (not dy/dx), "the integral of f of x, d x", \
"the limit as x tends to zero".
- Greek/letters: "theta", "pi", "delta x", "lambda" (spelled as words).
- BRACKETS / GROUPING: never say "open bracket", "close bracket" or "parenthesis". \
Speak grouped expressions the natural Indian-English way: \
(a + b)^2 → "a plus b whole squared"; (a + b)^3 → "a plus b whole cubed"; \
(a - b)^2 → "a minus b whole squared"; (a + b)(a - b) → "a plus b into a minus b"; \
a / (b + c) → "a divided by, b plus c whole" (or "a upon b plus c"). For a longer \
group say "the whole quantity ... " to make the grouping clear by voice.
- Chemistry: "H two O", "C O two", "sulphuric acid is H two S O four".
- Equations: read the whole thing in words. For x^2 + 3x = 0 write \
"x squared plus three x equals zero".
- Keep the maths WORDS (squared, plus, equals ...) in English even inside a \
{LANGUAGE_NAME} sentence. \
Example (Hindi): "तो equation बनेगा x squared plus three x equals zero।"
- If an expression is long, say it slowly and in small chunks rather than all at once.

SCOPE
- If asked about something outside class 11–12 Maths, Physics or Chemistry, \
gently say that's outside what you tutor, and steer back to their studies.

PACING
- A very short filler acknowledgement (like "अच्छा, देखते हैं") may already have \
been spoken to the student just before your reply. Do NOT open with another \
greeting or "अच्छा/हम्म" — get straight into the actual help.
"""


# ---------------------------------------------------------------------------
# Fast SLM for low-latency "semantic fillers"
# ---------------------------------------------------------------------------
# A small, fast model (GPT-4o-mini) speaks a tiny acknowledgement the instant the
# student stops talking — e.g. "अच्छा, चलो देखते हैं" — so the FIRST word is heard
# in ~300-500ms instead of waiting ~1-2s for GPT-4o's first token. The big model
# then delivers the real answer, spoken right after the filler.
#
# CENTRALIZED CONTEXT (shared both ways):
#   • One LLMContext is the single source of truth (user turns, fillers, answers).
#   • SLM -> LLM: the filler the SLM produces is written into that shared context,
#     so GPT-4o sees it and continues naturally (no repeated greeting).
#   • LLM -> SLM: after each answer a background call COMPACTS the conversation
#     into a one/two-line running note; the SLM reads only that note (+ the latest
#     student line), so it stays fast while staying aware of the conversation.
# SLM provider: "groq" (fast Llama) or "openai" (gpt-4o-mini). Groq is OpenAI-API
# compatible, so the same AsyncOpenAI client works with a different base_url + key.
SLM_PROVIDER = os.getenv("SLM_PROVIDER", "groq").strip().lower()
if SLM_PROVIDER == "groq":
    SLM_MODEL = os.getenv("SLM_MODEL", "llama-3.3-70b-versatile")
    SLM_BASE_URL = "https://api.groq.com/openai/v1"
    SLM_API_KEY_ENV = "GROQ_API_KEY"
else:
    SLM_MODEL = os.getenv("SLM_MODEL", "gpt-4o-mini")
    SLM_BASE_URL = None
    SLM_API_KEY_ENV = "OPENAI_API_KEY"

FILLER_SYSTEM_PROMPT = (
    "You are the INSTANT voice of a friendly young Hindi/English study buddy. The "
    "student just said something; a bigger model is preparing the full answer. Your "
    "ONLY job is to say ONE very short, natural filler acknowledgement (3 to 8 words) "
    "that buys a moment AND shows you heard THIS specific student. Make it CONTEXTUAL: "
    "briefly reflect their actual topic or mood, e.g. an integration doubt -> "
    "'अच्छा, integration वाला — देखते हैं', a tough one -> 'हम्म, अच्छा सवाल है, एक सेकंड', "
    "if they sound stuck -> 'कोई बात नहीं, समझाता हूँ'. "
    "CRITICAL SCRIPT RULE: write all Hindi words ONLY in Devanagari script (देवनागरी). "
    "NEVER use Roman/Latin letters for Hindi — never write 'theek hai' or 'dekhte hain'; "
    "write 'ठीक है', 'देखते हैं'. You MAY keep a genuine technical/subject word in English "
    "(integration, physics, velocity). "
    "Do NOT answer the question or give ANY maths/explanation; no lists; under 8 words; "
    "end cleanly with no trailing '...'."
)

SUMMARY_SYSTEM_PROMPT = (
    "You keep a very short running note (max 2 short sentences) of a tutoring "
    "conversation: the subject, the current doubt, and where things stand. This note "
    "is the compacted context handed to a fast filler model. Output ONLY the note."
)


class ConversationState:
    """Centralized state shared between the SLM (fillers) and the LLM (answers)."""

    def __init__(self):
        self.summary = ""            # LLM conversation, compacted for the SLM
        self.last_user_text = ""     # latest finalized student utterance
        self.last_filler_ms = None   # how long the SLM took to make the last filler


async def generate_filler(slm: AsyncOpenAI, state: "ConversationState", language_name: str) -> str:
    """Ask the fast SLM for a quick spoken filler, using the compacted context."""
    note = (
        f"Conversation note: {state.summary or 'just starting'}\n"
        f"Student just said: {state.last_user_text}\n"
        f"Give one short {language_name} filler now."
    )
    resp = await slm.chat.completions.create(
        model=SLM_MODEL,
        messages=[
            {"role": "system", "content": FILLER_SYSTEM_PROMPT},
            {"role": "user", "content": note},
        ],
        max_tokens=24,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


async def update_summary(slm: AsyncOpenAI, state: "ConversationState", assistant_text: str) -> None:
    """Compact the latest exchange into state.summary (runs in the background)."""
    prompt = (
        f"Previous note: {state.summary or 'none'}\n"
        f"Student said: {state.last_user_text}\n"
        f"Tutor answered: {assistant_text}\n"
        "Update the running note."
    )
    try:
        resp = await slm.chat.completions.create(
            model=SLM_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=90,
            temperature=0.3,
        )
        state.summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # never let compaction break the conversation
        logger.warning(f"Summary compaction failed: {e}")


def _latest_user_text(context) -> str:
    """The most recent student utterance in the shared context (dict messages)."""
    for msg in reversed(context.get_messages()):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):  # some providers use content parts
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                ).strip()
            return (content or "").strip()
    return ""


class FillerProcessor(FrameProcessor):
    """Speaks an instant, contextual SLM filler when the student finishes a turn.

    Sits between the user aggregator and the LLM. When the LLMContextFrame that
    triggers the LLM passes through AND the latest context message is the student's
    (i.e. a real turn, not the greeting), it:
      1) reads the student's CURRENT words straight from the shared context,
      2) gets a short, topic-aware filler from the fast SLM,
      3) writes a note into the shared context (so the LLM continues, no repeat), and
      4) speaks the filler immediately — then the LLM produces the full answer.

    Barge-in: interruptions are handled by the base processor (an InterruptionFrame
    cancels the in-flight filler generation and the TTS), so a spoken filler stops
    the moment the student talks over it.
    """

    def __init__(self, context, state: "ConversationState", slm: AsyncOpenAI, language_name: str):
        super().__init__()
        self._context = context
        self._state = state
        self._slm = slm
        self._language_name = language_name
        self._last_filled_for = None  # dedupe: the user text we already filled for

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            messages = self._context.get_messages()
            last = messages[-1] if messages else None
            is_user_turn = isinstance(last, dict) and last.get("role") == "user"
            user_text = _latest_user_text(self._context)

            if is_user_turn and user_text and user_text != self._last_filled_for:
                self._last_filled_for = user_text
                self._state.last_user_text = user_text  # keep shared state current
                filler = ""
                _t = time.monotonic()
                try:
                    filler = await asyncio.wait_for(
                        generate_filler(self._slm, self._state, self._language_name),
                        timeout=1.2,  # if the SLM is slow, skip rather than delay the answer
                    )
                    self._state.last_filler_ms = int((time.monotonic() - _t) * 1000)
                except Exception as e:  # (CancelledError from a barge-in is NOT caught here)
                    logger.warning(f"Filler skipped: {e}")

                if filler:
                    logger.info(f"SLM filler: {filler}")
                    # Share with the LLM via the centralized context (a note, so the
                    # LLM continues with the real answer instead of prefixing it).
                    self._context.add_message(
                        {
                            "role": "system",
                            "content": (
                                f'[You already spoke a brief filler to the student: "{filler}". '
                                "Now give the actual answer; do not repeat the filler.]"
                            ),
                        }
                    )
                    # Speak it now (append_to_context=False — we already added our note).
                    await self.push_frame(
                        TTSSpeakFrame(text=filler, append_to_context=False), direction
                    )

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Maths letter pronunciation (deterministic safety net)
# ---------------------------------------------------------------------------
# The voice (in Hindi mode) mis-reads a lone Latin letter used as a variable —
# e.g. "a" comes out as "ah" instead of the letter name "ay". This map forces a
# spoken spelling for each standalone single letter. Tune any value here if a
# letter still sounds wrong — this is the one place to edit, then restart.
# Devanagari letter-names: this is how the (Hindi-mode) voice pronounces each
# letter correctly — e.g. "a" -> "ए" gives the long "ay" /eɪ/ sound. If any single
# letter still sounds off, tweak just its value here and restart.
LETTER_SOUNDS = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई",
    "f": "एफ", "g": "जी", "h": "एच", "i": "आई", "j": "जे",
    "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ",
    "p": "पी", "q": "क्यू", "r": "आर", "s": "एस", "t": "टी",
    "u": "यू", "v": "वी", "w": "डब्ल्यू", "x": "एक्स", "y": "वाय",
    "z": "ज़ेड",
}

# Multi-letter maths terms (trig functions, Greek letters, etc.) that the
# Hindi-mode voice mis-reads when written in Latin. Spelling them in Devanagari
# makes the voice say them correctly — e.g. "cos theta" -> "कॉस थीटा" ("kos theeta").
# Add/adjust any term here if it sounds wrong, then restart.
MATH_TERM_SOUNDS = {
    "sin": "साइन", "cos": "कॉस", "tan": "टैन", "cosec": "कोसैक",
    "sec": "सेक", "cot": "कॉट", "sinh": "सिन्च", "cosh": "कॉश", "tanh": "टैन्च",
    "theta": "थीटा", "alpha": "अल्फ़ा", "beta": "बीटा", "gamma": "गामा",
    "delta": "डेल्टा", "lambda": "लैम्डा", "sigma": "सिग्मा", "omega": "ओमेगा",
    "phi": "फ़ाई", "psi": "साई", "mu": "म्यू", "pi": "पाई", "rho": "रो",
    "epsilon": "एप्सिलॉन", "eta": "ईटा", "tau": "टाउ", "nu": "न्यू",
    "naught": "नॉट",  # the "0" in subscripts like epsilon-naught, v-naught
    "log": "लॉग", "ln": "ऍल ऍन", "lim": "लिमिट",
}

# Match whole maths-terms (longest first so "cosec" wins over "cos"), case-insensitive.
_MATH_TERM = re.compile(
    r"\b(" + "|".join(sorted(MATH_TERM_SOUNDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Matches a single ASCII letter that is NOT touching another letter — i.e. a
# letter standing alone as a variable (surrounded by spaces, digits, operators
# or punctuation), not a letter inside a word.
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def spell_maths_for_voice(text: str) -> str:
    """Rewrite maths terms and lone variable letters into spoken Devanagari.

    Terms first (so "cos" -> कॉस before the single-letter pass runs), then any
    remaining standalone single letter (a -> ए, x -> एक्स ...). The Devanagari
    function and variable end up as separate space-separated words, which the
    voice says smoothly ("cosec x") without any inserted pause.
    """
    text = _MATH_TERM.sub(lambda m: MATH_TERM_SOUNDS[m.group(1).lower()], text)
    text = _STANDALONE_LETTER.sub(
        lambda m: LETTER_SOUNDS.get(m.group(1).lower(), m.group(1)), text
    )
    return text


class MathsTextFilter(BaseTextFilter):
    """Rewrites maths terms and lone variable letters to spoken Devanagari.

    Runs as a TTS *text filter*, which the TTS applies AFTER it aggregates the
    streamed tokens into a full sentence. That's essential: the LLM streams text
    token by token, so a word like "cosec" can arrive split as "cos" + "ec".
    A per-token rewrite would see "cos" alone and mangle it; this filter sees the
    whole sentence, so "cosec x" is intact.
    """

    async def filter(self, text: str) -> str:
        return spell_maths_for_voice(text)


# ---------------------------------------------------------------------------
# Transcript romanisation (Devanagari -> readable Latin, for the on-screen text)
# ---------------------------------------------------------------------------
# The voice needs Devanagari, but the transcript should be Latin. After the voice
# has been synthesised, we romanise the bot's spoken text for display only (the
# audio is already produced, so it is untouched).
from indic_transliteration import sanscript  # noqa: E402
from indic_transliteration.sanscript import transliterate  # noqa: E402

# Reverse of our maths maps: turn our Devanagari maths spellings back into clean
# Latin so the transcript shows "a", "b", "cos" — not romanised Devanagari.
_MATH_DEVA_TO_LATIN = {v: k for k, v in {**MATH_TERM_SOUNDS, **LETTER_SOUNDS}.items()}
_MATH_DEVA_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_MATH_DEVA_TO_LATIN, key=len, reverse=True))
)
_HAS_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def romanize_for_transcript(text: str) -> str:
    """Convert the bot's Devanagari text to readable Latin for the transcript."""
    # 1) our own maths spellings back to plain Latin (ए -> a, कॉस -> cos)
    text = _MATH_DEVA_RE.sub(lambda m: _MATH_DEVA_TO_LATIN[m.group(0)], text)
    # 2) romanise any remaining Devanagari (the actual Hindi words); English
    #    words already in the string are left untouched by the transliterator.
    if _HAS_DEVANAGARI.search(text):
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    # 3) tidy up ITRANS artefacts for a casual, readable look
    return text.lower().replace(".", "").replace("~", "")


class RomanizeTranscript(FrameProcessor):
    """Romanises the bot's spoken text for the transcript, after synthesis.

    Placed AFTER the TTS so the audio (already generated) is unaffected; only the
    text shown in the transcript is converted from Devanagari to Latin.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            frame.text = romanize_for_transcript(frame.text)
        await self.push_frame(frame, direction)


class FirstTokenLatency(FrameProcessor):
    """Tags the turn's transcript line with the perceived time-to-first-word
    (from 'you stopped speaking' to the first spoken token). The detailed
    stage-by-stage breakdown is emitted separately by LatencyObserver.
    """

    def __init__(self):
        super().__init__()
        self._t0 = None
        self._awaiting = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._t0 = time.monotonic()
            self._awaiting = True
        elif (
            isinstance(frame, TTSTextFrame)
            and self._awaiting
            and self._t0 is not None
            and frame.text
        ):
            ms = int((time.monotonic() - self._t0) * 1000)
            self._awaiting = False
            frame.text = f"⚡ {ms}ms · " + frame.text
        await self.push_frame(frame, direction)


class LatencyObserver(BaseObserver):
    """Measures the per-turn latency breakdown across the whole pipeline and emits
    it to the browser Events panel (type "latency-breakdown").

    Fix vs the first version: the student may PAUSE mid-sentence, which fires an
    early VAD-stop / partial transcript. So instead of grabbing the FIRST of each,
    we keep the LATEST pre-response marker and freeze them the moment the bot
    starts talking. All numbers are then measured forward from the confirmed turn
    end, so they're always positive and add up cleanly.

    Reported per turn (ms):
      endpoint_ms       = VAD silence detected -> turn confirmed (the ~0.2s wait)
      to_first_word_ms  = turn confirmed -> first spoken word (SLM + TTS)  (+word)
      to_llm_token_ms   = turn confirmed -> big LLM's first token           (+word)
      perceived_total_ms= VAD silence -> first spoken word (what you feel)
    """

    def __init__(self, rtvi, state):
        super().__init__()
        self._rtvi = rtvi
        self._state = state   # to read the SLM (filler) generation time
        self._reset()

    def _reset(self):
        self._t_vad = None          # latest VAD "stopped" before the response
        self._t_user = None         # latest turn-confirmed before the response
        self._t_first_word = None   # first spoken word (SLM filler)
        self._t_llm = None          # big LLM's first token
        self._first_word = None
        self._llm_word = None
        self._ttfb = {}             # per-service time-to-first-byte (stt/llm/tts)
        self._locked = False        # freeze pre-response markers once bot talks
        self._emitted = False

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        now = time.monotonic()

        if isinstance(frame, UserStartedSpeakingFrame):
            self._reset()  # a new turn begins (also covers barge-in)
            return

        # Per-service TTFB comes from Pipecat's own metrics (enable_metrics=True).
        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    p = d.processor  # e.g. "SarvamSTTService#0"
                    ms_val = int(d.value * 1000)
                    if "STT" in p:
                        self._ttfb["stt"] = ms_val
                    elif "LLM" in p:
                        self._ttfb["llm"] = ms_val
                    elif "TTS" in p:
                        self._ttfb["tts"] = ms_val
            return

        # Pre-response markers: keep the LATEST until the bot starts talking, so a
        # mid-sentence pause doesn't get mistaken for the end of the turn.
        if not self._locked:
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                self._t_vad = now
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._t_user = now

        # Response markers (only after a real turn end).
        if self._t_user is not None:
            if (
                isinstance(frame, TTSTextFrame)
                and self._t_first_word is None
                and (frame.text or "").strip()
            ):
                self._t_first_word = now
                self._first_word = frame.text.strip()[:60]
                self._locked = True
            elif (
                isinstance(frame, LLMTextFrame)
                and self._t_llm is None
                and (frame.text or "").strip()
            ):
                self._t_llm = now
                self._llm_word = frame.text.strip()[:60]
                self._locked = True

        if not self._emitted and self._t_user and self._t_first_word and self._t_llm:
            self._emitted = True
            await self._emit()

    async def _emit(self):
        def ms(a, b):
            return int((b - a) * 1000) if (a is not None and b is not None) else None

        breakdown = {
            "type": "latency-breakdown",
            # --- per-component latencies ---
            "vad_ms": ms(self._t_vad, self._t_user),        # silence detected -> turn confirmed
            "stt_ttfb_ms": self._ttfb.get("stt"),           # Sarvam speech-to-text
            "slm_ms": self._state.last_filler_ms,           # Groq filler generation
            "tts_ttfb_ms": self._ttfb.get("tts"),           # ElevenLabs first-audio
            "llm_ttfb_ms": self._ttfb.get("llm"),           # GPT-4o first token
            # --- what the student actually experiences ---
            "to_first_word_ms": ms(self._t_user, self._t_first_word),   # turn -> first spoken word
            "first_word": self._first_word,
            "to_llm_answer_ms": ms(self._t_user, self._t_llm),         # turn -> answer starts
            "llm_first_word": self._llm_word,
            "perceived_total_ms": ms(self._t_vad or self._t_user, self._t_first_word),
        }
        logger.info(f"Latency breakdown: {breakdown}")
        if self._rtvi is not None:
            try:
                await self._rtvi.send_server_message(breakdown)
            except Exception as e:
                logger.warning(f"Latency event failed: {e}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Wire up the services and run one conversation session."""
    logger.info(f"Starting voice tutor (language: {LANGUAGE_NAME})")

    sarvam_key = os.getenv("SARVAM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # 1) Speech-to-Text (the "ears"): Sarvam's "saaras" model is speech-to-text-
    # TRANSLATE — it auto-detects whatever language the student speaks (Hindi,
    # Telugu, Tamil, Kannada...) and outputs the transcript in ENGLISH. So the
    # transcript is always English even when the student talks in Hindi.
    stt = SarvamSTTService(
        api_key=sarvam_key,
        settings=SarvamSTTService.Settings(
            model="saaras:v2.5",
        ),
    )

    # 2) The "brain": GPT-4o reads the transcript and writes a reply.
    llm = OpenAILLMService(
        api_key=openai_key,
        model=os.getenv("LLM_MODEL", "gpt-4o"),
    )

    # 3) Text-to-Speech (the "voice"): ElevenLabs speaks GPT-4o's reply. The
    # voice itself is whichever ElevenLabs voice ID you picked (ELEVENLABS_VOICE_ID).
    # eleven_multilingual_v2 = the most natural / smoothest model (chosen for flow,
    # esp. across maths terms). Trade-off vs turbo: a bit more latency, and it has
    # no language-code — it auto-detects the language from the script the LLM writes,
    # so live language-switching still works because we reply in the native script.
    #
    # Voice settings tuned for SMOOTH delivery so Hindi<->English code-mixing
    # doesn't change speed/modulation abruptly:
    #   stability        higher = steadier, more even delivery (less jumpy)
    #   style            0 = no exaggeration, keeps tone consistent across words
    #   speed            fixed so the pace never lurches
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        # The maths-pronunciation filter runs on full aggregated sentences (so
        # multi-token terms like "cosec" stay intact before we rewrite them).
        text_filters=[MathsTextFilter()],
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model=os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            language=TTS_LANGUAGE,
            stability=0.6,
            similarity_boost=0.85,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
            apply_text_normalization="auto",
        ),
    )

    # --- Live language switching ---------------------------------------------
    # A "tool" GPT-4o can call when the student asks to change language. It
    # retunes Sarvam STT (the ears) and TTS (the voice) on the fly, so the very
    # next reply is heard and spoken in the new language.
    async def switch_language(params):
        target = str(params.arguments.get("language", "")).strip().lower()
        if target not in LANGUAGE_BY_NAME:
            await params.result_callback(
                {"status": "error", "message": f"Unsupported language: {target}"}
            )
            return

        name, _stt_lang, tts_lang = LANGUAGE_BY_NAME[target]
        logger.info(f"Switching language to {name}")

        # Retune only the VOICE (TTS) to the new language. The STT (saaras) auto-
        # detects whatever the student speaks and always outputs English, so it
        # needs no per-language switching.
        await params.pipeline_worker.queue_frames(
            [
                TTSUpdateSettingsFrame(delta=ElevenLabsTTSService.Settings(language=tts_lang)),
            ]
        )
        # Make the model keep using the new language for the rest of the chat.
        params.context.add_message(
            {
                "role": "system",
                "content": (
                    f"You are now speaking {name}. Continue the entire "
                    f"conversation in {name} until the student asks otherwise."
                ),
            }
        )
        await params.result_callback({"status": "ok", "language": name})

    switch_language_tool = FunctionSchema(
        name="switch_language",
        description=(
            "Switch the language the tutor listens in and speaks in. Call this "
            "whenever the student asks to talk in a different language."
        ),
        properties={
            "language": {
                "type": "string",
                "enum": ["Telugu", "Tamil", "Hindi", "Kannada"],
                "description": "The language to switch the conversation to.",
            }
        },
        required=["language"],
        handler=switch_language,
    )

    # Conversation memory. The aggregator also holds the voice-activity detector
    # (Silero VAD), which decides when the student has stopped speaking.
    context = LLMContext(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        tools=[switch_language_tool],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Fast SLM (Groq Llama by default) that produces the instant fillers. It shares
    # the same centralized `context` with the big LLM. `state` is the bookkeeping.
    state = ConversationState()
    slm = AsyncOpenAI(api_key=os.getenv(SLM_API_KEY_ENV), base_url=SLM_BASE_URL)
    logger.info(f"SLM: {SLM_PROVIDER} / {SLM_MODEL}")
    filler = FillerProcessor(context, state, slm, LANGUAGE_NAME)

    # Capture the student's finalized words (used by the background compaction).
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        state.last_user_text = getattr(message, "content", "") or ""

    # After the big model answers, compact the conversation for the SLM (background).
    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        assistant_text = getattr(message, "content", "") or ""
        asyncio.create_task(update_summary(slm, state, assistant_text))

    latency = FirstTokenLatency()        # measures time-to-first-word per turn

    # The pipeline = the assembly line the audio flows through, in order.
    pipeline = Pipeline(
        [
            transport.input(),           # microphone audio in from the browser
            stt,                         # speech  -> text
            user_aggregator,             # remember what the student said
            filler,                      # SLM: instant filler before the big model
            llm,                         # text    -> reply text
            tts,                         # single voice: filler + answer (multilingual)
            RomanizeTranscript(),        # bot transcript -> Latin (audio already made)
            latency,                     # tag + emit time-to-first-word event
            transport.output(),          # speaker audio out to the browser
            assistant_aggregator,        # remember what the bot said
        ]
    )

    # enable_metrics=True is what prints the per-stage latency (TTFB) numbers
    # you'll read in the terminal.
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    # Observe the whole pipeline and emit the per-turn latency breakdown to the UI.
    worker.add_observer(LatencyObserver(worker.rtvi, state))

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # The browser is connected and ready — have the tutor greet the student.
        context.add_message(
            {
                "role": "developer",
                "content": (
                    f"Greet the student in {LANGUAGE_NAME}. Start with a quick, "
                    "casual \"Hey,\" (no exclamation, not stretched — like a friend "
                    "saying hi) and flow straight into the sentence. Sound upbeat "
                    "but relaxed. In one short sentence say you can help with maths, "
                    "physics and chemistry doubts, and ask what they'd like help with."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Student connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Student disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point the Pipecat dev runner calls for each browser session."""
    transport_params = {
        # "webrtc" = self-hosted WebRTC — great for LOCAL testing (localhost:7860).
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        # "daily" = Daily's cloud transport — used when HOSTED so friends on any
        # network can connect (Daily handles the WebRTC/NAT plumbing). Requires
        # DAILY_API_KEY in the environment. Imported lazily so local runs without
        # the Daily SDK still work.
        "daily": lambda: _daily_params(),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def _daily_params():
    """Build DailyParams (lazy import so webrtc-only local runs don't need it)."""
    from pipecat.transports.daily.transport import DailyParams

    return DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
