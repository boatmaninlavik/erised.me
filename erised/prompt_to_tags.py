"""
Convert rich musical prompts/descriptions into HeartMuLa-compatible comma-separated tags.

Two strategies:
  1. LLM-based extraction (accurate, needs API key)
  2. Rule-based keyword extraction (fast, offline fallback)
"""

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a music tag extraction system for an AI music generator.
Given a musical description/prompt, extract the most relevant tags.

Output ONLY a JSON object with these keys (all values are lists of lowercase strings):
{
  "genre": [],
  "gender": [],
  "mood": [],
  "instrument": [],
  "timbre": [],
  "scene": [],
  "region": [],
  "topic": []
}

Rules:
- All tag values must be lowercase, 1-3 words each
- Only include tags clearly implied by the description
- If a specific artist or song is referenced, extract their STYLE attributes (genre, mood, production style) — never output the artist/song name as a tag
- For BPM references: <100 = slow, 100-120 = mid-tempo, 120-140 = upbeat, >140 = fast — include the tempo feel as a mood tag
- Keep each list to at most 4 tags
- If a category is not mentioned or implied, leave its list empty"""

TAG_VOCABULARY = {
    "genre": {
        "pop", "rock", "hiphop", "hip-hop", "rap", "r&b", "rnb", "jazz", "blues",
        "country", "electronic", "edm", "dance", "house", "techno", "trance",
        "classical", "folk", "indie", "alternative", "metal", "punk", "reggae",
        "soul", "funk", "gospel", "latin", "afrobeat", "k-pop", "j-pop",
        "drill", "trap", "lo-fi", "lofi", "ambient", "disco", "ska", "grunge",
        "new wave", "synthwave", "dubstep", "drum and bass", "garage",
        "grime", "uk drill", "afroswing", "dancehall", "bossa nova",
        "swing", "opera", "soundtrack", "ballad", "acoustic",
    },
    "gender": {"male", "female"},
    "mood": {
        "happy", "sad", "romantic", "melancholic", "energetic", "chill", "relaxed",
        "aggressive", "angry", "peaceful", "nostalgic", "dreamy", "dark", "bright",
        "uplifting", "emotional", "confident", "mysterious", "playful", "sensual",
        "epic", "triumphant", "somber", "bittersweet", "luxurious", "nonchalant",
        "boastful", "introspective", "euphoric", "moody", "gritty", "warm", "cool",
        "laid-back", "intense", "slow", "mid-tempo", "upbeat", "fast",
    },
    "instrument": {
        "piano", "guitar", "acoustic guitar", "electric guitar", "bass", "drums",
        "violin", "cello", "flute", "saxophone", "trumpet", "synthesizer",
        "808", "hi-hat", "snare", "kick", "strings", "brass", "woodwind",
        "harmonica", "banjo", "ukulele", "organ", "harp", "percussion",
        "marimba", "xylophone", "accordion", "mandolin", "sitar",
        "bass guitar", "drum machine", "sampler", "synth pad", "synth lead",
        "plucked strings", "bassline",
    },
    "scene": {
        "wedding", "party", "workout", "driving", "study", "sleep", "meditation",
        "cafe", "club", "festival", "road trip", "beach", "rain", "sunset",
        "morning", "night", "dinner", "gaming", "yoga", "running",
    },
    "timbre": {
        "raspy", "smooth", "powerful", "breathy", "warm", "bright", "husky",
        "nasal", "deep", "high-pitched", "soft", "crisp", "rich", "thin",
        "gravelly", "silky", "airy", "full", "light", "heavy", "clean", "dry",
    },
    "topic": {
        "love", "heartbreak", "party", "freedom", "wealth", "struggle",
        "nature", "friendship", "loss", "celebration", "rebellion",
        "self-discovery", "loneliness", "empowerment", "nostalgia",
        "adventure", "night life", "summer", "winter", "city life",
        "flex", "money", "hustle", "loyalty",
    },
    "region": {
        "american", "british", "korean", "japanese", "latin", "african",
        "french", "spanish", "brazilian", "caribbean", "indian",
        "middle eastern", "scandinavian", "australian", "chinese",
        "uk", "us", "mediterranean",
    },
}

# Phrases that map to tags not obvious from single-word matching
PHRASE_ALIASES = {
    "hip hop": "hiphop",
    "hip-hop": "hiphop",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "lo-fi": "lofi",
    "lo fi": "lofi",
    "uk drill": "drill,uk",
    "drum and bass": "drum and bass",
    "laid back": "laid-back",
    "laid-back": "laid-back",
    "back-and-forth": "conversational",
    "808 bass": "808,bass",
    "808 bassline": "808,bassline",
    "plucked string": "plucked strings",
    "syncopated hi-hats": "hi-hat",
    "syncopated hi hats": "hi-hat",
    "drill hi-hats": "hi-hat,drill",
    "male rapper": "male,rap",
    "male rappers": "male,rap",
    "female singer": "female",
    "male singer": "male",
    "female vocal": "female",
    "male vocal": "male",
}


class PromptToTags:
    def __init__(
        self,
        use_llm: bool = True,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.use_llm = use_llm and bool(api_key)
        self._client = None

        if self.use_llm:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url=base_url)
                self._model = model
                logger.info("PromptToTags: using LLM-based extraction (%s)", model)
            except ImportError:
                logger.warning("openai package not installed, falling back to rule-based extraction")
                self.use_llm = False

        if not self.use_llm:
            logger.info("PromptToTags: using rule-based extraction")

    def convert(self, prompt: str) -> str:
        """Convert a musical prompt to HeartMuLa-compatible comma-separated tags."""
        if self.use_llm:
            try:
                return self._llm_extract(prompt)
            except Exception as e:
                logger.warning("LLM extraction failed (%s), falling back to rule-based", e)
        return self._rule_extract(prompt)

    def _llm_extract(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON: %s", raw[:200])
            return self._rule_extract(prompt)

        tags = []
        for category in ("genre", "gender", "mood", "instrument", "timbre", "scene", "region", "topic"):
            vals = parsed.get(category, [])
            if isinstance(vals, list):
                tags.extend(v.strip().lower() for v in vals if v.strip())
            elif isinstance(vals, str) and vals.strip():
                tags.append(vals.strip().lower())

        tags = list(dict.fromkeys(tags))  # deduplicate, preserve order
        if not tags:
            return self._rule_extract(prompt)
        return ",".join(tags)

    def _rule_extract(self, prompt: str) -> str:
        text = prompt.lower()
        found_tags = []

        # Multi-word phrase aliases first (longer matches take priority)
        for phrase, mapped in sorted(PHRASE_ALIASES.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                for t in mapped.split(","):
                    t = t.strip()
                    if t and t not in found_tags:
                        found_tags.append(t)

        # Single-word / short-phrase vocabulary scan
        for _category, vocab in TAG_VOCABULARY.items():
            for tag in sorted(vocab, key=len, reverse=True):
                pattern = r"\b" + re.escape(tag) + r"\b"
                if re.search(pattern, text) and tag not in found_tags:
                    found_tags.append(tag)

        # BPM heuristic
        bpm_match = re.search(r"(\d{2,3})\s*bpm", text)
        if bpm_match:
            bpm = int(bpm_match.group(1))
            if bpm < 100 and "slow" not in found_tags:
                found_tags.append("slow")
            elif 100 <= bpm < 120 and "mid-tempo" not in found_tags:
                found_tags.append("mid-tempo")
            elif 120 <= bpm < 140 and "upbeat" not in found_tags:
                found_tags.append("upbeat")
            elif bpm >= 140 and "fast" not in found_tags:
                found_tags.append("fast")

        if not found_tags:
            found_tags = ["pop"]
            logger.warning("No tags extracted from prompt, defaulting to 'pop'")

        return ",".join(found_tags)
