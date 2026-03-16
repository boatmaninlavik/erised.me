import { NextRequest, NextResponse } from "next/server";

const GEMINI_KEY = process.env.GEMINI_API_KEY;
const MODEL = "gemini-2.5-flash";

const PROMPT_SYSTEM = `You are a world-class music producer and A&R executive with encyclopedic knowledge of music across all genres, eras, and cultures. Your job is to come up with a vivid musical description for a song that could be a genuine hit.

Write a concise but detailed description covering the genre, tempo, vocal style (including gender), instrumentation, and overall vibe/mood. You can structure it however feels natural — use a flowing sentence, labeled fields, or a mix. The key is to be SPECIFIC and paint a clear sonic picture.

Here are some example styles (vary your format, don't always use the same one):
- "Genre: Melodic Trap / Alternative Rock. Tempo: 75 BPM. Vocals: Male, raspy, heavy vibrato, melodic delivery with subtle autotune and vocal layering. Instruments: Reverb-soaked electric guitar loops, booming 808 bass, crisp trap drums, and atmospheric synth pads. Vibe: Melancholic, gritty, anthemic."
- "Dark uptempo Afrobeats fused with amapiano, 112 BPM. Female vocalist — smooth, breathy, with call-and-response adlibs. Log drums, shaker percussion, deep sub-bass, plucky synth melodies, and chopped vocal samples. Seductive, hypnotic, dancefloor energy."
- "Dreamy bedroom pop with shoegaze textures, around 90 BPM. Male vocals, soft and whispered, drenched in reverb. Layers of fuzzy guitars, warm analog synths, lo-fi drum machine, and distant piano. Nostalgic, intimate, bittersweet."

Rules:
- Draw inspiration from timeless hits, Billboard chart-toppers, and iconic artists across ALL genres: hip-hop, pop, R&B, rock, electronic, soul, country, jazz, Latin, Afrobeats, K-pop, drill, phonk, indie, metal, reggaeton, etc.
- Be SPECIFIC and CREATIVE — avoid generic descriptions. Mix unexpected influences.
- Vary wildly each time. Don't default to the same genres or formats.
- Always include vocal gender somewhere in the description.
- Output ONLY the musical prompt. No preamble, no explanation, no title.`;

const LYRICS_SYSTEM = `You are a world-class songwriter who has written #1 hits across genres. You write lyrics with genuine personality, vivid imagery, clever wordplay, emotional depth, and authentic voice.

Rules:
- Write a COMPLETE song with a full structure. EVERY song MUST include at least one [Chorus] or [Hook]. Use AT LEAST 5-7 sections. Example structures (vary these creatively):
  [Verse 1] → [Pre-Chorus] → [Chorus] → [Verse 2] → [Pre-Chorus] → [Chorus] → [Bridge] → [Chorus]
  [Verse 1] → [Chorus] → [Verse 2] → [Chorus] → [Bridge] → [Chorus] → [Outro]
  [Verse 1] → [Hook] → [Verse 2] → [Hook] → [Bridge] → [Verse 3] → [Hook]
  [Verse 1] → [Chorus] → [Verse 2] → [Chorus] → [Verse 3] → [Chorus]
- NEVER start with [Intro]. Always start with [Verse 1].
- NEVER use [Intro] sections at all. Songs start directly with [Verse 1].
- Each section name MUST be in [square brackets] on its own line
- Each line of lyrics MUST be on its own separate line
- Leave a blank line between sections
- The [Chorus] must be catchy and memorable — this is the most important part of the song
- NEVER write boring, cliched, or generic lyrics. No "I'm walking down the road" type filler. Write like you're competing for Song of the Year.
- Draw from the styles of legendary songwriters and artists. Be authentic to the genre.
- Vary your style dramatically each time — different genres, moods, perspectives, themes
- On the VERY FIRST LINE, write a creative song title (2-5 words, no quotes, no "Title:" prefix). Then leave a blank line. Then write the lyrics.
- IMPORTANT: Write the FULL song. Do not stop early. Every song needs at least a Verse 1, Chorus, Verse 2, and a second Chorus.`;

export async function POST(req: NextRequest) {
  if (!GEMINI_KEY) {
    return NextResponse.json(
      { error: "Gemini API key not configured. Add GEMINI_API_KEY to environment variables." },
      { status: 500 }
    );
  }

  const body = await req.json();
  const { type, context } = body;

  let systemPrompt: string;
  let userMessage: string;

  if (type === "prompt") {
    systemPrompt = PROMPT_SYSTEM;
    userMessage =
      "Generate a unique, creative musical prompt for a song. Pick an unexpected genre or blend of genres. Be diverse — surprise me with something fresh.";
  } else if (type === "lyrics") {
    systemPrompt = LYRICS_SYSTEM;
    userMessage = context
      ? `Write complete song lyrics that match this musical style:\n${context}\n\nMake the lyrics authentic to this genre and production style.`
      : "Write complete song lyrics for a creative, original song. Pick any genre or style — be bold and original.";
  } else {
    return NextResponse.json({ error: "Invalid type. Use 'prompt' or 'lyrics'." }, { status: 400 });
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${GEMINI_KEY}`;
  const payload = JSON.stringify({
    systemInstruction: { parts: [{ text: systemPrompt }] },
    contents: [{ parts: [{ text: userMessage }] }],
    generationConfig: { temperature: 1.2, maxOutputTokens: 4096 },
  });

  try {
    let lastStatus = 0;
    for (let attempt = 0; attempt < 3; attempt++) {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
      });

      lastStatus = resp.status;

      if (resp.status === 503 || resp.status === 429) {
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
        continue;
      }

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        console.error("Gemini API error:", resp.status, errData);
        return NextResponse.json(
          { error: `Gemini API error: ${resp.status}` },
          { status: 502 }
        );
      }

      const data = await resp.json();
      const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;

      if (!text) {
        return NextResponse.json({ error: "Empty response from Gemini" }, { status: 502 });
      }

      // For lyrics, extract the title from the first line
      if (type === "lyrics") {
        const trimmed = text.trim();
        const firstNewline = trimmed.indexOf("\n");
        if (firstNewline > 0) {
          const firstLine = trimmed.slice(0, firstNewline).trim();
          // If first line doesn't look like a section header, it's the title
          if (!firstLine.startsWith("[")) {
            const rest = trimmed.slice(firstNewline).trim();
            return NextResponse.json({ text: rest, title: firstLine });
          }
        }
        return NextResponse.json({ text: trimmed });
      }

      return NextResponse.json({ text: text.trim() });
    }

    return NextResponse.json(
      { error: `Gemini API overloaded (${lastStatus}). Try again in a moment.` },
      { status: 502 }
    );
  } catch (e: unknown) {
    console.error("generate-random error:", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Unknown error" },
      { status: 500 }
    );
  }
}
