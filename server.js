// ============================================================
// MyVoice — Render.com Backend Server v4.0
// Express.js + Groq SDK
//
// POST /analyze  — extracts style DNA from writing samples
// POST /rewrite  — rewrites text in your exact voice
// POST /humanize — same but pushes harder on human texture
// GET  /health   — uptime check
// ============================================================

const express = require("express");
const cors = require("cors");
const https = require("https");

const app = express();
const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL = "llama3-70b-8192";

// ---- Middleware ----
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ============================================================
// GROQ API CALL (native https — no SDK needed, works on Render free tier)
// ============================================================
async function callGroq(systemPrompt, userPrompt, maxTokens = 2000) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: MODEL,
      max_tokens: maxTokens,
      temperature: 0.72,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ]
    });

    const options = {
      hostname: "api.groq.com",
      path: "/openai/v1/chat/completions",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        "Content-Length": Buffer.byteLength(body)
      }
    };

    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", chunk => data += chunk);
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) return reject(new Error(parsed.error.message || "Groq error"));
          const text = parsed?.choices?.[0]?.message?.content || "";
          resolve(text);
        } catch (e) {
          reject(new Error("Failed to parse Groq response"));
        }
      });
    });

    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

// ============================================================
// LOCAL STYLE DNA — hard metrics extracted from raw text
// These concrete numbers get injected into every Groq prompt
// so the model has a real mathematical fingerprint, not just vibes
// ============================================================
function extractStyleDNA(samples) {
  const allText = samples.map(s => s.text).join(" ");
  const sentences = allText.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 5);
  const words = allText.toLowerCase().match(/\b[a-z']+\b/g) || [];
  const sentenceCount = Math.max(sentences.length, 1);

  // Sentence length
  const lengths = sentences.map(s => s.split(/\s+/).length);
  const avgLen = lengths.length ? Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length) : 15;
  const shortCount = lengths.filter(l => l <= 8).length;
  const longCount = lengths.filter(l => l >= 25).length;
  const rhythmLabel =
    shortCount > longCount * 2 ? "punchy and short — mostly under 10 words" :
    longCount > shortCount * 2 ? "sprawling and detailed — often 25+ words" :
    "varied — mixes short punchy lines with longer flowing ones";

  // Vocabulary
  const uniqueWords = new Set(words);
  const vocabRichness = words.length ? Math.round((uniqueWords.size / words.length) * 100) : 50;
  const avgWordLen = words.length ? Math.round(words.reduce((a, w) => a + w.length, 0) / words.length) : 5;

  // Contractions
  const contractionMatches = (allText.match(/\b(don't|doesn't|can't|won't|it's|i'm|i've|i'd|i'll|we're|we've|they're|you're|you've|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|wouldn't|shouldn't|couldn't|that's|there's|here's|let's|who's|what's)\b/gi) || []).length;
  const contractionRate = words.length ? Math.round((contractionMatches / words.length) * 100) : 0;
  const contractionLabel =
    contractionRate > 8 ? "heavy — writes the way they talk" :
    contractionRate > 3 ? "moderate — semi-casual" :
    "rare — leans more formal";

  // Punctuation
  const exclamations = (allText.match(/!/g) || []).length;
  const questions = (allText.match(/\?/g) || []).length;
  const ellipses = (allText.match(/\.\.\./g) || []).length;
  const dashes = (allText.match(/[—\-]{1,2}/g) || []).length;
  const commas = (allText.match(/,/g) || []).length;
  const punctuationNotes = [
    exclamations / sentenceCount > 0.3 ? "uses exclamation marks often" : null,
    ellipses / sentenceCount > 0.2 ? "uses ellipses for trailing or unfinished thoughts" : null,
    dashes / sentenceCount > 0.3 ? "uses dashes heavily for asides and interruptions" : null,
    questions / sentenceCount > 0.2 ? "asks rhetorical questions frequently" : null,
    commas / sentenceCount > 2 ? "comma-heavy — stacks clauses" : null,
  ].filter(Boolean).join("; ") || "standard punctuation — no strong habits";

  // Formality
  const formalWords = (allText.match(/\b(therefore|moreover|furthermore|subsequently|nevertheless|accordingly|consequently|albeit|notwithstanding|utilize|facilitate|implement|leverage|regarding|pertaining)\b/gi) || []).length;
  const casualWords = (allText.match(/\b(kinda|sorta|gonna|wanna|gotta|tbh|ngl|lowkey|highkey|literally|basically|honestly|like|just|really|super|totally|pretty|stuff|yeah|yep|nope|okay)\b/gi) || []).length;
  let formalityScore = 50 + (formalWords * 5) - (casualWords * 3) - (contractionRate * 2);
  formalityScore = Math.max(0, Math.min(100, formalityScore));

  // Fragment rate
  const fragments = sentences.filter(s => s.split(/\s+/).length <= 4).length;
  const fragmentRate = Math.round((fragments / sentenceCount) * 100);

  // Top signature words (skip stopwords)
  const stopwords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","shall","can","i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","what","which","who","when","where","why","how","all","each","every","not","no","so","if","as","by","from","up","about","than","then","just","also","there","here","very","more","some","any","only","out","into","get","got","go","going"]);
  const wordFreq = {};
  words.forEach(w => { if (!stopwords.has(w) && w.length > 2) wordFreq[w] = (wordFreq[w] || 0) + 1; });
  const topWords = Object.entries(wordFreq).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([w]) => w);

  // Top recurring phrases (2–3 word combos)
  const phraseFreq = {};
  for (let i = 0; i < words.length - 2; i++) {
    const [w1, w2, w3] = [words[i], words[i+1], words[i+2]];
    if (!stopwords.has(w1) || !stopwords.has(w2)) {
      const p2 = `${w1} ${w2}`;
      phraseFreq[p2] = (phraseFreq[p2] || 0) + 1;
      const p3 = `${w1} ${w2} ${w3}`;
      phraseFreq[p3] = (phraseFreq[p3] || 0) + 1;
    }
  }
  const topPhrases = Object.entries(phraseFreq)
    .filter(([, c]) => c >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([p]) => p);

  return {
    avgLen, rhythmLabel, vocabRichness, avgWordLen,
    contractionRate, contractionLabel, punctuationNotes, formalityScore,
    fragmentRate, topWords, topPhrases,
    exclamationsPerSentence: Math.round((exclamations / sentenceCount) * 10) / 10,
    ellipsesPerSentence: Math.round((ellipses / sentenceCount) * 10) / 10,
    dashesPerSentence: Math.round((dashes / sentenceCount) * 10) / 10,
  };
}

// ============================================================
// THE CORE REWRITE SYSTEM PROMPT
// This is the heart of the extension. It tells Groq exactly
// how to analyze a writing fingerprint and then imitate it.
// The 3-step framework is non-negotiable and runs every time.
// ============================================================
function buildSystemPrompt(dna, rawBlueprint, humanizeMode = false) {

  const dnaBlock = dna ? `
MEASURED WRITING FINGERPRINT (extracted mathematically from real samples):
- Average sentence length: ${dna.avgLen} words
- Sentence rhythm: ${dna.rhythmLabel}
- Fragment rate: ${dna.fragmentRate}% of sentences are very short (4 words or fewer)
- Contraction usage: ${dna.contractionRate}% of words — ${dna.contractionLabel}
- Formality score: ${dna.formalityScore}/100 (0 = texting a friend, 100 = academic journal)
- Punctuation profile: ${dna.punctuationNotes}
- Exclamation marks: ${dna.exclamationsPerSentence} per sentence on average
- Ellipses: ${dna.ellipsesPerSentence} per sentence on average
- Dashes: ${dna.dashesPerSentence} per sentence on average
- Vocabulary richness: ${dna.vocabRichness}% unique words (higher = more varied vocabulary)
- Average word length: ${dna.avgWordLen} characters
- Signature words (characteristic to this writer — use naturally when they fit):
  ${dna.topWords.length ? dna.topWords.join(", ") : "none strongly detected"}
- Recurring phrase patterns:
  ${dna.topPhrases.length ? dna.topPhrases.join(" | ") : "none strongly detected"}
` : "";

  const humanizeBlock = humanizeMode ? `

HUMANIZE MODE — THIS IS YOUR PRIMARY DIRECTIVE:
The previous version was too clean and AI-detectable. Push harder:
- Make paragraph lengths more chaotic — vary them wildly
- Add more intentional fragments where they feel natural
- Let a sentence run a little long if that fits this writer
- Use a couple more of their signature words and filler phrases than usual
- Break minor grammar rules the way a real person does when typing fast
- Make it feel like it was dashed off in five minutes, not polished for thirty
- Aim for a result that AI detectors would confidently call human` : "";

  return `You are a writing style cloner. Your only job is to rewrite text so it sounds exactly like a specific human wrote it — not an AI, not a generic writer, this one person.

You do not summarize. You do not explain. You do not add a preamble. You output only the rewritten text.

${dnaBlock}

Follow this exact three-step process internally before writing a single word of output:

============================================================
STEP 1: ANALYZE THE WRITING FINGERPRINT
============================================================
Before writing a single word, study the measured fingerprint and samples above. Identify ALL of these:

- Average sentence length: are they short (under 10 words), medium, long (25+), or mixed?
- Rhythm: is it choppy and punchy, smooth and flowing, reflective and slow, fast-paced, or something else?
- Tone: casual, formal, emotional, analytical, confident, uncertain — or a specific combination?
- Structure habits: how does this writer introduce an idea? Do they frontload the point or build to it? How do they conclude?
- Vocabulary level: simple and direct, advanced and technical, repetitive, or richly varied?
- Repeated patterns: any phrases, transitions, or sentence starters they keep returning to?
- Thinking style: do they think linearly, in layers, through storytelling, through explanation, or by asking questions?
- Punctuation habits: heavy commas? Fragments? Run-ons? Dashes? Ellipses? What is completely absent?

Summarize all of this into a clear internal style profile. Do not skip this step.

============================================================
STEP 2: BUILD THE STYLE MODEL
============================================================
From your analysis, build an internal writing model with these exact components:

- Sentence rhythm rules: how long are sentences typically, and how do they vary? Do short and long alternate?
- Tone rules: what does this voice feel like? What emotional register does it live in?
- Structure rules: how are paragraphs typically built? Short punchy blocks or longer developed ones?
- Word choice rules: what vocabulary level fits? What kind of words would this person actually use?
- DO NOT DO list: what would this writer NEVER say? What phrasing would feel completely wrong? What AI phrases are banned for this voice specifically?

Every sentence you write must pass through this model before it is output.

============================================================
STEP 3: MATCH THE STYLE — REWRITE THE TEXT
============================================================
Now rewrite the given text using your style model:

- Prioritize matching rhythm and flow ABOVE everything else — this matters more than exact vocabulary
- Preserve every fact and piece of meaning from the original — do not add or remove any information
- Avoid ALL generic AI phrasing — write the way this specific human actually thinks out loud
- Keep the same thinking patterns as the original writer — if they build to a point, build to it; if they front-load, front-load
- Do not copy sentences directly from the samples — generate fresh sentences that fit the fingerprint
- Make it feel like the same person sat down and typed this themselves, naturally, without thinking about style

IMPORTANT RULES — NON-NEGOTIABLE:
- Do not copy sentences directly from the writing samples
- Focus on imitating structure, rhythm, and flow — not just swapping vocabulary
- If uncertain about anything, default to matching sentence rhythm and tone first
- The output must feel like the same person is thinking out loud — not a robot doing an impression

============================================================
ABSOLUTE RULES — NEVER BREAK THESE
============================================================
- Output ONLY the rewritten text. No intro. No label. No "Here is the rewrite:". Nothing before or after the text itself.
- Never use these phrases — they are instant AI tells: "Furthermore", "Moreover", "Additionally", "It is worth noting", "In conclusion", "To summarize", "utilize", "leverage", "delve into", "it's important to note", "I hope this helps", "Certainly!", "Absolutely!", "Great question!", "As an AI"
- Do not make the writing cleaner or more polished than the original writer actually is
- Do not over-explain. Do not hedge. Match the writer's actual confidence and energy level.
- The result must be indistinguishable from something this person typed themselves
${humanizeBlock}

============================================================
QUALITATIVE STYLE BLUEPRINT (from AI analysis of their samples)
============================================================
${rawBlueprint || "No qualitative blueprint available — rely on the measured fingerprint above."}`;
}

// ============================================================
// GET /health — quick uptime check
// ============================================================
app.get("/health", (req, res) => {
  res.json({ status: "ok", model: MODEL, groqKeySet: !!GROQ_API_KEY });
});

// ============================================================
// POST /analyze
// Accepts: { samples: [{ label, text }] }
// Returns: { blueprint: "<json string with dna + rawBlueprint>" }
// ============================================================
app.post("/analyze", async (req, res) => {
  try {
    const { samples } = req.body;
    if (!samples || !samples.length) {
      return res.status(400).json({ error: "No samples provided" });
    }

    // Step A: Extract hard metrics locally (instant, no API cost)
    const dna = extractStyleDNA(samples);

    // Step B: Send samples to Groq for deep qualitative analysis
    const samplesText = samples
      .map((s, i) => `--- Sample ${i + 1}: ${s.label || "Untitled"} ---\n${s.text}`)
      .join("\n\n");

    const systemPrompt = `You are a forensic writing analyst. Your job is to identify the deep personality, tone, rhythm, and structural habits of a writer from their real samples. Be specific. Quote actual phrases as evidence for every point. Output ONLY the sections requested — no preamble, no sign-off, nothing extra.`;

    const userPrompt = `Study these writing samples carefully and extract this writer's style fingerprint. Output ONLY the following sections — nothing before or after:

${samplesText}

VOICE & PERSONALITY:
Describe their exact vibe and energy as a writer in 2–3 sentences. Quote one phrase from the samples as evidence.

TONE PATTERNS:
How does their tone shift across the samples? Do they use sarcasm, irony, or humor to make a point? Quote a specific example.

SENTENCE RHYTHM QUIRKS:
Go beyond length. Do they use fragments for punch? Answer their own rhetorical questions? Repeat a word for emphasis? Use one-word sentences? Quote examples from the samples.

PARAGRAPH & FLOW STYLE:
Do they frontload the main point or build toward it? Do they write in short punchy blocks or long flowing ones? How do they move between ideas?

PUNCTUATION PERSONALITY:
What does their punctuation say about them? What do they use heavily? What is clearly absent? Be specific.

VOCABULARY FINGERPRINT:
What words and phrases are distinctly theirs — not common filler, but real signature expressions? Quote them directly from the samples.

THINGS THEY NEVER DO:
Based on all the samples, what writing habits are clearly absent? What would sound completely wrong coming from this person?

PERSONALITY TRAITS:
List 5–8 single words that describe this writer's voice. Example: direct, sarcastic, warm, impatient, self-aware.

CLONING INSTRUCTIONS:
Write 2–3 paragraphs addressed directly to an AI that will imitate this person. Use second person: "When writing as this person, you should..." Cover their overall approach, their relationship with the reader, their energy level, and the single most important thing to get right about their voice.`;

    const rawBlueprint = await callGroq(systemPrompt, userPrompt, 1500);

    // Package hard metrics + qualitative analysis together
    const blueprint = JSON.stringify({ dna, rawBlueprint });

    res.json({ blueprint });

  } catch (err) {
    console.error("/analyze error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /rewrite
// Accepts: { text, blueprint }
// Returns: { rewritten }
// ============================================================
app.post("/rewrite", async (req, res) => {
  try {
    const { text, blueprint } = req.body;
    if (!text || !blueprint) {
      return res.status(400).json({ error: "Missing text or blueprint" });
    }

    // Parse blueprint — supports new JSON format and old string format
    let dna = null;
    let rawBlueprint = blueprint;
    try {
      const parsed = JSON.parse(blueprint);
      if (parsed.dna && parsed.rawBlueprint) {
        dna = parsed.dna;
        rawBlueprint = parsed.rawBlueprint;
      }
    } catch (e) { /* old string format — use as-is */ }

    const systemPrompt = buildSystemPrompt(dna, rawBlueprint, false);
    const userPrompt = `Rewrite this in this person's exact voice. Keep every fact and piece of meaning. Output ONLY the rewritten text — nothing else:\n\n${text}`;

    const rewritten = await callGroq(systemPrompt, userPrompt, 1500);

    res.json({ rewritten });

  } catch (err) {
    console.error("/rewrite error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /humanize
// Same as /rewrite but pushes harder on human messiness
// Accepts: { text, blueprint }
// Returns: { rewritten }
// ============================================================
app.post("/humanize", async (req, res) => {
  try {
    const { text, blueprint } = req.body;
    if (!text || !blueprint) {
      return res.status(400).json({ error: "Missing text or blueprint" });
    }

    let dna = null;
    let rawBlueprint = blueprint;
    try {
      const parsed = JSON.parse(blueprint);
      if (parsed.dna && parsed.rawBlueprint) {
        dna = parsed.dna;
        rawBlueprint = parsed.rawBlueprint;
      }
    } catch (e) { /* old string format */ }

    const systemPrompt = buildSystemPrompt(dna, rawBlueprint, true);
    const userPrompt = `This text needs to sound more human and less AI-generated. Rewrite it with more natural messiness while keeping this person's exact voice. Output ONLY the rewritten text:\n\n${text}`;

    const rewritten = await callGroq(systemPrompt, userPrompt, 1500);

    res.json({ rewritten });

  } catch (err) {
    console.error("/humanize error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// START SERVER
// ============================================================
app.listen(PORT, () => {
  console.log(`MyVoice server v4.0 running on port ${PORT}`);
  if (!GROQ_API_KEY) console.warn("WARNING: GROQ_API_KEY is not set — all Groq calls will fail");
});

