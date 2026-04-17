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
const MODEL = "llama-3.3-70b-versatile";

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
  // Build per-sample word sets to identify cross-sample style words
  // Words that only appear in ONE sample are likely topic-specific (e.g. "grandpa")
  // Words that appear across MULTIPLE samples are true style fingerprint words
  const sampleWordSets = samples.map(s => {
    const sWords = (s.text.toLowerCase().match(/\b[a-z']+\b/g) || []);
    return new Set(sWords);
  });

  const wordFreq = {};
  words.forEach(w => { if (!stopwords.has(w) && w.length > 2) wordFreq[w] = (wordFreq[w] || 0) + 1; });

  // Count how many samples each word appears in
  const wordSampleCount = {};
  Object.keys(wordFreq).forEach(w => {
    wordSampleCount[w] = sampleWordSets.filter(s => s.has(w)).length;
  });

  // Only keep words that appear in at least 2 samples (true style words, not topic words)
  // Exception: if there's only 1 sample, fall back to frequency-based selection
  const minSamples = samples.length > 1 ? 2 : 1;
  const topWords = Object.entries(wordFreq)
    .filter(([w]) => wordSampleCount[w] >= minSamples)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([w]) => w);

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
// Built to defeat AI detectors (GPTZero, Originality, Turnitin)
// while cloning the user's exact personal voice.
//
// HOW AI DETECTORS WORK (and how we beat each one):
//
// 1. PERPLEXITY — detectors flag text where word choices are
//    too predictable. Humans reach for unexpected-but-correct
//    words. We instruct Groq to use slightly surprising word
//    choices that still make sense in context.
//
// 2. BURSTINESS — detectors flag uniform sentence lengths.
//    Real humans write: "Yep." then a massive run-on thought.
//    We aggressively vary lengths, not just slightly.
//
// 3. ENTROPY / PATTERN RECOGNITION — detectors flag repeated
//    structural templates (Topic sentence → 3 points → conclusion).
//    We break structural predictability by mirroring the user's
//    actual idiosyncratic patterns from their samples.
//
// 4. TRANSITION FINGERPRINTS — "Furthermore", "Moreover",
//    "Additionally" are instant AI flags. We ban all of them.
//
// 5. HEDGING DENSITY — AI overuses hedges ("it's worth noting",
//    "it's important to understand"). We strip all of these.
//
// 6. PASSIVE VOICE RATE — AI overuses passive voice. We push
//    toward active voice matching the user's own tendency.
// ============================================================
function buildSystemPrompt(dna, rawBlueprint, humanizeMode = false) {

  const dnaBlock = dna ? `
MEASURED WRITING FINGERPRINT (mathematically extracted from this person's real writing):
- Average sentence length: ${dna.avgLen} words
- Sentence rhythm: ${dna.rhythmLabel}
- Fragment rate: ${dna.fragmentRate}% of sentences are very short (4 words or fewer)
- Contraction usage: ${dna.contractionRate}% of words — ${dna.contractionLabel}
- Formality score: ${dna.formalityScore}/100 (0=texting, 100=academic journal)
- Punctuation profile: ${dna.punctuationNotes}
- Exclamation marks: ${dna.exclamationsPerSentence} per sentence
- Ellipses: ${dna.ellipsesPerSentence} per sentence
- Dashes: ${dna.dashesPerSentence} per sentence
- Vocabulary richness: ${dna.vocabRichness}% unique words
- Average word length: ${dna.avgWordLen} characters
- Signature STYLE words (connectors, fillers, expression words ONLY — never topic nouns from samples):
  ${dna.topWords.length ? dna.topWords.join(", ") : "none strongly detected"}
- Recurring phrase structures (mirror the structure, never the specific subject matter):
  ${dna.topPhrases.length ? dna.topPhrases.join(" | ") : "none strongly detected"}
` : "";

  const humanizeBlock = humanizeMode ? `

══════════════════════════════════════════════════════════════
PUSH HARDER — SECOND PASS HUMANIZATION
The first rewrite was flagged as too clean. This time:
- Be more aggressive with sentence length variation — go wider
- Add one or two more natural imperfections: a fragment, a
  trailing thought, a self-correction, a parenthetical aside
- Use one more of their signature filler words or connectors
- Let one sentence run longer than it "should" — real people do this
- The goal: a human reading this should think "yeah that's a person"
══════════════════════════════════════════════════════════════` : "";

  return `You are a writing style cloner and AI-detection defeater. Your job is to rewrite text so it passes AI detectors (GPTZero, Originality.ai, Turnitin) while sounding exactly like one specific human — not a generic human, this particular person.

Output ONLY the rewritten text. No intro. No label. No explanation. Nothing before or after the text.

${dnaBlock}

══════════════════════════════════════════════════════════════
PHASE 1 — DECODE THIS PERSON'S VOICE
══════════════════════════════════════════════════════════════
Before writing a single word, internalize the fingerprint above. Answer these internally:
- What is the average sentence length, and how much does it vary?
- What is the tone — casual, formal, confident, hedged, warm, dry?
- How do they open and close ideas — front-loaded point or build-to-it?
- What punctuation habits define them — dashes? Ellipses? Fragments? Questions?
- What vocabulary level — everyday words, precise but not academic, technical?
- What would sound completely wrong coming from this person?

Build a clear internal model before proceeding.

══════════════════════════════════════════════════════════════
PHASE 2 — DEFEAT AI DETECTORS
══════════════════════════════════════════════════════════════
AI detectors flag these specific patterns. Actively eliminate every single one:

BURSTINESS (most important detector signal):
Real humans write sentences of wildly different lengths in the same paragraph.
AI writes sentences that cluster around 15-20 words — disturbingly uniform.
You must create strong length contrast: very short sentences (3-8 words) mixed
with longer ones (25-40 words). Not a gentle mix — an aggressive one.
Example of real human burstiness: "I get it. But the problem isn't that simple —
there's a whole layer of context that most people skip over entirely when they
try to explain why this keeps happening, and honestly it makes the whole
conversation worse. Not more informed. Worse."

PERPLEXITY (second most important):
AI picks the most statistically expected word every time. Humans reach for
slightly unexpected-but-correct choices. Where AI writes "significant", a human
might write "real" or "actual" or "noticeable". Where AI writes "demonstrate",
a human writes "show" or "prove" or "make clear". Pick the word a thoughtful
person would choose naturally — not the word an AI would predict.

TRANSITION FINGERPRINTS — these are instant detector flags, ban every one:
NEVER USE: "Furthermore", "Moreover", "Additionally", "In conclusion",
"To summarize", "It is worth noting", "It is important to note",
"It's crucial to understand", "Notably", "Significantly", "Subsequently",
"In summary", "To elaborate", "In other words", "As previously mentioned",
"Having said that", "With that being said", "It goes without saying",
"Needless to say", "Last but not least", "First and foremost",
"At the end of the day", "All things considered", "Upon reflection"

Instead use the transitions this person actually uses — pulled from their
fingerprint above. If none are obvious, use casual connectors: "but", "so",
"and", "because", "though", "which means", "the thing is", "honestly".

HEDGING DENSITY — AI over-hedges, detectors know this:
Never use: "it's worth noting", "it's important to understand", "one might argue",
"it could be said", "this suggests that", "arguably", "in many ways",
"to some extent", "it is generally accepted", "broadly speaking"
Match only the hedging level shown in this person's fingerprint.

PASSIVE VOICE RATE — AI overuses passive voice:
Convert passive constructions to active wherever this writer would naturally use active.
"It was found that X" → "X showed" or "X turned out to be"
"This can be seen in" → "You see this in" or "This shows up in"

STRUCTURAL PREDICTABILITY — AI always follows Topic→Points→Conclusion:
Break this template. Follow this person's actual structural fingerprint instead.
If they front-load conclusions — front-load. If they tell it like a story — narrate.
If they ask questions and answer them — do that. Never use the generic AI structure.

SENTENCE STARTER VARIETY — AI starts sentences the same way repeatedly:
Vary how sentences begin. Mix subject-first, conjunction-first ("But", "So", "And"),
clause-first ("When X happens,"), and fragment openers. Never start three
consecutive sentences with the same type of opener.

══════════════════════════════════════════════════════════════
PHASE 3 — REWRITE: VOICE + HUMAN TEXTURE COMBINED
══════════════════════════════════════════════════════════════
Now write the rewrite, combining this person's voice with human texture:

- Every single fact, idea, argument, and piece of information from the original must be preserved
- Do not add ideas that weren't there. Do not remove ideas that were.
- Prioritize sentence rhythm above vocabulary — rhythm is what sounds like a person
- Use this person's characteristic punctuation habits from the fingerprint
- Let one or two natural imperfections through: a slightly long sentence,
  a trailing thought, a fragment for punch, a parenthetical aside
- Do not sanitize their voice — if they're direct, be direct; if they ramble a little, let it ramble a little
- The result should make an AI detector say "human" with high confidence
- The result should make anyone who knows this person say "yep, that's them"

ABSOLUTE RULES:
- Output ONLY the rewritten text. Zero intro, zero label, zero preamble.
- Preserve full intellectual depth and sophistication of the original — do not simplify
- Do not make it cleaner or more polished than this writer actually is
- Never copy sentence structures directly from the writing samples
- Match this person's exact confidence and energy level — no more, no less
${humanizeBlock}

══════════════════════════════════════════════════════════════
QUALITATIVE VOICE BLUEPRINT (deep analysis of their writing samples)
══════════════════════════════════════════════════════════════
${rawBlueprint || "No qualitative blueprint — rely on the measured fingerprint above."}`;
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
    const userPrompt = `Read the text below. Absorb every idea, fact, argument, and detail completely. Then put it aside mentally and rewrite it entirely from scratch in this person's exact voice.

CRITICAL REQUIREMENTS (every single one must be met):

1. PRESERVE EVERYTHING — every fact, idea, argument, and detail stays. Nothing added, nothing cut.

2. BURSTINESS IS MANDATORY — this is the #1 detector signal. Within every paragraph, you must have:
   - At least one very short sentence (3-8 words)
   - At least one longer sentence (20-35+ words) that flows naturally
   - Sentence lengths must vary aggressively, not gently

3. FORBIDDEN TRANSITIONS — using any of these causes an instant fail:
   "Furthermore", "Moreover", "Additionally", "In conclusion", "To summarize",
   "It is worth noting", "Notably", "Subsequently", "Having said that",
   "With that being said", "It goes without saying", "First and foremost",
   "Last but not least", "At the end of the day", "All things considered"

4. WORD CHOICE — pick words a real person naturally reaches for, not the statistically
   expected word. "Significant" → "real" or "actual". "Demonstrate" → "show".
   "Utilize" → "use". Choose what feels natural, not what sounds impressive.

5. NATURAL IMPERFECTIONS — let through at least one: a fragment for punch, a slightly
   long sentence that runs a bit, a trailing thought, a parenthetical aside.

6. PRESERVE INTELLECTUAL DEPTH — do not simplify ideas or water down arguments.
   Match the sophistication level of the original exactly.

7. MATCH THIS PERSON'S VOICE — use their punctuation habits, rhythm, and tone from the fingerprint.

Output ONLY the rewritten text. No intro, no label, no preamble. The first word of your response must be the first word of the rewritten text.

Text to rewrite:\n\n${text}`;

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
    const userPrompt = `This text was rewritten but still reads slightly AI-generated. Your job is a full second-pass rewrite that pushes harder on every human quality.

The previous rewrite was too clean. This time:

BURSTINESS — go more extreme. Find every place where sentences are similar in length and break that pattern. Add a one-sentence paragraph. Let another sentence run longer than comfortable. Real humans do this constantly.

WORD CHOICES — anywhere a word feels like the "safe" or "expected" pick, swap it for what a real person would actually say out loud.

STRUCTURE — break any remaining predictable patterns. If you see Topic→Evidence→Conclusion, disrupt it. Real writing doesn't always follow that arc.

IMPERFECTIONS — add one more natural human element: a fragment, a casual aside in parentheses, a rhetorical question, a self-correction mid-sentence.

TRANSITIONS — if you see ANY of these, replace them immediately:
"Furthermore", "Moreover", "Additionally", "In conclusion", "To summarize",
"It is worth noting", "Notably", "Having said that", "With that being said"

Keep every idea, fact, and argument from the original. Preserve full intellectual depth.
Output ONLY the rewritten text. First word of response = first word of text.

Text to push harder on:\n\n${text}`;

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

