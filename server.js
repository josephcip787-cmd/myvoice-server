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
// THE CORE REWRITE SYSTEM PROMPT v5.0
// Strategy: SEMANTIC RECONSTRUCTION — not paraphrasing.
//
// WHY PARAPHRASING FAILS DETECTORS:
// Swapping words and reordering sentences doesn't change the
// underlying statistical shape of the text. GPTZero v4.1b
// measures perplexity (word predictability), burstiness
// (sentence length variance), vocabulary distribution, topic
// coherence patterns, and linguistic entropy simultaneously.
// Even well-paraphrased AI text keeps its AI-shaped skeleton.
//
// WHY SEMANTIC RECONSTRUCTION WORKS:
// We extract the MEANING, then build entirely new sentences
// from scratch using the human's voice fingerprint as the
// template. The output has genuinely human statistical
// properties — not disguised AI properties.
//
// The approach that drops GPTZero detection below 10%:
// High perplexity (surprising-but-correct word choices) +
// genuine burstiness (chaotic sentence length variation) +
// human structural patterns (not Topic→Points→Conclusion) +
// zero AI transition phrases + active voice dominance.
// ============================================================
function buildSystemPrompt(dna, rawBlueprint, humanizeMode = false) {

  const dnaBlock = dna ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S WRITING FINGERPRINT (measured from real samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sentence rhythm: ${dna.rhythmLabel}
Average sentence length: ${dna.avgLen} words
Fragment rate: ${dna.fragmentRate}% of sentences are 4 words or fewer
Contraction style: ${dna.contractionLabel} (${dna.contractionRate}% contraction rate)
Formality: ${dna.formalityScore}/100 (0=texting a friend, 100=academic journal)
Punctuation: ${dna.punctuationNotes}
Exclamations: ${dna.exclamationsPerSentence} per sentence
Ellipses: ${dna.ellipsesPerSentence} per sentence  
Dashes: ${dna.dashesPerSentence} per sentence
Vocabulary richness: ${dna.vocabRichness}% unique words
Avg word length: ${dna.avgWordLen} characters
Style connectors/fillers (use naturally, NOT topic-specific nouns):
  ${dna.topWords.length ? dna.topWords.join(", ") : "none detected"}
Recurring phrase structures (copy the STRUCTURE not the subject):
  ${dna.topPhrases.length ? dna.topPhrases.join(" | ") : "none detected"}
` : "";

  const pushHarderBlock = humanizeMode ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUSH HARDER MODE — second pass, go more extreme
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The previous version still reads slightly AI. Push further:
- Widen the sentence length range even more — add a 1-3 word
  sentence somewhere unexpected. Let another run 40+ words.
- Make word choices less safe. Where you wrote a predictable
  word, swap it for what this person would actually say.
- Break one more structural pattern. Start a paragraph with
  a conjunction. End one abruptly. Use a one-sentence paragraph.
- Add one more human texture element: a casual aside, a
  rhetorical question, a thought that trails off just slightly.
- Every sentence that still sounds "AI-clean" needs to be rebuilt.
` : "";

  return `You are a semantic reconstruction engine. Your job is to read a piece of text, fully absorb its meaning, then rebuild it completely from scratch in one specific human's exact voice — so that it passes GPTZero, ZeroGPT, and Originality.ai with a human score.

You do NOT paraphrase. You do NOT edit. You READ, ABSORB, then WRITE FRESH.

Output ONLY the reconstructed text. No intro. No label. No "here is the rewrite". Nothing. The very first token of your response must be the first word of the text.

${dnaBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — READ AND ABSORB (do not skip this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the text fully. Identify every idea, argument, fact, and
point in it. Hold all of that meaning in your working memory.
Now set the original text aside completely — you will not look
at it again. You will write entirely from the meaning you absorbed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — INTERNALIZE THIS PERSON'S VOICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Study the fingerprint above. Before writing, answer internally:
- How long are their sentences on average, and how much do they vary?
- What is their tone — casual, confident, warm, dry, direct?
- How do they structure ideas — point first, or building toward it?
- What punctuation habits are uniquely theirs?
- What vocabulary level — simple and direct, or precise?
- What would sound completely wrong from this person?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — WRITE FRESH, DEFEAT THE DETECTORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now write the text from scratch, using the meaning you absorbed
and this person's voice fingerprint. As you write, actively
target the exact signals that GPTZero and ZeroGPT measure:

── PERPLEXITY (word-level unpredictability) ──────────────────
AI always picks the statistically safest word. Detectors know
this. For every key word you write, ask: is this the most
predictable choice? If yes, reach for what this person would
actually say instead — not a bizarre word, just the natural,
slightly unexpected one a real person grabs without thinking.
  "significant" → "real" / "actual" / "noticeable"
  "demonstrate" → "show" / "prove" / "make clear"
  "obtain" → "get" / "grab" / "end up with"
  "utilize" → "use" / "work with"
  "individuals" → "people"
  "it is evident" → "clearly" / "you can tell" / "obviously"
  "therefore" → "so" / "which means" / "that's why"
Do this throughout the entire text — every paragraph must have
word choices that a language model would find surprising.

── BURSTINESS (sentence length chaos) ────────────────────────
This is the single biggest detector signal. AI produces
sentences of eerily similar length — typically 15-22 words,
paragraph after paragraph. Real humans do not.

In EVERY paragraph you write, you must have:
  • At least one sentence under 8 words (can be as short as 2-3)
  • At least one sentence over 28 words
  • The other sentences can be anything in between

The variation must be real and aggressive, not cosmetic.
Bad (AI-typical): "This approach has several advantages. It allows for greater flexibility in implementation. The results tend to be more consistent over time."
Good (human burstiness): "This works. And the reason it works is actually pretty straightforward — when you give people more flexibility in how they implement something, they stop fighting the process and start adapting it, which means the results are more consistent not because you forced them to be, but because the people doing the work actually believe in what they're doing."

── STRUCTURAL UNPREDICTABILITY ───────────────────────────────
AI follows Topic Sentence → Supporting Points → Conclusion.
Every time. Detectors recognize this structure instantly.

Break it. Use the patterns this person actually uses from their
fingerprint. Options that score as human:
  • Start with a question, then answer it
  • Open with the conclusion, then explain why
  • Begin mid-thought, as if continuing a conversation
  • Use a one-sentence paragraph for emphasis
  • Let a paragraph end without wrapping up neatly
  • Start a sentence with "But" / "And" / "So" / "Because"

── BANNED PHRASES (instant AI flags — never use any of these) ──
Furthermore, Moreover, Additionally, In addition, In conclusion,
To summarize, To elaborate, In summary, It is worth noting,
It is important to note, It's crucial to understand, Notably,
Significantly, Subsequently, Consequently, In other words,
As previously mentioned, Having said that, With that being said,
It goes without saying, Needless to say, Last but not least,
First and foremost, At the end of the day, All things considered,
Upon reflection, This highlights, This demonstrates, This suggests,
One might argue, It could be said, To some extent, Broadly speaking,
It is generally accepted, This underscores, It is clear that,
This illustrates, Plays a crucial role, It is essential to,
It is important to, When it comes to, In terms of, In light of

── ACTIVE VOICE ───────────────────────────────────────────────
Convert passive voice to active wherever natural:
  "It was found that X" → "X showed" / "turns out X"
  "This can be seen in" → "You see this in"
  "It has been shown that" → "Research shows" / "we know"

── ZERO HEDGING UNLESS THEY HEDGE ────────────────────────────
AI overloads text with hedges detectors recognize immediately.
Unless this person's fingerprint shows high hedging, avoid:
"one might argue", "it could be said", "arguably",
"in many ways", "to some extent", "it appears that"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NON-NEGOTIABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Preserve EVERY idea, fact, argument, and piece of information
   from the original — nothing added, nothing removed
2. Preserve the full intellectual depth — do not simplify ideas
   or water down arguments under any circumstances
3. Match this person's exact confidence and energy level
4. No grammatical errors — natural imperfections (fragments,
   run-ons that feel intentional) are fine; actual mistakes are not
5. Output ONLY the text — zero intro, zero label, zero preamble
6. First word of your response = first word of the text
${pushHarderBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITATIVE VOICE BLUEPRINT (from deep analysis of their samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${rawBlueprint || "No qualitative blueprint — rely on the fingerprint above."}`;
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
    const userPrompt = `STEP 1 — READ AND ABSORB:
Read every word of the text below. Understand every idea, argument, fact, and point it makes. Hold all of that meaning in your head.

STEP 2 — SET IT ASIDE:
Now forget the original sentences completely. You are not going to edit them, paraphrase them, or reorder them. You are going to write something entirely new that conveys the same meaning.

STEP 3 — WRITE FROM SCRATCH IN THIS PERSON'S VOICE:
Write the text fresh, as if you just thought of these ideas yourself and are explaining them in this person's natural voice. Every sentence you write must be brand new — not a variation of an original sentence, a genuinely new sentence that carries the same meaning.

As you write, actively check each paragraph:
✓ Does it have at least one very short sentence (under 8 words)?
✓ Does it have at least one longer sentence (over 28 words)?
✓ Does it contain any banned AI phrases? (Furthermore, Moreover, Additionally, In conclusion, It is worth noting, etc.) → remove them all
✓ Are the word choices what this person would naturally say, or are they suspiciously "correct"? → make them more natural
✓ Does the structure follow Topic→Points→Conclusion? → break it

Output ONLY the reconstructed text. Zero intro. Zero label. The very first word of your response must be the first word of the text itself.

Text to absorb and reconstruct:\n\n${text}`;

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
    const userPrompt = `This text was reconstructed but is still being flagged by AI detectors. Do a full second-pass reconstruction — not a light edit, a complete rebuild.

Read it. Absorb every idea in it. Then write it entirely from scratch again, this time pushing harder on every human signal:

Go through each paragraph and ask:
1. BURSTINESS — is there a sentence under 6 words? Is there one over 30? If not, restructure until there is.
2. WORD CHOICES — find every word that feels like the "safe AI pick" and replace it with what a real person would actually say
3. STRUCTURE — does any paragraph follow the predictable AI pattern of Topic→Evidence→Wrap-up? Break it completely.
4. BANNED PHRASES — scan for: Furthermore, Moreover, Additionally, In conclusion, It is worth noting, Having said that, With that being said, It goes without saying, This highlights, This demonstrates, This suggests, Plays a crucial role. Replace every single one.
5. ONE NEW HUMAN ELEMENT — add something that wasn't in the previous version: a one-sentence paragraph, a rhetorical question, a thought that starts with "And" or "But", a brief aside in parentheses.

Preserve every idea and fact. Preserve full depth and sophistication.
Output ONLY the reconstructed text. First word of response = first word of text.

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

