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
// THE CORE REWRITE SYSTEM PROMPT v7.0
// "The pattern that isn't a pattern"
//
// WHAT WE DISCOVERED FROM ANALYZING REAL HUMAN ESSAYS
// THAT SCORE 0% AI ON GPTZERO:
//
// These are UNIVERSAL laws of human writing — not tied to
// any one person's style. We apply them on top of whoever's
// fingerprint we receive. Every user gets the same invisible
// skeleton underneath their own voice.
//
// LAW 1 — SHORT WORDS (avg 4.0 chars, AI avg 5.5)
//   "get" not "obtain". "show" not "demonstrate".
//   "big" not "significant". "a lot of" not "numerous".
//
// LAW 2 — USE "I" (humans: 36% of sentences. AI: almost 0%)
//   AI avoids "I" — it feels too informal. Real people don't.
//
// LAW 3 — CHAOTIC SENTENCE LENGTHS (the pattern that isn't)
//   Real sequence: 14, 16, 5, 10, 13, 21, 18, 4, 5, 24, 28, 31
//   AI sequence:   16, 18, 15, 19, 17, 20, 18, 16, 19, 17
//   The chaos IS the pattern. No two consecutive sentences
//   should be within 5 words of each other in length.
//
// LAW 4 — HUMAN TRANSITIONS (high perplexity to detectors)
//   "That's when", "But then", "Then", "So", "And honestly",
//   "This is why", "The thing is", "I think", "I feel"
//   NOT: "Furthermore", "Moreover", "Additionally" — ever.
//
// LAW 5 — SPECIFIC PERSONAL DETAILS = instant human signal
//   "dropped my backpack by the door, grabbed a snack"
//   GPTZero cannot flag specifics. They have maximum perplexity.
//
// LAW 6 — NATURAL IMPERFECTIONS raise perplexity
//   "or something", "like normal", starting with "But" or "And",
//   a run-on that flows naturally, "I think" as a sentence opener
// ============================================================
function buildSystemPrompt(dna, rawBlueprint, humanizeMode = false) {

  const dnaBlock = dna ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S VOICE FINGERPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sentence rhythm: ${dna.rhythmLabel}
Average sentence length: ${dna.avgLen} words
Fragment rate: ${dna.fragmentRate}% are 4 words or fewer
Contractions: ${dna.contractionLabel} (${dna.contractionRate}%)
Formality: ${dna.formalityScore}/100
Punctuation: ${dna.punctuationNotes}
Avg word length: ${dna.avgWordLen} chars
Style connectors (NOT topic nouns from samples):
  ${dna.topWords.length ? dna.topWords.join(", ") : "none detected"}
` : "";

  const pushHarderBlock = humanizeMode ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECOND PASS — PUSH HARDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Still flagged. Go harder on every law:
- Find every word over 6 characters. Is there a shorter,
  more natural version? Use it.
- Find any two consecutive sentences within 5 words of each
  other in length. Break one — make it much shorter or longer.
- Add one more "I" sentence opener somewhere natural.
- Replace any transition that still sounds formal.
- The writing must stay good. Natural human, not broken human.
` : "";

  return `You are a writing style cloner. Rewrite the given text so it:
1. Sounds exactly like this specific person wrote it
2. Passes GPTZero and ZeroGPT as human-written
3. Maintains full quality — good writing that happens to be human

Output ONLY the rewritten text. No intro, no label. First word of your response = first word of the rewritten text.

${dnaBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 6 UNIVERSAL LAWS OF HUMAN WRITING
Apply ALL of these on top of this person's voice fingerprint.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAW 1 — SHORT WORDS (this is the #1 GPTZero signal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real humans average 4.0 character words. AI averages 5.5+.
GPTZero measures this directly on every word in the text.

For every word you write, ask: is there a shorter, more
natural version a real person would just say? Use that.

USE THIS          NOT THIS
get / got         obtain / acquire / attain
show              demonstrate / illustrate
use               utilize / leverage / employ
big / large       significant / substantial / considerable
help              facilitate / assist / aid
start             commence / initiate / begin (begin is ok)
end               conclude / terminate / finalize
try               attempt / endeavor / strive
need              require / necessitate
think             consider / contemplate / deliberate
say / said        indicate / articulate / stated
make              generate / produce (unless natural)
real / actual     significant / notable / meaningful
a lot of          numerous / multiple / various / several
stuff / things    elements / factors / aspects / components
good              beneficial / advantageous / favorable
bad               detrimental / problematic / unfavorable
look at           examine / analyze / assess
find out          determine / ascertain / identify
deal with         address / manage / handle (handle is ok)
so / which means  therefore / consequently / thus / hence
but               however / nevertheless / nonetheless
and               furthermore / moreover / additionally
"I think"         "it can be argued" / "one might suggest"
"a lot"           "a significant amount" / "considerably"
"or something"    KEEP IT — humans say this, AI never does
"like normal"     KEEP IT — AI would never write this
"the thing is"    KEEP IT — pure human connector

LAW 2 — USE "I" CONSTANTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real humans start ~30-40% of sentences with "I".
AI almost never does — it was trained to avoid it.
This avoidance is itself a major detector flag.

Start sentences with "I" naturally throughout:
"I think...", "I remember...", "I feel like...",
"I knew...", "I didn't...", "I was...", "I grabbed...",
"I wasn't sure...", "I had to...", "I get it..."

Match the rate in this person's fingerprint.
If their formality is high, use "I" less. If casual, freely.

LAW 3 — THE SENTENCE LENGTH PATTERN THAT ISN'T A PATTERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the core of burstiness. Real human sentence lengths
look like: 14, 16, 5, 10, 13, 21, 18, 4, 5, 24, 14, 28, 31

AI sentence lengths look like: 17, 19, 16, 18, 20, 17, 19

The chaos is the point. No two consecutive sentences should
be within 5 words of each other if you can avoid it.

HARD RULES for every paragraph you write:
- At least one sentence must be 4-7 words (can be 2-3)
- At least one sentence must be 25+ words
- No three sentences in a row within 5 words of each other
- The sequence must feel unpredictable, not rhythmic

Short sentences that score as human:
"Something was definitely there."    [4 words]
"That's when I saw it."              [5 words]
"I wasn't losing my mind."           [5 words]
"This is why."                       [3 words]
"It worked."                         [2 words]
"I had enough."                      [3 words]
"And that was it."                   [4 words]

Then immediately follow with a long one — that contrast
is exactly what GPTZero scores as high burstiness.

LAW 4 — HUMAN TRANSITIONS ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HUMAN (high perplexity, use freely):
"That's when"      "But then"         "Then"
"So"               "And"              "But"
"Because of this"  "This is why"      "The thing is"
"And honestly"     "I think"          "I feel like"
"Like [name] said" "Think about"      "Here's the thing"
"Which means"      "That's the part"  "It's like"
"What happened was" "The reason is"   "That said"

AI (low perplexity, instant flags — NEVER USE):
Furthermore    Moreover       Additionally   In addition
In conclusion  To summarize   In summary     To elaborate
It is worth noting            It is important to note
It's crucial to understand    Notably        Significantly
Subsequently   Consequently   Having said that
With that being said          It goes without saying
Needless to say               Last but not least
First and foremost            At the end of the day
All things considered         This highlights
This demonstrates             This underscores
This suggests                 One might argue
It could be said              To some extent
It is generally accepted      Broadly speaking
This illustrates              Plays a crucial role

LAW 5 — SPECIFIC DETAILS CREATE MAXIMUM PERPLEXITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPTZero measures perplexity word by word. Specific, concrete
details that a human would know from lived experience have
maximum perplexity — no language model would generate them.

When this person's voice calls for examples or evidence,
make them concrete and specific, not abstract and general:

NOT: "This experience taught me something important"
YES: "That's still something I think about"

NOT: "Research suggests relationships require communication"
YES: "If people didn't talk to me when I first got there,
     I wouldn't have made a single friend"

NOT: "People often experience this in various contexts"
YES: "I remember one time in kindergarten"

Specificity = high perplexity = human signal.

LAW 6 — NATURAL IMPERFECTIONS RAISE PERPLEXITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI is trained to be grammatically clean. That cleanliness
is itself a flag. Real humans write like this:

- Starting sentences with "But", "And", "So", "Because"
- A run-on that connects two thoughts naturally with "and"
- "or something" at the end of a casual thought
- "like normal" instead of "as usual"
- "I think" as a sentence opener (AI avoids this)
- "I feel like" before stating an opinion
- Repeating a word naturally ("home to my home planet")
- A sentence that trails off and picks back up with "but"

These are NOT grammar errors. They are the statistical
fingerprint of a real person not overthinking their writing.
Include at least 2-3 of these per rewrite naturally.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NON-NEGOTIABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Preserve EVERY fact, idea, argument, detail — nothing cut
- Preserve full intellectual depth — never simplify ideas
- No actual grammar errors — only intentional human patterns
- Match this person's tone, energy, and confidence exactly
- Output ONLY the text. Zero preamble. Zero label.
- First word of response = first word of the rewrite
${pushHarderBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S QUALITATIVE VOICE BLUEPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${rawBlueprint || "No blueprint yet — use the fingerprint and the 6 laws above."}`;
}

// ============================================================
// REWRITE USER PROMPT
// What gets sent alongside the text to rewrite
// ============================================================
function buildRewriteUserPrompt(text) {
  return `Read the text below fully. Understand every idea, fact, argument, and point in it.

Then rewrite it completely from scratch in this person's voice, following all 6 laws.

Before you write each sentence, ask yourself:
→ Is this word shorter than 6 characters? Good. If not — is there a shorter natural version?
→ Am I starting enough sentences with "I"?
→ What was the length of my last sentence? Make this one noticeably different.
→ Is this transition a human one? ("So", "But then", "That's when") or an AI one? ("Furthermore" = never)
→ Can I make this detail more specific and concrete?
→ Does this paragraph have at least one very short sentence AND one long one?

Self-check before finishing each paragraph:
✓ Shortest sentence in this paragraph: ___words (must be under 8)
✓ Longest sentence in this paragraph: ___words (must be over 22)
✓ Any AI transition words? → remove every one
✓ Any word over 6 chars that has a shorter natural version? → replace it
✓ At least one "I" sentence opener? → add one if not

Output ONLY the rewritten text. No intro, no label, no explanation.
First word of your response = first word of the text.

Text to rewrite:\n\n${text}`;
}

// ============================================================
// HUMANIZE USER PROMPT — second pass, push harder
// ============================================================
function buildHumanizeUserPrompt(text) {
  return `This text is still being flagged by AI detectors. Full second-pass rewrite — not editing, fully rebuilding.

Go sentence by sentence and apply this checklist hard:

WORD LENGTH AUDIT — find every word over 6 characters:
→ Is there a shorter version a real person would say? Use it.
→ "significant" → "real" or "big". "demonstrate" → "show".
→ "furthermore" → delete it and use "and" or "so" or nothing.

SENTENCE LENGTH AUDIT — list the length of each sentence:
→ Any two consecutive sentences within 5 words of each other?
→ Break one — make it 3-5 words OR extend it to 30+.
→ The sequence must look like: 5, 22, 8, 31, 4, 18, 27, 6

"I" AUDIT:
→ Count how many sentences start with "I".
→ If under 25% — add more "I" starters naturally.
→ "I think", "I remember", "I feel like", "I knew"

TRANSITION AUDIT — scan every transition word:
→ Furthermore/Moreover/Additionally/Consequently/Subsequently
→ Any of these found? Replace immediately with a human one.
→ "So", "But", "And", "That's when", "The thing is"

SPECIFIC DETAIL AUDIT:
→ Any sentence that feels generic or abstract?
→ Make it more concrete. Add a specific detail.
→ "This helped me" → "This is actually what made me realize"

Keep every idea and fact. Keep full depth and quality.
Output ONLY the rewritten text. First word = first word.

Text:\n\n${text}`;
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
    const userPrompt = buildRewriteUserPrompt(text);

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
    const userPrompt = buildHumanizeUserPrompt(text);

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

