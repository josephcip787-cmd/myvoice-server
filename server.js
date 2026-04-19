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
// GROQ API CALL
// temperature: 0.72 for analysis (consistent, factual)
//              1.0  for rewrites (creative, varied, unpredictable)
// Higher temperature = more varied word choices = higher perplexity
// = looks more human to detectors. 0.72 was making every rewrite
// sound the same — safe, uniform, detectable.
// ============================================================
async function callGroq(systemPrompt, userPrompt, maxTokens = 2000, temperature = 0.72) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: MODEL,
      max_tokens: maxTokens,
      temperature: temperature,
      top_p: 0.95,
      frequency_penalty: 0.3,
      presence_penalty: 0.3,
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
// DEEP STYLE DNA — extracts everything GPTZero actually measures
// Goes beyond sentence length into syntax variety, creativity
// markers, emotional warmth, argument structure, literary devices
// ============================================================
function extractStyleDNA(samples) {
  const allText = samples.map(s => s.text).join(" ");
  const sentences = allText.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 5);
  const words = allText.toLowerCase().match(/\b[a-z']+\b/g) || [];
  const sentenceCount = Math.max(sentences.length, 1);
  const wordCount = Math.max(words.length, 1);

  // ---- Sentence length distribution ----
  const lengths = sentences.map(s => s.split(/\s+/).length);
  const avgLen = Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length);
  const minLen = Math.min(...lengths);
  const maxLen = Math.max(...lengths);
  const shortCount = lengths.filter(l => l <= 8).length;
  const longCount = lengths.filter(l => l >= 25).length;
  const medCount = lengths.filter(l => l > 8 && l < 25).length;
  const variance = Math.round(lengths.reduce((acc, l) => acc + Math.pow(l - avgLen, 2), 0) / lengths.length);
  const rhythmLabel =
    shortCount > longCount * 2 ? "punchy — mostly short sentences under 10 words" :
    longCount > shortCount * 2 ? "flowing — often long sentences over 25 words" :
    variance > 40 ? "wildly varied — unpredictable mix of very short and very long" :
    "balanced — mixes short and medium sentences";

  // ---- Syntax variety (what GPTZero calls "Monotonous Syntax") ----
  const startsWithI = sentences.filter(s => /^I\s/i.test(s)).length;
  const startsWithBut = sentences.filter(s => /^(But|And|So|Because|Yet|Or)\s/i.test(s)).length;
  const startsWithTime = sentences.filter(s => /^(When|After|Before|While|Once|As|Then)\s/i.test(s)).length;
  const startsWithIt = sentences.filter(s => /^(It|That|This|There)\s/i.test(s)).length;
  const startsWithVerb = sentences.filter(s => /^(Getting|Having|Doing|Being|Taking|Making|Looking|Going)\s/i.test(s)).length;
  const declarativeCount = sentences.filter(s => /\.$/.test(s.trim())).length;
  const questionCount = sentences.filter(s => /\?$/.test(s.trim())).length;
  const exclamCount = sentences.filter(s => /!$/.test(s.trim())).length;

  // ---- Rhythm patterns (GPTZero: "Predictable Rhythm") ----
  // Detect if writer uses same length repeatedly
  let rhythmRepeat = 0;
  for (let i = 1; i < lengths.length; i++) {
    if (Math.abs(lengths[i] - lengths[i-1]) <= 3) rhythmRepeat++;
  }
  const rhythmRepeatRate = Math.round((rhythmRepeat / sentenceCount) * 100);

  // ---- Creativity markers (GPTZero: "Lacks Creativity") ----
  const metaphors = (allText.match(/\b(like|as if|as though|feels like|seems like|reminds me|kind of like|sort of like)\b/gi) || []).length;
  const personalMemories = (allText.match(/\b(I remember|I recall|back when|one time|I used to|growing up|when I was|I once)\b/gi) || []).length;
  const selfCorrections = (allText.match(/\b(I mean|well|actually|honestly|or rather|you know|I guess|kind of|sort of)\b/gi) || []).length;
  const rhetoricalQs = questionCount;
  const tangents = (allText.match(/\b(by the way|speaking of|that reminds me|which is funny|the weird thing|the thing is)\b/gi) || []).length;
  const opinions = (allText.match(/\b(I think|I feel|I believe|I don't know|I wonder|I get|in my opinion|to me|for me)\b/gi) || []).length;

  // ---- Emotional warmth (GPTZero: "Detached Warmth") ----
  const warmthWords = (allText.match(/\b(love|hate|excited|nervous|scared|happy|sad|frustrated|proud|worried|amazed|honestly|actually|really|feel|felt|emotion|heart|care|matter)\b/gi) || []).length;
  const personalPronouns = (allText.match(/\b(I|me|my|mine|myself|we|us|our)\b/gi) || []).length;
  const warmthScore = Math.round(((warmthWords + personalPronouns) / wordCount) * 100);

  // ---- Vocabulary profile ----
  const uniqueWords = new Set(words);
  const vocabRichness = Math.round((uniqueWords.size / wordCount) * 100);
  const avgWordLen = Math.round(words.reduce((a, w) => a + w.length, 0) / wordCount);
  const longWords = words.filter(w => w.length > 8).length;
  const longWordRate = Math.round((longWords / wordCount) * 100);

  // ---- Contractions ----
  const contractionMatches = (allText.match(/\b(don't|doesn't|can't|won't|it's|i'm|i've|i'd|i'll|we're|we've|they're|you're|you've|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|wouldn't|shouldn't|couldn't|that's|there's|here's|let's|who's|what's|didn't|couldn't|she's|he's)\b/gi) || []).length;
  const contractionRate = Math.round((contractionMatches / wordCount) * 100);
  const contractionLabel =
    contractionRate > 8 ? "heavy — writes exactly how they talk" :
    contractionRate > 3 ? "moderate — relaxed and semi-casual" :
    "rare — tends toward more formal phrasing";

  // ---- Punctuation fingerprint ----
  const exclamations = (allText.match(/!/g) || []).length;
  const questions = (allText.match(/\?/g) || []).length;
  const ellipses = (allText.match(/\.\.\./g) || []).length;
  const dashes = (allText.match(/[—\-]{1,2}/g) || []).length;
  const commas = (allText.match(/,/g) || []).length;
  const semicolons = (allText.match(/;/g) || []).length;
  const punctuationNotes = [
    exclamations / sentenceCount > 0.3 ? "uses exclamation marks freely" : null,
    ellipses / sentenceCount > 0.15 ? "uses ellipses — thoughts trail off naturally" : null,
    dashes / sentenceCount > 0.3 ? "uses dashes for asides and interruptions" : null,
    questions / sentenceCount > 0.15 ? "asks questions — rhetorical and real" : null,
    commas / sentenceCount > 2.5 ? "comma-heavy — stacks clauses and lists" : null,
    semicolons > 3 ? "uses semicolons to connect related thoughts" : null,
  ].filter(Boolean).join("; ") || "clean punctuation — no dramatic habits";

  // ---- Formality ----
  const formalWords = (allText.match(/\b(therefore|moreover|furthermore|subsequently|nevertheless|accordingly|consequently|albeit|notwithstanding|utilize|facilitate|implement|leverage|regarding|pertaining|demonstrate|indicate|significant|substantial|numerous|considerable)\b/gi) || []).length;
  const casualWords = (allText.match(/\b(kinda|sorta|gonna|wanna|gotta|tbh|ngl|lowkey|literally|basically|honestly|like|just|really|super|totally|pretty|stuff|yeah|yep|nope|okay|cool|crazy|huge|wild|nuts)\b/gi) || []).length;
  let formalityScore = 50 + (formalWords * 5) - (casualWords * 3) - (contractionRate * 2);
  formalityScore = Math.max(0, Math.min(100, formalityScore));

  // ---- Argument/structure style ----
  const usesExamples = (allText.match(/\b(for example|for instance|like when|such as|one time|I remember when)\b/gi) || []).length;
  const frontLoads = sentences.filter((s, i) => {
    if (i === 0) return false;
    const prev = sentences[i-1];
    return prev && prev.split(/\s+/).length < 10 && s.split(/\s+/).length > 20;
  }).length;
  const buildsToPoint = sentences.filter((s, i) => {
    if (i < 2) return false;
    return /\b(so|which means|that's why|this is why|because of this|and that's)\b/i.test(s);
  }).length;

  // ---- Literary devices ----
  const similes = (allText.match(/\b(like a|like an|like the|as a|as if|as though)\b/gi) || []).length;
  const repetition = (allText.match(/\b(\w+)\s+\1\b/gi) || []).length;
  const directAddress = (allText.match(/\b(you|your|yourself)\b/gi) || []).length;

  // ---- Signature style words (cross-sample only) ----
  const stopwords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","shall","can","i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","what","which","who","when","where","why","how","all","each","every","not","no","so","if","as","by","from","up","about","than","then","just","also","there","here","very","more","some","any","only","out","into","get","got","go","going","like","one","know","think","feel","time","way","make","take","come","see","look","want","give","use","find","tell","work","call","try","ask","need","seem","leave","show","keep","let","begin","long","never","always","often","back","still","around","even","well","new","good","old","right","big","high","different","small","large","next","early","young","important","few","public","bad","same","able"]);

  const sampleWordSets = samples.map(s => new Set((s.text.toLowerCase().match(/\b[a-z']+\b/g) || [])));
  const wordFreq = {};
  words.forEach(w => { if (!stopwords.has(w) && w.length > 3) wordFreq[w] = (wordFreq[w] || 0) + 1; });
  const minSamples = samples.length > 1 ? 2 : 1;
  const topWords = Object.entries(wordFreq)
    .filter(([w]) => sampleWordSets.filter(s => s.has(w)).length >= minSamples)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([w]) => w);

  // ---- Recurring phrase patterns ----
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
    // Sentence rhythm
    avgLen, minLen, maxLen, variance, rhythmLabel,
    shortCount, medCount, longCount, sentenceCount,
    rhythmRepeatRate,
    // Syntax variety
    startsWithIRate: Math.round((startsWithI / sentenceCount) * 100),
    startsWithConjRate: Math.round((startsWithBut / sentenceCount) * 100),
    startsWithTimeRate: Math.round((startsWithTime / sentenceCount) * 100),
    questionRate: Math.round((questionCount / sentenceCount) * 100),
    exclamRate: Math.round((exclamCount / sentenceCount) * 100),
    // Creativity markers
    metaphorRate: Math.round((metaphors / sentenceCount) * 100),
    personalMemoryCount: personalMemories,
    selfCorrectionCount: selfCorrections,
    tangentCount: tangents,
    opinionCount: opinions,
    hasLiteraryDevices: similes > 2 || repetition > 1,
    directAddressRate: Math.round((directAddress / wordCount) * 100),
    // Warmth & personality
    warmthScore, contractionRate, contractionLabel,
    // Vocabulary
    vocabRichness, avgWordLen, longWordRate,
    // Punctuation
    punctuationNotes,
    exclamationsPerSentence: Math.round((exclamations / sentenceCount) * 10) / 10,
    ellipsesPerSentence: Math.round((ellipses / sentenceCount) * 10) / 10,
    dashesPerSentence: Math.round((dashes / sentenceCount) * 10) / 10,
    // Formality & structure
    formalityScore,
    usesExamples: usesExamples > 1,
    buildsToPoint: buildsToPoint > 1,
    // Signature patterns
    topWords, topPhrases,
    // Fragment rate
    fragmentRate: Math.round((sentences.filter(s => s.split(/\s+/).length <= 4).length / sentenceCount) * 100),
  };
}

// ============================================================
// STEP 1 — VOICE PROMPT
// Groq's ONLY job here is to sound like this person.
// Zero detector rules. Zero banned phrase lists.
// Just: read their samples, clone their voice, write the text.
// The post-processor handles detector signals after.
// ============================================================
function buildSystemPrompt(dna, rawBlueprint) {

  const dnaBlock = dna ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S WRITING FINGERPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sentence rhythm: ${dna.rhythmLabel}
Average length: ${dna.avgLen} words | Range: ${dna.minLen}–${dna.maxLen}
Short sentences (≤8w): ${dna.shortCount} | Long (25+w): ${dna.longCount}
Starts with "I": ${dna.startsWithIRate}% | Conjunctions: ${dna.startsWithConjRate}%
Questions: ${dna.questionRate}% of sentences
Personal memories: ${dna.personalMemoryCount} | Opinions: ${dna.opinionCount}
Contractions: ${dna.contractionLabel} (${dna.contractionRate}%)
Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100
Avg word length: ${dna.avgWordLen} chars
Punctuation: ${dna.punctuationNotes}
Signature style words: ${dna.topWords.length ? dna.topWords.join(", ") : "none"}
` : "";

  return `You are a writing style cloner. Your ONLY job is to rewrite the given text so it sounds exactly like this specific person wrote it.

Do NOT think about AI detectors. Do NOT follow rules about sentence length. Do NOT add or remove personality artificially. Just write the way this person writes — their rhythm, their tone, their vocabulary, their way of thinking through ideas.

Output ONLY the rewritten text. No intro, no label. First word of response = first word of the rewritten text.

${dnaBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO CLONE THIS PERSON'S VOICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing, study the blueprint below and answer:
- What is their tone? Casual, formal, direct, warm, dry?
- How do they structure ideas — point first or build to it?
- What words do they actually use? What would they never say?
- How long are their sentences, and how do they vary?
- What makes their voice feel like THEM specifically?

Then write as if you ARE that person explaining these ideas.
Match their rhythm. Match their vocabulary level. Match how
they connect one thought to the next. Preserve every fact
and argument from the original — just say it their way.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S VOICE BLUEPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${rawBlueprint || "No blueprint yet — use the fingerprint above."}`;
}

function buildRewriteUserPrompt(text) {
  return `Read the text below. Understand every idea and argument in it.

Now rewrite it completely in this person's voice — the way THEY would explain these exact ideas. Every fact and point stays. The way it's said changes to match their voice.

Output ONLY the rewritten text. First word of response = first word of text.

Text:\n\n${text}`;
}

function buildHumanizeUserPrompt(text) {
  return `This is already written in this person's voice. Your job is to make it sound MORE like them — more natural, less polished, more the way they actually talk and think.

Find any sentence that still sounds slightly formal or generic and make it more personal and specific. Find any place where the rhythm feels too smooth and break it up. Add a bit more of their natural imperfections — the way they actually write, not the way a textbook says to write.

Keep every idea. Keep every fact. Just make it more authentically them.

Output ONLY the result. First word of response = first word of text.

Text:\n\n${text}`;
}

// ============================================================
// STEP 2 — POST-PROCESSOR
// Pure code. Runs after Groq returns the voice rewrite.
// Makes targeted surgical edits to fix GPTZero signals
// WITHOUT touching the voice or meaning.
//
// This is deterministic — same input, same fixes, every time.
// No prompting. No guessing. Just math.
// ============================================================
function postProcess(text, dna) {
  let result = text;

  // ---- 1. STRIP BANNED AI PHRASES ----
  // These are instant flags regardless of everything else
  const bannedPhrases = [
    [/\bFurthermore,?\s*/gi, ""],
    [/\bMoreover,?\s*/gi, ""],
    [/\bAdditionally,?\s*/gi, ""],
    [/\bIn addition,?\s*/gi, ""],
    [/\bIn conclusion,?\s*/gi, ""],
    [/\bTo summarize,?\s*/gi, ""],
    [/\bIn summary,?\s*/gi, ""],
    [/\bNotably,?\s*/gi, ""],
    [/\bSignificantly,?\s*/gi, ""],
    [/\bSubsequently,?\s*/gi, ""],
    [/\bConsequently,?\s*/gi, "So "],
    [/\bHaving said that,?\s*/gi, "But "],
    [/\bWith that being said,?\s*/gi, "That said, "],
    [/\bIt goes without saying( that)?,?\s*/gi, ""],
    [/\bNeedless to say,?\s*/gi, ""],
    [/\bIt is worth noting( that)?,?\s*/gi, ""],
    [/\bIt is important to note( that)?,?\s*/gi, ""],
    [/\bIt('s| is) crucial to( understand)?,?\s*/gi, ""],
    [/\bThis highlights\b/gi, "This shows"],
    [/\bThis demonstrates\b/gi, "This shows"],
    [/\bThis underscores\b/gi, "This shows"],
    [/\bThis illustrates\b/gi, "This shows"],
    [/\bThis suggests\b/gi, "This means"],
    [/\bone might argue\b/gi, "some people think"],
    [/\bit could be argued\b/gi, "you could say"],
    [/\bplays a crucial role\b/gi, "matters a lot"],
    [/\bplays an important role\b/gi, "matters"],
    [/\bAt the end of the day,?\s*/gi, ""],
    [/\bAll things considered,?\s*/gi, ""],
    [/\bFirst and foremost,?\s*/gi, "First, "],
    [/\bLast but not least,?\s*/gi, "And "],
    [/\bis more than just\b/gi, "is not just"],
    [/\bThat's when I realized\b/gi, "That's when I got it —"],
  ];

  for (const [pattern, replacement] of bannedPhrases) {
    result = result.replace(pattern, replacement);
  }

  // ---- 2. WORD SUBSTITUTIONS ----
  // Replace long AI words with short human ones
  // Only swaps whole words, preserves capitalization
  const wordSwaps = [
    ["utilize", "use"], ["utilizes", "uses"], ["utilized", "used"],
    ["leverage", "use"], ["leverages", "uses"], ["leveraged", "used"],
    ["facilitate", "help"], ["facilitates", "helps"], ["facilitated", "helped"],
    ["demonstrate", "show"], ["demonstrates", "shows"], ["demonstrated", "showed"],
    ["obtain", "get"], ["obtains", "gets"], ["obtained", "got"],
    ["acquire", "get"], ["acquires", "gets"], ["acquired", "got"],
    ["commence", "start"], ["commences", "starts"], ["commenced", "started"],
    ["endeavor", "try"], ["endeavors", "tries"], ["endeavored", "tried"],
    ["individuals", "people"], ["individual", "person"],
    ["implement", "use"], ["implements", "uses"], ["implemented", "used"],
    ["significant", "real"], ["significant", "big"],
    ["substantial", "big"], ["considerable", "big"],
    ["numerous", "many"], ["multiple", "many"],
    ["assist", "help"], ["assists", "helps"], ["assisted", "helped"],
    ["require", "need"], ["requires", "needs"], ["required", "needed"],
    ["purchase", "buy"], ["purchases", "buys"], ["purchased", "bought"],
    ["attempt", "try"], ["attempts", "tries"], ["attempted", "tried"],
    ["pertaining to", "about"],
    ["regarding", "about"],
    ["in order to", "to"],
    ["due to the fact that", "because"],
    ["in the event that", "if"],
  ];

  for (const [ai, human] of wordSwaps) {
    const regex = new RegExp(`\\b${ai}\\b`, "gi");
    result = result.replace(regex, (match) => {
      // Preserve capitalization
      if (match[0] === match[0].toUpperCase()) {
        return human.charAt(0).toUpperCase() + human.slice(1);
      }
      return human;
    });
  }

  // ---- 3. BURSTINESS FIX ----
  // Find consecutive sentences of similar length and break the pattern
  // Split into sentences, check lengths, insert short punchy sentences
  // where rhythm is too uniform
  const sentenceRegex = /([^.!?]+[.!?]+)/g;
  const sentences = result.match(sentenceRegex) || [result];

  if (sentences.length >= 3) {
    const fixed = [];
    for (let i = 0; i < sentences.length; i++) {
      fixed.push(sentences[i]);

      // Check if 3 consecutive sentences are all similar length
      if (i >= 2) {
        const lens = [
          sentences[i-2].trim().split(/\s+/).length,
          sentences[i-1].trim().split(/\s+/).length,
          sentences[i].trim().split(/\s+/).length,
        ];
        const allSimilar = Math.max(...lens) - Math.min(...lens) < 6;
        const allMedium = lens.every(l => l >= 10 && l <= 22);

        // If 3 in a row are all medium length, flag — post-processor
        // can't add content, but we track this for the humanize pass
        if (allSimilar && allMedium) {
          // Mark this location — the humanize pass will target it
          // We can't add sentences without knowing the context,
          // so we leave a soft signal in the output for now
        }
      }
    }
  }

  // ---- 4. CLEAN UP ----
  // Fix any double spaces or weird artifacts from replacements
  result = result
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,!?])/g, "$1")
    .replace(/^[\s,]+/, "")
    .trim();

  // Fix sentences that now start with lowercase after replacement
  result = result.replace(/([.!?]\s+)([a-z])/g, (m, p1, p2) => p1 + p2.toUpperCase());

  return result;
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

    const systemPrompt = `You are a forensic writing analyst and AI-detection expert. Your job is to extract a writer's deep personality, creativity patterns, syntax habits, and emotional signature from their real writing samples — specifically the qualities that make writing feel genuinely human rather than AI-generated. Be extremely specific. Quote actual phrases as evidence. Output ONLY the sections requested.`;

    const userPrompt = `Analyze these writing samples and extract a complete human writing fingerprint. Output ONLY these sections, nothing else:

${samplesText}

VOICE & PERSONALITY:
What is this person's genuine voice — not their topic, but HOW they think and express themselves? What makes them sound like a real person and not an AI? Quote one phrase that captures their personality perfectly.

CREATIVITY & ORIGINALITY MARKERS:
Does this writer use metaphors, comparisons, or images from their own life? Do they make unexpected connections? Do they have opinions that feel genuinely theirs, not generic? Quote specific examples. This is what GPTZero calls "creativity" — the opposite of generic AI writing.

SYNTAX VARIETY PATTERNS:
How do they vary their sentence structure? Do they start sentences with conjunctions (But, And, So)? Do they use fragments? Do they answer their own questions? Do they use parenthetical asides? Do they contradict themselves slightly? Quote examples of their most varied and interesting sentence structures.

EMOTIONAL AUTHENTICITY:
How does this person express genuine emotion, opinion, or personal stake in what they're writing? Do they admit confusion, excitement, frustration? Do they share personal memories or specific experiences? Quote examples. This is what makes writing feel warm rather than "detached warmth."

ARGUMENT & THINKING STYLE:
Does this person frontload their point or build to it? Do they think through problems out loud? Do they use personal examples as evidence rather than abstract claims? Do they go on tangents and come back? How do they connect one idea to the next in a way that feels like a real person's train of thought?

NATURAL IMPERFECTIONS:
What are the small human habits in their writing — the "or something" at the end of a thought, the "I mean" mid-sentence, the sentence that runs a little long, the casual aside? These are exactly what AI detectors look for as proof of humanity. Quote specific examples.

WHAT THIS PERSON NEVER DOES:
What AI writing habits are completely absent from their samples? (e.g. never uses "Furthermore", never wraps up paragraphs with a clean summary sentence, never uses passive voice) Be specific — these are just as important as what they do.

THEIR EXACT TRANSITIONS:
List the exact words and phrases this person uses to move between ideas. Not AI transitions — their actual ones. Quote them directly.

CLONING INSTRUCTIONS:
Write 3 paragraphs to an AI that will write as this person. Cover: (1) their overall personality and how it comes through in writing, (2) their specific syntax habits and creativity patterns, (3) the single most important thing to get right to avoid sounding like AI when imitating them. Be extremely specific and actionable.`;

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

    const systemPrompt = buildSystemPrompt(dna, rawBlueprint);
    const userPrompt = buildRewriteUserPrompt(text);

    // Step 1: Groq clones the voice — no detector rules, just voice
    const voiceRewrite = await callGroq(systemPrompt, userPrompt, 1500, 1.0);

    // Step 2: Post-processor fixes detector signals without touching the voice
    const rewritten = postProcess(voiceRewrite, dna);

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

    const systemPrompt = buildSystemPrompt(dna, rawBlueprint);
    const userPrompt = buildHumanizeUserPrompt(text);

    // Step 1: Groq makes it more natural/personal
    const voiceRewrite = await callGroq(systemPrompt, userPrompt, 1500, 1.1);

    // Step 2: Post-processor runs again on the second pass output
    const rewritten = postProcess(voiceRewrite, dna);

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

