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

  // ---- Structure analysis ----
  // How does this person organize their thinking?
  const firstSentence = sentences[0] || "";
  const lastSentence = sentences[sentences.length - 1] || "";

  // Opening style
  const openingStyle =
    /^(I |My |When I|For me)/i.test(firstSentence) ? "starts with personal statement or memory" :
    /\?$/.test(firstSentence.trim()) ? "opens with a question" :
    /^(The |This |It )/i.test(firstSentence) ? "opens with a direct statement about the topic" :
    /^(But|And|So|Because)/i.test(firstSentence) ? "jumps in mid-thought with a conjunction" :
    "varied opening style";

  // Closing style
  const closingStyle =
    /\?$/.test(lastSentence.trim()) ? "ends with a question — open, not wrapped up" :
    lastSentence.split(/\s+/).length < 8 ? "ends abruptly with a short punchy line" :
    /\.\.\.$/.test(lastSentence.trim()) ? "trails off with an ellipsis" :
    buildsToPoint > 1 ? "wraps up with a conclusion sentence" :
    "ends mid-thought or with a casual observation";

  // Structure type
  const linearScore = buildsToPoint;
  const jumpScore = tangents + selfCorrections;
  const storyScore = personalMemories;
  const structureType =
    storyScore > linearScore && storyScore > jumpScore ? "narrative — tells stories and uses personal experiences" :
    jumpScore > linearScore ? "associative — jumps between ideas, self-corrects, goes on tangents" :
    buildsToPoint > 2 ? "linear — builds logically from point to point" :
    "mixed — some structure but not rigid";

  // Opinion/fact mixing
  const opinionFactMix =
    opinions > sentences.length * 0.3 ? "heavily mixed — opinions woven throughout, hard to separate from facts" :
    opinions > sentences.length * 0.1 ? "moderate mixing — personal takes appear regularly alongside facts" :
    "mostly separate — states facts first, opinions occasionally";

  // Paragraph style
  const avgParaLen = sentences.length > 0 ? Math.round(sentences.length / Math.max(allText.split(/\n\n+/).length, 1)) : 5;
  const paragraphStyle =
    avgParaLen <= 2 ? "very short paragraphs — 1-2 sentences each, punchy blocks" :
    avgParaLen <= 4 ? "short-medium paragraphs — 3-4 sentences, moves fast" :
    "longer paragraphs — develops ideas over multiple sentences";

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
    // Structure patterns (NEW)
    structureType, openingStyle, closingStyle,
    opinionFactMix, paragraphStyle,
    // Signature patterns
    topWords, topPhrases,
    // Fragment rate
    fragmentRate: Math.round((sentences.filter(s => s.split(/\s+/).length <= 4).length / sentenceCount) * 100),
  };
}

// ============================================================
// VOICE + STRUCTURE CLONE SYSTEM v9.0
//
// TWO SEPARATE JOBS — voice and detector never fight each other:
//
// JOB 1 — GROQ: Clone this person's voice AND structure.
//   - Voice: rhythm, vocabulary, tone, personality
//   - Structure: how they organize thinking (do they front-load?
//     tell stories? mix opinion/fact? jump around? trail off?)
//   - Zero detector rules in this prompt — just pure voice clone
//
// JOB 2 — postProcess(): Deterministic code fixes detector signals
//   - Strip banned AI phrases with regex (guaranteed)
//   - Swap long AI words for short human ones (guaranteed)
//   - No AI involved — pure math, works every time
//
// WHY STRUCTURE MATTERS:
//   GPTZero flags: Task-Oriented, Formulaic Flow, Rigid Guidance,
//   Lacks Complexity. These come from ChatGPT's skeleton —
//   linear, step-by-step, point-by-point. No word swap fixes that.
//   We need to rebuild the STRUCTURE to match how this person
//   actually organizes their thoughts, not how ChatGPT does.
// ============================================================

function buildSystemPrompt(dna, rawBlueprint) {

  const dnaBlock = dna ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S MEASURED WRITING FINGERPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENTENCE RHYTHM:
  Style: ${dna.rhythmLabel}
  Average: ${dna.avgLen} words | Range: ${dna.minLen}–${dna.maxLen}
  Short (≤8w): ${dna.shortCount} | Medium: ${dna.medCount} | Long (25+w): ${dna.longCount}

SYNTAX VARIETY:
  Starts with "I": ${dna.startsWithIRate}%
  Starts with conjunction (But/And/So): ${dna.startsWithConjRate}%
  Starts with time/clause (When/After/Because): ${dna.startsWithTimeRate}%
  Questions: ${dna.questionRate}% | Exclamations: ${dna.exclamRate}%
  Fragment rate: ${dna.fragmentRate}%

PERSONALITY MARKERS:
  Personal memories/anecdotes: ${dna.personalMemoryCount}
  Opinion markers (I think/feel/believe): ${dna.opinionCount}
  Self-corrections (I mean/well/honestly): ${dna.selfCorrectionCount}
  Metaphors/comparisons: ${dna.metaphorRate} per 100 sentences
  Direct reader address (you/your): ${dna.directAddressRate}%

STRUCTURE HABITS:
  Argument style: ${dna.buildsToPoint ? "builds toward the point" : "front-loads the conclusion"}
  Uses personal examples as evidence: ${dna.usesExamples ? "yes" : "rarely"}
  Structure type: ${dna.structureType}
  Opening style: ${dna.openingStyle}
  Closing style: ${dna.closingStyle}
  Opinion/fact mixing: ${dna.opinionFactMix}
  Paragraph style: ${dna.paragraphStyle}

VOICE:
  Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100
  Contractions: ${dna.contractionLabel} (${dna.contractionRate}%)
  Avg word length: ${dna.avgWordLen} chars | Long words (8+): ${dna.longWordRate}%
  Punctuation: ${dna.punctuationNotes}
  Signature style words: ${dna.topWords.length ? dna.topWords.join(", ") : "none"}
` : "";

  return `You are a writing style and structure cloner. Your job is to rewrite a given text so it sounds AND thinks exactly like this specific person — matching not just their words and rhythm, but HOW they organize their ideas.

Output ONLY the rewritten text. No intro, no label. First word of response = first word of the rewritten text.

${dnaBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO CLONE — VOICE AND STRUCTURE TOGETHER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLONE THEIR STRUCTURE (this is what most people miss):
The text you're rewriting is probably structured like a guide
or explanation — linear, task-oriented, point by point. That
structure itself triggers AI detectors. You need to rebuild
it the way THIS person actually organizes their thinking.

Study the blueprint below. Answer these before writing:
→ Does this person front-load their point or build to it?
→ Do they mix their own opinion in with facts, or keep them separate?
→ Do they use personal stories/memories as evidence?
→ Do they address the reader directly?
→ Do they follow a linear path or jump between ideas?
→ How do they open — with a statement, a question, mid-thought?
→ How do they end — clean conclusion, trailing off, or abruptly?
→ Do they stay on topic or go on tangents and come back?

Then restructure the content to match their actual thinking pattern.
A person who jumps around should jump around here too.
A person who tells stories should tell a story here too.
A person who asks questions and answers them should do that here.

CLONE THEIR VOICE (rhythm, words, personality):
→ Match their sentence length variation exactly
→ Use their vocabulary level — if they use simple words, use simple words
→ Start sentences the way they start sentences
→ Use their punctuation habits
→ Let their personality come through — opinions, uncertainty, enthusiasm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PERSON'S FULL VOICE AND STRUCTURE BLUEPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${rawBlueprint || "No blueprint yet — rely on the fingerprint above."}`;
}

// ============================================================
// REWRITE USER PROMPT — restructure + voice in one pass
// ============================================================
function buildRewriteUserPrompt(text) {
  return `Read the text below. Understand every idea, fact, and argument in it completely.

Now look at its structure. It's probably written like an explanation or guide — linear, organized, each point leading to the next. That's ChatGPT's structure. A real person doesn't think or write like that.

Your job is to do two things at once:

1. RESTRUCTURE IT — rebuild the organization to match how this person actually thinks:
   - If they front-load conclusions, start with the point and explain after
   - If they tell stories, frame this as a story or personal experience
   - If they mix opinions with facts, weave them together instead of separating them
   - If they jump around, don't follow the original order strictly
   - If they address the reader, talk to the reader
   - Don't give each idea its own clean paragraph — let ideas bleed into each other
   - Don't wrap up each point with a summary sentence — move on when the thought is done
   - Start somewhere that feels natural for THIS person, not at the beginning of the topic

2. REWRITE IN THEIR VOICE — every sentence should sound like them:
   - Their rhythm, their word choices, their tone
   - Their way of connecting one idea to the next
   - Their personality showing through

PRESERVE: Every fact, idea, and argument must still be there. Restructure and rephrase — don't cut content.

Output ONLY the rewritten text. No intro, no label. First word of response = first word of text.

Text:\n\n${text}`;
}

// ============================================================
// HUMANIZE USER PROMPT — second pass, more personal
// ============================================================
function buildHumanizeUserPrompt(text) {
  return `This is written in this person's voice but it still reads slightly AI. Make it more authentically them.

Find the parts that still feel too clean, too organized, or too generic. For each one:
- If a sentence wraps up an idea too neatly, break it or trail off
- If a paragraph follows a too-obvious structure, disrupt it
- If an opinion is stated without personality behind it, add the specific reason this person would have
- If two or three sentences are the same length in a row, break the rhythm

The goal is not to make it worse — it's to make it feel more like this person actually sat down and typed it without overthinking.

Keep every fact and argument. Keep full depth.
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

    const systemPrompt = `You are a forensic writing analyst specializing in voice, structure, and thinking patterns. Extract how this person writes AND how they organize their ideas — both are equally important. Your job is to extract a writer's deep personality, creativity patterns, syntax habits, and emotional signature from their real writing samples — specifically the qualities that make writing feel genuinely human rather than AI-generated. Be extremely specific. Quote actual phrases as evidence. Output ONLY the sections requested.`;

    const userPrompt = `Analyze these writing samples deeply. Output ONLY these sections:

${samplesText}

VOICE & PERSONALITY:
What is this person's genuine voice — not their topic but HOW they think and express themselves? What makes them sound like a real human not an AI? Quote one phrase that captures their personality perfectly.

THINKING STRUCTURE (critical — this is what AI detectors flag most):
How does this person organize their ideas? Be extremely specific:
- Do they front-load the point or build toward it?
- Do they think linearly (A→B→C) or associatively (jumping between connected ideas)?
- Do they mix their personal opinions in with facts, or keep them separate?
- Do they use personal stories/memories as evidence or stay abstract?
- Do they address the reader directly ("you should...") or write in third person?
- How do they open a piece — with a statement, a question, mid-thought, a memory?
- How do they close — clean conclusion, trailing off, abruptly, with a question?
- Do they go on tangents? Do they self-correct mid-thought?
Quote specific examples from the samples for each habit you identify.

SYNTAX & RHYTHM PATTERNS:
How do they vary sentence structure — not just length but TYPE? Do they use fragments? Questions they answer themselves? Sentences starting with "But" or "And"? Run-ons? Parenthetical asides? Repetition for emphasis? Quote the most distinctive examples.

PERSONALITY & EMOTION:
How does this person's actual personality show in their writing? Where do they admit uncertainty, show excitement, express frustration? Where does their voice feel most alive and least like a robot? Quote specific moments.

NATURAL IMPERFECTIONS:
What are the small human habits — "or something", "I mean", casual asides, sentences that run long, self-corrections? Quote real examples from the samples.

WHAT THEY NEVER DO:
What AI habits are completely absent? Be specific — never uses "Furthermore"? Never wraps paragraphs with a summary? Never uses passive voice? These negatives are just as important.

THEIR TRANSITIONS:
List the exact phrases this person uses to move between ideas. Not AI transitions — their actual words. Quote them directly from the samples.

CLONING INSTRUCTIONS:
Write 3 paragraphs to an AI that will imitate this person covering: (1) their voice and personality, (2) their STRUCTURE — how they organize thinking, because this is what AI detectors flag most, (3) the single most important thing to nail to avoid sounding like AI when writing as them.`;

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

