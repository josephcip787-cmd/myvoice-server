// ============================================================
// MyVoice — Render.com Backend Server v10.0
// The clean rebuild. Two-call architecture.
//
// HOW IT WORKS:
// POST /analyze  — deep analysis of writing samples
//   Step A: extractStyleDNA() — local code, instant, mathematical
//   Step B: Groq qualitative analysis — personality, structure,
//           what they never do, cloning instructions
//   Both stored together as the blueprint.
//
// POST /rewrite  — two-call reconstruction
//   Call 1: Extract content from ChatGPT response as bullet points
//           (strips the AI skeleton completely)
//   Call 2: Reconstruct from bullet points using full DNA + blueprint
//           (Groq never sees ChatGPT's sentences when writing)
//
// POST /humanize — second pass on the rewrite
//   Same two-call approach, pushed harder on personal specificity
//
// GET  /health   — uptime check
// ============================================================

const express = require("express");
const cors = require("cors");
const https = require("https");

const app = express();
const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL = "llama-3.3-70b-versatile";

app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ============================================================
// GROQ API CALL
// temperature param:
//   0.4 = extraction (consistent, factual, just pull the content)
//   0.7 = analysis (balanced, thorough)
//   1.0 = rewrite (varied, creative, human-feeling)
//   1.1 = humanize second pass (push harder)
// ============================================================
async function callGroq(systemPrompt, userPrompt, maxTokens = 2000, temperature = 0.7) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: MODEL,
      max_tokens: maxTokens,
      temperature,
      top_p: 0.95,
      frequency_penalty: 0.4,
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
          resolve(text.trim());
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
// STEP A — LOCAL STYLE DNA
// Pure JavaScript. No API. Instant. Mathematical.
// Captures every measurable dimension of how someone writes.
// ============================================================
function extractStyleDNA(samples) {
  const allText = samples.map(s => s.text).join("\n\n");
  const paragraphs = allText.split(/\n\n+/).filter(p => p.trim().length > 10);
  const sentences = allText.match(/[^.!?]+[.!?]+/g) || [];
  const cleanSentences = sentences.map(s => s.trim()).filter(s => s.split(/\s+/).length > 1);
  const words = allText.toLowerCase().match(/\b[a-z']+\b/g) || [];
  const sc = Math.max(cleanSentences.length, 1);
  const wc = Math.max(words.length, 1);

  // ---- Sentence lengths ----
  const lengths = cleanSentences.map(s => s.split(/\s+/).length);
  const avgLen = Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length);
  const minLen = Math.min(...lengths);
  const maxLen = Math.max(...lengths);
  const shortCount = lengths.filter(l => l <= 8).length;
  const medCount = lengths.filter(l => l > 8 && l < 25).length;
  const longCount = lengths.filter(l => l >= 25).length;
  const fragmentRate = Math.round((lengths.filter(l => l <= 4).length / sc) * 100);

  // Variance between consecutive sentences (burstiness)
  let consecDiff = 0;
  for (let i = 1; i < lengths.length; i++) {
    consecDiff += Math.abs(lengths[i] - lengths[i - 1]);
  }
  const avgConsecDiff = lengths.length > 1 ? Math.round(consecDiff / (lengths.length - 1)) : 0;

  // Run-on detection (sentences with 4+ commas)
  const runOnCount = cleanSentences.filter(s => (s.match(/,/g) || []).length >= 4).length;
  const runOnRate = Math.round((runOnCount / sc) * 100);

  // ---- Sentence openers ----
  const openerI = cleanSentences.filter(s => /^I\s/i.test(s)).length;
  const openerConj = cleanSentences.filter(s => /^(But|And|So|Because|Yet|Or|Nor)\s/i.test(s)).length;
  const openerClause = cleanSentences.filter(s => /^(When|After|Before|While|Once|If|Since|Although|Though|Because|As)\s/i.test(s)).length;
  const openerFiller = cleanSentences.filter(s => /^(Well,|I mean,|Honestly,|Look,|See,|Now,|Okay,|Right,|Actually,)/i.test(s)).length;
  const openerQuestion = cleanSentences.filter(s => /^(What|How|Why|When|Where|Who|Is|Are|Do|Does|Can|Have|Has|Did)\s/i.test(s) && s.trim().endsWith("?")).length;
  const openerIt = cleanSentences.filter(s => /^(It|That|This|There)\s/i.test(s)).length;

  // Most common first word
  const firstWords = {};
  cleanSentences.forEach(s => {
    const fw = s.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "");
    if (fw) firstWords[fw] = (firstWords[fw] || 0) + 1;
  });
  const topOpener = Object.entries(firstWords).sort((a, b) => b[1] - a[1])[0]?.[0] || "i";

  // ---- Transitions actually used ----
  const transitionList = [
    "so", "but", "and", "because", "then", "though", "although",
    "however", "which means", "that's why", "this is why", "the thing is",
    "i mean", "you know", "honestly", "actually", "basically", "like",
    "anyway", "still", "also", "plus", "even", "yet", "well",
    "i guess", "i think", "i feel", "i remember", "i know",
    "and so", "but then", "so then", "and then", "but also"
  ];
  const usedTransitions = transitionList.filter(t => {
    const re = new RegExp(`\\b${t.replace(/\s/g, "\\s")}\\b`, "gi");
    return (allText.match(re) || []).length >= 2;
  });

  // ---- Vocabulary ----
  const avgWordLen = Math.round(words.reduce((a, w) => a + w.length, 0) / wc);
  const longWordRate = Math.round((words.filter(w => w.length > 8).length / wc) * 100);
  const uniqueWords = new Set(words);
  const vocabRichness = Math.round((uniqueWords.size / wc) * 100);

  // Casual words
  const casualList = ["kinda","sorta","gonna","wanna","gotta","literally","basically","honestly","like","just","really","super","totally","pretty","stuff","yeah","yep","nope","okay","cool","crazy","wild","nuts","awesome","dude","man","wait","hold on","i mean","you know","or something","kind of","sort of","a lot","tons","loads"];
  const casualHits = casualList.filter(w => new RegExp(`\\b${w}\\b`, "gi").test(allText));

  // Formal words
  const formalList = ["therefore","moreover","furthermore","subsequently","nevertheless","accordingly","consequently","albeit","utilize","facilitate","implement","leverage","regarding","pertaining","demonstrate","significant","substantial","numerous"];
  const formalHits = formalList.filter(w => new RegExp(`\\b${w}\\b`, "gi").test(allText));

  // Contractions
  const contractionTypes = ["don't","doesn't","can't","won't","it's","i'm","i've","i'd","i'll","we're","they're","you're","isn't","aren't","wasn't","weren't","hasn't","haven't","wouldn't","couldn't","shouldn't","that's","there's","let's","didn't","she's","he's","what's","who's","here's","i'd"];
  const usedContractions = contractionTypes.filter(c => new RegExp(`\\b${c.replace("'", "\\'")}\\b`, "gi").test(allText));
  const contractionRate = Math.round((usedContractions.length > 0 ? (allText.match(/\b\w+'\w+\b/g) || []).length : 0) / wc * 100);

  // ---- Punctuation ----
  const exclamations = (allText.match(/!/g) || []).length;
  const questions = (allText.match(/\?/g) || []).length;
  const ellipses = (allText.match(/\.\.\./g) || []).length;
  const dashes = (allText.match(/—|-{2}/g) || []).length;
  const commas = (allText.match(/,/g) || []).length;
  const semicolons = (allText.match(/;/g) || []).length;
  const parens = (allText.match(/\(/g) || []).length;
  const commaPerSentence = parseFloat((commas / sc).toFixed(2));

  // Comma splices (sentence ends without period — just comma joining two independent clauses)
  const commaSplices = (allText.match(/[a-z],\s+[a-z]/g) || []).length;

  // ---- Tone scores ----
  const hedgeWords = (allText.match(/\b(maybe|perhaps|probably|possibly|might|could|i think|i feel|i guess|i suppose|sort of|kind of|somewhat|a bit|a little)\b/gi) || []).length;
  const hedgeRate = Math.round((hedgeWords / sc) * 100);
  const opinionWords = (allText.match(/\b(i think|i feel|i believe|i know|i guess|i wonder|in my opinion|for me|to me|personally)\b/gi) || []).length;
  const memoryWords = (allText.match(/\b(i remember|i recall|back when|one time|when i was|i used to|i once|growing up)\b/gi) || []).length;
  const selfCorrections = (allText.match(/\b(i mean|well,|actually|wait,|no wait|or rather|you know|i guess)\b/gi) || []).length;
  const directAddress = (allText.match(/\b(you|your|yourself)\b/gi) || []).length;
  const personalPronouns = (allText.match(/\b(i|me|my|mine|myself|we|us|our)\b/gi) || []).length;
  const emotionWords = (allText.match(/\b(love|hate|scared|nervous|excited|happy|sad|frustrated|proud|worried|amazed|angry|hurt|embarrassed|shocked|surprised|feel|felt|feeling)\b/gi) || []).length;

  let formalityScore = 50 + (formalHits.length * 4) - (casualHits.length * 2) - (contractionRate * 1.5);
  formalityScore = Math.max(0, Math.min(100, Math.round(formalityScore)));
  const warmthScore = Math.min(100, Math.round(((emotionWords + personalPronouns) / wc) * 200));

  // ---- Paragraph style ----
  const paraLengths = paragraphs.map(p => (p.match(/[^.!?]+[.!?]+/g) || []).length);
  const avgParaLen = paraLengths.length ? Math.round(paraLengths.reduce((a, b) => a + b, 0) / paraLengths.length) : 3;

  // ---- Structure patterns ----
  const firstSentence = cleanSentences[0] || "";
  const lastSentence = cleanSentences[cleanSentences.length - 1] || "";
  const openingStyle =
    /^I\s/i.test(firstSentence) ? "starts with personal I statement" :
    /^(But|And|So)\s/i.test(firstSentence) ? "jumps in with a conjunction" :
    /\?$/.test(firstSentence.trim()) ? "opens with a question" :
    /^(Well|I mean|Honestly)/i.test(firstSentence) ? "starts with a filler/aside" :
    "starts with a direct statement about the topic";

  const closingStyle =
    /\?$/.test(lastSentence.trim()) ? "ends with a question" :
    lastSentence.split(/\s+/).length < 7 ? "ends with a short punchy line" :
    /\.\.\.$/.test(lastSentence.trim()) ? "trails off with ellipsis" :
    "ends with a complete thought";

  // ---- Signature style words (appear in 2+ samples) ----
  const stopwords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","what","which","who","when","where","why","how","all","each","not","no","so","if","as","by","from","up","about","than","then","just","also","there","here","very","more","some","any","only","out","into","get","got","go","going","one","like","know","think","feel","time","way","make","take","come","see","look","want","back","still","even","well","new","good","old","right","big","really","very","much","many","same","own","too","now","after","before","where","while","though","because","since","although","never","always","every","both","few","most","other","such","than","then","its","into","over","also","after","before","between","through","during","without","within","along","across","behind","beyond","plus","except","up","down","off","yet","once","twice"]);
  const sampleWordSets = samples.map(s => new Set((s.text.toLowerCase().match(/\b[a-z']+\b/g) || [])));
  const wordFreq = {};
  words.forEach(w => { if (!stopwords.has(w) && w.length > 3) wordFreq[w] = (wordFreq[w] || 0) + 1; });
  const minSampleAppearance = samples.length > 1 ? 2 : 1;
  const signatureWords = Object.entries(wordFreq)
    .filter(([w]) => sampleWordSets.filter(s => s.has(w)).length >= minSampleAppearance)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([w]) => w);

  // Recurring phrases
  const phraseFreq = {};
  for (let i = 0; i < words.length - 2; i++) {
    const [w1, w2, w3] = [words[i], words[i + 1], words[i + 2]];
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
    // Rhythm
    avgLen, minLen, maxLen, shortCount, medCount, longCount,
    fragmentRate, runOnRate, avgConsecDiff,
    sentenceCount: sc, wordCount: wc,

    // Openers
    openerIRate: Math.round((openerI / sc) * 100),
    openerConjRate: Math.round((openerConj / sc) * 100),
    openerClauseRate: Math.round((openerClause / sc) * 100),
    openerFillerRate: Math.round((openerFiller / sc) * 100),
    openerQuestionRate: Math.round((openerQuestion / sc) * 100),
    openerItRate: Math.round((openerIt / sc) * 100),
    topOpener,

    // Transitions
    usedTransitions,

    // Vocabulary
    avgWordLen, longWordRate, vocabRichness,
    casualWords: casualHits,
    formalWords: formalHits,
    usedContractions: usedContractions.slice(0, 10),
    contractionRate,

    // Punctuation
    exclamPerSentence: parseFloat((exclamations / sc).toFixed(2)),
    questionPerSentence: parseFloat((questions / sc).toFixed(2)),
    ellipsisPerSentence: parseFloat((ellipses / sc).toFixed(2)),
    dashPerSentence: parseFloat((dashes / sc).toFixed(2)),
    commaPerSentence,
    semicolonCount: semicolons,
    parenCount: parens,
    commaSpliceRate: Math.round((commaSplices / sc) * 100),

    // Personality
    formalityScore, warmthScore,
    hedgeRate, opinionCount: opinionWords,
    memoryCount: memoryWords,
    selfCorrectionCount: selfCorrections,
    directAddressRate: Math.round((directAddress / wc) * 100),
    emotionWordCount: emotionWords,

    // Structure
    avgParaLen, openingStyle, closingStyle,

    // Signatures
    signatureWords, topPhrases,
  };
}

// ============================================================
// STEP B — GROQ QUALITATIVE ANALYSIS PROMPT
// Captures what code can't measure:
// personality, humor, argument style, what they never do,
// cloning instructions written directly to the rewrite model
// ============================================================
function buildAnalysisPrompt(samplesText, dna) {
  return {
    system: `You are a forensic writing analyst. Your job is to study real human writing samples and produce an extremely precise profile that another AI can use to perfectly clone this person's voice and writing structure. Every observation must be specific and grounded in actual evidence from the samples. Quote real phrases. Be forensic, not generic.`,

    user: `Study these writing samples and produce a complete writer profile. Output ONLY the sections below — nothing else, no preamble.

${samplesText}

---

MEASURED FINGERPRINT (for context — already calculated):
- Avg sentence length: ${dna.avgLen} words (range: ${dna.minLen}–${dna.maxLen})
- Short sentences (≤8w): ${dna.shortCount} | Medium: ${dna.medCount} | Long (25+w): ${dna.longCount}
- Starts with "I": ${dna.openerIRate}% | Conjunctions: ${dna.openerConjRate}% | Questions: ${dna.openerQuestionRate}%
- Run-on rate: ${dna.runOnRate}% | Fragment rate: ${dna.fragmentRate}%
- Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100
- Contractions used: ${dna.usedContractions.join(", ") || "few"}
- Casual words: ${dna.casualWords.slice(0, 8).join(", ") || "few"}
- Commas per sentence: ${dna.commaPerSentence}
- Personal memories: ${dna.memoryCount} | Self-corrections: ${dna.selfCorrectionCount}
- Opinion markers: ${dna.opinionCount} | Hedge rate: ${dna.hedgeRate}%

---

VOICE & PERSONALITY:
In 3-4 sentences describe this writer's exact personality as it comes through in writing. What makes them feel like a real specific person and not a generic writer? Quote one phrase that captures them perfectly.

THINKING STRUCTURE:
How does this person organize their ideas? Be precise:
- Linear (A→B→C) or associative (jumping between connected ideas)?
- Do they state their point first then explain, or build toward it?
- Do they mix opinion and fact together or keep them separate?
- Do they follow the original topic or go on tangents?
- How long are their paragraphs typically?
Quote one example of their structural pattern from the samples.

HOW THEY OPEN:
How does this person start a piece or a new idea? Do they jump in mid-thought? Start with a personal memory? Ask a question? Make a bold statement? Quote an actual opening from their samples.

HOW THEY CLOSE:
How does this person end a thought or a paragraph? Do they trail off? End abruptly? Wrap up cleanly? Ask a question? Quote an actual closing from their samples.

PERSONAL STORY PATTERN:
Do they use personal memories and stories? If so — how? Do they name specific details (ages, people, places)? Do they drop into the story immediately or introduce it? How long are the stories? Quote an example.

HUMOR & PERSONALITY QUIRKS:
What is their humor style if any — dry, self-deprecating, absurd, none? What personality quirks show up consistently? Sarcasm? Hyperbole? Self-correction mid-thought? Quote examples.

THEIR EXACT TRANSITIONS:
List the exact words and phrases this person uses to move between ideas. Quote them directly from the samples. Do NOT list generic transitions — only ones that actually appear.

WHAT THEY NEVER DO:
List 5-8 specific writing habits that are completely absent. Be specific — not "they don't use formal language" but "they never use 'Furthermore' or 'Moreover', never end paragraphs with a summary sentence, never use passive voice."

OPINION STYLE:
When they state an opinion, do they give reasons? Do they hedge ("I think maybe") or state confidently? Do they invite the reader to agree or just state their view? Quote an example.

CLONING INSTRUCTIONS:
Write 4 paragraphs directly to the AI that will rewrite content as this person. Use "you should..." and "when writing as this person...". Cover:
1. Their voice, personality, and energy — what makes every sentence sound like them
2. Their structure — how they organize ideas, how they open, how they close
3. Their personal story pattern — when and how to use specific memories
4. The single most important thing to get right, and the single biggest mistake to avoid`
  };
}

// ============================================================
// CALL 1 — CONTENT EXTRACTION
// Strips ChatGPT's skeleton completely.
// Output: clean bullet points of just the meaning.
// Groq never looks at these sentences again when rewriting.
// ============================================================
function buildExtractionPrompt(text) {
  return {
    system: `You are a content extractor. Your only job is to pull the core meaning out of a piece of writing — the facts, arguments, ideas, and logical relationships — and list them as clean bullet points. Strip ALL phrasing, structure, and wording. Just the meaning. Be complete — don't miss any ideas. Output ONLY the bullet points, nothing else.`,

    user: `Extract every idea, fact, argument, and logical relationship from this text as bullet points. Strip all phrasing — just the raw meaning. Be complete.

Text:
${text}

Output format:
• [idea/fact/argument]
• [idea/fact/argument]
...`
  };
}

// ============================================================
// CALL 2 — RECONSTRUCTION PROMPT
// Groq gets: bullet points (meaning) + full DNA + blueprint
// Groq does NOT get: ChatGPT's original sentences
// Result: built entirely from this person's writing patterns
// ============================================================
function buildReconstructionPrompt(bulletPoints, dna, rawBlueprint, pushHarder = false) {

  const fingerprint = `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATHEMATICAL FINGERPRINT — match these numbers precisely
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENTENCE RHYTHM:
  Average length: ${dna.avgLen} words | Range: ${dna.minLen}–${dna.maxLen}
  Short (≤8w): ${Math.round((dna.shortCount/dna.sentenceCount)*100)}% | Medium: ${Math.round((dna.medCount/dna.sentenceCount)*100)}% | Long (25+w): ${Math.round((dna.longCount/dna.sentenceCount)*100)}%
  Avg change between consecutive sentences: ${dna.avgConsecDiff} words
  Fragment rate: ${dna.fragmentRate}% | Run-on rate: ${dna.runOnRate}%

SENTENCE OPENERS:
  Starts with "I": ${dna.openerIRate}%
  Starts with conjunction (But/And/So): ${dna.openerConjRate}%
  Starts with clause (When/After/If): ${dna.openerClauseRate}%
  Starts with filler (Well/I mean/Honestly): ${dna.openerFillerRate}%
  Most common first word: "${dna.topOpener}"

VOCABULARY:
  Avg word length: ${dna.avgWordLen} chars | Long words (8+ chars): ${dna.longWordRate}%
  Casual words they use: ${dna.casualWords.slice(0,8).join(", ") || "standard"}
  Contractions they use: ${dna.usedContractions.join(", ") || "few"}
  Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100

PUNCTUATION:
  Exclamations: ${dna.exclamPerSentence}/sentence
  Questions: ${dna.questionPerSentence}/sentence
  Ellipsis: ${dna.ellipsisPerSentence}/sentence
  Dashes: ${dna.dashPerSentence}/sentence
  Commas: ${dna.commaPerSentence}/sentence
  Semicolons total: ${dna.semicolonCount}
  Parenthetical asides: ${dna.parenCount}

PERSONALITY MARKERS:
  Personal memories: ${dna.memoryCount} in samples
  Opinion markers (I think/feel): ${dna.opinionCount}
  Self-corrections (I mean/well/actually): ${dna.selfCorrectionCount}
  Hedge rate: ${dna.hedgeRate}%
  Direct reader address: ${dna.directAddressRate}%

STRUCTURE:
  Avg paragraph length: ${dna.avgParaLen} sentences
  Opening style: ${dna.openingStyle}
  Closing style: ${dna.closingStyle}

TRANSITIONS THEY ACTUALLY USE:
  ${dna.usedTransitions.join(", ") || "standard"}

SIGNATURE WORDS (appear across multiple samples):
  ${dna.signatureWords.join(", ") || "none detected"}`;

  const pushBlock = pushHarder ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECOND PASS — push harder on these specifically
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The first version was close but still slightly AI. Fix:
- Any sentence using "not just X, but Y" → rewrite it
- Any paragraph ending with a clean summary → break it
- Any point made without a personal angle → add one
- Any two consecutive sentences of similar length → disrupt
- Any formal word that has a shorter natural version → swap it` : "";

  return {
    system: `You are a writing style cloner. You reconstruct content in a specific human's exact writing voice. You have their complete mathematical fingerprint and qualitative profile. Your job is to write the given content exactly as THEY would have written it — matching every measurable and qualitative dimension of their writing. The output should be statistically indistinguishable from their real writing.

Output ONLY the written text. No intro, no label, no explanation. First word of your response = first word of the text.`,

    user: `Write the following content in this person's exact voice. You are not rewriting someone else's text — you are writing these ideas FROM SCRATCH as if this person just sat down and wrote about this topic themselves.

${fingerprint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITATIVE VOICE & STRUCTURE BLUEPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${rawBlueprint || "Use the fingerprint above as your guide."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — non-negotiable
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PERSONAL STORIES ARE THE #1 HUMAN SIGNAL
   Weave in at least one specific personal detail per paragraph.
   A real memory, a specific moment, a named detail.
   "I remember when..." or just drop into the story directly.
   This is what GPTZero cannot flag — real specificity.

2. CONTRAST PHRASING IS THE #1 AI SIGNAL — NEVER USE IT
   Never write: "not just X, but Y"
   Never write: "not just about X, but about Y"
   Never write: "it's not just X, it's Y"
   Never write: "more than just X"
   Never write: "not about X, but about Y"
   These get flagged every single time even in casual writing.
   Instead say the positive directly: "it's about X and Y"

3. DESTROY THE AI STRUCTURE
   Do NOT write: topic sentence → explanation → conclusion
   Do NOT give each idea its own clean paragraph
   Do NOT wrap up paragraphs with summary sentences
   DO follow this person's actual structure from the blueprint

4. PRESERVE ALL CONTENT
   Every idea and fact from the bullet points must appear.
   Nothing added, nothing removed.

5. SHORT WORDS OVER LONG ONES
   "get" not "obtain" | "show" not "demonstrate"
   "use" not "utilize" | "help" not "facilitate"
   "need" not "require" | "try" not "attempt"

${pushBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT TO WRITE ABOUT (extracted meaning — build from these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${bulletPoints}

Now write it as this person. First word of your response = first word of the text.`
  };
}

// ============================================================
// POST-PROCESSOR
// Deterministic code cleanup after Groq writes.
// Strips any remaining AI phrases and word choices.
// Never touches meaning — only surface-level signals.
// ============================================================
function postProcess(text) {
  let r = text;

  // Banned transitions
  const banned = [
    [/\bFurthermore,?\s*/gi, ""],
    [/\bMoreover,?\s*/gi, ""],
    [/\bAdditionally,?\s*/gi, ""],
    [/\bIn addition,?\s*/gi, ""],
    [/\bIn conclusion,?\s*/gi, ""],
    [/\bTo summarize,?\s*/gi, ""],
    [/\bIn summary,?\s*/gi, ""],
    [/\bNotably,?\s*/gi, ""],
    [/\bSubsequently,?\s*/gi, ""],
    [/\bConsequently,?\s*/gi, "So "],
    [/\bHaving said that,?\s*/gi, "But "],
    [/\bWith that being said,?\s*/gi, "That said, "],
    [/\bIt goes without saying( that)?,?\s*/gi, ""],
    [/\bNeedless to say,?\s*/gi, ""],
    [/\bIt is worth noting( that)?,?\s*/gi, ""],
    [/\bIt is important to note( that)?,?\s*/gi, ""],
    [/\bThis highlights\b/gi, "This shows"],
    [/\bThis demonstrates\b/gi, "This shows"],
    [/\bThis underscores\b/gi, "This shows"],
    [/\bThis illustrates\b/gi, "This shows"],
    [/\bThis suggests\b/gi, "This means"],
    [/\bone might argue\b/gi, "some people think"],
    [/\bplays a crucial role\b/gi, "matters a lot"],
    [/\bAt the end of the day,?\s*/gi, ""],
    [/\bAll things considered,?\s*/gi, ""],
    [/\bFirst and foremost,?\s*/gi, "First, "],
    [/\bLast but not least,?\s*/gi, "And "],
    // Contrast phrasing — #1 AI flag
    [/\bnot just about ([^,\.]+),?\s*but (also )?about\b/gi, "about $1 and"],
    [/\bit'?s not just ([^,\.]+),?\s*(it'?s|but) (also )?/gi, "it's "],
    [/\bthey'?re not just ([^,\.]+),?\s*(they'?re|but) (also )?/gi, "they're "],
    [/\bnot just ([^,\.]{3,40}),?\s*but (also )?/gi, ""],
    [/\bmore than just\b/gi, "really"],
    [/\bis more than just\b/gi, "is really"],
  ];

  for (const [p, rep] of banned) r = r.replace(p, rep);

  // Word swaps
  const swaps = [
    ["utilize","use"],["utilizes","uses"],["utilized","used"],
    ["leverage","use"],["leverages","uses"],["leveraged","used"],
    ["facilitate","help"],["facilitates","helps"],["facilitated","helped"],
    ["demonstrate","show"],["demonstrates","shows"],["demonstrated","showed"],
    ["obtain","get"],["obtains","gets"],["obtained","got"],
    ["acquire","get"],["acquires","gets"],["acquired","got"],
    ["commence","start"],["commences","starts"],["commenced","started"],
    ["endeavor","try"],["endeavors","tries"],["endeavored","tried"],
    ["individuals","people"],["individual","person"],
    ["implement","use"],["implements","uses"],["implemented","used"],
    ["substantial","big"],["considerable","big"],
    ["numerous","many"],["multiple","many"],
    ["assist","help"],["assists","helps"],["assisted","helped"],
    ["require","need"],["requires","needs"],["required","needed"],
    ["purchase","buy"],["purchases","buys"],["purchased","bought"],
    ["attempt","try"],["attempts","tries"],["attempted","tried"],
    ["pertaining to","about"],["regarding","about"],
    ["in order to","to"],["due to the fact that","because"],
    ["in the event that","if"],["at this point in time","now"],
    ["in the near future","soon"],["a number of","some"],
    ["the majority of","most"],["a significant amount","a lot"],
  ];

  for (const [ai, human] of swaps) {
    const re = new RegExp(`\\b${ai.replace(/ /g, "\\s+")}\\b`, "gi");
    r = r.replace(re, m => m[0] === m[0].toUpperCase() ? human[0].toUpperCase() + human.slice(1) : human);
  }

  // Cleanup
  r = r.replace(/\s{2,}/g, " ").replace(/\s+([.,!?])/g, "$1").trim();
  r = r.replace(/([.!?]\s+)([a-z])/g, (m, p1, p2) => p1 + p2.toUpperCase());

  return r;
}

// ============================================================
// GET /health
// ============================================================
app.get("/health", (req, res) => {
  res.json({ status: "ok", model: MODEL, groqKeySet: !!GROQ_API_KEY, version: "10.0" });
});

// ============================================================
// POST /analyze
// ============================================================
app.post("/analyze", async (req, res) => {
  try {
    const { samples } = req.body;
    if (!samples?.length) return res.status(400).json({ error: "No samples provided" });

    // Step A — local mathematical fingerprint
    const dna = extractStyleDNA(samples);

    // Step B — Groq qualitative analysis
    const samplesText = samples.map((s, i) => `--- Sample ${i + 1}: ${s.label || "Untitled"} ---\n${s.text}`).join("\n\n");
    const { system, user } = buildAnalysisPrompt(samplesText, dna);
    const rawBlueprint = await callGroq(system, user, 2000, 0.7);

    const blueprint = JSON.stringify({ dna, rawBlueprint });
    res.json({ blueprint });

  } catch (err) {
    console.error("/analyze error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /rewrite — two-call reconstruction
// ============================================================
app.post("/rewrite", async (req, res) => {
  try {
    const { text, blueprint } = req.body;
    if (!text || !blueprint) return res.status(400).json({ error: "Missing text or blueprint" });

    let dna = null, rawBlueprint = blueprint;
    try {
      const parsed = JSON.parse(blueprint);
      if (parsed.dna && parsed.rawBlueprint) { dna = parsed.dna; rawBlueprint = parsed.rawBlueprint; }
    } catch (e) {}

    // Call 1 — extract content as bullet points (strips AI skeleton)
    const { system: extSys, user: extUser } = buildExtractionPrompt(text);
    const bulletPoints = await callGroq(extSys, extUser, 800, 0.4);

    // Call 2 — reconstruct from bullet points in this person's voice
    const { system: recSys, user: recUser } = buildReconstructionPrompt(bulletPoints, dna, rawBlueprint, false);
    const voiceRewrite = await callGroq(recSys, recUser, 2000, 1.0);

    // Post-process — strip any remaining AI signals
    const rewritten = postProcess(voiceRewrite);
    res.json({ rewritten });

  } catch (err) {
    console.error("/rewrite error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /humanize — second pass, push harder
// ============================================================
app.post("/humanize", async (req, res) => {
  try {
    const { text, blueprint } = req.body;
    if (!text || !blueprint) return res.status(400).json({ error: "Missing text or blueprint" });

    let dna = null, rawBlueprint = blueprint;
    try {
      const parsed = JSON.parse(blueprint);
      if (parsed.dna && parsed.rawBlueprint) { dna = parsed.dna; rawBlueprint = parsed.rawBlueprint; }
    } catch (e) {}

    // Call 1 — extract content again cleanly
    const { system: extSys, user: extUser } = buildExtractionPrompt(text);
    const bulletPoints = await callGroq(extSys, extUser, 800, 0.4);

    // Call 2 — reconstruct with pushHarder = true
    const { system: recSys, user: recUser } = buildReconstructionPrompt(bulletPoints, dna, rawBlueprint, true);
    const voiceRewrite = await callGroq(recSys, recUser, 2000, 1.1);

    const rewritten = postProcess(voiceRewrite);
    res.json({ rewritten });

  } catch (err) {
    console.error("/humanize error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// START
// ============================================================
app.listen(PORT, () => {
  console.log(`MyVoice v10.0 running on port ${PORT}`);
  if (!GROQ_API_KEY) console.warn("WARNING: GROQ_API_KEY not set");
});

