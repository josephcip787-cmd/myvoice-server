// ============================================================
// MyVoice v2 — Backend Server
// Writes content FROM SCRATCH in the user's exact voice.
// No ChatGPT rewriting. No dependency on chatgpt.com.
//
// PIPELINE:
// POST /write   — generate text from scratch in user's voice
//   Call 1: Groq writes the content in their voice
//   Call 2: Groq humanizes using grubby method
//   Final:  Regex cleanup
//
// POST /analyze — deep analysis of writing samples
// POST /quiz    — analyze Voice Capture Quiz answers
// GET  /health  — uptime check
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
// ============================================================
async function callGroq(systemPrompt, userPrompt, maxTokens = 2000, temperature = 0.9) {
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
// LOCAL STYLE DNA
// Pure JS — no API, instant, mathematical
// ============================================================
function extractStyleDNA(samples) {
  const allText = samples.map(s => s.text).join("\n\n");
  const paragraphs = allText.split(/\n\n+/).filter(p => p.trim().length > 10);
  const sentences = allText.match(/[^.!?]+[.!?]+/g) || [];
  const cleanSentences = sentences.map(s => s.trim()).filter(s => s.split(/\s+/).length > 1);
  const words = allText.toLowerCase().match(/\b[a-z']+\b/g) || [];
  const sc = Math.max(cleanSentences.length, 1);
  const wc = Math.max(words.length, 1);

  const lengths = cleanSentences.map(s => s.split(/\s+/).length);
  const avgLen = Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length);
  const minLen = Math.min(...lengths);
  const maxLen = Math.max(...lengths);
  const shortCount = lengths.filter(l => l <= 8).length;
  const medCount = lengths.filter(l => l > 8 && l < 25).length;
  const longCount = lengths.filter(l => l >= 25).length;
  const fragmentRate = Math.round((lengths.filter(l => l <= 4).length / sc) * 100);
  const runOnCount = cleanSentences.filter(s => (s.match(/,/g) || []).length >= 4).length;
  const runOnRate = Math.round((runOnCount / sc) * 100);

  let consecDiff = 0;
  for (let i = 1; i < lengths.length; i++) consecDiff += Math.abs(lengths[i] - lengths[i-1]);
  const avgConsecDiff = lengths.length > 1 ? Math.round(consecDiff / (lengths.length - 1)) : 0;

  const openerI = cleanSentences.filter(s => /^I\s/i.test(s)).length;
  const openerConj = cleanSentences.filter(s => /^(But|And|So|Because|Yet|Or)\s/i.test(s)).length;
  const openerClause = cleanSentences.filter(s => /^(When|After|Before|While|Once|If|Since|Although|Though|As)\s/i.test(s)).length;
  const openerFiller = cleanSentences.filter(s => /^(Well,|I mean,|Honestly,|Look,|See,|Okay,|Actually,)/i.test(s)).length;

  const firstWords = {};
  cleanSentences.forEach(s => {
    const fw = s.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "");
    if (fw) firstWords[fw] = (firstWords[fw] || 0) + 1;
  });
  const topOpener = Object.entries(firstWords).sort((a, b) => b[1] - a[1])[0]?.[0] || "i";

  const transitionList = ["so","but","and","because","then","though","although","which means","that's why","the thing is","i mean","you know","honestly","actually","basically","like","anyway","still","also","well","i guess","i think","i feel","i remember","but then","and then","but also"];
  const usedTransitions = transitionList.filter(t => {
    const re = new RegExp(`\\b${t.replace(/\s/g, "\\s")}\\b`, "gi");
    return (allText.match(re) || []).length >= 2;
  });

  const avgWordLen = Math.round(words.reduce((a, w) => a + w.length, 0) / wc);
  const longWordRate = Math.round((words.filter(w => w.length > 8).length / wc) * 100);
  const uniqueWords = new Set(words);
  const vocabRichness = Math.round((uniqueWords.size / wc) * 100);

  const casualList = ["kinda","sorta","gonna","wanna","gotta","literally","basically","honestly","like","just","really","super","totally","pretty","stuff","yeah","yep","nope","okay","cool","crazy","wild","awesome","i mean","you know","or something","kind of","sort of","a lot","tons"];
  const casualWords = casualList.filter(w => new RegExp(`\\b${w}\\b`, "gi").test(allText));

  const formalList = ["therefore","moreover","furthermore","subsequently","nevertheless","accordingly","consequently","utilize","facilitate","implement","leverage","regarding","pertaining","demonstrate","significant","substantial","numerous"];
  const formalWords = formalList.filter(w => new RegExp(`\\b${w}\\b`, "gi").test(allText));

  const contractionTypes = ["don't","doesn't","can't","won't","it's","i'm","i've","i'd","i'll","we're","they're","you're","isn't","aren't","wasn't","weren't","hasn't","haven't","wouldn't","couldn't","shouldn't","that's","there's","let's","didn't","she's","he's","what's","who's","here's"];
  const usedContractions = contractionTypes.filter(c => new RegExp(`\\b${c.replace("'", "\\'")}\\b`, "gi").test(allText));
  const contractionCount = (allText.match(/\b\w+'\w+\b/g) || []).length;
  const contractionRate = Math.round((contractionCount / wc) * 100);

  const exclamations = (allText.match(/!/g) || []).length;
  const questions = (allText.match(/\?/g) || []).length;
  const ellipses = (allText.match(/\.\.\./g) || []).length;
  const dashes = (allText.match(/—|-{2}/g) || []).length;
  const commas = (allText.match(/,/g) || []).length;
  const semicolons = (allText.match(/;/g) || []).length;
  const parens = (allText.match(/\(/g) || []).length;

  const hedgeWords = (allText.match(/\b(maybe|perhaps|probably|possibly|might|could|i think|i feel|i guess|sort of|kind of|somewhat|a bit)\b/gi) || []).length;
  const opinionWords = (allText.match(/\b(i think|i feel|i believe|i know|i guess|in my opinion|for me|to me|personally)\b/gi) || []).length;
  const memoryWords = (allText.match(/\b(i remember|i recall|back when|one time|when i was|i used to|i once|growing up)\b/gi) || []).length;
  const selfCorrections = (allText.match(/\b(i mean|well,|actually|wait,|you know|i guess)\b/gi) || []).length;
  const directAddress = (allText.match(/\b(you|your|yourself)\b/gi) || []).length;
  const personalPronouns = (allText.match(/\b(i|me|my|mine|myself|we|us|our)\b/gi) || []).length;
  const emotionWords = (allText.match(/\b(love|hate|scared|nervous|excited|happy|sad|frustrated|proud|worried|amazed|angry|hurt|feel|felt)\b/gi) || []).length;

  let formalityScore = 50 + (formalWords.length * 4) - (casualWords.length * 2) - (contractionRate * 1.5);
  formalityScore = Math.max(0, Math.min(100, Math.round(formalityScore)));
  const warmthScore = Math.min(100, Math.round(((emotionWords + personalPronouns) / wc) * 200));

  const paraLengths = paragraphs.map(p => (p.match(/[^.!?]+[.!?]+/g) || []).length);
  const avgParaLen = paraLengths.length ? Math.round(paraLengths.reduce((a, b) => a + b, 0) / paraLengths.length) : 3;

  const firstSentence = cleanSentences[0] || "";
  const lastSentence = cleanSentences[cleanSentences.length - 1] || "";
  const openingStyle =
    /^I\s/i.test(firstSentence) ? "starts with personal I statement" :
    /^(But|And|So)\s/i.test(firstSentence) ? "jumps in with a conjunction" :
    /\?$/.test(firstSentence.trim()) ? "opens with a question" :
    "starts with a direct statement";
  const closingStyle =
    /\?$/.test(lastSentence.trim()) ? "ends with a question" :
    lastSentence.split(/\s+/).length < 7 ? "ends with a short punchy line" :
    /\.\.\.$/.test(lastSentence.trim()) ? "trails off with ellipsis" :
    "ends with a complete thought";

  const stopwords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","what","which","who","when","where","why","how","all","each","not","no","so","if","as","by","from","up","about","than","then","just","also","there","here","very","more","some","any","only","out","into","get","got","go","going","one","like","know","think","feel","time","way","make","take","come","see","look","want","back","still","even","well","new","good","old","right","big","really","much","many","same","own","too","now","after","before","never","always","every","both","few","most","other","once"]);
  const sampleWordSets = samples.map(s => new Set((s.text.toLowerCase().match(/\b[a-z']+\b/g) || [])));
  const wordFreq = {};
  words.forEach(w => { if (!stopwords.has(w) && w.length > 3) wordFreq[w] = (wordFreq[w] || 0) + 1; });
  const minSampleAppearance = samples.length > 1 ? 2 : 1;
  const signatureWords = Object.entries(wordFreq)
    .filter(([w]) => sampleWordSets.filter(s => s.has(w)).length >= minSampleAppearance)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([w]) => w);

  const longestRunOn = cleanSentences.filter(s => (s.match(/,/g) || []).length >= 3).sort((a, b) => b.split(/\s+/).length - a.split(/\s+/).length)[0] || "";
  const shortFragments = cleanSentences.filter(s => s.split(/\s+/).length <= 6).slice(0, 3);
  const personalSentences = cleanSentences.filter(s => /^I\s/i.test(s) && /remember|recall|used to|one time|my coach|my friend|my/i.test(s)).slice(0, 3);
  const conjunctionSentences = cleanSentences.filter(s => /^(But|And|So|Because|Yet)\s/i.test(s)).slice(0, 2);
  const mostDistinctive = cleanSentences.filter(s => /\bI\b/i.test(s) && s.split(/\s+/).length > 15 && s.split(/\s+/).length < 50).sort(() => Math.random() - 0.5).slice(0, 3);

  return {
    avgLen, minLen, maxLen, shortCount, medCount, longCount,
    fragmentRate, runOnRate, avgConsecDiff, sentenceCount: sc, wordCount: wc,
    openerIRate: Math.round((openerI/sc)*100),
    openerConjRate: Math.round((openerConj/sc)*100),
    openerClauseRate: Math.round((openerClause/sc)*100),
    openerFillerRate: Math.round((openerFiller/sc)*100),
    topOpener, usedTransitions,
    avgWordLen, longWordRate, vocabRichness,
    casualWords, formalWords,
    usedContractions: usedContractions.slice(0, 10), contractionRate,
    exclamPerSentence: parseFloat((exclamations/sc).toFixed(2)),
    questionPerSentence: parseFloat((questions/sc).toFixed(2)),
    ellipsisPerSentence: parseFloat((ellipses/sc).toFixed(2)),
    dashPerSentence: parseFloat((dashes/sc).toFixed(2)),
    commaPerSentence: parseFloat((commas/sc).toFixed(2)),
    semicolonCount: semicolons, parenCount: parens,
    formalityScore, warmthScore,
    hedgeRate: Math.round((hedgeWords/sc)*100),
    opinionCount: opinionWords, memoryCount: memoryWords,
    selfCorrectionCount: selfCorrections,
    directAddressRate: Math.round((directAddress/wc)*100),
    emotionWordCount: emotionWords,
    avgParaLen, openingStyle, closingStyle,
    signatureWords,
    exampleSentences: { longestRunOn, shortFragments, personalSentences, conjunctionSentences, mostDistinctive }
  };
}

// ============================================================
// WRITING PROMPT — Call 1
// Groq writes from scratch in this person's voice.
// No AI text to rewrite — just the prompt and the profile.
// ============================================================
function buildWritePrompt(userRequest, dna, rawBlueprint, quizProfile) {
  const ex = dna?.exampleSentences || {};

  const fingerprintBlock = dna ? `
MEASURED WRITING FINGERPRINT:
- Avg sentence: ${dna.avgLen} words | Range: ${dna.minLen}–${dna.maxLen}
- Short (≤8w): ${Math.round((dna.shortCount/dna.sentenceCount)*100)}% | Long (25+w): ${Math.round((dna.longCount/dna.sentenceCount)*100)}%
- Starts with "I": ${dna.openerIRate}% | Conjunctions: ${dna.openerConjRate}%
- Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100
- Contractions: ${dna.usedContractions.join(", ") || "few"}
- Casual words: ${dna.casualWords.slice(0,8).join(", ") || "standard"}
- Run-on rate: ${dna.runOnRate}% | Fragment rate: ${dna.fragmentRate}%
- Commas/sentence: ${dna.commaPerSentence}
- Personal memories: ${dna.memoryCount} | Self-corrections: ${dna.selfCorrectionCount}
- Transitions they use: ${dna.usedTransitions.slice(0,8).join(", ") || "standard"}
- Signature words: ${dna.signatureWords.join(", ") || "none"}
${ex.longestRunOn ? `\nEXAMPLE run-on: "${ex.longestRunOn}"` : ""}
${ex.shortFragments?.length ? `\nEXAMPLE short lines: ${ex.shortFragments.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.personalSentences?.length ? `\nEXAMPLE personal sentences: ${ex.personalSentences.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.mostDistinctive?.length ? `\nMOST DISTINCTIVE: ${ex.mostDistinctive.map(s => `"${s}"`).join(" | ")}` : ""}` : "";

  const quizBlock = quizProfile ? `
VOICE QUIZ PROFILE (how this person thinks and speaks):
${quizProfile}` : "";

  return {
    system: `You are a writing agent. You write content from scratch in a specific person's exact voice — matching their vocabulary, rhythm, sentence construction, and personality perfectly. You do not announce what you're doing. You just write.

Output ONLY the written content. No intro, no label, no "Here is your essay:". First word of your response = first word of the content.`,

    user: `Write the following for this specific person, in their exact voice:

REQUEST: ${userRequest}

${fingerprintBlock}

VOICE PROFILE:
${rawBlueprint || "Use the fingerprint above."}
${quizBlock}

HOW TO WRITE AS THIS PERSON:
- Use their vocabulary — their casual words, their contractions, their specific fillers
- Match their sentence construction from the examples above
- Match their natural sentence length variation — short punchy sentences mixed with longer ones
- Write the way they actually think — if they tell stories, tell a story; if they state opinions freely, state opinions
- Use "I think", "I feel", "I remember" naturally at the rate their fingerprint shows
- Do NOT start multiple sentences with "And," or "But," — use conjunctions mid-sentence instead
- Do NOT over-inject fillers — match the rate in their fingerprint, not more
- Match their opening and closing style from the profile
- Write with their energy level — casual, formal, enthusiastic, dry — whatever matches

Write the complete content now. First word = first word of content.`
  };
}

// ============================================================
// GRUBBY HUMANIZATION PROMPT — Call 2
// Takes the written content and humanizes it statistically.
// Raises perplexity, burstiness, injects human markers.
// ============================================================
function buildGrubbyPrompt(text) {
  return {
    system: `You are an AI text humanizer. Transform text to read as genuinely human — raising perplexity and burstiness, removing AI fingerprints, injecting natural human style. Preserve all meaning exactly.

Output ONLY the humanized text. No intro, no label. First word = first word of text.`,

    user: `Humanize this text using the grubby pipeline. Work through each step carefully.

STEP 1 — IDENTIFY AI PATTERNS:
Flag: uniform sentence lengths, formal AI words, clean summary sentences, AI transition openers, missing contractions.

STEP 2 — LEXICAL SUBSTITUTION (raise perplexity):
Replace these AI-prone words with natural human alternatives:
"accelerated"→"sped up" | "sophisticated"→"complex" | "leverage"→"use"
"demonstrate"→"show" | "utilize"→"use" | "significant"→"big" or "real"
"obtain"→"get" | "frequently"→"often" | "individuals"→"people"
"communicate"→"talk" | "comprehend"→"understand" | "subsequently"→"then"
"implement"→"use" | "sufficient"→"enough" | "attempt"→"try"
"require"→"need" | "provide"→"give" | "assist"→"help" | "purchase"→"buy"
"numerous"→"a lot of" | "substantial"→"big" | "regarding"→"about"
"additionally"→"also" | "furthermore"→delete | "moreover"→delete
"in conclusion"→delete | "to summarize"→delete
"at the same time"→"but" | "in the end"→delete
"on the other hand"→"but" | "as a result"→"so"
"therefore"→"so" | "thus"→"so" | "however"→"but"
"in addition"→"also" | "nevertheless"→"but still"
Replace ANY formal/robotic word with what a real person would naturally say.

STEP 3 — SYNTACTIC VARIATION (raise burstiness):
Mix sentence lengths aggressively — this is the #1 GPTZero signal:
- Find paragraphs with similar-length sentences — disrupt them
- After a long sentence (20+ words), add a short punchy one (4-7 words)
- After a short sentence, let the next one run longer
- Split one medium sentence per paragraph at a natural break point
- Merge the final paragraph if it ends with 3+ short clean sentences
- Target varied sequence: 14, 5, 22, 8, 31, 6, 19, 4, 28

STEP 4 — STYLISTIC NOISE (human markers — balanced):
- Add contractions: "it is"→"it's", "they are"→"they're", "do not"→"don't",
  "I am"→"I'm", "would not"→"wouldn't", "cannot"→"can't",
  "that is"→"that's", "there is"→"there's", "did not"→"didn't"
- Strip ALL em-dashes (—) — replace with comma or remove
- Fillers: embed naturally MID-SENTENCE at most 1-2 per paragraph
  "words are, honestly, more powerful than people think" ✓
  "Honestly, words are powerful" — do NOT start sentences this way ✗
- NEVER start sentences with "So, like," or "And, I mean,"
- NEVER end sentences with "you know?" — embed mid-sentence instead

STEP 5 — PUNCTUATION NORMALIZATION:
- Remove em-dashes → comma or nothing
- Semicolons → period or comma where natural
- Keep contractions consistent throughout

PRESERVE: Every fact, idea, meaning. Nothing added, nothing removed.

Text to humanize:
${text}`
  };
}

// ============================================================
// REGEX CLEANUP — Final pass
// ============================================================
function regexCleanup(text) {
  let r = text;

  const openers = [
    /^In the end,?\s*/gim, /^At the same time,?\s*/gim,
    /^On the other hand,?\s*/gim, /^Because of this,?\s*/gim,
    /^As a result,?\s*/gim, /^With this in mind,?\s*/gim,
    /^All in all,?\s*/gim, /^Overall,?\s*/gim,
    /^In conclusion,?\s*/gim, /^To conclude,?\s*/gim,
    /^In summary,?\s*/gim, /^To summarize,?\s*/gim,
    /^Ultimately,?\s*/gim, /^Furthermore,?\s*/gim,
    /^Moreover,?\s*/gim, /^Additionally,?\s*/gim,
    /^Subsequently,?\s*/gim, /^Consequently,?\s*/gim,
    /^Nevertheless,?\s*/gim, /^Having said that,?\s*/gim,
    /^That being said,?\s*/gim, /^Needless to say,?\s*/gim,
    /^First and foremost,?\s*/gim, /^Last but not least,?\s*/gim,
    /^At the end of the day,?\s*/gim, /^All things considered,?\s*/gim,
  ];

  for (const p of openers) r = r.replace(p, "");

  const fixes = [
    [/\bThis highlights\b/gi, "This shows"],
    [/\bThis demonstrates\b/gi, "This shows"],
    [/\bThis underscores\b/gi, "This means"],
    [/\bplays a crucial role\b/gi, "matters a lot"],
    [/\bnot just about ([^,\.]+),?\s*but (also )?about\b/gi, "about $1 and"],
    [/\bmore than just\b/gi, "really"],
    [/—/g, ", "],
  ];

  for (const [p, rep] of fixes) r = r.replace(p, rep);

  r = r
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,!?])/g, "$1")
    .replace(/([.!?]\s+)([a-z])/g, (m, p1, p2) => p1 + p2.toUpperCase())
    .replace(/^([a-z])/, m => m.toUpperCase())
    .trim();

  return r;
}

// ============================================================
// GET /health
// ============================================================
app.get("/health", (req, res) => {
  res.json({ status: "ok", model: MODEL, groqKeySet: !!GROQ_API_KEY, version: "2.0" });
});

// ============================================================
// POST /analyze
// ============================================================
app.post("/analyze", async (req, res) => {
  try {
    const { samples } = req.body;
    if (!samples?.length) return res.status(400).json({ error: "No samples provided" });

    const dna = extractStyleDNA(samples);
    const samplesText = samples.map((s, i) => `--- Sample ${i+1}: ${s.label || "Untitled"} ---\n${s.text}`).join("\n\n");
    const ex = dna.exampleSentences;

    const analysisPrompt = {
      system: `You are a forensic writing analyst. Study real writing samples and produce an extremely precise voice profile that another AI can use to write as this person. Quote real phrases. Be specific and forensic.`,
      user: `Study these writing samples and extract a complete writer profile. Output ONLY the sections below.

${samplesText}

---
MEASURED FINGERPRINT:
- Avg sentence: ${dna.avgLen} words | Range: ${dna.minLen}–${dna.maxLen}
- Starts with "I": ${dna.openerIRate}% | Conjunction: ${dna.openerConjRate}%
- Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100
- Contractions: ${dna.usedContractions.join(", ") || "few"}
- Casual words: ${dna.casualWords.slice(0,8).join(", ") || "standard"}
- Run-on rate: ${dna.runOnRate}% | Fragment rate: ${dna.fragmentRate}%
- Personal memories: ${dna.memoryCount} | Self-corrections: ${dna.selfCorrectionCount}
${ex.longestRunOn ? `- Run-on example: "${ex.longestRunOn}"` : ""}
${ex.shortFragments?.length ? `- Short fragments: ${ex.shortFragments.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.personalSentences?.length ? `- Personal sentences: ${ex.personalSentences.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.mostDistinctive?.length ? `- Most distinctive: ${ex.mostDistinctive.map(s => `"${s}"`).join(" | ")}` : ""}
---

VOICE & PERSONALITY:
3 sentences. What is their energy — casual, earnest, dry, enthusiastic? Quote one phrase that captures them perfectly.

VOCABULARY:
What words are distinctly theirs? What formal words do they never use? What casual words appear constantly?

SENTENCE CONSTRUCTION:
How do they build sentences? Run-ons? Fragments? Self-interruptions? Quote 2 actual examples.

THINKING PATTERN:
Linear or associative? Point first or build to it? Quote one example.

PERSONAL STORY PATTERN:
Do they use personal memories? How specific? Quote an example.

WHAT THEY NEVER DO:
5 specific absent habits.

THEIR TRANSITIONS:
Exact transition words from their samples only.

OPENING AND CLOSING:
Quote actual opening and closing.

WRITING INSTRUCTIONS:
3 paragraphs to the AI writing as this person:
1. Their vocabulary — what to use, what to never use
2. Their sentence construction — exactly how they build sentences
3. Single most important thing to nail + biggest mistake to avoid`
    };

    const rawBlueprint = await callGroq(analysisPrompt.system, analysisPrompt.user, 2000, 0.7);
    res.json({ blueprint: JSON.stringify({ dna, rawBlueprint }) });

  } catch (err) {
    console.error("/analyze error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /quiz
// Analyzes the 5 Voice Capture Quiz answers
// ============================================================
app.post("/quiz", async (req, res) => {
  try {
    const { answers } = req.body;
    if (!answers?.length) return res.status(400).json({ error: "No answers provided" });

    const answersText = answers.map((a, i) =>
      `Question ${i+1}: ${a.question}\nAnswer: ${a.answer}`
    ).join("\n\n");

    const systemPrompt = `You are a voice analyst. Analyze quiz answers to extract how someone naturally thinks, speaks, and writes. Be specific. Quote actual phrases from their answers.`;

    const userPrompt = `Analyze these Voice Capture Quiz answers and extract a writing profile.

${answersText}

Extract:

NATURAL VOCABULARY:
What words and phrases do they naturally reach for? Quote specific words and expressions from their answers.

TONE RANGE:
Are they dry/ironic? Earnest and direct? Dramatic? Humorous? What tone showed up across multiple answers?

SENTENCE STYLE:
From their answers — do they write in short punchy sentences or long flowing ones? Do they self-correct? Use fragments? Quote examples.

PERSUASION STYLE:
When they tried to convince (Q4) — did they use logic, emotion, humor, or personal stories?

UNDER PRESSURE VOICE:
From Q5 — how do they handle tension? Are they conciliatory, direct, diplomatic, or assertive?

PERSONAL SPECIFICITY:
Did they use specific details, names, places in their answers? Or stay abstract?

WRITING INSTRUCTIONS:
2 paragraphs to the AI that will write as this person. Cover what their quiz answers reveal about their natural voice that writing samples alone might miss — especially their tone range, their persuasion style, and any distinctive phrases they reach for naturally.`;

    const quizProfile = await callGroq(systemPrompt, userPrompt, 1500, 0.7);
    res.json({ quizProfile });

  } catch (err) {
    console.error("/quiz error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /write
// Write content from scratch in user's voice
// Call 1: Write in their voice
// Call 2: Grubby humanization
// Final: Regex cleanup
// ============================================================
app.post("/write", async (req, res) => {
  try {
    const { prompt, blueprint } = req.body;
    if (!prompt) return res.status(400).json({ error: "No prompt provided" });

    let dna = null, rawBlueprint = null, quizProfile = null;
    try {
      const parsed = JSON.parse(blueprint || "{}");
      // Blueprint can be nested (from popup) or flat (from content script)
      if (parsed.blueprint) {
        const inner = JSON.parse(parsed.blueprint);
        dna = inner.dna;
        rawBlueprint = inner.rawBlueprint;
      }
      quizProfile = parsed.quizProfile || null;
    } catch (e) {}

    // Call 1 — Write in their voice
    const { system: ws, user: wu } = buildWritePrompt(prompt, dna, rawBlueprint, quizProfile);
    const voiceWrite = await callGroq(ws, wu, 2000, 0.95);

    // Call 2 — Grubby humanization
    const { system: gs, user: gu } = buildGrubbyPrompt(voiceWrite);
    const humanized = await callGroq(gs, gu, 2000, 1.0);

    // Final cleanup
    const text = regexCleanup(humanized);
    res.json({ text });

  } catch (err) {
    console.error("/write error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// START
// ============================================================
app.listen(PORT, () => {
  console.log(`MyVoice v2.0 running on port ${PORT}`);
  if (!GROQ_API_KEY) console.warn("WARNING: GROQ_API_KEY not set");
});

