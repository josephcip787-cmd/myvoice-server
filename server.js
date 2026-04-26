// ============================================================
// MyVoice for ChatGPT — Backend Server v11.0
// Clean rebuild. Two Groq calls + regex cleanup.
//
// PIPELINE:
// 1. POST /analyze  — deep analysis of writing samples
//    A) Local code extracts mathematical fingerprint
//    B) Groq extracts qualitative voice profile
//
// 2. POST /rewrite  — two Groq calls
//    Call 1: Voice clone — reword ChatGPT's text in person's style
//    Call 2: Grubby pipeline — humanize the voice clone
//    Final:  Regex cleanup — strip remaining AI openers
//
// 3. POST /humanize — same pipeline, pushed harder
//
// 4. GET  /health   — uptime check
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
// Pure JavaScript. No API. Instant. Mathematical.
// ============================================================
function extractStyleDNA(samples) {
  const allText = samples.map(s => s.text).join("\n\n");
  const paragraphs = allText.split(/\n\n+/).filter(p => p.trim().length > 10);
  const sentences = allText.match(/[^.!?]+[.!?]+/g) || [];
  const cleanSentences = sentences.map(s => s.trim()).filter(s => s.split(/\s+/).length > 1);
  const words = allText.toLowerCase().match(/\b[a-z']+\b/g) || [];
  const sc = Math.max(cleanSentences.length, 1);
  const wc = Math.max(words.length, 1);

  // Sentence lengths
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

  // Consecutive variance
  let consecDiff = 0;
  for (let i = 1; i < lengths.length; i++) consecDiff += Math.abs(lengths[i] - lengths[i-1]);
  const avgConsecDiff = lengths.length > 1 ? Math.round(consecDiff / (lengths.length - 1)) : 0;

  // Sentence openers
  const openerI = cleanSentences.filter(s => /^I\s/i.test(s)).length;
  const openerConj = cleanSentences.filter(s => /^(But|And|So|Because|Yet|Or)\s/i.test(s)).length;
  const openerClause = cleanSentences.filter(s => /^(When|After|Before|While|Once|If|Since|Although|Though|As)\s/i.test(s)).length;
  const openerFiller = cleanSentences.filter(s => /^(Well,|I mean,|Honestly,|Look,|See,|Now,|Okay,|Right,|Actually,)/i.test(s)).length;
  const openerQuestion = cleanSentences.filter(s => /\?$/.test(s.trim()) && /^(What|How|Why|When|Where|Who|Is|Are|Do|Does|Can|Did)\s/i.test(s)).length;

  const firstWords = {};
  cleanSentences.forEach(s => {
    const fw = s.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "");
    if (fw) firstWords[fw] = (firstWords[fw] || 0) + 1;
  });
  const topOpener = Object.entries(firstWords).sort((a, b) => b[1] - a[1])[0]?.[0] || "i";

  // Transitions
  const transitionList = ["so","but","and","because","then","though","although","which means","that's why","this is why","the thing is","i mean","you know","honestly","actually","basically","like","anyway","still","also","plus","well","i guess","i think","i feel","i remember","and so","but then","so then","and then","but also"];
  const usedTransitions = transitionList.filter(t => {
    const re = new RegExp(`\\b${t.replace(/\s/g, "\\s")}\\b`, "gi");
    return (allText.match(re) || []).length >= 2;
  });

  // Vocabulary
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

  // Punctuation
  const exclamations = (allText.match(/!/g) || []).length;
  const questions = (allText.match(/\?/g) || []).length;
  const ellipses = (allText.match(/\.\.\./g) || []).length;
  const dashes = (allText.match(/—|-{2}/g) || []).length;
  const commas = (allText.match(/,/g) || []).length;
  const semicolons = (allText.match(/;/g) || []).length;
  const parens = (allText.match(/\(/g) || []).length;

  // Tone
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

  // Paragraph style
  const paraLengths = paragraphs.map(p => (p.match(/[^.!?]+[.!?]+/g) || []).length);
  const avgParaLen = paraLengths.length ? Math.round(paraLengths.reduce((a, b) => a + b, 0) / paraLengths.length) : 3;

  // Opening/closing
  const firstSentence = cleanSentences[0] || "";
  const lastSentence = cleanSentences[cleanSentences.length - 1] || "";
  const openingStyle =
    /^I\s/i.test(firstSentence) ? "starts with personal I statement" :
    /^(But|And|So)\s/i.test(firstSentence) ? "jumps in with a conjunction" :
    /\?$/.test(firstSentence.trim()) ? "opens with a question" :
    /^(Well|I mean|Honestly)/i.test(firstSentence) ? "starts with a filler" :
    "starts with a direct statement";

  const closingStyle =
    /\?$/.test(lastSentence.trim()) ? "ends with a question" :
    lastSentence.split(/\s+/).length < 7 ? "ends with a short punchy line" :
    /\.\.\.$/.test(lastSentence.trim()) ? "trails off with ellipsis" :
    "ends with a complete thought";

  // Signature words
  const stopwords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","this","that","these","those","what","which","who","when","where","why","how","all","each","not","no","so","if","as","by","from","up","about","than","then","just","also","there","here","very","more","some","any","only","out","into","get","got","go","going","one","like","know","think","feel","time","way","make","take","come","see","look","want","back","still","even","well","new","good","old","right","big","really","much","many","same","own","too","now","after","before","never","always","every","both","few","most","other","once"]);
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
    .slice(0, 8)
    .map(([p]) => p);

  // Example sentences
  const longestRunOn = cleanSentences
    .filter(s => (s.match(/,/g) || []).length >= 3)
    .sort((a, b) => b.split(/\s+/).length - a.split(/\s+/).length)[0] || "";
  const shortFragments = cleanSentences.filter(s => s.split(/\s+/).length <= 6).slice(0, 3);
  const personalSentences = cleanSentences.filter(s => /^I\s/i.test(s) && /remember|recall|used to|one time|my coach|my friend|my/i.test(s)).slice(0, 3);
  const conjunctionSentences = cleanSentences.filter(s => /^(But|And|So|Because|Yet)\s/i.test(s)).slice(0, 2);
  const selfCorrectionSentences = cleanSentences.filter(s => /\bI mean\b|\bwell,\b|\bactually\b|\bor something\b|\byou know\b/i.test(s)).slice(0, 2);
  const mostDistinctive = cleanSentences.filter(s => /\bI\b/i.test(s) && s.split(/\s+/).length > 15 && s.split(/\s+/).length < 50).sort(() => Math.random() - 0.5).slice(0, 3);

  return {
    avgLen, minLen, maxLen, shortCount, medCount, longCount,
    fragmentRate, runOnRate, avgConsecDiff, sentenceCount: sc, wordCount: wc,
    openerIRate: Math.round((openerI/sc)*100),
    openerConjRate: Math.round((openerConj/sc)*100),
    openerClauseRate: Math.round((openerClause/sc)*100),
    openerFillerRate: Math.round((openerFiller/sc)*100),
    openerQuestionRate: Math.round((openerQuestion/sc)*100),
    topOpener,
    usedTransitions,
    avgWordLen, longWordRate, vocabRichness,
    casualWords, formalWords: formalWords,
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
    signatureWords, topPhrases,
    exampleSentences: { longestRunOn, shortFragments, personalSentences, conjunctionSentences, selfCorrectionSentences, mostDistinctive }
  };
}

// ============================================================
// GROQ ANALYSIS PROMPT
// Qualitative profile — what code can't measure
// ============================================================
function buildAnalysisPrompt(samplesText, dna) {
  const ex = dna.exampleSentences;
  return {
    system: `You are a forensic writing analyst. Study real writing samples and produce an extremely precise profile that another AI can use to perfectly clone this person's voice. Every observation must be specific and grounded in actual evidence. Quote real phrases. Be forensic, not generic.`,

    user: `Study these writing samples and produce a complete writer profile. Output ONLY the sections below.

${samplesText}

---
MEASURED FINGERPRINT (already calculated):
- Avg sentence: ${dna.avgLen} words | Range: ${dna.minLen}–${dna.maxLen}
- Short (≤8w): ${dna.shortCount} | Medium: ${dna.medCount} | Long (25+w): ${dna.longCount}
- Starts with "I": ${dna.openerIRate}% | Conjunction: ${dna.openerConjRate}% | Filler: ${dna.openerFillerRate}%
- Formality: ${dna.formalityScore}/100 | Warmth: ${dna.warmthScore}/100
- Contractions: ${dna.usedContractions.join(", ") || "few"}
- Casual words: ${dna.casualWords.slice(0,8).join(", ") || "standard"}
- Run-on rate: ${dna.runOnRate}% | Fragment rate: ${dna.fragmentRate}%
- Personal memories: ${dna.memoryCount} | Self-corrections: ${dna.selfCorrectionCount}
- Commas/sentence: ${dna.commaPerSentence}
${ex.longestRunOn ? `- Longest run-on: "${ex.longestRunOn}"` : ""}
${ex.shortFragments?.length ? `- Short fragments: ${ex.shortFragments.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.personalSentences?.length ? `- Personal sentences: ${ex.personalSentences.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.conjunctionSentences?.length ? `- Conjunction openers: ${ex.conjunctionSentences.map(s => `"${s}"`).join(" | ")}` : ""}
${ex.mostDistinctive?.length ? `- Most distinctive: ${ex.mostDistinctive.map(s => `"${s}"`).join(" | ")}` : ""}
---

VOICE & PERSONALITY:
Describe this writer's exact personality in 3 sentences. What energy do they have — casual, earnest, dry, enthusiastic? Quote one phrase that captures them perfectly.

VOCABULARY FINGERPRINT:
What words does this person actually use that are distinctly theirs? What formal words do they never use? What casual words appear constantly? Be specific — list actual words from the samples.

SENTENCE CONSTRUCTION:
How do they build sentences? Run-ons connected by "and" or "because"? Fragments for punch? Self-interruptions mid-sentence? Comma stacking? Quote 2 actual examples showing their most distinctive construction.

THINKING PATTERN:
Do they think linearly or jump between ideas? Do they state the point first or build to it? Do they mix opinion and fact? Quote one example showing their thinking pattern.

PERSONAL STORY PATTERN:
Do they use personal memories? How specific — named people, ages, places? Do they drop into stories or introduce them? Quote an example.

WHAT THEY NEVER DO:
List 5 specific habits completely absent. E.g. "never wraps paragraphs with summary sentences, never uses Furthermore, never uses passive voice."

THEIR TRANSITIONS:
Quote the exact words/phrases they use to connect ideas — only real ones from the samples.

HOW THEY OPEN AND CLOSE:
Quote an actual opening and closing from their samples.

CLONING INSTRUCTIONS:
Write 3 paragraphs to the AI that will rewrite as this person. Cover:
1. Their vocabulary and word choice — what words they use, what they never use
2. Their sentence construction — exactly how they build sentences
3. The single most important thing to nail and the biggest mistake to avoid`
  };
}

// ============================================================
// CALL 1 — VOICE CLONE PROMPT
// One job: reword ChatGPT's text in this person's vocabulary
// and style. Keep all facts. Keep the structure. Just change
// the wording to match how this person actually talks and writes.
// ============================================================
function buildVoiceClonePrompt(text, dna, rawBlueprint) {
  const ex = dna?.exampleSentences || {};

  const fingerprintBlock = dna ? `
THEIR MEASURED WRITING PATTERNS:
- Sentence length: avg ${dna.avgLen} words, range ${dna.minLen}–${dna.maxLen}
- Starts with "I": ${dna.openerIRate}% | Conjunctions: ${dna.openerConjRate}% | Fillers: ${dna.openerFillerRate}%
- Formality: ${dna.formalityScore}/100 (${dna.formalityScore < 40 ? "very casual" : dna.formalityScore < 60 ? "conversational" : "formal"})
- Contractions they use: ${dna.usedContractions.join(", ") || "few"}
- Casual words they use: ${dna.casualWords.slice(0,10).join(", ") || "standard"}
- Commas/sentence: ${dna.commaPerSentence} | Run-ons: ${dna.runOnRate}%
- Personal memories: ${dna.memoryCount} instances | Self-corrections: ${dna.selfCorrectionCount}
- Their transitions: ${dna.usedTransitions.slice(0,10).join(", ") || "standard"}
- Signature words: ${dna.signatureWords.join(", ") || "none"}
${ex.longestRunOn ? `\nEXAMPLE of their run-on style:\n"${ex.longestRunOn}"` : ""}
${ex.shortFragments?.length ? `\nEXAMPLE of their short punchy lines:\n${ex.shortFragments.map(s => `"${s}"`).join("\n")}` : ""}
${ex.personalSentences?.length ? `\nEXAMPLE of their personal sentences:\n${ex.personalSentences.map(s => `"${s}"`).join("\n")}` : ""}
${ex.mostDistinctive?.length ? `\nMOST DISTINCTIVE sentences (their voice at its clearest):\n${ex.mostDistinctive.map(s => `"${s}"`).join("\n")}` : ""}` : "";

  return {
    system: `You are a writing style cloner. Your only job is to reword text in a specific person's vocabulary and style. You keep all the facts, all the content, all the structure — you only change the wording to match how this person actually writes and talks.

Output ONLY the reworded text. No intro, no label. First word of your response = first word of the text.`,

    user: `Reword the text below so it sounds like this specific person wrote it.

Keep everything: all facts, all ideas, all paragraphs, same order. Only change the wording.

${fingerprintBlock}

THEIR VOICE PROFILE:
${rawBlueprint || "Use the fingerprint above to guide the voice."}

HOW TO REWORD:
- Swap formal/AI words for what this person would actually say
- Use their casual words, their contractions, their fillers
- Match their sentence construction from the examples above
- If they use run-ons — let sentences run. If they use fragments — use fragments.
- Use "I think", "I feel", "I remember" where it fits naturally
- Replace formal openers like "In the end," "At the same time," with how they actually connect ideas
- Keep the meaning exactly the same — just say it their way

Text to reword:
${text}`
  };
}

// ============================================================
// CALL 2 — GRUBBY PIPELINE PROMPT
// Takes the voice clone and humanizes it using grubby's exact
// pipeline: identify AI patterns → lexical substitution →
// syntactic variation → stylistic noise → punctuation cleanup
// ============================================================
function buildGrubbyPrompt(text, pushHarder = false) {
  const intensity = pushHarder ? "very aggressively" : "thoroughly";

  return {
    system: `You are an AI text humanizer. You take text that may still have AI statistical patterns and transform it so it reads as genuinely human — raising perplexity and burstiness, injecting natural human style, and removing AI fingerprints. You preserve all meaning while transforming the statistical properties of the text.

Output ONLY the humanized text. No intro, no label. First word = first word of text.`,

    user: `Humanize this text ${intensity} using this exact pipeline. Work through each step in order.

STEP 1 — IDENTIFY AI PATTERNS:
Before changing anything, mentally flag:
- Sentences that are too similar in length (uniform cadence)
- Formal or AI-typical words that survived the voice clone
- Clean summary sentences that wrap up paragraphs too neatly
- Sentence openers that still sound like AI transitions
- Missing contractions where a real person would use them

STEP 2 — LEXICAL SUBSTITUTION (raise perplexity):
Replace AI-prone words with casual human alternatives throughout.
These specific words are AI fingerprints — replace every one you find:
"accelerated" → "sped up" | "sophisticated" → "complex" or "tricky"
"leverage" → "use" | "demonstrate" → "show" | "utilize" → "use"
"significant" → "big" or "real" | "obtain" → "get"
"frequently" → "often" or "a lot" | "individuals" → "people"
"communicate" → "talk" or "say" | "comprehend" → "understand"
"subsequently" → "then" or "after that" | "implement" → "use"
"sufficient" → "enough" | "attempt" → "try" | "require" → "need"
"provide" → "give" | "assist" → "help" | "purchase" → "buy"
"numerous" → "a lot of" | "substantial" → "big" | "regarding" → "about"
"additionally" → delete or use "also" | "furthermore" → delete
"it is important to note" → delete | "it is worth noting" → delete
"in conclusion" → delete | "to summarize" → delete
"at the same time" → "but" or delete
"in the end" → delete | "on the other hand" → "but"
"as a result" → "so" | "therefore" → "so" | "thus" → "so"
"however" → "but" | "nevertheless" → "but still"
"in addition" → "also" or delete | "moreover" → delete
Also replace ANY word that feels formal, academic, or robotic with what a real person would say.

STEP 3 — SYNTACTIC VARIATION (raise burstiness):
Mix sentence lengths aggressively — this is the #1 thing GPTZero measures.
- Find paragraphs where sentences are similar length — disrupt them
- After a long sentence (20+ words), add a short punchy one (4-7 words)
- After a short sentence, let the next one run long and winding
- Split one medium sentence per paragraph into short + long at a natural break point like "because", "which", "and this", "so"
- Merge the final paragraph if it has 3+ short clean sentences — combine into one long flowing sentence
- Target sentence length sequence: something like 14, 5, 22, 8, 31, 6, 19, 4, 28

STEP 4 — STYLISTIC NOISE (inject human markers):
Add natural human style throughout:
- Add contractions wherever missing: "it is" → "it's", "they are" → "they're", "do not" → "don't", "I am" → "I'm", "would not" → "wouldn't", "cannot" → "can't", "that is" → "that's", "there is" → "there's"
- Strip ALL em-dashes (—) — replace with a comma or just remove
- Add occasional human fillers at natural points: "honestly", "you know", "I mean", "actually", "kind of", "sort of" — use sparingly, 1-2 per paragraph max
- Where a sentence starts with "This is" or "That is" — consider starting with "And honestly," or "I mean," instead
- Let one sentence per paragraph start with "And" or "But" for punch

STEP 5 — PUNCTUATION NORMALIZATION:
- Remove em-dashes — replace with comma or rewrite the clause
- Replace semicolons with periods or commas where natural
- Make sure contractions are used consistently
- Remove any overly formal punctuation patterns

PRESERVE: Every fact, idea, and argument from the original. Nothing added, nothing removed.

${pushHarder ? `
PUSH HARDER NOTE:
This is a second pass. The first version still had AI patterns. Be more aggressive:
- Find every sentence that still sounds clean and AI-polished — break it
- Find every formal word that survived — replace it
- Create more extreme length contrast — shorter shorts, longer longs
- Add more contractions and fillers` : ""}

Text to humanize:
${text}`
  };
}

// ============================================================
// REGEX CLEANUP
// Final pass — strips any remaining AI openers that
// survived both Groq calls. Pure deterministic code.
// ============================================================
function regexCleanup(text) {
  let r = text;

  // Strip AI sentence openers
  const openers = [
    /^In the end,?\s*/gim,
    /^At the same time,?\s*/gim,
    /^On the other hand,?\s*/gim,
    /^Because of this,?\s*/gim,
    /^As a result,?\s*/gim,
    /^For this reason,?\s*/gim,
    /^With this in mind,?\s*/gim,
    /^All in all,?\s*/gim,
    /^Overall,?\s*/gim,
    /^In conclusion,?\s*/gim,
    /^To conclude,?\s*/gim,
    /^In summary,?\s*/gim,
    /^To summarize,?\s*/gim,
    /^Ultimately,?\s*/gim,
    /^Furthermore,?\s*/gim,
    /^Moreover,?\s*/gim,
    /^Additionally,?\s*/gim,
    /^Subsequently,?\s*/gim,
    /^Consequently,?\s*/gim,
    /^Nevertheless,?\s*/gim,
    /^Having said that,?\s*/gim,
    /^That being said,?\s*/gim,
    /^It is worth noting that\s*/gim,
    /^Needless to say,?\s*/gim,
    /^First and foremost,?\s*/gim,
    /^Last but not least,?\s*/gim,
    /^At the end of the day,?\s*/gim,
    /^All things considered,?\s*/gim,
  ];

  for (const p of openers) r = r.replace(p, "");

  // Fix remaining AI phrases
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

  // Fix capitalization
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
  res.json({ status: "ok", model: MODEL, groqKeySet: !!GROQ_API_KEY, version: "11.0" });
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
    const { system, user } = buildAnalysisPrompt(samplesText, dna);
    const rawBlueprint = await callGroq(system, user, 2000, 0.7);

    res.json({ blueprint: JSON.stringify({ dna, rawBlueprint }) });
  } catch (err) {
    console.error("/analyze error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /rewrite
// Call 1: Voice clone — reword in person's style
// Call 2: Grubby pipeline — humanize statistically
// Final:  Regex cleanup
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

    // Call 1 — Voice clone
    const { system: vs, user: vu } = buildVoiceClonePrompt(text, dna, rawBlueprint);
    const voiceClone = await callGroq(vs, vu, 2000, 0.9);

    // Call 2 — Grubby humanization
    const { system: gs, user: gu } = buildGrubbyPrompt(voiceClone, false);
    const humanized = await callGroq(gs, gu, 2000, 1.0);

    // Final regex cleanup
    const rewritten = regexCleanup(humanized);
    res.json({ rewritten });
  } catch (err) {
    console.error("/rewrite error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============================================================
// POST /humanize — second pass, pushed harder
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

    // Call 1 — Voice clone again on the already-rewritten text
    const { system: vs, user: vu } = buildVoiceClonePrompt(text, dna, rawBlueprint);
    const voiceClone = await callGroq(vs, vu, 2000, 1.0);

    // Call 2 — Grubby pushed harder
    const { system: gs, user: gu } = buildGrubbyPrompt(voiceClone, true);
    const humanized = await callGroq(gs, gu, 2000, 1.1);

    const rewritten = regexCleanup(humanized);
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
  console.log(`MyVoice v11.0 running on port ${PORT}`);
  if (!GROQ_API_KEY) console.warn("WARNING: GROQ_API_KEY not set");
});

