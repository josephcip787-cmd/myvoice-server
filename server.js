// ============================================================
// MyVoice for ChatGPT — Backend Server v11.1
// Base: v11.0 (the one that got 5 human sentences)
// Fix: conjunctions starting sentences + over-conversational tone
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
  const openerFiller = cleanSentences.filter(s => /^(Well,|I mean,|Honestly,|Look,|See,|Now,|Okay,|Right,|Actually,)/i.test(s)).length;
  const openerQuestion = cleanSentences.filter(s => /\?$/.test(s.trim()) && /^(What|How|Why|When|Where|Who|Is|Are|Do|Does|Can|Did)\s/i.test(s)).length;

  const firstWords = {};
  cleanSentences.forEach(s => {
    const fw = s.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "");
    if (fw) firstWords[fw] = (firstWords[fw] || 0) + 1;
  });
  const topOpener = Object.entries(firstWords).sort((a, b) => b[1] - a[1])[0]?.[0] || "i";

  const transitionList = ["so","but","and","because","then","though","although","which means","that's why","this is why","the thing is","i mean","you know","honestly","actually","basically","like","anyway","still","also","plus","well","i guess","i think","i feel","i remember","and so","but then","so then","and then","but also"];
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
    /^(Well|I mean|Honestly)/i.test(firstSentence) ? "starts with a filler" :
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
    .slice(0, 15)
    .map(([w]) => w);

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

  const longestRunOn = cleanSentences.filter(s => (s.match(/,/g) || []).length >= 3).sort((a, b) => b.split(/\s+/).length - a.split(/\s+/).length)[0] || "";
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
    signatureWords, topPhrases,
    exampleSentences: { longestRunOn, shortFragments, personalSentences, conjunctionSentences, selfCorrectionSentences, mostDistinctive }
  };
}

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
Describe this writer's exact personality in 3 sentences. What energy do they have? Quote one phrase that captures them perfectly.

VOCABULARY FINGERPRINT:
What words are distinctly theirs? What formal words do they never use? What casual words appear constantly?

SENTENCE CONSTRUCTION:
How do they build sentences? Run-ons? Fragments? Self-interruptions? Quote 2 actual examples.

THINKING PATTERN:
Linear or associative? Point first or build to it? Mix opinion and fact? Quote one example.

PERSONAL STORY PATTERN:
Do they use personal memories? How specific? Quote an example.

WHAT THEY NEVER DO:
List 5 specific absent habits.

THEIR TRANSITIONS:
Quote exact transition words/phrases from the samples only.

HOW THEY OPEN AND CLOSE:
Quote an actual opening and closing.

CLONING INSTRUCTIONS:
3 paragraphs to the AI that will rewrite as this person covering: vocabulary, sentence construction, most important thing to nail.`
  };
}

function buildVoiceClonePrompt(text, dna, rawBlueprint) {
  const ex = dna?.exampleSentences || {};

  const fingerprintBlock = dna ? `
THEIR MEASURED WRITING PATTERNS:
- Sentence length: avg ${dna.avgLen} words, range ${dna.minLen}–${dna.maxLen}
- Starts with "I": ${dna.openerIRate}% | Conjunctions: ${dna.openerConjRate}% | Fillers: ${dna.openerFillerRate}%
- Formality: ${dna.formalityScore}/100
- Contractions they use: ${dna.usedContractions.join(", ") || "few"}
- Casual words they use: ${dna.casualWords.slice(0,10).join(", ") || "standard"}
- Commas/sentence: ${dna.commaPerSentence} | Run-ons: ${dna.runOnRate}%
- Personal memories: ${dna.memoryCount} | Self-corrections: ${dna.selfCorrectionCount}
- Their transitions: ${dna.usedTransitions.slice(0,10).join(", ") || "standard"}
- Signature words: ${dna.signatureWords.join(", ") || "none"}
${ex.longestRunOn ? `\nEXAMPLE run-on style:\n"${ex.longestRunOn}"` : ""}
${ex.shortFragments?.length ? `\nEXAMPLE short lines:\n${ex.shortFragments.map(s => `"${s}"`).join("\n")}` : ""}
${ex.personalSentences?.length ? `\nEXAMPLE personal sentences:\n${ex.personalSentences.map(s => `"${s}"`).join("\n")}` : ""}
${ex.mostDistinctive?.length ? `\nMOST DISTINCTIVE sentences:\n${ex.mostDistinctive.map(s => `"${s}"`).join("\n")}` : ""}` : "";

  return {
    system: `You are a writing style cloner. Reword text in a specific person's vocabulary and style. Keep all facts, all content, all structure — only change the wording to match how this person writes.

Output ONLY the reworded text. No intro, no label. First word = first word of text.`,

    user: `Reword the text below so it sounds like this specific person wrote it.
Keep everything: all facts, all ideas, all paragraphs, same order. Only change the wording.

${fingerprintBlock}

THEIR VOICE PROFILE:
${rawBlueprint || "Use the fingerprint above."}

HOW TO REWORD:
- Swap formal/AI words for what this person would actually say
- Use their contractions and casual vocabulary
- Match their sentence construction from the examples
- If they use run-ons — let sentences run. If they use fragments — use them.
- Use "I think", "I feel", "I remember" where it fits naturally
- Replace AI openers like "In the end," "At the same time," with their actual transitions

CRITICAL — CONJUNCTION RULE:
Do NOT start sentences with "And,", "But,", "So," as standalone openers.
GPTZero flags "Artificial Simplicity" when sentences start with conjunctions repeatedly.
Instead, use conjunctions MID-SENTENCE: "Words are small, but they carry real weight"
NOT: "But words carry real weight."
Maximum ONE conjunction sentence opener in the entire piece.

CRITICAL — CONVERSATIONAL BALANCE:
Match this person's natural tone — don't make it MORE casual than their samples show.
Fillers like "like", "you know", "I mean" should appear at the same rate as their samples.
Do not inject extra casual markers beyond what their fingerprint shows.

Text to reword:
${text}`
  };
}

function buildGrubbyPrompt(text, pushHarder = false) {
  return {
    system: `You are an AI text humanizer. Transform text to read as genuinely human — raising perplexity and burstiness, injecting natural human style, removing AI fingerprints. Preserve all meaning.

Output ONLY the humanized text. No intro, no label. First word = first word of text.`,

    user: `Humanize this text using the grubby pipeline. Work through each step.

STEP 1 — IDENTIFY AI PATTERNS:
Flag: uniform sentence lengths, formal AI words, clean summary sentences, AI transition openers, missing contractions.

STEP 2 — LEXICAL SUBSTITUTION (raise perplexity):
Replace every AI-prone word with a casual human alternative:
"accelerated"→"sped up" | "sophisticated"→"complex" | "leverage"→"use"
"demonstrate"→"show" | "utilize"→"use" | "significant"→"big" or "real"
"obtain"→"get" | "frequently"→"often" | "individuals"→"people"
"communicate"→"talk" | "comprehend"→"understand" | "subsequently"→"then"
"implement"→"use" | "sufficient"→"enough" | "attempt"→"try"
"require"→"need" | "provide"→"give" | "assist"→"help" | "purchase"→"buy"
"numerous"→"a lot of" | "substantial"→"big" | "regarding"→"about"
"additionally"→"also" or delete | "furthermore"→delete | "moreover"→delete
"it is important to note"→delete | "in conclusion"→delete
"at the same time"→"but" | "in the end"→delete | "on the other hand"→"but"
"as a result"→"so" | "therefore"→"so" | "thus"→"so" | "however"→"but"
"in addition"→"also" | "nevertheless"→"but still"
Replace ANY formal/robotic word with what a real person would say.

STEP 3 — SYNTACTIC VARIATION (raise burstiness):
Mix sentence lengths aggressively:
- Find paragraphs with uniform sentence lengths — disrupt them
- After a long sentence (20+ words), add a short one (4-7 words)
- After a short sentence, let the next run long
- Split one medium sentence per paragraph at a natural break ("because", "which", "so")
- Merge the final paragraph if it has 3+ short clean sentences into one long flowing one
- Target: varied sequence like 14, 5, 22, 8, 31, 6, 19, 4, 28

STEP 4 — STYLISTIC NOISE (inject human markers — carefully):
- Add contractions everywhere natural: "it is"→"it's", "they are"→"they're", "do not"→"don't", "I am"→"I'm", "would not"→"wouldn't", "cannot"→"can't", "that is"→"that's", "there is"→"there's"
- Strip ALL em-dashes (—) — replace with comma or remove
- Fillers: embed naturally MID-SENTENCE not at sentence starts. "words are, honestly, more powerful than people think" not "Honestly, words are powerful"
- NEVER start sentences with "So, like," or "And, I mean," or "But, you know," — GPTZero flags these as Artificial Simplicity
- NEVER end sentences with "you know?" — embed mid-sentence instead
- Maximum 1-2 filler words per paragraph, placed naturally mid-sentence

STEP 5 — PUNCTUATION NORMALIZATION:
- Remove em-dashes → comma or nothing
- Semicolons → period or comma
- Keep contractions consistent
- Remove formal punctuation patterns

${pushHarder ? `
PUSH HARDER:
Second pass — be more aggressive:
- Break every sentence that still sounds AI-polished
- Replace every formal word that survived
- More extreme length contrast
- More contractions throughout` : ""}

PRESERVE: Every fact, idea, argument. Nothing added, nothing removed.

Text:
${text}`
  };
}

function regexCleanup(text) {
  let r = text;

  const openers = [
    /^In the end,?\s*/gim, /^At the same time,?\s*/gim,
    /^On the other hand,?\s*/gim, /^Because of this,?\s*/gim,
    /^As a result,?\s*/gim, /^For this reason,?\s*/gim,
    /^With this in mind,?\s*/gim, /^All in all,?\s*/gim,
    /^Overall,?\s*/gim, /^In conclusion,?\s*/gim,
    /^To conclude,?\s*/gim, /^In summary,?\s*/gim,
    /^To summarize,?\s*/gim, /^Ultimately,?\s*/gim,
    /^Furthermore,?\s*/gim, /^Moreover,?\s*/gim,
    /^Additionally,?\s*/gim, /^Subsequently,?\s*/gim,
    /^Consequently,?\s*/gim, /^Nevertheless,?\s*/gim,
    /^Having said that,?\s*/gim, /^That being said,?\s*/gim,
    /^It is worth noting that\s*/gim, /^Needless to say,?\s*/gim,
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

  // Fix sentences starting with conjunctions — move conjunction mid-sentence
  // "And words are powerful." → "Words are, and they're powerful."
  // Actually just capitalize cleanly after removals
  r = r
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,!?])/g, "$1")
    .replace(/([.!?]\s+)([a-z])/g, (m, p1, p2) => p1 + p2.toUpperCase())
    .replace(/^([a-z])/, m => m.toUpperCase())
    .trim();

  return r;
}

app.get("/health", (req, res) => {
  res.json({ status: "ok", model: MODEL, groqKeySet: !!GROQ_API_KEY, version: "11.1" });
});

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

app.post("/rewrite", async (req, res) => {
  try {
    const { text, blueprint } = req.body;
    if (!text || !blueprint) return res.status(400).json({ error: "Missing text or blueprint" });

    let dna = null, rawBlueprint = blueprint;
    try {
      const parsed = JSON.parse(blueprint);
      if (parsed.dna && parsed.rawBlueprint) { dna = parsed.dna; rawBlueprint = parsed.rawBlueprint; }
    } catch (e) {}

    const { system: vs, user: vu } = buildVoiceClonePrompt(text, dna, rawBlueprint);
    const voiceClone = await callGroq(vs, vu, 2000, 0.9);

    const { system: gs, user: gu } = buildGrubbyPrompt(voiceClone, false);
    const humanized = await callGroq(gs, gu, 2000, 1.0);

    const rewritten = regexCleanup(humanized);
    res.json({ rewritten });
  } catch (err) {
    console.error("/rewrite error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post("/humanize", async (req, res) => {
  try {
    const { text, blueprint } = req.body;
    if (!text || !blueprint) return res.status(400).json({ error: "Missing text or blueprint" });

    let dna = null, rawBlueprint = blueprint;
    try {
      const parsed = JSON.parse(blueprint);
      if (parsed.dna && parsed.rawBlueprint) { dna = parsed.dna; rawBlueprint = parsed.rawBlueprint; }
    } catch (e) {}

    const { system: vs, user: vu } = buildVoiceClonePrompt(text, dna, rawBlueprint);
    const voiceClone = await callGroq(vs, vu, 2000, 1.0);

    const { system: gs, user: gu } = buildGrubbyPrompt(voiceClone, true);
    const humanized = await callGroq(gs, gu, 2000, 1.1);

    const rewritten = regexCleanup(humanized);
    res.json({ rewritten });
  } catch (err) {
    console.error("/humanize error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`MyVoice v11.1 running on port ${PORT}`);
  if (!GROQ_API_KEY) console.warn("WARNING: GROQ_API_KEY not set");
});

