// ============================================================
// MyVoice — Render Backend v3.0
// Groq-powered: analyze, chat, generate, rewrite, quiz
// Deploy to Render. Set GROQ_API_KEY as env variable.
// ============================================================

const express = require("express");
const cors    = require("cors");
const Groq    = require("groq-sdk");

const app  = express();
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const PORT = process.env.PORT || 3000;
const MODEL = "llama-3.3-70b-versatile"; // Best quality on Groq free tier

app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ---- Health check ----
app.get("/", (req, res) => res.json({ status: "MyVoice backend v3.0 running" }));

// ================================================================
// POST /analyze
// Input:  { samples: [{label, text}] }
// Output: { blueprint: string }
// Groq reads all writing samples and produces a detailed style blueprint
// ================================================================
app.post("/analyze", async (req, res) => {
  const { samples } = req.body;
  if (!samples || samples.length === 0) return res.status(400).json({ error: "No samples" });

  const combined = samples.map((s) => `[${s.label}]\n${s.text}`).join("\n\n---\n\n");

  try {
    const completion = await groq.chat.completions.create({
      model: MODEL,
      max_tokens: 1200,
      messages: [
        {
          role: "system",
          content: `You are a writing style analyst. Analyze writing samples and produce a precise, actionable style blueprint. Be specific and concrete — no vague adjectives. Focus on patterns that an AI can actually replicate.`
        },
        {
          role: "user",
          content: `Analyze these writing samples and create a detailed style blueprint:\n\n${combined}\n\nWrite a style blueprint covering:\n1. Sentence rhythm and length patterns\n2. Tone (formality, warmth, confidence, humor)\n3. How they open sentences and paragraphs\n4. Punctuation habits (dashes, ellipsis, exclamations, comma density)\n5. Transition words and connectors they favor\n6. Vocabulary level and word choice tendencies\n7. Whether they use lists, questions, asides in parentheses\n8. 3-5 signature phrases or patterns\n9. What makes their voice UNIQUE and recognizable\n\nFormat as a clear blueprint an AI can follow when writing new content.`
        }
      ]
    });

    const blueprint = completion.choices[0]?.message?.content || "";
    res.json({ blueprint });
  } catch (err) {
    console.error("/analyze error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// POST /chat
// Input:  { messages: [{role, content}], blueprint?: string, context?: string }
// Output: { reply: string }
// Conversational AI that writes in the user's voice
// ================================================================
app.post("/chat", async (req, res) => {
  const { messages, blueprint, context } = req.body;
  if (!messages || messages.length === 0) return res.status(400).json({ error: "No messages" });

  const systemParts = [
    `You are MyVoice, a personal AI writing assistant. Your job is to write content EXACTLY as the user would write it — capturing their voice, rhythm, style, and personality.`,
    ``,
    `CRITICAL RULES:`,
    `- Write as if YOU are the user. Match their voice completely.`,
    `- Do NOT sound like a generic AI assistant. Sound like a specific human.`,
    `- If asked to write something, just write it — no preamble, no "Here's your email:", no meta-commentary.`,
    `- If you need to clarify something, ask one short question.`,
    `- Keep outputs human and natural. No AI-isms like "Certainly!", "Great question!", "I'd be happy to".`
  ];

  if (context) systemParts.push(``, `CURRENT CONTEXT: ${context}`);

  if (blueprint) {
    systemParts.push(``, `USER'S WRITING STYLE — follow this precisely:`, blueprint);
  } else {
    systemParts.push(``, `No style profile available. Write in a clear, natural, conversational style that doesn't sound like an AI.`);
  }

  try {
    const completion = await groq.chat.completions.create({
      model: MODEL,
      max_tokens: 1000,
      messages: [
        { role: "system", content: systemParts.join("\n") },
        ...messages.slice(-20) // Keep last 20 messages for context
      ]
    });

    const reply = completion.choices[0]?.message?.content || "";
    res.json({ reply });
  } catch (err) {
    console.error("/chat error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// POST /generate
// Input:  { prompt: string, blueprint?: string, context?: string }
// Output: { text: string }
// One-shot generation (used for quick writes, not chat)
// ================================================================
app.post("/generate", async (req, res) => {
  const { prompt, blueprint, context } = req.body;
  if (!prompt) return res.status(400).json({ error: "No prompt" });

  const systemParts = [
    `You are a writing assistant that writes exactly as the user would. Output ONLY the requested content — no intros, no labels, no explanations.`,
  ];

  if (context) systemParts.push(`Context: ${context}`);
  if (blueprint) systemParts.push(`\nUser's writing style:\n${blueprint}`);

  try {
    const completion = await groq.chat.completions.create({
      model: MODEL,
      max_tokens: 800,
      messages: [
        { role: "system", content: systemParts.join("\n") },
        { role: "user", content: prompt }
      ]
    });

    const text = completion.choices[0]?.message?.content || "";
    res.json({ text });
  } catch (err) {
    console.error("/generate error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// POST /rewrite
// Input:  { text: string, blueprint?: string, instruction?: string }
// Output: { rewritten: string }
// Rewrite existing text in the user's voice
// ================================================================
app.post("/rewrite", async (req, res) => {
  const { text, blueprint, instruction } = req.body;
  if (!text) return res.status(400).json({ error: "No text to rewrite" });

  const userMsg = instruction
    ? `Rewrite this text with the following adjustment: ${instruction}\n\nText:\n${text}`
    : `Rewrite this text in my voice. Preserve all the information but make it sound exactly like how I write:\n\n${text}`;

  const systemParts = [
    `You rewrite text to match the user's exact writing voice. Output ONLY the rewritten text — no labels or commentary.`
  ];
  if (blueprint) systemParts.push(`\nUser's voice:\n${blueprint}`);

  try {
    const completion = await groq.chat.completions.create({
      model: MODEL,
      max_tokens: 1000,
      messages: [
        { role: "system", content: systemParts.join("\n") },
        { role: "user", content: userMsg }
      ]
    });

    const rewritten = completion.choices[0]?.message?.content || "";
    res.json({ rewritten });
  } catch (err) {
    console.error("/rewrite error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// POST /quiz
// Input:  { original: string, option_a: string, option_b: string, blueprint: string }
// Output: { result: { winner, feedback } }
// Evaluates which quiz option better matches the user's style
// ================================================================
app.post("/quiz", async (req, res) => {
  const { original, option_a, option_b, blueprint } = req.body;

  try {
    const completion = await groq.chat.completions.create({
      model: MODEL,
      max_tokens: 200,
      messages: [
        {
          role: "system",
          content: `You are a writing style evaluator. Given a user's style blueprint, pick which of two options better matches their natural writing style. Respond in JSON: {"winner":"a" or "b","feedback":"one short sentence why"}`
        },
        {
          role: "user",
          content: `Style blueprint:\n${blueprint}\n\nScenario: ${original}\n\nOption A: ${option_a}\n\nOption B: ${option_b}\n\nWhich matches the user's style better?`
        }
      ]
    });

    let result;
    try {
      const raw = completion.choices[0]?.message?.content || "{}";
      result = JSON.parse(raw.replace(/```json|```/g, "").trim());
    } catch {
      result = { winner: "a", feedback: "Both options seem reasonable for this style." };
    }

    res.json({ result });
  } catch (err) {
    console.error("/quiz error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---- Start ----
app.listen(PORT, () => {
  console.log(`MyVoice backend v3.0 running on port ${PORT}`);
});

