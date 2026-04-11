// ============================================================
// MyVoice — Render.com Backend Server
// Holds your Groq API key privately.
// Two endpoints:
//   POST /analyze  — forensic style analysis of writing samples
//   POST /rewrite  — rewrite text using a saved style blueprint
// ============================================================

const http = require("http");
const https = require("https");

const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const MODEL = "llama3-70b-8192"; // best free Groq model

// ---- CORS headers — allow requests from ChatGPT tabs ----
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Content-Type": "application/json"
};

// ---- Call Groq API ----
async function callGroq(systemPrompt, userPrompt, maxTokens = 2000) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: MODEL,
      max_tokens: maxTokens,
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
      res.on("data", (chunk) => data += chunk);
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data);
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

// ---- HTTP Server ----
const server = http.createServer(async (req, res) => {

  // Handle preflight
  if (req.method === "OPTIONS") {
    res.writeHead(200, CORS);
    res.end();
    return;
  }

  // Only accept POST
  if (req.method !== "POST") {
    res.writeHead(405, CORS);
    res.end(JSON.stringify({ error: "Method not allowed" }));
    return;
  }

  // Read request body
  let body = "";
  req.on("data", (chunk) => body += chunk);
  req.on("end", async () => {
    try {
      const data = JSON.parse(body);

      // ---- /analyze — forensic style analysis ----
      if (req.url === "/analyze") {
        const { samples } = data;
        if (!samples || !samples.length) {
          res.writeHead(400, CORS);
          res.end(JSON.stringify({ error: "No samples provided" }));
          return;
        }

        // Format samples for the prompt
        const samplesText = samples.map((s, i) =>
          `Sample ${i + 1} (${s.label || "Untitled"}):\n${s.text}`
        ).join("\n\n---\n\n");

        const systemPrompt = `You are the world's best forensic writing-style analyst and imitator. Your only job is to reverse-engineer someone's EXACT personal writing style from real samples, then create a bulletproof blueprint that another LLM (Groq) can use to perfectly imitate them in real time. Be brutally specific. Quote actual phrases from their samples as evidence for every single point. Do NOT be generic or vague. Output ONLY what is requested — nothing before or after.`;

        const userPrompt = `Here are my real writing samples:\n\n${samplesText}\n\nStep 1: Deep Forensic Analysis\nAnalyze every sample line-by-line. Quote exact phrases as evidence. Extract and list:\n- Voice & Personality (exact vibe)\n- Vocabulary & Word Choice (overused words, slang, filler words, signature phrases)\n- Sentence Structure & Rhythm (lengths, fragments, run-ons, starting with and/but/so, questions as statements)\n- Paragraph Flow & Transitions\n- Punctuation & Formatting Habits (emojis, CAPS, dashes, ellipsis, parentheses, line breaks)\n- Grammar Quirks broken on purpose\n- Humor, Tone Shifts, Energy Level\n- All personal tics and patterns\n\nStep 2: Create MY STYLE BLUEPRINT\nOutput ONLY this exact format (nothing before or after):\n\n**MY STYLE BLUEPRINT**\n- Always:\n- Never:\n- Vocabulary rules:\n- Sentence rules:\n- Flow & structure rules:\n- Punctuation & formatting rules:\n- Tone & personality rules:\n- Unique quirks to copy exactly:\n\nMake every bullet insanely specific and technical so Groq can follow it with zero leakage of its own style. Use real quoted examples in every bullet.\n\nStep 3: Groq Rewrite System Prompt\nCreate the exact system prompt my Chrome extension will send to Groq every time it rewrites text. It must:\n- Bake the entire blueprint directly into it\n- Be short, strict, and ruthless (Groq is fast but needs zero room for interpretation)\n- Force Groq to output ONLY the rewritten text, nothing else\n- Preserve every fact and meaning from the original text\n- Sound 100% like me, zero AI slime\n\nOutput it in this exact format:\n**GROQ REWRITE SYSTEM PROMPT:**\n[the system prompt here]`;

        const blueprint = await callGroq(systemPrompt, userPrompt, 2000);

        res.writeHead(200, CORS);
        res.end(JSON.stringify({ blueprint }));
        return;
      }

      // ---- /rewrite — rewrite text using style blueprint ----
      if (req.url === "/rewrite") {
        const { text, blueprint, groqSystemPrompt } = data;
        if (!text || !blueprint) {
          res.writeHead(400, CORS);
          res.end(JSON.stringify({ error: "Missing text or blueprint" }));
          return;
        }

        // Use the AI-generated Groq system prompt if available
        // Otherwise fall back to a generic one
        const systemPrompt = groqSystemPrompt ||
          `You are a writing style imitator. Rewrite the text given to you to perfectly match the style blueprint below. Keep ALL the same information and meaning — only change how it sounds. Output ONLY the rewritten text. No explanations, no preamble, nothing else.\n\nSTYLE BLUEPRINT:\n${blueprint}`;

        const userPrompt = `Rewrite this text in my exact style. Output only the rewritten text, nothing else:\n\n${text}`;

        const rewritten = await callGroq(systemPrompt, userPrompt, 1500);

        res.writeHead(200, CORS);
        res.end(JSON.stringify({ rewritten }));
        return;
      }

      // ---- health check ----
      if (req.url === "/health") {
        res.writeHead(200, CORS);
        res.end(JSON.stringify({ status: "ok" }));
        return;
      }

      res.writeHead(404, CORS);
      res.end(JSON.stringify({ error: "Not found" }));

    } catch (err) {
      res.writeHead(500, CORS);
      res.end(JSON.stringify({ error: err.message }));
    }
  });
});

server.listen(PORT, () => {
  console.log(`MyVoice server running on port ${PORT}`);
  if (!GROQ_API_KEY) console.warn("WARNING: GROQ_API_KEY not set!");
});

