// Server-side proxy to Gemini. The API key lives only here, as a Vercel
// environment variable (GEMINI_API_KEY) - it is never sent to the browser.
// The client posts { system, messages: [{role, content}, ...] } and gets
// back a stream of newline-delimited JSON objects: {"content":"...piece..."}
//
// Runs on the Edge Runtime rather than Node.js: this function has zero npm
// dependencies and only ever touches fetch/TextEncoder/TextDecoder (all
// Web-standard), so it's a clean fit - Edge functions run in a V8 isolate
// with near-zero cold start versus Node's real (if small) init cost. Pinned
// to iad1 rather than left to run at whichever edge location is nearest the
// requester, so the outbound hop to Gemini stays on the exact network path
// already measured by hand (~1.5-1.6s end to end) - Vercel's other edge
// locations are generally well-peered to Google's API network too, but that
// wasn't worth gambling on for every geography without measuring it first.
export const config = { runtime: 'edge', regions: ['iad1'] };

export default async function handler(request) {
  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'content-type': 'application/json' }
    });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'Server misconfigured: GEMINI_API_KEY is not set' }), {
      status: 500,
      headers: { 'content-type': 'application/json' }
    });
  }

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return new Response(JSON.stringify({ error: 'Invalid JSON body' }), {
      status: 400,
      headers: { 'content-type': 'application/json' }
    });
  }

  const system = String((body && body.system) || '');
  const messages = Array.isArray(body && body.messages) ? body.messages : [];
  const contents = messages.map(function (m) {
    return {
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: String(m.content || '') }]
    };
  });

  const model = process.env.GEMINI_MODEL || 'gemini-3.5-flash-lite';
  const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model +
    ':streamGenerateContent?alt=sse&key=' + apiKey;

  let upstream;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: system }] },
        contents: contents,
        // Trimmed from 220: this is a ceiling, not a multiplier, so it only
        // costs time on turns where the model ignores the "1-3 short
        // sentences" persona instruction and runs long - a tail-latency
        // guard rather than something that changes the median reply.
        generationConfig: { maxOutputTokens: 150 }
      })
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: 'Could not reach Gemini: ' + e.message }), {
      status: 502,
      headers: { 'content-type': 'application/json' }
    });
  }

  if (!upstream.ok || !upstream.body) {
    const errText = await upstream.text().catch(function () { return ''; });
    return new Response(JSON.stringify({ error: 'Gemini ' + upstream.status + ': ' + errText.slice(0, 500) }), {
      status: upstream.status || 502,
      headers: { 'content-type': 'application/json' }
    });
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder('utf-8');
  const reader = upstream.body.getReader();

  const stream = new ReadableStream({
    async start(controller) {
      let carry = '';
      try {
        while (true) {
          const chunk = await reader.read();
          if (chunk.done) break;
          carry += decoder.decode(chunk.value, { stream: true });
          const lines = carry.split('\n');
          carry = lines.pop();
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line.startsWith('data:')) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr || jsonStr === '[DONE]') continue;
            let obj;
            try { obj = JSON.parse(jsonStr); } catch (e) { continue; }
            const cand = obj && obj.candidates && obj.candidates[0];
            const part = cand && cand.content && cand.content.parts && cand.content.parts[0];
            const text = part && part.text;
            if (text) controller.enqueue(encoder.encode(JSON.stringify({ content: text }) + '\n'));
          }
        }
      } catch (e) {
        // client disconnected mid-stream; nothing to do
      }
      controller.close();
    }
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-cache'
    }
  });
}
