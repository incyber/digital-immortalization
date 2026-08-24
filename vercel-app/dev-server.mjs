// Local runner for the Vercel app, so it can be exercised on a dev machine
// without deploying and without linking the folder to a Vercel project.
//
// It does the two things the Vercel platform does in production:
//   1. Serves everything in public/ as static files (public/index.html at "/").
//   2. Routes POST /api/chat to the handler exported by api/chat.js.
//
// api/chat.js is written against Web standards (Request, Response,
// ReadableStream, fetch), which Node exposes as globals - so the same file that
// runs on the Edge runtime in production runs here unmodified. Nothing in this
// file is used in production; Vercel never reads it.
//
// Config: GEMINI_API_KEY is required by api/chat.js and is read from the
// environment or from vercel-app/.env.local (gitignored). GEMINI_MODEL
// optionally overrides the model name that api/chat.js defaults to.
//
// Run:  node dev-server.mjs      then open http://localhost:3000

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const PUBLIC_DIR = join(ROOT, 'public');
const PORT = Number(process.env.PORT) || 3000;

// Load vercel-app/.env.local if present - the local stand-in for the
// environment variables that are set on the Vercel project in production.
const envFile = join(ROOT, '.env.local');
if (existsSync(envFile)) process.loadEnvFile(envFile);

const { default: chatHandler } = await import('./api/chat.js');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon'
};

// Turn an incoming Node request into the Web Request that the handler expects.
async function toWebRequest(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return new Request('http://localhost:' + PORT + req.url, {
    method: req.method,
    headers: req.headers,
    body: chunks.length ? Buffer.concat(chunks) : undefined
  });
}

// Copy a Web Response back onto the Node response, streaming the body so the
// token-by-token output of /api/chat still arrives incrementally.
async function sendWebResponse(webRes, res) {
  res.writeHead(webRes.status, Object.fromEntries(webRes.headers));
  if (!webRes.body) return res.end();
  Readable.fromWeb(webRes.body).pipe(res);
}

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');

    if (url.pathname === '/api/chat') {
      const webRes = await chatHandler(await toWebRequest(req));
      return sendWebResponse(webRes, res);
    }

    // Static files. normalize() strips any "../" so the served tree stays
    // inside public/.
    const rel = url.pathname === '/' ? 'index.html' : normalize(url.pathname).replace(/^([/\\.])+/, '');
    const filePath = join(PUBLIC_DIR, rel);
    const body = await readFile(filePath);
    res.writeHead(200, { 'content-type': MIME[extname(filePath)] || 'application/octet-stream' });
    res.end(body);
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      res.writeHead(404, { 'content-type': 'text/plain' });
      return res.end('Not found');
    }
    res.writeHead(500, { 'content-type': 'text/plain' });
    res.end('Server error: ' + (err && err.message));
  }
}).listen(PORT, () => {
  if (!process.env.GEMINI_API_KEY) {
    console.log('Warning: GEMINI_API_KEY is not set - /api/chat will return 500.');
    console.log('Set it in vercel-app/.env.local as GEMINI_API_KEY=...');
  }
  console.log('Running on http://localhost:' + PORT + '/');
});
