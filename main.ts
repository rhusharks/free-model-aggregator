// free-model-aggregator-02's starter kit: one server, two client wires, and a real free model behind
// every answer.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth a fraction of the score: use it to see the shape of a trial, then
// replace its routing, its failover and its catalog with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's standard
//   library alone (nothing is installed; TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the key you pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its weight
//       (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a unique token
//       the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This starter sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the published ratings. It takes the first free model in the brief, in
//     the brief's order, that can meet the request's requirements. TODO(quality): rank the free
//     candidates by rating_milli at the prompt's requirements and prefer the best.
//   - failover (10%): a gateway call answered with a 4xx or a 5xx becomes a 502 to the client, and that
//     prompt is lost. TODO(failover): retry the prompt on the next free model that meets it, and count
//     every call it costs, because the tie-break is the number of gateway calls.
//   - rate limits: after a 429 it keeps picking the same first model, so one rate-limited model takes
//     every later prompt down with it. TODO(cooldowns): track each model's refusals and back off per
//     model, and never spend a call on a model that is cooling down.
//   - the catalog (5%): it lists every free model the brief names, which is exactly the fourteen ids
//     this column scores. TODO(catalog): keep it exact — register that same set and answer GET /v1/models
//     with it on both auth headers; the column checks the register body and both answers against exactly
//     those fourteen ids, so listing fewer loses it as surely as listing more.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Serve one prompt: the caller's own failover loop picks the best-rated free model that meets the
 * request and retries the next best on a refusal (see serveWithFailover).
 */

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Refusals this process has seen per model id: one refusal backs a model off for the rest of the trial. */
const coolingDown = new Set<string>();
/** The last model that served a requirement class well: tried first for the same class, fewer wasted calls. */
const lastGoodByClass = new Map<string, string>();
/** Rows for the optional `x-makefun-upstream-failures` trail, in the order the calls refused. */
const failureTrail: { status: number; code: string; model: string | null }[] = [];

/** Record one refused gateway call: cool the model down and add a trail row. */
function recordRefusal(model: string | null, status: number, code: string): void {
  // A 429 is the account's shared rate limit, not that model's fault: cool it down only for real refusals.
  if (model !== null && status !== 429) coolingDown.add(model);
  if (failureTrail.length < 8) failureTrail.push({ status, code, model });
}

/** The trail header's value, or undefined when nothing refused. */
function trailHeader(): string | undefined {
  if (failureTrail.length === 0) return undefined;
  try {
    const text = JSON.stringify(failureTrail);
    return text.length <= 8192 ? text : undefined;
  } catch {
    return undefined;
  }
}

/** The best-rated free model for a prompt, skipping cooling-down ones; null when none is left. */
function routeFor(requires: Requirement): ModelRow | null {
  let best: ModelRow | null = null;
  for (const model of freeModels(brief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    if (coolingDown.has(model.id)) continue;
    if (best === null || model.rating_milli > best.rating_milli) best = model;
  }
  return best;
}

/** Every free model that meets a request's requirements, best-rated first (failover order). */
function routesFor(requires: Requirement): ModelRow[] {
  const candidates = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  candidates.sort((a, b) => b.rating_milli - a.rating_milli);
  return candidates;
}

/** A refusal body's error code, in either wire's shape. */
function errorCode(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      const err = isRecord(value.error) ? value.error : value;
      if (typeof err.code === 'string') return err.code;
      if (typeof err.type === 'string') return err.type;
    }
  } catch {
    // not JSON: fall through to the generic code
  }
  return 'upstream_error';
}

/** A stable key for one prompt's requirement set: the memo's class. */
function requirementClass(requires: Requirement): string {
  return JSON.stringify({ tools: requires.tools === true, vision: requires.vision === true, json: requires.json === true });
}

/**
 * Serve one prompt with failover: dial the best-rated model that meets the request, and on a 4xx or
 * 5xx refuse-and-retry the next best-rated one that does, until one answers or the candidates run
 * out. Deterministic: the order is the ratings and the brief's order, never the wall clock.
 */
async function serveWithFailover(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  requires: Requirement,
  gatewayBody: (model: ModelRow) => Json,
  deliver: (upstream: Response, model: ModelRow) => Promise<void>,
  attempts = 0,
): Promise<void> {
  if (attempts >= 3) {
    return sendError(res, wire, 429, wire === 'anthropic' ? 'rate_limit_error' : 'rate_limit_exceeded', 'every free model refused (rate limited) after backoff');
  }
  const candidates = routesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'api_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  // A model that answered a same-class prompt before is tried first: fewer calls on later prompts.
  const preferred = lastGoodByClass.get(requirementClass(requires));
  const ordered = preferred === undefined ? candidates : [candidates.find((m) => m.id === preferred), ...candidates.filter((m) => m.id !== preferred)].filter((m) => m !== undefined);
  let lastStatus = 502;
  let lastText = '';
  for (const candidate of ordered) {
    if (coolingDown.has(candidate.id)) continue;
    let upstream: Response | null = null;
    try {
      upstream = await dial(candidate.id, gatewayBody(candidate));
    } catch (err) {
      recordRefusal(candidate.id, 0, 'upstream_unreachable');
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      lastGoodByClass.set(requirementClass(requires), candidate.id);
      const trail = trailHeader();
      if (trail !== undefined) res.setHeader('x-makefun-upstream-failures', trail);
      await deliver(upstream, candidate);
      return;
    }
    const text = upstream.body === null ? '' : await upstream.text();
    const code = errorCode(text);
    recordRefusal(candidate.id, upstream.status, code);
    lastStatus = upstream.status;
    lastText = text;
    // A client-shaped 400 (bad request) will not improve on another model: stop at once.
    if (upstream.status === 400 && code === 'invalid_request_error') break;
    // A 429 is the account's own rate limit: spraying more models this second burns the same budget.
    // Back off, then retry the SAME class again from the top (the next run of this loop serves it).
    if (upstream.status === 429) {
      await new Promise((resolve) => setTimeout(resolve, 4000));
      // Retry the same candidate set once more after the backoff, from the best model again.
      return serveWithFailover(res, wire, requires, gatewayBody, deliver, attempts + 1);
    }
  }
  if (lastStatus === 0) sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_unreachable', lastText);
  else sendError(res, wire, lastStatus >= 400 && lastStatus < 600 ? lastStatus : 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', `every free model refused: HTTP ${lastStatus}${lastText === '' ? '' : `: ${lastText.slice(0, 400)}`}`);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered with failover across free models. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  await serveWithFailover(res, 'openai', requires, (model) => ({ ...body, model: model.id }), async (upstream, model) => {
    if (body.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      try {
        for await (const frame of openAIStreamWithModel(upstream.body!, model.id)) res.write(frame);
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await upstream.json();
    sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model.id));
  });
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions, answered with failover. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  await serveWithFailover(res, 'anthropic', requires, (model) => anthropicToOpenAI(body, model.id), async (upstream, model) => {
    if (body.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      try {
        for await (const event of anthropicEventsFromOpenAI(upstream.body!, model.id)) res.write(event);
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await upstream.json();
    sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model.id));
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
