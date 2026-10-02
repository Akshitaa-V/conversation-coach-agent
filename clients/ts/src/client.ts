// Minimal client for the roleplay API: start a session, stream a scripted
// conversation over WebSocket, then request structured feedback.
//
//   npm run build && node dist/client.js http://127.0.0.1:8000
//   COACH_TOKEN=$(gcloud auth print-identity-token) node dist/client.js https://<service>.run.app

type ServerEvent =
  | { type: "start" }
  | { type: "delta"; text: string }
  | { type: "end"; model: string; fallback: boolean; degraded: boolean; interrupted: boolean; ttft_ms: number }
  | { type: "error"; detail: string }
  | { type: "session_ended" };

interface SessionInfo {
  session_id: string;
  scenario_id: string;
  variant: string;
  opening_line: string;
}

interface FeedbackResponse {
  overall: number;
  feedback: {
    scores: { dimension: string; score: number; evidence: string }[];
    strengths: string[];
    next_focus: string;
  };
  previous_focus: string[];
  cost_usd: number;
}

const SCRIPT = [
  "Thanks for making the time. What does your current setup look like today?",
  "I hear you. Is it the price itself, or the timing within your budget year?",
  "Teams your size usually save around 6 hours a week on manual reporting.",
  "Let's book 30 minutes next week, I'll send a calendar invite now.",
];

const baseUrl = (process.argv[2] ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const token = process.env.COACH_TOKEN;
const headers: Record<string, string> = { "content-type": "application/json" };
if (token) headers.authorization = `Bearer ${token}`;

async function post<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}: ${await response.text()}`);
  return (await response.json()) as T;
}

function nextEvent(socket: WebSocket, queue: ServerEvent[], waiters: ((e: ServerEvent) => void)[]): Promise<ServerEvent> {
  const queued = queue.shift();
  if (queued) return Promise.resolve(queued);
  return new Promise((resolve) => waiters.push(resolve));
}

async function roleplay(session: SessionInfo): Promise<void> {
  const socket = new WebSocket(`${baseUrl.replace(/^http/, "ws")}/sessions/${session.session_id}/stream`);
  const queue: ServerEvent[] = [];
  const waiters: ((e: ServerEvent) => void)[] = [];
  socket.addEventListener("message", (message) => {
    const event = JSON.parse(String(message.data)) as ServerEvent;
    const waiter = waiters.shift();
    if (waiter) waiter(event);
    else queue.push(event);
  });
  await new Promise<void>((resolve, reject) => {
    socket.addEventListener("open", () => resolve(), { once: true });
    socket.addEventListener("error", () => reject(new Error("WebSocket connection failed")), { once: true });
  });

  console.log(`Customer: ${session.opening_line}`);
  for (const line of SCRIPT) {
    console.log(`You:      ${line}`);
    socket.send(JSON.stringify({ type: "message", text: line }));
    process.stdout.write("Customer: ");
    for (;;) {
      const event = await nextEvent(socket, queue, waiters);
      if (event.type === "delta") process.stdout.write(event.text);
      else if (event.type === "end") {
        const flags = [event.fallback && "fallback model", event.degraded && "degraded"].filter(Boolean);
        console.log(`   [${Math.round(event.ttft_ms)} ms to first token${flags.length ? ", " + flags.join(", ") : ""}]`);
        break;
      } else if (event.type === "error") throw new Error(event.detail);
    }
  }
  socket.send(JSON.stringify({ type: "end_session" }));
  await nextEvent(socket, queue, waiters);
  socket.close();
}

async function main(): Promise<void> {
  const session = await post<SessionInfo>("/sessions", {
    trainee_id: "ts-client-demo",
    scenario_id: "renewal_price_increase",
  });
  console.log(`Session ${session.session_id} (variant: ${session.variant})\n`);
  await roleplay(session);

  const result = await post<FeedbackResponse>(`/sessions/${session.session_id}/feedback`);
  console.log(`\nOverall ${result.overall}/5, cost $${result.cost_usd.toFixed(5)}`);
  for (const s of result.feedback.scores) {
    console.log(`  ${s.dimension.padEnd(19)} ${s.score}/5  ${s.evidence ? `"${s.evidence}"` : ""}`);
  }
  console.log(`Next focus: ${result.feedback.next_focus}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
