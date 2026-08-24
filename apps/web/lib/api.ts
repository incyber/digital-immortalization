// Gateway client.
//
// The gateway is the only thing that can produce a room token, because it is
// the only thing that checks consent. The browser never talks to LiveKit
// without going through it first.

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

export type SessionDetails = {
  session_id: string;
  room: string;
  url: string;
  token: string;
  avatar_id: string;
};

export class ConsentRefused extends Error {}

export async function openSession(avatarId: string): Promise<SessionDetails> {
  const response = await fetch(`${GATEWAY}/api/sessions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ avatar_id: avatarId }),
  });

  if (response.status === 403) {
    // The avatar may well exist. What is missing is documented permission,
    // and the customer needs to be told which so they can fix it.
    const body = await response.json().catch(() => ({ detail: "consent refused" }));
    throw new ConsentRefused(body.detail);
  }
  if (!response.ok) {
    throw new Error(`gateway returned ${response.status}`);
  }
  return response.json();
}
