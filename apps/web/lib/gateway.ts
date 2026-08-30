// Gateway client.
//
// credentials: "include" on every call so the httpOnly session cookie is sent.
// The cookie is not readable from here by design — an injected script must not
// be able to lift a session that reaches a family's photographs.

// Where the API is.
//
// Empty means same origin, and same origin is the deployed shape: the gateway
// serves this site as well as the API, so every path below is relative and the
// session cookie is first-party. That is the point of it — cross-origin, the
// cookie has to be SameSite=None, Safari refuses those outright and Chrome
// refuses them in common configurations, and the result was a sign-in that
// returned 200 and bounced straight back to the sign-in page.
//
// Development is the exception, and it is a real one: `next dev` serves this
// on 3100 while the gateway runs on 8000, so there the absolute URL is
// correct. NODE_ENV is inlined by the compiler along with the variable, so
// the branch does not survive into the built bundle.
//
// NEXT_PUBLIC_GATEWAY_URL overrides both, and is read at build time rather
// than at runtime — changing it means rebuilding the site.
const GATEWAY =
  process.env.NEXT_PUBLIC_GATEWAY_URL ??
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${GATEWAY}${path}`, {
    ...init,
    credentials: "include",
    // A custom header on every call. Several endpoints take no body, and a
    // bodyless POST is a request a browser sends cross-origin without asking
    // first - so the session cookie would ride along on a form submitted by
    // any other site. A custom header cannot be set that way, which is what
    // makes it a defence rather than a decoration. The server requires it.
    headers: { "x-avatar-client": "web", ...(init.headers ?? {}) },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail ?? `request failed (${response.status})`, response.status);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export type Requirements = {
  minimum: number;
  recommended_min: number;
  recommended_max: number;
  half_body_threshold: number;
  note: string;
  shots: { label: string; count: string }[];
  rules: string[];
};

export type PhotoVerdict = {
  id: string;
  filename: string;
  accepted: boolean;
  reasons: string[];
  half_body: boolean;
};

export type Requirement = {
  key: string;
  label: string;
  current: number;
  target: number;
  met: boolean;
  hint: string;
  // Informational requirements report progress without gating the build.
  blocking: boolean;
};

export type PhotoSet = {
  id: string;
  status: string;
  usable_count: number;
  half_body_count: number;
  framing: "head" | "half_body";
  framing_label: string;
  problems: string[];
  requirements: Requirement[];
  photos: PhotoVerdict[];
};

/**
 * A clip being turned into frames, while it happens.
 *
 * The upload itself returns immediately now; the work is a job on the server,
 * because doing it inside the request stalled every other request in the
 * process and returned 502 to everybody on the site. So this is what the page
 * polls, and the counts in it are counted rather than estimated: the frame
 * total is read from the clip's own metadata before any decoding starts.
 */
export type VideoJob = {
  id: string;
  photo_set_id: string;
  status: string;
  filename: string;
  progress: number;
  frames_planned: number;
  frames_examined: number;
  frames_usable: number;
  error: string | null;
};

export type VideoStarted = {
  job_id: string;
  photo_set_id: string;
  status: string;
  size_bytes: number;
};

export type Country = {
  code: string;
  name: string;
  locale: string;
  crisis_line: string;
};

export type PersonInput = {
  display_name: string;
  locale: string;
  country: string;
  biography: string;
  voice_description: string;
  boundaries: string;
};

// The body, in words. Someone who knew the person can say "solid"; nobody can
// say 0.63, so nothing below is a number except the height they knew.
export type Build = "slight" | "average" | "solid" | "heavy";
export type ShoulderWidth = "narrow" | "average" | "broad";
export type Posture = "upright" | "relaxed" | "stooped";

// null is a real answer: it means nobody said, and the gateway keeps it that
// way rather than filling something in on the family's behalf.
export type StatedBody = {
  height_cm: number | null;
  build: Build | null;
  shoulders: ShoulderWidth | null;
  posture: Posture | null;
};

export type Body = {
  stated: StatedBody;
  // What the build falls back on wherever a question was left blank. Kept
  // apart from `stated` so a fallback can never be read back as something the
  // family told us.
  in_use: {
    height_cm: number;
    build: Build;
    shoulders: ShoulderWidth;
    posture: Posture;
  };
};

export type AvatarInput = PersonInput & StatedBody;

export type Avatar = PersonInput & {
  id: string;
  body: Body;
  photo_set_id: string | null;
  has_assets: boolean;
  // Generated from the name, never supplied by the customer.
  disclosure: string;
  crisis_line: { name: string; number: string } | null;
  callable: boolean;
};

// --- the splat build ------------------------------------------------------
//
// Two routes, one artefact, and the customer chooses neither: whatever they
// were able to upload decides it. The third outcome is a refusal, which is a
// legitimate answer rather than a failure — it comes back 200 with a sentence
// naming what is still needed, and the page shows it as guidance.

export type SplatRoute = "reconstruct" | "generate";

export type SplatRefusal = {
  status: "refused";
  buildable: false;
  reasoning: string;
  // What the customer must supply, itemised in their own counts.
  missing: string[];
  // The same thing as one sentence somebody can act on.
  guidance: string;
  considered: string[];
};

export type SplatStarted = {
  status: "building";
  buildable: true;
  job_id: string;
  avatar_id: string;
  route: SplatRoute;
  reasoning: string;
  considered: string[];
};

export type SplatStart = SplatStarted | SplatRefusal;

// Every shape that reports a finished likeness carries the disclosure and the
// fraction of it that was actually measured. There is no response from this
// API that hands back a likeness without them.
export type SplatJob = {
  id: string;
  status: string;
  backend: string;
  progress: number;
  error: string | null;
  avatar_id: string | null;
  splat_key: string | null;
  route: SplatRoute | null;
  reasoning: string | null;
  disclosure: string | null;
  measured_fraction: number | null;
  generated_fraction: number | null;
  concerns: string[];
  gaussians: number;
  size_bytes: number;
};

export type AvatarSplat = {
  avatar_id: string;
  built: boolean;
  splat_key: string | null;
  route: SplatRoute | null;
  reasoning: string | null;
  disclosure: string | null;
  // null means no likeness has been built. It is never 0, which would be a
  // claim that none of one was measured.
  measured_fraction: number | null;
  generated_fraction: number | null;
  concerns: string[];
  gaussians: number;
  size_bytes: number;
  backend: string | null;
  // A short-lived signed URL for the asset itself, straight from the object
  // store. Optional because it is the store's answer, not the database's: a
  // gateway that cannot sign one leaves it out and the client falls back to
  // reading the asset through the gateway. See splatAssetUrl.
  asset_url?: string | null;
};

/** Where a built likeness is read from, and whether the read carries the session. */
export type SplatAsset = {
  url: string;
  // A signed store URL authenticates itself and must be fetched anonymously —
  // sending a cookie to a third-party bucket is both useless and a leak. The
  // gateway route is same-origin and does need the session cookie. Two
  // different answers for two different hosts, which is why this is carried
  // with the URL rather than fixed.
  credentials: RequestCredentials;
  /** Sent with the request. Undefined for a store URL, which wants none. */
  headers?: Record<string, string>;
};

// The same header request() sets, in a frozen shared object so that a caller
// holding it across renders holds one identity. Anything that keys an effect
// on these headers would otherwise rebuild a viewer on every render.
const GATEWAY_HEADERS: Record<string, string> = Object.freeze({
  "x-avatar-client": "web",
});

/**
 * The asset behind a built likeness, or null if there is not one to read.
 *
 * Nothing here is public: the store holds photographs of dead people, so the
 * bytes come either from a signed URL that expires or from the gateway with
 * the session attached. There is no unauthenticated path to a likeness.
 */
export function splatAssetUrl(splat: AvatarSplat): SplatAsset | null {
  if (!splat.built) return null;
  if (splat.asset_url) return { url: splat.asset_url, credentials: "omit" };
  if (!splat.splat_key) return null;
  return {
    url: `${GATEWAY}/api/avatars/${splat.avatar_id}/splat/asset`,
    credentials: "include",
    headers: GATEWAY_HEADERS,
  };
}

export const api = {
  // What the site must know before it renders anything. Only demo_mode today,
  // and it is not cosmetic: in demo mode there is one shared account, and the
  // interface has to say so on every screen. See components/DemoBanner.tsx.
  config: () => request<{ demo_mode: boolean }>("/api/config"),

  register: (email: string, password: string) =>
    request<{ id: string }>("/api/auth/register", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<{ id: string }>("/api/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),

  me: () => request<{ id: string }>("/api/me"),

  requirements: () => request<Requirements>("/api/photo-sets/requirements"),

  countries: () => request<{ countries: Country[] }>("/api/countries"),

  languages: () =>
    request<{ languages: { code: string; name: string; voice: string }[] }>("/api/languages"),

  listAvatars: () => request<{ avatars: Avatar[] }>("/api/avatars"),

  readAvatar: (id: string) => request<Avatar>(`/api/avatars/${id}`),

  createAvatar: (input: AvatarInput) =>
    request<Avatar>("/api/avatars", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
    }),

  updateAvatar: (id: string, input: AvatarInput) =>
    request<Avatar>(`/api/avatars/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
    }),

  attachPhotoSet: (avatarId: string, photoSetId: string) =>
    request<Avatar>(`/api/avatars/${avatarId}/photo-set/${photoSetId}`, {
      method: "POST",
    }),

  recordConsent: (
    avatarId: string,
    body: {
      rights_holder_name: string;
      relationship_to_subject: string;
      jurisdiction: string;
    },
  ) =>
    request<{ status: string }>(`/api/avatars/${avatarId}/consent`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),

  createPhotoSet: () => request<{ id: string }>("/api/photo-sets", { method: "POST" }),

  readPhotoSet: (id: string) => request<PhotoSet>(`/api/photo-sets/${id}`),

  uploadPhoto: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<PhotoVerdict>(`/api/photo-sets/${id}/photos`, {
      method: "POST",
      body: form,
    });
  },

  // A clip is turned into frames on the server, each checked exactly as an
  // uploaded photograph is. This returns as soon as the bytes have landed and
  // hands back a job to watch: the checking is minutes of work and used to be
  // done inside this request, which took the whole site down with it.
  uploadVideo: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<VideoStarted>(`/api/photo-sets/${id}/video`, {
      method: "POST",
      body: form,
    });
  },

  videoJob: (id: string) => request<VideoJob>(`/api/video-jobs/${id}`),

  evaluate: (id: string) => request<PhotoSet>(`/api/photo-sets/${id}/evaluate`, { method: "POST" }),

  // Re-runs the current checks over images already uploaded, so a validator
  // fix does not mean gathering the photographs again.
  revalidate: (id: string) =>
    request<PhotoSet>(`/api/photo-sets/${id}/revalidate`, { method: "POST" }),

  train: (id: string) =>
    request<{ job_id: string; status: string }>(`/api/photo-sets/${id}/train`, {
      method: "POST",
    }),

  // Starts the splat build, or explains what is missing. A refusal is not an
  // error and does not throw: check `buildable` on what comes back.
  buildSplat: (photoSetId: string) =>
    request<SplatStart>(`/api/photo-sets/${photoSetId}/splat`, { method: "POST" }),

  splatJob: (id: string) => request<SplatJob>(`/api/splat-jobs/${id}`),

  cancelSplat: (id: string) =>
    request<{ id: string; status: string }>(`/api/splat-jobs/${id}/cancel`, {
      method: "POST",
    }),

  // What was built for this person and what must be said about it, long after
  // the page that started the build has gone.
  avatarSplat: (avatarId: string) => request<AvatarSplat>(`/api/avatars/${avatarId}/splat`),

  job: (id: string) =>
    request<{
      id: string;
      status: string;
      error: string | null;
      progress: number;
      avatar_id: string | null;
    }>(`/api/training-jobs/${id}`),
};

export type SessionDetails = {
  session_id: string;
  room: string;
  url: string;
  token: string;
  avatar_id: string;
};

export class ConsentRefused extends Error {}

/** Nothing has been built for this person yet, so there is nothing to show. */
export class NoLikeness extends Error {}

export async function openSession(avatarId: string): Promise<SessionDetails> {
  const response = await fetch(`${GATEWAY}/api/sessions`, {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ avatar_id: avatarId }),
  });

  if (response.status === 403) {
    // The avatar is theirs and real; what is missing is verified consent.
    const body = await response.json().catch(() => ({ detail: "consent refused" }));
    throw new ConsentRefused(body.detail);
  }
  if (response.status === 409) {
    // Distinct from a consent refusal: this is the one gate the person on the
    // other side can clear themselves, by uploading a video or photographs.
    const body = await response.json().catch(() => ({ detail: "no likeness yet" }));
    throw new NoLikeness(body.detail);
  }
  if (!response.ok) throw new ApiError(`gateway returned ${response.status}`, response.status);
  return response.json();
}
