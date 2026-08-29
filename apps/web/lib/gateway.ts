// Gateway client.
//
// credentials: "include" on every call so the httpOnly session cookie is sent.
// The cookie is not readable from here by design — an injected script must not
// be able to lift a session that reaches a family's photographs.

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

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
  // gateway route is a different origin from this page and needs the session
  // cookie, which is the only reason this is not a constant.
  credentials: RequestCredentials;
};

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
  };
}

export const api = {
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
  // uploaded photograph is, so this returns how many of them survived.
  uploadVideo: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{
      frames_examined: number;
      accepted: number;
      photos: PhotoVerdict[];
    }>(`/api/photo-sets/${id}/video`, { method: "POST", body: form });
  },

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
  if (!response.ok) throw new ApiError(`gateway returned ${response.status}`, response.status);
  return response.json();
}
