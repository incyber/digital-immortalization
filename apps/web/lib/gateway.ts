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

export type AvatarInput = {
  display_name: string;
  locale: string;
  country: string;
  biography: string;
  voice_description: string;
  boundaries: string;
};

export type Avatar = AvatarInput & {
  id: string;
  photo_set_id: string | null;
  has_assets: boolean;
  // Generated from the name, never supplied by the customer.
  disclosure: string;
  crisis_line: { name: string; number: string } | null;
  callable: boolean;
};

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
    request<Avatar>(`/api/avatars/${avatarId}/photo-set/${photoSetId}`, { method: "POST" }),

  recordConsent: (
    avatarId: string,
    body: { rights_holder_name: string; relationship_to_subject: string; jurisdiction: string },
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

  evaluate: (id: string) =>
    request<PhotoSet>(`/api/photo-sets/${id}/evaluate`, { method: "POST" }),

  // Re-runs the current checks over images already uploaded, so a validator
  // fix does not mean gathering the photographs again.
  revalidate: (id: string) =>
    request<PhotoSet>(`/api/photo-sets/${id}/revalidate`, { method: "POST" }),

  train: (id: string) =>
    request<{ job_id: string; status: string }>(`/api/photo-sets/${id}/train`, {
      method: "POST",
    }),

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
