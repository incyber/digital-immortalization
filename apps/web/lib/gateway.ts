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
  minimum_half_body: number;
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

export type PhotoSet = {
  id: string;
  status: string;
  usable_count: number;
  half_body_count: number;
  problems: string[];
  photos: PhotoVerdict[];
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

  train: (id: string) =>
    request<{ job_id: string; status: string }>(`/api/photo-sets/${id}/train`, {
      method: "POST",
    }),

  job: (id: string) =>
    request<{ id: string; status: string; error: string | null }>(`/api/training-jobs/${id}`),
};
