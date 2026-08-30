import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A static export: HTML, CSS and JavaScript, no Node process. The gateway
  // serves these files itself (src/avatar/gateway/web.py), which is what puts
  // the site and the API on one origin and makes the session cookie
  // first-party. Cross-origin, it had to be SameSite=None — which Safari
  // blocks outright and Chrome blocks in common configurations, so sign-in
  // returned 200, the cookie was discarded, and the app bounced back to the
  // sign-in page forever.
  //
  // Exporting is possible because every page here is a client component:
  // there are no server actions, no server-side data loading and no
  // request-time rendering anywhere in the app. Everything it needs comes
  // from the gateway over fetch, at runtime, in the browser. Running a second
  // Node process beside the gateway would have been a process to supervise, a
  // port to keep private and a restart to sequence, in exchange for nothing a
  // visitor could tell apart.
  //
  // The one thing an export cannot do is a dynamic route segment whose values
  // are not known at build time, which is why a call is /call?avatar=<id>
  // rather than /call/<id> — see app/call/page.tsx.
  output: "export",

  // Written into the export as `<route>/index.html`. Chosen over `<route>.html`
  // because it is the layout that also works if these files are ever put
  // behind a plain object store, where a bucket serves index.html for a prefix
  // and knows nothing about stripping suffixes.
  trailingSlash: true,
};

export default nextConfig;
