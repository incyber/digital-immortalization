// Browser-side Gaussian splat loading, rendering and teardown.
//
// This module exists to answer one commercial question: can a viewer's own
// device render the likeness, so the product does not have to rent a GPU for
// the length of every call. Everything here is therefore built to be measured
// — frame rate, byte counts and timings are first-class outputs, not debug
// noise — because a viewer that renders but cannot be measured tells us
// nothing about whether the economics work.
//
// Renderer: Spark (@sparkjsdev/spark, MIT, WebGL2). Preferred over a
// self-contained splat viewer because it is a THREE.js citizen: the splat is
// an Object3D in an ordinary scene graph, so a head pose driven by the call
// pipeline is a transform on a node rather than a fight with a closed viewer's
// camera controller. It also keeps the door open to compositing meshes,
// lighting or a background behind the same sort.
//
// Nothing in this file touches the DOM or WebGL until createSplatViewer runs,
// and Spark and THREE are imported dynamically inside it: together they are
// several megabytes, and a page that merely links to a splat should not pay
// for the renderer.

import type * as ThreeNS from "three";
import type { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

/** Head orientation in degrees, in the convention a face rig uses. */
export interface HeadPose {
  yaw: number;
  pitch: number;
  roll: number;
}

/**
 * Camera framing. `distance` and `height` are multiples of the asset's own
 * bounding radius rather than metres, because splat files carry whatever
 * scale their trainer happened to use and an absolute distance would frame
 * one asset correctly and put the next one inside the camera.
 */
export interface CameraFraming {
  distance: number;
  height: number;
  /** Azimuth around the subject, degrees. */
  orbit: number;
  /** Vertical field of view, degrees. */
  fov: number;
}

export const DEFAULT_POSE: HeadPose = { yaw: 0, pitch: 0, roll: 0 };

export const DEFAULT_FRAMING: CameraFraming = {
  distance: 2.4,
  height: 0.1,
  orbit: 0,
  fov: 45,
};

export type SplatPhase =
  | "idle"
  | "unsupported"
  | "downloading"
  | "decoding"
  | "ready"
  | "failed";

export interface SplatStatus {
  phase: SplatPhase;
  /** Failure text or the reason the device cannot render. Never invented. */
  message: string | null;
  bytesLoaded: number;
  /** Null when the server sends no Content-Length; the UI must not guess. */
  bytesTotal: number | null;
  /** Null whenever a true percentage is unknowable. */
  percent: number | null;
  splatCount: number | null;
  downloadMs: number | null;
  decodeMs: number | null;
  timeToFirstFrameMs: number | null;
}

export interface FrameStats {
  /** Frames per second over the last measurement window. */
  fps: number;
  /** Frames rendered since the viewer was created. */
  frames: number;
  /** Slowest single frame in the window, milliseconds. */
  worstFrameMs: number;
  /** Drawing-buffer size, which is what the GPU actually shades. */
  drawWidth: number;
  drawHeight: number;
  pixelRatio: number;
  /** From THREE's own bookkeeping; a climb across load cycles is a leak. */
  textures: number;
  geometries: number;
}

export interface SplatViewer {
  setPose(pose: HeadPose): void;
  setFraming(framing: CameraFraming): void;
  dispose(): void;
}

export interface SplatViewerOptions {
  container: HTMLElement;
  url: string;
  pose?: HeadPose;
  framing?: CameraFraming;
  /**
   * Upper bound on device pixel ratio. Phones ship ratios of 3 and 4, and
   * shading nine to sixteen times the pixels turns a viable frame rate into
   * an unviable one for detail nobody can resolve at arm's length.
   */
  maxPixelRatio?: number;
  /**
   * Credential mode for the asset fetch. Defaults to "omit", which is right
   * for a signed URL: sending a session cookie to an object store is useless
   * there and hands the cookie to a host that has no business with it. A
   * caller reading the asset back through its own API passes "include".
   */
  credentials?: RequestCredentials;
  onStatus?: (status: SplatStatus) => void;
  onFrameStats?: (stats: FrameStats) => void;
}

const IDLE_STATUS: SplatStatus = {
  phase: "idle",
  message: null,
  bytesLoaded: 0,
  bytesTotal: null,
  percent: null,
  splatCount: null,
  downloadMs: null,
  decodeMs: null,
  timeToFirstFrameMs: null,
};

export function idleStatus(): SplatStatus {
  return { ...IDLE_STATUS };
}

export interface Webgl2Support {
  supported: boolean;
  /** Populated only when unsupported. */
  reason: string | null;
  /** Best-effort GPU string; browsers are free to withhold or mask it. */
  renderer: string | null;
}

/**
 * Splats are drawn with WebGL2-only features, so a device without it cannot be
 * made to work by degrading quality — it can only be told. The probe context
 * is explicitly destroyed: browsers cap the number of live contexts per page
 * at around sixteen, and a probe that lingers spends one of them.
 */
export function detectWebgl2(): Webgl2Support {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return { supported: false, reason: "No browser environment", renderer: null };
  }

  const canvas = document.createElement("canvas");
  let gl: WebGL2RenderingContext | null = null;
  try {
    gl = canvas.getContext("webgl2");
  } catch {
    gl = null;
  }

  if (!gl) {
    return {
      supported: false,
      reason:
        "This browser has no WebGL2. Splats are rendered with WebGL2-only features, so there is no reduced-quality version of this that would work here.",
      renderer: null,
    };
  }

  let renderer: string | null = null;
  try {
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    renderer = info
      ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
  } catch {
    renderer = null;
  }

  gl.getExtension("WEBGL_lose_context")?.loseContext();
  return { supported: true, reason: null, renderer };
}

/**
 * Measures the display's refresh ceiling by timing empty animation frames.
 *
 * Without this a measured "58 fps" is unreadable: on a 60Hz panel it means the
 * renderer is keeping up, and on a 120Hz panel it means it is missing half the
 * frames. Gate 3 is a number on a phone, and phones ship both.
 */
export function measureRefreshHz(sampleMs = 400): Promise<number> {
  return new Promise((resolve) => {
    if (typeof window === "undefined") {
      resolve(0);
      return;
    }
    const intervals: number[] = [];
    let previous = performance.now();
    const started = previous;

    const tick = (now: number) => {
      intervals.push(now - previous);
      previous = now;
      if (now - started < sampleMs) {
        requestAnimationFrame(tick);
        return;
      }
      if (intervals.length < 3) {
        resolve(0);
        return;
      }
      // Median rather than mean: the first frames after a rAF loop starts are
      // routinely long, and one 40ms outlier would halve the reported ceiling.
      const sorted = intervals.slice(1).sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      resolve(median > 0 ? Math.round(1000 / median) : 0);
    };

    requestAnimationFrame(tick);
  });
}

export interface DeviceProfile extends Webgl2Support {
  /** Null when it could not be measured, or when there is nothing to render on. */
  refreshHz: number | null;
}

/**
 * Everything about the device that a frame-rate number has to be read against.
 *
 * Ordered so that an unsupported device answers immediately: there is no point
 * timing animation frames on hardware that cannot draw a splat, and the caller
 * should be able to say so without waiting on a measurement.
 */
export async function probeDevice(): Promise<DeviceProfile> {
  const support = detectWebgl2();
  if (!support.supported) return { ...support, refreshHz: null };
  return { ...support, refreshHz: await measureRefreshHz() };
}

/**
 * Formats a byte count for people rather than for machines. Splat assets span
 * three orders of magnitude, so a single unit would be unreadable at one end.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Multi-file formats reference sibling files relative to their manifest, so
 * they cannot be fetched as a single blob. Spark resolves those siblings
 * itself when handed a URL, at the cost of byte-accurate progress.
 */
function isSingleFileAsset(url: string): boolean {
  const path = url.split(/[?#]/)[0].toLowerCase();
  return /\.(spz|ply|splat|ksplat|zip|sog)$/.test(path);
}

interface Downloaded {
  bytes: Uint8Array;
  /** Distinct from bytes.length only when the server misreported the length. */
  contentLength: number | null;
}

/**
 * Streams the asset while reporting byte progress.
 *
 * Done here rather than left to the renderer for three reasons: these files
 * run to hundreds of megabytes and a silent wait is not acceptable; an
 * abortable fetch means leaving the page actually stops the transfer instead
 * of paying for it in the background; and the exact transferred size is one of
 * the numbers this whole exercise exists to collect.
 */
async function downloadWithProgress(
  url: string,
  signal: AbortSignal,
  credentials: RequestCredentials,
  onProgress: (loaded: number, total: number | null) => void,
): Promise<Downloaded> {
  const response = await fetch(url, { signal, mode: "cors", credentials });
  if (!response.ok) {
    throw new Error(`Server answered ${response.status} ${response.statusText}`);
  }

  const header = response.headers.get("content-length");
  const parsed = header === null ? Number.NaN : Number(header);
  const contentLength = Number.isFinite(parsed) && parsed > 0 ? parsed : null;

  if (!response.body) {
    // Some environments expose no readable stream. A correct load with no
    // progress beats refusing to load at all.
    const buffer = await response.arrayBuffer();
    onProgress(buffer.byteLength, contentLength);
    return { bytes: new Uint8Array(buffer), contentLength };
  }

  const reader = response.body.getReader();
  // Preallocating when the length is known avoids holding the chunk list and
  // the joined copy at the same time. Note this does not make peak memory 1x
  // the asset: the renderer copies the buffer again to hand it to its decode
  // worker, so peak is at least 2x either way. If that ceiling ever becomes
  // the binding constraint at production sizes, the renderer also accepts a
  // ReadableStream, which would avoid buffering the asset here at all.
  let preallocated = contentLength === null ? null : new Uint8Array(contentLength);
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  // Diverges from contentLength the moment the body proves that header wrong.
  let reportedTotal = contentLength;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;

    if (preallocated && loaded + value.length <= preallocated.length) {
      preallocated.set(value, loaded);
    } else if (preallocated) {
      // The body outgrew the declared length, which happens when the response
      // was compressed on the wire. Fall back rather than truncate the asset.
      chunks.push(preallocated.subarray(0, loaded).slice());
      chunks.push(value);
      preallocated = null;
      // The declared length is now known to be false, so quoting it would show
      // "12.0 MB of 4.0 MB". An unknown total already has an honest rendering.
      reportedTotal = null;
    } else {
      chunks.push(value);
    }

    loaded += value.length;
    onProgress(loaded, reportedTotal);
  }

  if (preallocated) {
    // Exact-size buffer rather than a view: a decoder that reaches through to
    // .buffer would otherwise read the unwritten tail of a short response as
    // trailing zeroes, corrupting the asset instead of failing to parse it.
    return {
      bytes: loaded === preallocated.length ? preallocated : preallocated.slice(0, loaded),
      contentLength,
    };
  }

  const joined = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  return { bytes: joined, contentLength };
}

/**
 * Awaits a mesh's decode, disposing it if that decode throws.
 *
 * A SplatMesh allocates during construction, so one whose initialisation
 * rejects still holds resources while being unreachable from the viewer's own
 * teardown, which only knows about meshes that finished successfully.
 */
async function disposeOnFailure(mesh: SplatMesh): Promise<void> {
  try {
    await mesh.initialized;
  } catch (error) {
    mesh.dispose();
    throw error;
  }
}

function percentOf(loaded: number, total: number | null): number | null {
  if (total === null || total <= 0) return null;
  // Clamped because a compressed response reports fewer declared bytes than it
  // delivers, and a progress bar reading 118% reads as a bug.
  return Math.min(100, Math.round((loaded / total) * 100));
}

const FPS_WINDOW_MS = 500;

/**
 * Creates a splat viewer inside `container` and starts loading `url`.
 *
 * Returns synchronously so that a React effect's cleanup is always able to
 * dispose it, including when the component unmounts before the asset — or even
 * the renderer module — has finished arriving.
 *
 * The viewer owns its own canvas rather than borrowing one from the caller.
 * Teardown forces the WebGL context to be lost, which permanently poisons the
 * canvas it ran on; owning the element means a remount gets a clean one, so
 * repeated visits cannot walk into the browser's per-page context limit.
 */
export function createSplatViewer(options: SplatViewerOptions): SplatViewer {
  const {
    container,
    url,
    maxPixelRatio = 2,
    credentials = "omit",
    onStatus,
    onFrameStats,
  } = options;

  let pose: HeadPose = { ...(options.pose ?? DEFAULT_POSE) };
  let framing: CameraFraming = { ...(options.framing ?? DEFAULT_FRAMING) };

  let disposed = false;
  const abort = new AbortController();

  let status: SplatStatus = idleStatus();
  const publish = (patch: Partial<SplatStatus>) => {
    if (disposed) return;
    status = { ...status, ...patch };
    onStatus?.(status);
  };

  const support = detectWebgl2();
  if (!support.supported) {
    // Reported through the same channel as every other outcome so the caller
    // has one place to render failure, and reported before any allocation.
    queueMicrotask(() => {
      publish({ phase: "unsupported", message: support.reason });
    });
    return {
      setPose: () => {},
      setFraming: () => {},
      dispose: () => {
        disposed = true;
      },
    };
  }

  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  container.appendChild(canvas);

  // Populated once the renderer module resolves. Held in this closure so that
  // dispose can tear down whatever exists at the moment it is called.
  let three: typeof ThreeNS | null = null;
  let renderer: ThreeNS.WebGLRenderer | null = null;
  let scene: ThreeNS.Scene | null = null;
  let camera: ThreeNS.PerspectiveCamera | null = null;
  let spark: SparkRenderer | null = null;
  let mesh: SplatMesh | null = null;
  let pivot: ThreeNS.Group | null = null;
  let resizeObserver: ResizeObserver | null = null;

  /** Bounding radius of the loaded asset; framing is expressed in multiples. */
  let subjectRadius = 1;
  let appliedPixelRatio = 1;

  const applyPose = () => {
    if (!three || !pivot) return;
    const rad = Math.PI / 180;
    // YXZ: yaw about the world vertical first, then pitch, then roll — the
    // order in which a head actually articulates, and the order the driving
    // pipeline will hand these angles over in.
    const euler = new three.Euler(pose.pitch * rad, pose.yaw * rad, pose.roll * rad, "YXZ");
    pivot.quaternion.setFromEuler(euler);
  };

  const applyFraming = () => {
    if (!camera) return;
    const azimuth = (framing.orbit * Math.PI) / 180;
    const radius = subjectRadius * framing.distance;
    camera.position.set(
      Math.sin(azimuth) * radius,
      subjectRadius * framing.height,
      Math.cos(azimuth) * radius,
    );
    camera.lookAt(0, 0, 0);
    camera.fov = framing.fov;
    // Clip planes tied to the subject's own size: fixed planes would either
    // clip a small asset away entirely or crush depth precision on a large one.
    camera.near = Math.max(0.001, subjectRadius * 0.01);
    camera.far = Math.max(10, subjectRadius * 100);
    camera.updateProjectionMatrix();
  };

  const resize = () => {
    if (!renderer || !camera) return;
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    appliedPixelRatio = Math.min(window.devicePixelRatio || 1, maxPixelRatio);
    renderer.setPixelRatio(appliedPixelRatio);
    // updateStyle=false: CSS already stretches the canvas to the container, so
    // THREE only needs to size the drawing buffer. This is what makes the
    // render sharp on a high-DPI screen instead of upscaled.
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  const startedAt = performance.now();
  let frames = 0;
  let windowFrames = 0;
  let windowWorst = 0;
  let firstFrameReported = false;
  // Seeded here only to satisfy the compiler. The real seeding happens where
  // the animation loop is installed, several megabytes of dynamic import
  // later; timing from here would charge that import to the first frame.
  let windowStart = 0;
  let lastFrameAt = 0;
  let firstLoopFrame = true;

  // INVARIANT, and it is invisible: every allocation below sits in an unbroken
  // synchronous run following a `disposed` check. dispose() can only interleave
  // at an await, so nothing can be created after teardown and then orphaned.
  // Introducing an await between a guard and the objects it protects — even an
  // innocuous-looking one — reintroduces exactly that leak.
  const setup = async () => {
    const [threeModule, sparkModule] = await Promise.all([
      import("three"),
      import("@sparkjsdev/spark"),
    ]);
    if (disposed) return;
    three = threeModule;

    renderer = new threeModule.WebGLRenderer({
      canvas,
      // Spark's own guidance: multisampling does nothing for splats, which are
      // already soft-edged, and costs a large fraction of the frame budget.
      antialias: false,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x000000, 0);

    scene = new threeModule.Scene();
    camera = new threeModule.PerspectiveCamera(framing.fov, 1, 0.01, 1000);

    spark = new sparkModule.SparkRenderer({ renderer });
    scene.add(spark);

    pivot = new threeModule.Group();
    scene.add(pivot);

    resize();
    applyFraming();
    applyPose();

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    // Render from the first frame, before the asset exists. The frame-rate
    // meter therefore starts producing numbers immediately, and an empty
    // stage is visibly a stage rather than a broken rectangle.
    windowStart = performance.now();
    lastFrameAt = windowStart;
    renderer.setAnimationLoop(() => {
      if (disposed || !renderer || !scene || !camera) return;
      const now = performance.now();
      const delta = now - lastFrameAt;
      lastFrameAt = now;

      renderer.render(scene, camera);
      frames += 1;

      if (firstLoopFrame) {
        // The first frame carries shader compilation and the gap since the
        // loop was installed. Both are real costs, but neither is a rendered
        // frame, and letting them close the first window would publish a
        // fabricated "1 fps" while the asset was still downloading.
        firstLoopFrame = false;
        windowStart = now;
      } else {
        windowFrames += 1;
        if (delta > windowWorst) windowWorst = delta;
      }

      if (mesh && !firstFrameReported) {
        firstFrameReported = true;
        publish({ timeToFirstFrameMs: Math.round(performance.now() - startedAt) });
      }

      const elapsed = now - windowStart;
      if (elapsed >= FPS_WINDOW_MS) {
        // Re-read the pixel ratio here rather than with a media query listener:
        // moving a window between a laptop panel and an external monitor
        // changes it, and half a second of staleness costs nothing.
        if (Math.min(window.devicePixelRatio || 1, maxPixelRatio) !== appliedPixelRatio) {
          resize();
        }
        const size = new threeModule.Vector2();
        renderer.getDrawingBufferSize(size);
        onFrameStats?.({
          fps: Math.round((windowFrames / elapsed) * 1000),
          frames,
          worstFrameMs: Math.round(windowWorst * 10) / 10,
          drawWidth: size.x,
          drawHeight: size.y,
          pixelRatio: appliedPixelRatio,
          textures: renderer.info.memory.textures,
          geometries: renderer.info.memory.geometries,
        });
        windowFrames = 0;
        windowWorst = 0;
        windowStart = now;
      }
    });

    publish({ phase: "downloading" });
    const downloadStarted = performance.now();

    let loadedMesh: SplatMesh;
    if (isSingleFileAsset(url)) {
      const { bytes } = await downloadWithProgress(url, abort.signal, credentials, (loaded, total) => {
        publish({
          bytesLoaded: loaded,
          bytesTotal: total,
          percent: percentOf(loaded, total),
        });
      });
      if (disposed) return;
      const downloadMs = Math.round(performance.now() - downloadStarted);
      publish({ phase: "decoding", downloadMs, bytesLoaded: bytes.byteLength, percent: 100 });

      const decodeStarted = performance.now();
      loadedMesh = new sparkModule.SplatMesh({
        fileBytes: bytes,
        // Given so Spark picks the reader from the extension; it falls back to
        // sniffing the header, so an extensionless URL still works.
        fileName: url.split(/[?#]/)[0].split("/").pop() ?? "asset.spz",
      });
      // A mesh whose decode throws is still holding whatever its partial
      // initialisation allocated, and the catch below has no reference to it.
      await disposeOnFailure(loadedMesh);
      if (disposed) {
        loadedMesh.dispose();
        return;
      }
      publish({ decodeMs: Math.round(performance.now() - decodeStarted) });
    } else {
      // A multi-file bundle: the renderer must resolve the siblings itself, so
      // byte progress is genuinely unavailable and is left null rather than
      // faked. It also cannot carry `credentials`: the renderer issues those
      // requests itself, so a multi-file bundle has to be reachable by a
      // signed URL rather than through an authenticated API. This branch is
      // additionally NOT abortable — the transfer happens inside
      // the renderer's own worker with no cancellation channel — so unmounting
      // mid-load lets it run to completion. Single-file assets, which is every
      // format this product will ship, do abort correctly.
      publish({ bytesTotal: null, percent: null });
      loadedMesh = new sparkModule.SplatMesh({
        url,
        onProgress: (event: ProgressEvent) => {
          publish({
            bytesLoaded: event.loaded,
            bytesTotal: event.lengthComputable ? event.total : null,
            percent: event.lengthComputable ? percentOf(event.loaded, event.total) : null,
          });
        },
      });
      await disposeOnFailure(loadedMesh);
      if (disposed) {
        loadedMesh.dispose();
        return;
      }
      publish({ downloadMs: Math.round(performance.now() - downloadStarted) });
    }

    mesh = loadedMesh;
    // Splat trainers overwhelmingly emit Y-down; this is the 180° flip about X
    // that Spark's own examples apply, without which every asset renders
    // upside down.
    mesh.quaternion.set(1, 0, 0, 0);

    // centers_only: the box spans splat centres, not their ±scale extents, so
    // a handful of oversized stray splats cannot blow the framing out.
    const box = mesh.getBoundingBox(/* centers_only */ true);
    const centre = box.getCenter(new threeModule.Vector3());
    const size = box.getSize(new threeModule.Vector3());
    subjectRadius = Math.max(0.001, Math.max(size.x, size.y, size.z) * 0.5);
    // Recentre so the pose pivot turns the subject about itself. An asset
    // whose origin sits metres from the subject would otherwise swing through
    // an arc when yawed, which reads as a camera move, not a head turn.
    mesh.position.copy(centre.applyQuaternion(mesh.quaternion)).negate();

    pivot!.add(mesh);
    applyFraming();
    applyPose();

    publish({ phase: "ready", splatCount: mesh.numSplats, percent: 100 });
  };

  setup().catch((error: unknown) => {
    if (disposed) return;
    // An abort is the caller leaving, not a failure; reporting it would put a
    // red error on screen every time somebody navigates away mid-download.
    if (error instanceof DOMException && error.name === "AbortError") return;
    publish({
      phase: "failed",
      message: error instanceof Error ? error.message : "Could not load this splat",
    });
  });

  return {
    setPose(next: HeadPose) {
      pose = { ...next };
      applyPose();
    },
    setFraming(next: CameraFraming) {
      framing = { ...next };
      applyFraming();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      abort.abort();

      resizeObserver?.disconnect();
      resizeObserver = null;

      renderer?.setAnimationLoop(null);

      mesh?.dispose();
      // SparkRenderer.dispose frees its render targets and workers but not the
      // geometry and material it built as a THREE.Mesh. Harmless while the
      // forced context loss below follows, but this must not silently become a
      // per-cycle leak if that call is ever dropped to reuse the canvas.
      spark?.geometry?.dispose();
      spark?.material?.dispose();
      spark?.dispose();
      scene?.clear();

      // dispose() releases the GPU objects; forceContextLoss releases the
      // context itself. Only the second one prevents a page that mounts this
      // repeatedly from exhausting the browser's per-page context budget and
      // silently killing the oldest viewer.
      renderer?.dispose();
      renderer?.forceContextLoss();

      canvas.remove();

      mesh = null;
      spark = null;
      scene = null;
      camera = null;
      pivot = null;
      renderer = null;
      three = null;
    },
  };
}
