// Which renderer draws the person, decided once, before the room connects.
//
// Two ways a face can reach the visitor. A built Gaussian splat rendered on
// their own hardware and moved by 148 bytes a frame, or the video track the
// server renders and publishes. The first is why this product can be sold at
// a price a family would pay; the second is what has always worked.
//
// The decision is made here, ahead of the connection, and then frozen. Not
// for tidiness: a source that could change while the call is running would
// change while somebody was mid-sentence with their father, and the swap
// would be the most visible thing on the screen.
//
// It is also silent. Nothing the visitor sees says which renderer they got.
// A family sitting down to this does not need to be told that their device
// failed a capability check, and there is no version of that sentence which
// is kind at the moment they are about to see a face they have been missing.

import {
  api,
  splatAssetUrl,
  type AvatarSplat,
  type SplatAsset,
} from "@/lib/gateway";
import { detectWebgl2 } from "@/lib/splat";

export type LikenessRenderer = "splat" | "video";

export interface LikenessPlan {
  renderer: LikenessRenderer;
  /** Non-null exactly when the renderer is "splat". */
  asset: SplatAsset | null;
  /**
   * What was built for this person, when anything was. Carried whichever
   * renderer won, because the disclosure it holds is about the likeness, not
   * about how this device happened to draw it — falling back to video must
   * not quietly retract a statement about how much of a face was generated.
   */
  splat: AvatarSplat | null;
}

const VIDEO_ONLY: LikenessPlan = { renderer: "video", asset: null, splat: null };

/**
 * Chooses the source for one avatar's call.
 *
 * Never rejects. Every reason a splat cannot be used — none built, no asset
 * behind it, no WebGL2 on this machine, or a gateway that does not answer the
 * question at all — resolves to the video track, which is the outcome that
 * has always worked. A thrown error here would put a failure on a screen
 * where nothing has actually failed.
 */
export async function resolveLikeness(avatarId: string): Promise<LikenessPlan> {
  let splat: AvatarSplat;
  try {
    splat = await api.avatarSplat(avatarId);
  } catch {
    // Includes a gateway with no splat route on it yet. Not knowing whether a
    // likeness was built is the same situation as none having been.
    return VIDEO_ONLY;
  }

  if (!splat.built) return { renderer: "video", asset: null, splat };

  const asset = splatAssetUrl(splat);
  if (asset === null) return { renderer: "video", asset: null, splat };

  // Checked before connecting rather than discovered by the viewer, because
  // the whole point of deciding early is that this answer is already known
  // when the room comes up. The probe destroys its own context; see splat.ts.
  if (!detectWebgl2().supported) return { renderer: "video", asset: null, splat };

  return { renderer: "splat", asset, splat };
}
