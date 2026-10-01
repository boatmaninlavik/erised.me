import type { NextRequest } from "next/server";
import { audioKey, enabled, latestVotes, loadPairSets, notFound } from "@/lib/judge-lab";
import type { BlindSet } from "@/lib/judge-lab-types";

export const dynamic = "force-dynamic";

// Pairs for blind rating: no model names, no file names. Your past votes come back with the
// A/B layout you heard them in, so you can find a take again.
export async function GET(req: NextRequest) {
  if (!enabled()) return notFound();
  const id = req.nextUrl.searchParams.get("id");
  const set = (await loadPairSets()).find((s) => s.id === id);
  if (!set) return notFound();
  const latest = [...(await latestVotes(set.id)).values()];
  const body: BlindSet = {
    id: set.id,
    title: set.title,
    rated: latest.map((v) => v.uid),
    history: latest
      .map(({ uid, choice, left, ts }) => ({ uid, choice, left, ts }))
      .sort((x, y) => y.ts.localeCompare(x.ts)),
    pairs: set.pairs.map((p) => ({
      uid: p.uid,
      prompt: p.prompt,
      lyrics: p.lyrics,
      audio: {
        a: audioKey({ set: set.id, uid: p.uid, side: "a" }),
        b: audioKey({ set: set.id, uid: p.uid, side: "b" }),
      },
    })),
  };
  return Response.json(body);
}
