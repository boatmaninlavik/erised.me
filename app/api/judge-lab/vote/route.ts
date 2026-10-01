import { enabled, loadPairSets, notFound, queueArchive, saveVote } from "@/lib/judge-lab";
import type { Choice, Side } from "@/lib/judge-lab-types";

export const dynamic = "force-dynamic";

const isSide = (x: unknown): x is Side => x === "a" || x === "b";
const isChoice = (x: unknown): x is Choice => isSide(x) || x === "tie";

// Stores one blind vote, then (in the background) copies both takes + the ratings list into the
// permanent B2 archive. Nothing about which model made which take comes back.
export async function POST(req: Request) {
  if (!enabled()) return notFound();
  const body = await req.json().catch(() => null);
  if (!body || typeof body.set !== "string" || typeof body.uid !== "string" ||
      !isChoice(body.choice) || !isSide(body.left)) {
    return Response.json({ error: "bad vote" }, { status: 400 });
  }
  const set = (await loadPairSets()).find((s) => s.id === body.set);
  if (!set?.pairs.some((p) => p.uid === body.uid)) return Response.json({ error: "unknown pair" }, { status: 404 });

  const ms = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? Math.max(0, Math.round(x)) : 0);
  await saveVote({
    set: body.set,
    uid: body.uid,
    choice: body.choice,
    left: body.left,
    ts: new Date().toISOString(),
    listened_ms: { a: ms(body.listened_ms?.a), b: ms(body.listened_ms?.b) },
    rater: "sean",
    ...(body.revised === true ? { revised: true } : {}),
  });
  void queueArchive(body.set);
  return Response.json({ ok: true });
}
