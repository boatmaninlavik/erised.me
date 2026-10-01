import type { NextRequest } from "next/server";
import { enabled, notFound, revealPair } from "@/lib/judge-lab";

export const dynamic = "force-dynamic";

// Which model made each take of ONE pair — refuses pairs you haven't voted on yet.
export async function GET(req: NextRequest) {
  if (!enabled()) return notFound();
  const q = req.nextUrl.searchParams;
  const r = await revealPair(q.get("set") ?? "", q.get("uid") ?? "");
  return r ? Response.json(r) : notFound();
}
