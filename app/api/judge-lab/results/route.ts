import type { NextRequest } from "next/server";
import { buildResults, enabled, notFound } from "@/lib/judge-lab";

export const dynamic = "force-dynamic";

// Unblinded: which model made each take, next to your pick. The page only asks for this once
// you've rated every pair, or after you explicitly unlock it early.
export async function GET(req: NextRequest) {
  if (!enabled()) return notFound();
  const results = await buildResults(req.nextUrl.searchParams.get("id") ?? "");
  return results ? Response.json(results) : notFound();
}
