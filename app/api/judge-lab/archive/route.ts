import type { NextRequest } from "next/server";
import { enabled, notFound, queueArchive } from "@/lib/judge-lab";

export const dynamic = "force-dynamic";

// Manual re-sync of the permanent archive (votes already trigger it automatically).
export async function POST(req: NextRequest) {
  if (!enabled()) return notFound();
  await queueArchive(req.nextUrl.searchParams.get("set") ?? "");
  return Response.json({ ok: true });
}
