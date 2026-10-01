import { enabled, notFound, setSummaries } from "@/lib/judge-lab";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!enabled()) return notFound();
  return Response.json(await setSummaries());
}
