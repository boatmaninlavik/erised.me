import { NextRequest, NextResponse } from "next/server";
import { GEN_URL } from "@/lib/backend";

// Fetches a finished song from the generator so the browser can save it to Supabase.
export async function GET(req: NextRequest) {
  const file = req.nextUrl.searchParams.get("file");
  if (!file) {
    return NextResponse.json({ error: "Missing file parameter" }, { status: 400 });
  }

  // Sanitize filename to prevent path traversal
  const sanitized = file.replace(/[^a-zA-Z0-9._-]/g, "");
  if (!sanitized) {
    return NextResponse.json({ error: "Invalid filename" }, { status: 400 });
  }

  try {
    const resp = await fetch(`${GEN_URL}/audio/${sanitized}`, {
      signal: AbortSignal.timeout(30000),
    });
    if (!resp.ok) {
      return NextResponse.json(
        { error: `Audio file not found (${resp.status})` },
        { status: resp.status }
      );
    }

    const arrayBuffer = await resp.arrayBuffer();
    const contentType = resp.headers.get("content-type") || "audio/mpeg";

    return new NextResponse(arrayBuffer, {
      headers: { "Content-Type": contentType },
    });
  } catch {
    return NextResponse.json(
      { error: "Could not reach GPU backend" },
      { status: 502 }
    );
  }
}
