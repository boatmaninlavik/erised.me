import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

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

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  const { data } = await supabase
    .from("erised_config")
    .select("value")
    .eq("key", "backend_url")
    .single();

  const backendUrl = data?.value?.trim();
  if (!backendUrl) {
    return NextResponse.json({ error: "GPU backend is offline" }, { status: 503 });
  }

  try {
    const resp = await fetch(`${backendUrl}/audio/${sanitized}`, {
      signal: AbortSignal.timeout(30000),
    });
    if (!resp.ok) {
      return NextResponse.json(
        { error: `Audio file not found (${resp.status})` },
        { status: resp.status }
      );
    }

    const arrayBuffer = await resp.arrayBuffer();
    const contentType = resp.headers.get("content-type") || "audio/wav";

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
