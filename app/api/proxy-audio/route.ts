import { NextRequest, NextResponse } from "next/server";

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

  const RUNPOD_URL = process.env.NEXT_PUBLIC_RUNPOD_URL;
  const MODAL_URL = "https://boatmaninlavik--erised-gpu-serve.modal.run";

  // Try RunPod first, fall back to Modal
  let backendUrl: string | null = null;
  if (RUNPOD_URL) {
    try {
      const h = await fetch(`${RUNPOD_URL}/health`, { signal: AbortSignal.timeout(2000) });
      if (h.ok) backendUrl = RUNPOD_URL;
    } catch {}
  }
  if (!backendUrl) backendUrl = MODAL_URL;

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
