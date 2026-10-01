import { Readable } from "node:stream";
import type { NextRequest } from "next/server";
import { b2Fetch, bucket, enabled, notFound, parseUri, resolveAudio } from "@/lib/judge-lab";

export const dynamic = "force-dynamic";

const TYPES: Record<string, string> = { mp3: "audio/mpeg", m4a: "audio/mp4", wav: "audio/wav", webm: "audio/webm" };
const sizes = new Map<string, number>();

// Streams audio straight from B2 or GCS with HTTP Range support (seeking; Safari needs 206s).
export async function GET(req: NextRequest) {
  if (!enabled()) return notFound();
  const uri = await resolveAudio(req.nextUrl.searchParams.get("k") ?? "");
  if (!uri) return notFound();
  const { scheme, bucket: name, path } = parseUri(uri);
  const type = TYPES[path.split(".").pop()!.toLowerCase()] ?? "application/octet-stream";
  const rangeHeader = req.headers.get("range");

  if (scheme === "b2") {
    const res = await b2Fetch(name, path, rangeHeader);
    if (!res.ok) return new Response(null, { status: res.status === 416 ? 416 : 502 });
    const headers = new Headers({ "Content-Type": type, "Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600" });
    for (const h of ["Content-Length", "Content-Range"]) {
      const v = res.headers.get(h);
      if (v) headers.set(h, v);
    }
    return new Response(res.body, { status: res.status, headers });
  }

  const file = bucket(name).file(path);
  let size = sizes.get(uri);
  if (size === undefined) {
    const [meta] = await file.getMetadata();
    size = Number(meta.size);
    sizes.set(uri, size);
  }

  let start = 0;
  let end = size - 1;
  const range = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader ?? "");
  if (range) {
    if (range[1] === "") start = Math.max(0, size - Number(range[2]));
    else {
      start = Number(range[1]);
      if (range[2] !== "") end = Math.min(Number(range[2]), size - 1);
    }
    if (start > end || start >= size) {
      return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
    }
  }

  const headers = new Headers({
    "Content-Type": type,
    "Content-Length": String(end - start + 1),
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, max-age=3600",
  });
  if (range) headers.set("Content-Range", `bytes ${start}-${end}/${size}`);
  const stream = file.createReadStream({ start, end, validation: false });
  return new Response(Readable.toWeb(stream) as ReadableStream, { status: range ? 206 : 200, headers });
}
