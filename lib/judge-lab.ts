// Server-only helpers for the local blind A/B page (/judge).
// Pair sets + votes live in gs://erised-dpo/judge_lab/{pairsets,votes}/; the audio itself stays in
// Backblaze B2 (b2://bucket/path) or GCS (gs://bucket/path) and is streamed, never stored locally.
// Every route is disabled unless JUDGE_LAB=1 is set (only in .env.local), so a deploy
// without that env serves nothing — the audio must stay private.

import { Storage, type Bucket } from "@google-cloud/storage";
import type { Choice, ResultPair, Results, SetSummary, Side } from "./judge-lab-types";

export const enabled = () => process.env.JUDGE_LAB === "1";

const HOME_BUCKET = "erised-dpo";
const ROOT = "judge_lab";
/** Only blind model-vs-model sets are shown; older judge-testing sets stay in GCS untouched. */
const KIND = "model-ab";

// ---------- raw GCS shapes ----------

export interface Pair {
  uid: string;
  prompt: string;
  lyrics: string;
  a: string;
  b: string;
  /** which model made take a / take b — never sent to the page before the results */
  model: Record<Side, string>;
}

export interface PairSet {
  id: string;
  kind: string;
  title: string;
  blurb: string;
  models: Record<string, string>;
  pairs: Pair[];
}

export interface Vote {
  set: string;
  uid: string;
  choice: Choice;
  left: Side;
  ts: string;
  listened_ms: Record<Side, number>;
  rater: string;
  /** re-vote after a misclick */
  revised?: boolean;
}

// ---------- GCS access with a short in-memory cache ----------

let storage: Storage | null = null;
export const bucket = (name = HOME_BUCKET): Bucket => (storage ??= new Storage()).bucket(name);

const TTL_MS = 60_000;
const cache = new Map<string, { at: number; value: Promise<unknown> }>();

function cached<T>(key: string, load: () => Promise<T>): Promise<T> {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.value as Promise<T>;
  const value = load().catch((e) => {
    cache.delete(key);
    throw e;
  });
  cache.set(key, { at: Date.now(), value });
  return value;
}

async function readAllJson<T>(prefix: string): Promise<T[]> {
  const [files] = await bucket().getFiles({ prefix });
  const jsons = files.filter((f) => f.name.endsWith(".json"));
  const out: T[] = [];
  for (let i = 0; i < jsons.length; i += 24) {
    const chunk = await Promise.all(
      jsons.slice(i, i + 24).map(async (f) => JSON.parse((await f.download())[0].toString("utf8")) as T),
    );
    out.push(...chunk);
  }
  return out;
}

export const loadPairSets = () =>
  cached("pairsets", async () =>
    (await readAllJson<PairSet>(`${ROOT}/pairsets/`)).filter((s) => s.kind === KIND).sort((a, b) => b.id.localeCompare(a.id)),
  );

const loadSetVotes = (setId: string) => cached(`votes/${setId}`, () => readAllJson<Vote>(`${ROOT}/votes/${safe(setId)}/`));

const safe = (s: string) => s.replace(/[^A-Za-z0-9_-]/g, "_");

export async function saveVote(v: Vote) {
  const path = `${ROOT}/votes/${safe(v.set)}/${safe(v.uid)}__${v.ts.replace(/[:.]/g, "-")}.json`;
  await bucket().file(path).save(JSON.stringify(v), { contentType: "application/json", resumable: false });
  // Append to the cached list instead of re-downloading every vote file.
  const key = `votes/${v.set}`;
  const hit = cache.get(key);
  if (hit) cache.set(key, { at: hit.at, value: (hit.value as Promise<Vote[]>).then((vs) => [...vs, v]) });
}

/** The newest vote per pair is the one that counts; older ones stay in GCS. */
export async function latestVotes(setId: string) {
  const latest = new Map<string, Vote>();
  for (const v of await loadSetVotes(setId)) {
    const prev = latest.get(v.uid);
    if (!prev || prev.ts < v.ts) latest.set(v.uid, v);
  }
  return latest;
}

export async function setSummaries(): Promise<SetSummary[]> {
  const sets = await loadPairSets();
  return Promise.all(
    sets.map(async (s) => ({ id: s.id, title: s.title, blurb: s.blurb, n: s.pairs.length, rated: (await latestVotes(s.id)).size })),
  );
}

// ---------- opaque audio keys (the page never sees bucket paths or file names) ----------

type AudioRef = { set: string; uid: string; side: Side };

export const audioKey = (ref: AudioRef) => Buffer.from(JSON.stringify(ref)).toString("base64url");

export async function resolveAudio(key: string): Promise<string | null> {
  let ref: AudioRef;
  try {
    ref = JSON.parse(Buffer.from(key, "base64url").toString("utf8"));
  } catch {
    return null;
  }
  if (ref.side !== "a" && ref.side !== "b") return null;
  const set = (await loadPairSets()).find((s) => s.id === ref.set);
  return set?.pairs.find((p) => p.uid === ref.uid)?.[ref.side] ?? null;
}

export function parseUri(uri: string) {
  const m = /^(gs|b2):\/\/([^/]+)\/(.+)$/.exec(uri);
  if (!m) throw new Error(`not a gs:// or b2:// uri: ${uri}`);
  return { scheme: m[1] as "gs" | "b2", bucket: m[2], path: m[3] };
}

// ---------- Backblaze B2 (native API: one login per ~day, then plain GETs with Range) ----------

type B2Login = { token: string; downloadUrl: string; apiUrl: string; accountId: string; at: number };
let b2Login: Promise<B2Login> | null = null;
const B2_TOKEN_MS = 20 * 3600_000; // tokens last 24 h

async function b2Auth(fresh: boolean): Promise<B2Login> {
  const prev = !fresh && b2Login ? await b2Login.catch(() => null) : null;
  if (prev && Date.now() - prev.at < B2_TOKEN_MS) return prev;
  const id = process.env.B2_KEY_ID, key = process.env.B2_APP_KEY;
  if (!id || !key) throw new Error("B2_KEY_ID / B2_APP_KEY missing from .env.local");
  b2Login = fetch("https://api.backblazeb2.com/b2api/v3/b2_authorize_account", {
    headers: { Authorization: `Basic ${Buffer.from(`${id}:${key}`).toString("base64")}` },
  })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`B2 login ${r.status}`))))
    .then((j) => ({
      token: j.authorizationToken, downloadUrl: j.apiInfo.storageApi.downloadUrl,
      apiUrl: j.apiInfo.storageApi.apiUrl, accountId: j.accountId, at: Date.now(),
    }));
  return b2Login;
}

async function b2Api<T>(op: string, body: unknown): Promise<T> {
  for (const fresh of [false, true]) {
    const auth = await b2Auth(fresh);
    const res = await fetch(`${auth.apiUrl}/b2api/v3/${op}`, {
      method: "POST", headers: { Authorization: auth.token }, body: JSON.stringify(body),
    });
    if (res.status === 401) continue;
    if (!res.ok) throw new Error(`B2 ${op} ${res.status}: ${await res.text()}`);
    return res.json() as Promise<T>;
  }
  throw new Error("B2 refused the login twice");
}

const bucketIds = new Map<string, Promise<string>>();
function b2BucketId(name: string) {
  if (!bucketIds.has(name)) {
    bucketIds.set(name, b2Auth(false).then((a) =>
      b2Api<{ buckets: { bucketId: string }[] }>("b2_list_buckets", { accountId: a.accountId, bucketName: name })
        .then((r) => r.buckets[0]?.bucketId ?? Promise.reject(new Error(`no B2 bucket ${name}`)))));
  }
  return bucketIds.get(name)!;
}

async function b2List(bucketName: string, prefix: string) {
  const r = await b2Api<{ files: { fileName: string; fileId: string }[] }>("b2_list_file_names", {
    bucketId: await b2BucketId(bucketName), prefix, maxFileCount: 1000,
  });
  return new Map(r.files.map((f) => [f.fileName, f.fileId]));
}

async function b2Upload(bucketName: string, path: string, body: string, contentType: string) {
  const { createHash } = await import("node:crypto");
  const up = await b2Api<{ uploadUrl: string; authorizationToken: string }>("b2_get_upload_url", { bucketId: await b2BucketId(bucketName) });
  const res = await fetch(up.uploadUrl, {
    method: "POST",
    headers: {
      Authorization: up.authorizationToken,
      "X-Bz-File-Name": path.split("/").map(encodeURIComponent).join("/"),
      "Content-Type": contentType,
      "X-Bz-Content-Sha1": createHash("sha1").update(body).digest("hex"),
    },
    body,
  });
  if (!res.ok) throw new Error(`B2 upload ${path} ${res.status}: ${await res.text()}`);
}

/** GET a B2 file, forwarding the Range header; logs in again once if the token was rejected. */
export async function b2Fetch(bucketName: string, path: string, range: string | null) {
  for (const fresh of [false, true]) {
    const auth = await b2Auth(fresh);
    const url = `${auth.downloadUrl}/file/${bucketName}/${path.split("/").map(encodeURIComponent).join("/")}`;
    const res = await fetch(url, { headers: { Authorization: auth.token, ...(range ? { Range: range } : {}) } });
    if (res.status !== 401) return res;
  }
  throw new Error("B2 refused the login twice");
}

// ---------- permanent archive of everything you rated ----------
// B2 <bucket>/blind_ab/<set>/: songs/<uid>_<model>.mp3 (server-side copies of the exact files you heard)
// + ratings.jsonl (one line per rated pair, newest vote) + README.md. Rebuilt after every vote.

const ARCHIVE_README = `# Blind A/B ratings archive

Written by the local /judge page (erised.me) after every vote.

- songs/<pair>_<model>.mp3 — both takes of every pair you rated, exactly as you heard them
  (server-side copies, so they survive even if the eval folders are cleaned up)
- ratings.jsonl — one line per rated pair (your newest vote): prompt, lyrics, which take was shown
  as A and B, which model made each, your pick, seconds listened, time of the vote
- Every individual vote (including changed picks) also stays in gs://erised-dpo/judge_lab/votes/<set>/
`;

let archiveChain: Promise<void> = Promise.resolve();

/** Queue a full (idempotent) archive sync; runs one at a time and never fails a vote. */
export function queueArchive(setId: string) {
  archiveChain = archiveChain
    .then(async () => {
      const r = await archiveSet(setId);
      console.log(`[judge-lab] archive ${setId}: ${r.rated} rated pairs, ${r.copied} new song copies`);
    })
    .catch((e) => console.error(`[judge-lab] archive of ${setId} failed:`, e));
  return archiveChain;
}

export async function archiveSet(setId: string) {
  const set = (await loadPairSets()).find((s) => s.id === setId);
  if (!set) return { copied: 0, rated: 0 };
  const first = set.pairs.map((p) => parseUri(p.a)).find((u) => u.scheme === "b2");
  if (!first) return { copied: 0, rated: 0 };
  const home = first.bucket;
  const prefix = `blind_ab/${set.id}/`;
  const latest = await latestVotes(set.id);
  const have = await b2List(home, prefix);
  let copied = 0;
  const lines: string[] = [];

  for (const p of set.pairs) {
    const v = latest.get(p.uid);
    if (!v) continue;
    const files: Partial<Record<Side, string>> = {};
    for (const side of ["a", "b"] as const) {
      const src = parseUri(p[side]);
      const dst = `${prefix}songs/${p.uid}_${p.model[side]}.${src.path.split(".").pop()}`;
      files[side] = `b2://${home}/${dst}`;
      if (have.has(dst) || src.scheme !== "b2") continue;
      const srcId = (await b2List(src.bucket, src.path)).get(src.path);
      if (!srcId) throw new Error(`missing source ${p[side]}`);
      await b2Api("b2_copy_file", {
        sourceFileId: srcId, fileName: dst,
        ...(src.bucket !== home ? { destinationBucketId: await b2BucketId(home) } : {}),
      });
      copied++;
    }
    const shown = { A: v.left, B: v.left === "a" ? "b" : "a" } as const;
    const take = (s: Side) => ({ model: p.model[s], name: set.models[p.model[s]], file: files[s] });
    lines.push(JSON.stringify({
      set: set.id, uid: p.uid, voted_at: v.ts, rater: v.rater, revised: v.revised ?? false,
      pick: v.choice === "tie" ? "tie" : v.choice === shown.A ? "A" : "B",
      picked_model: v.choice === "tie" ? "tie" : p.model[v.choice],
      takes: { A: take(shown.A), B: take(shown.B) },
      listened_s: { A: Math.round(v.listened_ms[shown.A] / 1000), B: Math.round(v.listened_ms[shown.B] / 1000) },
      prompt: p.prompt, lyrics: p.lyrics,
    }));
  }
  await b2Upload(home, `${prefix}ratings.jsonl`, lines.join("\n") + "\n", "application/x-ndjson");
  if (!have.has(`${prefix}README.md`)) await b2Upload(home, `${prefix}README.md`, ARCHIVE_README, "text/markdown");
  return { copied, rated: lines.length };
}

// ---------- reveal one rated pair ----------

/** Which model made each take — only for pairs you've already voted on. */
export async function revealPair(setId: string, uid: string) {
  const set = (await loadPairSets()).find((s) => s.id === setId);
  const pair = set?.pairs.find((p) => p.uid === uid);
  if (!set || !pair || !(await latestVotes(set.id)).has(uid)) return null;
  const ids = Object.keys(set.models);
  const info = (s: Side) => ({ id: pair.model[s], name: set.models[pair.model[s]], newer: pair.model[s] === ids[ids.length - 1] });
  return { a: info("a"), b: info("b") };
}

// ---------- results (unblinded) ----------

export async function buildResults(setId: string): Promise<Results | null> {
  const set = (await loadPairSets()).find((s) => s.id === setId);
  if (!set) return null;
  const latest = await latestVotes(set.id);
  const wins: Record<string, number> = Object.fromEntries(Object.keys(set.models).map((m) => [m, 0]));
  let ties = 0;
  const pairs: ResultPair[] = set.pairs.map((p) => {
    const choice = latest.get(p.uid)?.choice ?? null;
    if (choice === "tie") ties++;
    else if (choice) wins[p.model[choice]] = (wins[p.model[choice]] ?? 0) + 1;
    return {
      uid: p.uid,
      prompt: p.prompt,
      model: p.model,
      audio: { a: audioKey({ set: set.id, uid: p.uid, side: "a" }), b: audioKey({ set: set.id, uid: p.uid, side: "b" }) },
      choice,
    };
  });
  return {
    set: { id: set.id, title: set.title },
    models: set.models,
    wins, ties, rated: latest.size, total: set.pairs.length, pairs,
  };
}

export const notFound = () => new Response("Not found", { status: 404 });
