// Shapes shared by the blind A/B API routes and the /judge page.
// Pair sets in GCS are written by the eval jobs (e.g. shao_dpo/eval_sft.py::listen_prep).

export type Side = "a" | "b";
export type Choice = Side | "tie";

export interface SetSummary {
  id: string;
  title: string;
  blurb: string;
  n: number;
  rated: number;
}

export interface BlindPair {
  uid: string;
  prompt: string;
  lyrics: string;
  audio: Record<Side, string>; // opaque keys for /api/judge-lab/audio
}

/** One of your votes, with the A/B layout you heard it in (`left` was shown as A). */
export interface HistoryItem {
  uid: string;
  choice: Choice;
  left: Side;
  ts: string;
}

export interface BlindSet {
  id: string;
  title: string;
  pairs: BlindPair[];
  rated: string[];
  /** newest first */
  history: HistoryItem[];
}

/** Unblinded results: only fetched once you finish the set (or unlock early). */
export interface ResultPair {
  uid: string;
  prompt: string;
  /** which model made each take */
  model: Record<Side, string>;
  audio: Record<Side, string>;
  choice: Choice | null;
}

export interface Results {
  set: { id: string; title: string };
  /** model id -> display name */
  models: Record<string, string>;
  /** model id -> pairs where your pick was that model's take */
  wins: Record<string, number>;
  ties: number;
  rated: number;
  total: number;
  pairs: ResultPair[];
}

/** Which model made each take of a pair you've voted on (newer = the model being tested). */
export type Revealed = Record<Side, { id: string; name: string; newer: boolean }>;
