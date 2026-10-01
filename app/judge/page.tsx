import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { enabled } from "@/lib/judge-lab";
import { JudgeLab } from "./judge-lab";

/**
 * LOCAL-ONLY blind A/B page (not linked from the navbar). Needs in .env.local:
 *   JUDGE_LAB=1
 *   GOOGLE_APPLICATION_CREDENTIALS=<service-account key with access to gs://erised-dpo>  (pair sets + votes)
 *   B2_KEY_ID, B2_APP_KEY                                                               (audio stored in B2)
 * then `npm run dev` and open /judge. Without JUDGE_LAB=1 the page and its API 404.
 * Pair sets: gs://erised-dpo/judge_lab/pairsets/*.json with kind "model-ab" (e.g. shao_dpo/eval_sft.py::listen_prep).
 */
export const metadata: Metadata = { title: "Blind A/B · Erised" };
export const dynamic = "force-dynamic";

export default function JudgeLabPage() {
  if (!enabled()) notFound();
  return <JudgeLab />;
}
