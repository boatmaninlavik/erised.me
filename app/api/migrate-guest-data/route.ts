import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { createClient } from "@supabase/supabase-js";

export async function POST(req: Request) {
  const cookieStore = await cookies();
  const guestId = cookieStore.get("erised_guest_id")?.value;

  // Verify the user's identity via their Supabase access token
  const token = req.headers.get("authorization")?.replace("Bearer ", "");
  if (!token) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser(token);

  if (authError || !user) {
    return NextResponse.json({ error: "Invalid session" }, { status: 401 });
  }

  let migrated = 0;

  // 1. Link guest songs to this user account
  if (guestId) {
    const { data, error } = await supabase
      .from("dpo-songs")
      .update({ user_id: user.id })
      .eq("guest_id", guestId)
      .is("user_id", null)
      .select("id");

    if (!error && data) migrated += data.length;
  }

  // 2. Claim any unclaimed legacy songs (no guest_id, no user_id).
  //    These are pre-system rows (Sean's original data).
  //    Only the first user to sign up gets them.
  const { data: legacy, error: legacyErr } = await supabase
    .from("dpo-songs")
    .update({ user_id: user.id })
    .is("guest_id", null)
    .is("user_id", null)
    .select("id");

  if (!legacyErr && legacy) migrated += legacy.length;

  return NextResponse.json({ migrated });
}
