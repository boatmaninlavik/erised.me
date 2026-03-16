import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface DpoSong {
  id: string;
  title: string | null;
  prompt: string;
  lyrics: string | null;
  tags: string | null;
  audio_url: string;
  num_frames: number | null;
  model: string;
  guest_id: string | null;
  user_id: string | null;
  created_at: string;
}
