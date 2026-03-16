-- Run this in your Supabase SQL Editor before deploying the auth update.
-- Dashboard: https://supabase.com/dashboard/project/zllpoyoumsfnwgfecryh/sql/new

-- 1. Add identity columns to dpo-songs table
ALTER TABLE "dpo-songs" ADD COLUMN IF NOT EXISTS guest_id TEXT;
ALTER TABLE "dpo-songs" ADD COLUMN IF NOT EXISTS user_id UUID;

-- 2. Index for fast per-user queries
CREATE INDEX IF NOT EXISTS idx_dpo_songs_guest_id ON "dpo-songs" (guest_id);
CREATE INDEX IF NOT EXISTS idx_dpo_songs_user_id ON "dpo-songs" (user_id);

-- 3. (Optional) Enable Supabase Auth settings:
--    - Go to Authentication > URL Configuration
--    - Add to "Redirect URLs": https://erised.me/reset-password
--    - Also add: http://localhost:3000/reset-password (for local dev)
--
--    - Go to Authentication > Providers > Email
--    - Toggle OFF "Confirm email" if you want immediate sign-in after signup
