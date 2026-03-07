import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-black relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-white/[0.02] rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-3xl w-full text-center space-y-10 z-10 px-6">
        <div className="space-y-6">
          <h1 className="text-6xl md:text-8xl font-semibold tracking-tighter text-white">
            Erised
          </h1>
          <p className="text-2xl md:text-3xl text-zinc-400 font-medium tracking-tight max-w-xl mx-auto leading-snug">
            The music of your desire.
          </p>
        </div>

        <div className="flex flex-col items-center gap-6 pt-4">
          <Link
            href="/generate"
            className="px-10 py-4 bg-white text-black font-semibold rounded-full text-lg tracking-tight hover:bg-zinc-100 transition-colors"
          >
            Try it out
          </Link>

          <div className="flex items-center gap-8 text-sm text-zinc-500">
            <Link href="/library" className="hover:text-zinc-300 transition-colors tracking-tight">
              My Library
            </Link>
            <span>·</span>
            <Link href="/rate" className="hover:text-zinc-300 transition-colors tracking-tight">
              Rate songs — let Erised know your taste
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
