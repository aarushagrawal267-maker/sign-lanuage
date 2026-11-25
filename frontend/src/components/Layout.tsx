import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  Activity, BookOpen, Boxes, Database, FileVideo, FlaskConical, Gauge, History, LayoutDashboard, Menu, ScanFace, Video, X,
} from "lucide-react";
import { useApi } from "../lib/hooks";
import { IS_STATIC, type ModelInfo, type SystemInfo } from "../api/client";
import { Badge, cx } from "./ui";

const NAV: { group: string; items: { to: string; label: string; icon: ReactNode }[] }[] = [
  {
    group: "Recognise",
    items: [
      { to: "/", label: "Overview", icon: <LayoutDashboard className="size-4" /> },
      { to: "/live", label: "Live Recognition", icon: <Video className="size-4" /> },
      { to: "/analyze", label: "Video Analysis", icon: <FileVideo className="size-4" /> },
    ],
  },
  {
    group: "Explore",
    items: [
      { to: "/vocabulary", label: "Vocabulary", icon: <Boxes className="size-4" /> },
      { to: "/studio", label: "Landmark Studio", icon: <ScanFace className="size-4" /> },
      { to: "/dataset", label: "Dataset", icon: <Database className="size-4" /> },
    ],
  },
  {
    group: "Model",
    items: [
      { to: "/lab", label: "Model Laboratory", icon: <FlaskConical className="size-4" /> },
      { to: "/evaluation", label: "Evaluation", icon: <Gauge className="size-4" /> },
      { to: "/experiments", label: "Experiment History", icon: <History className="size-4" /> },
    ],
  },
  {
    group: "Learn",
    items: [{ to: "/research", label: "Research & Docs", icon: <BookOpen className="size-4" /> }],
  },
];

const TITLES: Record<string, string> = Object.fromEntries(NAV.flatMap((g) => g.items.map((it) => [it.to, it.label])));

/** Mobile navigation drawer: modal dialog that takes focus, traps Tab, closes on Escape and returns focus to the menu button. */
function NavDrawer({ onClose }: { onClose: () => void }) {
  const panel = useRef<HTMLElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeBtn.current?.focus();
    return () => opener?.focus();
  }, []);
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
    if (e.key !== "Tab" || !panel.current) return;
    const items = panel.current.querySelectorAll<HTMLElement>("a[href], button:not([disabled])");
    if (!items.length) return;
    const first = items[0], last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };
  return (
    <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation" id="nav-drawer" onKeyDown={onKeyDown}>
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <aside ref={panel} className="absolute inset-y-0 left-0 flex w-72 flex-col overflow-y-auto border-r border-ink-700 bg-ink-900 px-3 py-5">
        <button ref={closeBtn} className="absolute right-3 top-4 rounded-md p-1.5 text-fg-2 hover:bg-ink-800" onClick={onClose} aria-label="Close navigation">
          <X className="size-5" />
        </button>
        <SidebarContent onNavigate={onClose} />
      </aside>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
        <rect width="32" height="32" rx="8" fill="#1b1f25" />
        <g fill="none" stroke="#4c8dff" strokeWidth="2.2" strokeLinecap="round"><path d="M11 23V12M15 23V9M19 23V10M23 23v-8" /></g>
        <g fill="#2dd4bf"><circle cx="11" cy="12" r="1.8" /><circle cx="15" cy="9" r="1.8" /><circle cx="19" cy="10" r="1.8" /><circle cx="23" cy="15" r="1.8" /></g>
      </svg>
      <div className="leading-tight">
        <div className="text-[13px] font-semibold text-fg">Sign Intelligence</div>
        <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-3">Studio · WLASL100</div>
      </div>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <Brand />
      <nav className="mt-6 flex flex-col gap-5" aria-label="Main">
        {NAV.map((g) => (
          <div key={g.group}>
            <div className="mb-1.5 px-3 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-3">{g.group}</div>
            <ul className="flex flex-col gap-0.5">
              {g.items.map((it) => (
                <li key={it.to}>
                  <NavLink
                    to={it.to}
                    end={it.to === "/"}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      cx(
                        "flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] transition-colors",
                        isActive ? "bg-ink-800 font-medium text-fg shadow-[inset_2px_0_0_var(--color-accent)]" : "text-fg-2 hover:bg-ink-800/60 hover:text-fg",
                      )
                    }
                  >
                    {it.icon}
                    {it.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div className="mt-auto rounded-xl border border-ink-700 bg-ink-850 p-3 text-[11px] leading-relaxed text-fg-3">
        Educational &amp; research implementation. Recognises isolated WLASL100 signs as ASL <span className="text-fg-2">gloss labels</span> — not
        sentence translation. Not a replacement for interpreters.
      </div>
    </>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const { data: health, error } = useApi<{ ok: boolean }>("/api/health", 10000);
  const { data: model } = useApi<ModelInfo | { experiment_id: null }>("/api/models/active", 15000);
  const { data: sys } = useApi<SystemInfo>("/api/system");
  const loc = useLocation();
  const m = model && model.experiment_id !== null ? (model as ModelInfo) : null;
  const mainRef = useRef<HTMLElement>(null);
  const prevPath = useRef(loc.pathname);

  // Give every view its own title (WCAG 2.4.2) and, after in-app navigation, move focus to the new content
  // so keyboard and screen-reader users start there instead of at the link they activated.
  useEffect(() => {
    const page = TITLES[loc.pathname];
    document.title = page ? `${page} · Sign Intelligence Studio` : "Sign Intelligence Studio";
    if (prevPath.current === loc.pathname) return; // first render (effects run twice in StrictMode)
    prevPath.current = loc.pathname;
    mainRef.current?.focus({ preventScroll: true });
  }, [loc.pathname]);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_1fr]">
      <a href="#main" className="sr-only rounded-lg bg-ink-800 px-3 py-2 text-sm font-medium text-fg focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50"
        onClick={(e) => { e.preventDefault(); mainRef.current?.focus(); }}>
        Skip to main content
      </a>
      <aside className="sticky top-0 hidden h-screen flex-col overflow-y-auto border-r border-ink-700 bg-ink-900 px-3 py-5 lg:flex">
        <SidebarContent />
      </aside>

      {open && <NavDrawer onClose={() => setOpen(false)} />}

      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-ink-700 bg-ink-950/85 px-4 backdrop-blur md:px-8">
          <button className="rounded-md p-1.5 text-fg-2 hover:bg-ink-800 lg:hidden" onClick={() => setOpen(true)} aria-label="Open navigation" aria-expanded={open} aria-controls="nav-drawer">
            <Menu className="size-5" />
          </button>
          <div className="flex min-w-0 flex-1 items-center gap-2 text-xs text-fg-3">
            <Activity className="size-3.5" aria-hidden />
            <span className="truncate font-mono">{loc.pathname === "/" ? "overview" : loc.pathname.slice(1)}</span>
          </div>
          <div className="flex items-center gap-2">
            {m ? (
              <Badge tone={m.is_synthetic ? "warn" : "accent"}>
                <span className="hidden sm:inline">model</span> #{m.experiment_id} · {m.architecture.toUpperCase()}
                {m.is_synthetic && " · synthetic"}
              </Badge>
            ) : (
              <Badge>no active model</Badge>
            )}
            <span
              className={cx("inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px]", health?.ok && !error ? "border-good/40 text-[#5fd35f]" : "border-danger/40 text-danger")}
              title={error ?? (IS_STATIC ? "Model runs in this browser" : "Backend reachable")}
            >
              <span className={cx("size-1.5 rounded-full", health?.ok && !error ? "bg-good" : "bg-danger")} />
              {health?.ok && !error ? (IS_STATIC ? "in-browser" : "backend") : "backend offline"}
            </span>
          </div>
        </header>
        {sys?.public_demo && (
          <div role="note" className="border-b border-warn/30 bg-warn/[0.07] px-4 py-2 text-xs text-fg-2 md:px-8">
            <span className="font-medium text-warn">{sys.static ? "Online demo — runs entirely in your browser." : "Public read-only demo."}</span>{" "}
            {sys.static ? "Camera frames, landmarks and the model all stay on your device." : "Your camera is processed in your browser only."}{" "}
            Downloading WLASL data, training{sys.static ? " and video-file analysis" : ""} are disabled here — run the project locally for those.
          </div>
        )}
        <main id="main" ref={mainRef} tabIndex={-1} className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 outline-none md:px-8 md:py-8">{children}</main>
        <footer className="border-t border-ink-700 px-4 py-4 text-[11px] text-fg-3 md:px-8">
          WLASL data © the WLASL authors, used under the Computational Use of Data Agreement (C-UDA) — academic &amp; computational use only, no
          commercial use. Li et al., WACV 2020. Landmarks: Google MediaPipe (Apache-2.0).
        </footer>
      </div>
    </div>
  );
}
