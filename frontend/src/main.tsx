import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, HashRouter, Route, Routes } from "react-router-dom";
import "./styles/index.css";
import { Layout } from "./components/Layout";
import { UiLangProvider } from "./i18n";
import { SignLanguageProvider } from "./state/language";

const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const LivePage = lazy(() => import("./pages/LivePage"));
const AnalyzePage = lazy(() => import("./pages/AnalyzePage"));
const VocabularyPage = lazy(() => import("./pages/VocabularyPage"));
const StudioPage = lazy(() => import("./pages/StudioPage"));
const DatasetPage = lazy(() => import("./pages/DatasetPage"));
const LabPage = lazy(() => import("./pages/LabPage"));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage"));
const ExperimentsPage = lazy(() => import("./pages/ExperimentsPage"));
const ResearchPage = lazy(() => import("./pages/ResearchPage"));
const MySignsPage = lazy(() => import("./pages/MySignsPage"));

// Static hosting (e.g. a Hugging Face static Space) cannot rewrite deep links, so use hash URLs there.
const Router = import.meta.env.VITE_STATIC === "1" ? HashRouter : BrowserRouter;

function NotFound() {
  return <p className="text-sm text-fg-2">Page not found.</p>;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <UiLangProvider>
    <SignLanguageProvider>
    <Router>
      <Layout>
        <Suspense fallback={<p className="text-sm text-fg-3">Loading…</p>}>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/live" element={<LivePage />} />
            <Route path="/my-signs" element={<MySignsPage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/vocabulary" element={<VocabularyPage />} />
            <Route path="/studio" element={<StudioPage />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/lab" element={<LabPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/research" element={<ResearchPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Layout>
    </Router>
    </SignLanguageProvider>
    </UiLangProvider>
  </StrictMode>,
);
