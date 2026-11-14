import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./styles/index.css";
import { Layout } from "./components/Layout";

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

function NotFound() {
  return <p className="text-sm text-fg-2">Page not found.</p>;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Layout>
        <Suspense fallback={<p className="text-sm text-fg-3">Loading…</p>}>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/live" element={<LivePage />} />
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
    </BrowserRouter>
  </StrictMode>,
);
