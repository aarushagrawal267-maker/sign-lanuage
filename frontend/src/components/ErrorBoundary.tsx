import { Component, type ReactNode } from "react";

/** Keeps the layout and navigation usable when one page fails to render; resets when the page (resetKey) changes. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidUpdate(prev: { resetKey: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }
  componentDidCatch(error: Error) {
    console.error("Page error", error);
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="rounded-xl border border-danger/40 bg-danger/[0.07] px-4 py-3 text-sm text-fg-2">
        <div className="font-medium text-fg">This page hit an error / इस पेज में त्रुटि हुई</div>
        <p className="mt-1">{this.state.error.message}</p>
        <button type="button" className="mt-2 text-accent underline" onClick={() => this.setState({ error: null })}>Try again / फिर से कोशिश करें</button>
      </div>
    );
  }
}
