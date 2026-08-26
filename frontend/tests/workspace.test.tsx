import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { WorkspaceProvider, useWorkspace } from "@/state/workspace";
import TabBar from "@/components/TabBar";
import { DOC } from "./fixtures";

function fetchJson(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

/** Probe that records the active tab id. */
function TabProbe() {
  const { state } = useWorkspace();
  return <span data-testid="active-tab">{state.tab}</span>;
}

describe("workspace shell", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn((url: string) => {
      if (url.includes("/api/documents")) {
        return Promise.resolve(fetchJson([DOC]));
      }
      if (url.includes("/pages/")) {
        return Promise.resolve(fetchJson({ document_id: DOC.document_id }));
      }
      return Promise.resolve(fetchJson({}, 404));
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  function makeUi() {
    return (
      <WorkspaceProvider>
        <TabProbe />
        <TabBar />
      </WorkspaceProvider>
    );
  }

  it("loads the preferred document on mount", async () => {
    render(makeUi());
    await waitFor(() => {
      expect(screen.getByTestId("tab-document")).toHaveAttribute(
        "aria-selected",
        "true",
      );
    });
    // preferred = The Undiscovered Self
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/documents"),
      expect.anything(),
    );
  });

  it("shows an explicit backend error when unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("network down")),
    );
    const { default: LibrarySidebar } = await import("@/components/LibrarySidebar");
    render(
      <WorkspaceProvider>
        <LibrarySidebar />
      </WorkspaceProvider>,
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(alert.textContent).toContain("backend unreachable");
  });

  it("switches tabs by click", async () => {
    const user = userEvent.setup();
    render(makeUi());
    await waitFor(() =>
      expect(screen.getByTestId("tab-chunks")).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId("tab-chunks"));
    expect(screen.getByTestId("tab-chunks")).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByTestId("active-tab")).toHaveTextContent("chunks");
  });

  it("supports keyboard arrow navigation between tabs", async () => {
    const user = userEvent.setup();
    render(makeUi());
    await waitFor(() =>
      expect(screen.getByTestId("tab-document")).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId("tab-document"));
    screen.getByTestId("tab-document").focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByTestId("tab-structure")).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByTestId("tab-document")).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("exposes roving tabindex for accessibility", async () => {
    const user = userEvent.setup();
    render(makeUi());
    await waitFor(() =>
      expect(screen.getByTestId("tab-retrieval")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("tab-document")).toHaveAttribute("tabindex", "0");
    expect(screen.getByTestId("tab-structure")).toHaveAttribute("tabindex", "-1");
    await user.click(screen.getByTestId("tab-retrieval"));
    expect(screen.getByTestId("tab-retrieval")).toHaveAttribute("tabindex", "0");
    expect(screen.getByTestId("tab-document")).toHaveAttribute("tabindex", "-1");
  });
});
