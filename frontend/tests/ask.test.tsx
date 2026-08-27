import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, afterEach } from "vitest";
import AskTab from "@/components/tabs/AskTab";
import { WorkspaceProvider } from "@/state/workspace";
import { ASK_RESPONSE } from "./fixtures";

function stubAskFetch(responseBody: unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(() => {
      return Promise.resolve(new Response(JSON.stringify(responseBody), { status }));
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function renderAskTab() {
  return render(
    <WorkspaceProvider>
      <AskTab />
    </WorkspaceProvider>,
  );
}

describe("AskTab", () => {
  it("renders the ASK header, input, and submit button", () => {
    renderAskTab();
    expect(screen.getByText("ASK JUNG ARCHIVE")).toBeInTheDocument();
    expect(screen.getByTestId("ask-input")).toBeInTheDocument();
    expect(screen.getByTestId("ask-submit")).toHaveTextContent("ASK");
  });

  it("submit is enabled when query is non-empty", async () => {
    const user = userEvent.setup();
    renderAskTab();
    const btn = screen.getByTestId("ask-submit");
    const input = screen.getByTestId("ask-input");

    expect(btn).toBeDisabled();
    await user.type(input, "How does Jung describe the Self?");
    expect(btn).not.toBeDisabled();
  });

  it("Enter submits and Shift+Enter inserts newline", async () => {
    const user = userEvent.setup();
    renderAskTab();
    const input = screen.getByTestId("ask-input") as HTMLTextAreaElement;

    // Enter submits (prevented by handler), Shift+Enter inserts newline.
    stubAskFetch(ASK_RESPONSE);
    await user.type(input, "individuation{Enter}");
    await user.type(input, "{Shift>}{Enter}{/Shift}");
    expect(input).toHaveValue("individuation\n");
  });

  it("renders answer and source cards after a successful ask", async () => {
    const user = userEvent.setup();
    stubAskFetch(ASK_RESPONSE);

    renderAskTab();
    const input = screen.getByTestId("ask-input");
    const btn = screen.getByTestId("ask-submit");

    await user.type(input, "How does Jung describe the Self?");
    await user.click(btn);

    expect(await screen.findByTestId("answer")).toBeInTheDocument();
    expect(screen.getByTestId("answer").textContent).toContain(
      ASK_RESPONSE.answer.slice(0, 40),
    );
    expect(screen.getByTestId("source-S1")).toBeInTheDocument();
    expect(screen.getByTestId("provider-mode")).toHaveTextContent(
      ASK_RESPONSE.local_or_remote,
    );
  });

  it("shows citation valid/unknown status badges", async () => {
    const user = userEvent.setup();
    stubAskFetch(ASK_RESPONSE);

    renderAskTab();
    const input = screen.getByTestId("ask-input");
    const btn = screen.getByTestId("ask-submit");

    await user.type(input, "test");
    await user.click(btn);

    // S1 is in the evidence pack so its source card is rendered with status "valid".
    // S2 is cited but not in the pack, surfaced as a citation-validation warning.
    expect(await screen.findByTestId("source-S1")).toBeInTheDocument();
    expect(screen.getByText("valid")).toBeInTheDocument();
    expect(screen.getByText(/unknown citation\(s\): S2/)).toBeInTheDocument();
  });

  it("shows API error on non-200", async () => {
    const user = userEvent.setup();
    stubAskFetch({ detail: "generation endpoint unreachable" }, 502);

    renderAskTab();
    const input = screen.getByTestId("ask-input");
    const btn = screen.getByTestId("ask-submit");

    await user.type(input, "test");
    await user.click(btn);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("502");
  });

  it("shows retrieval trace with stage data", async () => {
    const user = userEvent.setup();
    stubAskFetch(ASK_RESPONSE);

    renderAskTab();
    const input = screen.getByTestId("ask-input");
    const btn = screen.getByTestId("ask-submit");

    await user.type(input, "test");
    await user.click(btn);

    const summary = await screen.findByText("VIEW RETRIEVAL TRACE");
    expect(summary).toBeInTheDocument();
    expect(screen.getByText("hybrid")).toBeInTheDocument();
    expect(screen.getByText(/1634 \/ 2500/)).toBeInTheDocument();
  });

  it("shows warnings when present", async () => {
    const user = userEvent.setup();
    stubAskFetch(ASK_RESPONSE);

    renderAskTab();
    const input = screen.getByTestId("ask-input");
    const btn = screen.getByTestId("ask-submit");

    await user.type(input, "test");
    await user.click(btn);

    expect(await screen.findByRole("status")).toBeInTheDocument();
    expect(screen.getByText(/generation provider is REMOTE/)).toBeInTheDocument();
  });
});
