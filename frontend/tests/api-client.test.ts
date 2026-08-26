import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { DOC } from "./fixtures";

describe("api client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetches documents list", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify([DOC]), { status: 200 }),
      ),
    );
    const docs = await api.documents();
    expect(docs).toHaveLength(1);
    expect(docs[0].title).toBe("The Undiscovered Self");
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/documents"),
      expect.anything(),
    );
  });

  it("throws typed ApiError on missing page", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "page out of range" }), {
          status: 404,
        }),
      ),
    );
    await expect(api.page(DOC.document_id, 9999)).rejects.toThrow(ApiError);
  });

  it("reports unreachable backend explicitly", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("network down")),
    );
    const err = await api.documents().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toContain("backend unreachable");
  });
});
