import { describe, expect, it } from "vitest";
import {
  computeScale,
  PAD_X,
  PAD_Y,
} from "@/lib/viewerMath";

const PAGE_W = 360;
const PAGE_H = 576;

describe("PdfViewer fit math", () => {
  it("fit-width fills the available width minus padding", () => {
    const s = computeScale("width", 1.4, PAGE_W, PAGE_H, 792, 600);
    expect(s).toBeCloseTo((792 - PAD_X) / PAGE_W, 6);
  });

  it("fit-width ignores height (vertical scroll allowed)", () => {
    // very short box: width-fit still uses width
    const s = computeScale("width", 1.4, PAGE_W, PAGE_H, 792, 100);
    expect(s).toBeCloseTo((792 - PAD_X) / PAGE_W, 6);
  });

  it("fit-page fits BOTH axes so the whole page is visible", () => {
    const s = computeScale("page", 1.4, PAGE_W, PAGE_H, 792, 400);
    expect(s).toBeCloseTo(Math.min(
      (792 - PAD_X) / PAGE_W,
      (400 - PAD_Y) / PAGE_H,
    ), 6);
    expect(s * PAGE_H).toBeLessThanOrEqual(400 - PAD_Y);
  });

  it("fit-page picks the width constraint when that binds first", () => {
    const s = computeScale("page", 1.4, PAGE_W, PAGE_H, 500, 2000);
    expect(s).toBeCloseTo(Math.min(
      (500 - PAD_X) / PAGE_W,
      (2000 - PAD_Y) / PAGE_H,
    ), 6);
  });

  it("custom zoom passes through unchanged", () => {
    expect(computeScale("custom", 2.2, PAGE_W, PAGE_H, 792, 600)).toBe(2.2);
  });

  it("recomputes on resize: shrinking the box shrinks the scale", () => {
    const before = computeScale("width", 1.4, PAGE_W, PAGE_H, 1000, 800);
    const after = computeScale("width", 1.4, PAGE_W, PAGE_H, 500, 800);
    expect(after).toBeLessThan(before);
    expect(after).toBeCloseTo((500 - PAD_X) / PAGE_W, 6);
  });

  it("degenerate inputs never produce zero/negative/NaN scales", () => {
    expect(computeScale("width", 1, PAGE_W, PAGE_H, 10, 10)).toBeGreaterThan(0);
    expect(Number.isNaN(computeScale("page", 1, 0, PAGE_H, 800, 600))).toBe(false);
    expect(computeScale("page", 1, 0, 0, 0, 0)).toBe(1);
  });
});
