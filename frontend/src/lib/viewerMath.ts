// Pure sizing math for the PDF viewer. No React/pdfjs imports so this is
// cheap to unit-test in jsdom/node.

export type FitMode = "width" | "page" | "custom";

export const PAD_X = 32;
export const PAD_Y = 24;

/**
 * Compute the render scale for the current fit mode.
 *
 * fit-width fills the available width (vertical scrolling allowed);
 * fit-page fits BOTH axes so the whole page stays visible;
 * custom returns the user zoom verbatim.
 */
export function computeScale(
  fitMode: FitMode,
  zoom: number,
  pageWidthPt: number,
  pageHeightPt: number,
  boxWidth: number,
  boxHeight: number,
): number {
  if (fitMode === "custom") return zoom;
  const w = Math.max(0, boxWidth - PAD_X);
  const h = Math.max(0, boxHeight - PAD_Y);
  if (pageWidthPt <= 0 || pageHeightPt <= 0) return 1;
  if (w <= 0 && h <= 0) return 1;
  if (fitMode === "page") {
    const byHeight = h > 0 ? h / pageHeightPt : Infinity;
    const byWidth = w > 0 ? w / pageWidthPt : Infinity;
    return Math.min(byWidth, byHeight);
  }
  if (w <= 0) return 1;
  return w / pageWidthPt;
}
