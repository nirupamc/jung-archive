import "@testing-library/jest-dom/vitest";

// jsdom does not implement ResizeObserver; components that observe layout
// (PdfViewer, ForceGraphView, workspace panes) need a deterministic stub.
class ResizeObserverStub {
  callback: ResizeObserverCallback;
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }
  observe(el: Element) {
    // Fire once with the observed size so initial measurements work.
    this.callback(
      [{
        target: el,
        contentRect: {
          width: (el as HTMLElement).clientWidth || 800,
          height: (el as HTMLElement).clientHeight || 600,
        } as DOMRectReadOnly,
      }] as unknown as ResizeObserverEntry[],
      this as unknown as ResizeObserver,
    );
  }
  unobserve() {}
  disconnect() {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  (globalThis as Record<string, unknown>).ResizeObserver = ResizeObserverStub;
}

// jsdom also lacks canvas/WebGL; keep getContext null-safe.
if (
  typeof HTMLCanvasElement !== "undefined" &&
  !HTMLCanvasElement.prototype.getContext
) {
  HTMLCanvasElement.prototype.getContext = (() => null) as never;
}
