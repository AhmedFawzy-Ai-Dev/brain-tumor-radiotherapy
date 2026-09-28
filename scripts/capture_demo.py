"""Record the README's demo media from a running app (a dev tool; needs Playwright).

    python app.py                        # in another terminal
    pip install playwright pypdfium2
    python scripts/make_examples.py      # held-out cases to show
    python scripts/capture_demo.py       # -> docs/images/demo.gif, demo_3d.png, demo_2d.png,
                                         #    report_pdf.png

Drives the app in the installed Microsoft Edge (Playwright's "msedge" channel, so
no browser download): picks a held-out BraTS patient from the example list, films
the report being generated, then a 2-D clinical image, and renders page 1 of the
PDF the 3-D run wrote.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "examples"


def _timing(page) -> str:
    """The timing line's own text (its wrapper also holds Gradio's progress overlay)."""
    return page.locator('[data-testid="markdown"].timing').inner_text().strip()


def wait_for_result(page, previous: str, frames: list | None = None,
                    timeout: float = 300) -> None:
    """Wait until the timing line (written last) shows a new run's result."""
    start = time.time()
    while time.time() - start < timeout:
        if frames is not None:
            frames.append((page.screenshot(), 500))
        text = _timing(page)
        if "s on CPU" in text and text != previous:
            time.sleep(1.5)                        # let the image finish painting
            return
        time.sleep(0.5)
    raise TimeoutError("the report did not finish in time")


def run_case(page, example: str, frames: list | None = None) -> None:
    """Load one of the app's examples (by file name) and generate its report."""
    previous = _timing(page)
    page.get_by_text(example, exact=True).click()
    time.sleep(2.5)                                # the example file loads into the input
    if frames is not None:
        frames.append((page.screenshot(), 1200))
    page.get_by_role("button", name="Generate report").click()
    wait_for_result(page, previous, frames)


def save_gif(frames: list[tuple[bytes, int]], path: Path, width: int = 1000) -> None:
    from PIL import Image

    images, durations, previous = [], [], None
    for raw, duration in frames:
        if raw == previous:                        # merge identical frames
            durations[-1] += duration
            continue
        previous = raw
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)))
        images.append(img.quantize(colors=160, method=Image.Quantize.MEDIANCUT))
        durations.append(duration)
    durations[-1] = 4000                           # hold the end
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True)


def render_pdf_page(pdf: Path, out: Path, scale: float = 1.6) -> None:
    import pypdfium2 as pdfium

    page = pdfium.PdfDocument(str(pdf))[0]
    page.render(scale=scale).to_pil().save(out)


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser(description="Record the README demo media.")
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    volumes = sorted(EXAMPLES.glob("*.nii.gz"))
    images = sorted(EXAMPLES.glob("glioma_*"))
    if not volumes or not images:
        raise SystemExit("no examples: run scripts/make_examples.py first")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                device_scale_factor=1.25)
        for _ in range(40):                        # the app may still be starting
            try:
                page.goto(args.url, timeout=10000)
                break
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(3)
        page.get_by_role("button", name="Generate report").wait_for(timeout=60000)

        frames: list = [(page.screenshot(), 1500)]
        run_case(page, volumes[0].name, frames)
        frames.append((page.screenshot(), 2500))
        for _ in range(6):                         # scroll down to the findings
            page.mouse.wheel(0, 160)
            time.sleep(0.25)
            frames.append((page.screenshot(), 250))
        frames.append((page.screenshot(), 2500))
        save_gif(frames, out / "demo.gif")

        page.mouse.wheel(0, -4000)
        time.sleep(0.5)
        page.screenshot(path=out / "demo_3d.png", full_page=True)
        pdf = ROOT / "reports" / "_ui" / "report.pdf"
        if pdf.exists():
            render_pdf_page(pdf, out / "report_pdf.png")

        run_case(page, images[0].name)
        page.screenshot(path=out / "demo_2d.png", full_page=True)
        browser.close()
    print(f"Wrote demo.gif, demo_3d.png, demo_2d.png and report_pdf.png to {out}")


if __name__ == "__main__":
    main()
