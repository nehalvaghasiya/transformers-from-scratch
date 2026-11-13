"""Rehearse or record the running TinySQL Gradio app with real model outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Page, expect, sync_playwright

TRACK = Path(__file__).resolve().parents[1]
STEPS = (
    (
        "retrieve names for the customers in oh",
        "Step 1 - Ask both models for customer names in Ohio",
    ),
    (
        "i need number of completed orders for ny customers",
        "Step 2 - Count completed orders for New York customers",
    ),
    (
        "i am interested in names of people from wa sort by name",
        "Step 3 - Try unfamiliar wording: the adapter can still fail",
    ),
)


def click(page: Page, label: str, role: str) -> None:
    target = page.get_by_role(role, name=label, exact=True)
    expect(target).to_be_visible()
    bounds = target.bounding_box()
    assert bounds is not None, f"No bounds for {label}"
    page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2, steps=12)
    target.click()


def walkthrough(page: Page, url: str, recording: bool) -> list[dict[str, str]]:
    page.goto(url, wait_until="networkidle")
    question = page.get_by_role("textbox", name="Question", exact=True)
    original = page.get_by_role("textbox", name="Frozen base SQL", exact=True)
    adapted = page.get_by_role("textbox", name="LoRA SQL", exact=True)
    expect(question).to_be_visible()
    if recording:
        page.evaluate(
            """() => {
                const caption = document.createElement('div');
                caption.id = 'demo-caption';
                caption.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:99998;'
                    + 'padding:14px 24px;background:#102430;color:white;'
                    + 'font:18px/1.5 system-ui;text-align:center;pointer-events:none';
                document.body.appendChild(caption);
                const cursor = document.createElement('div');
                cursor.style.cssText = 'position:fixed;z-index:99999;pointer-events:none';
                cursor.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24">'
                    + '<path d="M5 3L19 12L12 13L9 20L5 3Z" fill="white" '
                    + 'stroke="black" stroke-width="1.5"/></svg>';
                document.body.appendChild(cursor);
                document.addEventListener('mousemove', e => {
                    cursor.style.left = e.clientX + 'px';
                    cursor.style.top = e.clientY + 'px';
                });
            }"""
        )

    def caption(text: str) -> None:
        print(text, flush=True)
        if recording:
            page.locator("#demo-caption").evaluate("(el, text) => el.textContent = text", text)
            page.wait_for_timeout(1000)

    caption("TinySQL - one frozen base, one LoRA adapter, live SQL execution")
    outputs = []
    for text, subtitle in STEPS:
        caption(subtitle)
        previous = adapted.input_value()
        click(page, "Question", "textbox")
        question.fill("")
        question.press_sequentially(text, delay=35 if recording else 0)
        click(page, "Generate and execute", "button")
        expect(adapted).not_to_have_value(previous, timeout=60000)
        expect(original).not_to_have_value("", timeout=60000)
        expect(page.locator("#base-results")).to_contain_text("[", timeout=60000)
        expect(page.locator("#lora-results")).to_contain_text("[", timeout=60000)
        outputs.append(
            {
                "question": text,
                "caption": subtitle,
                "base_sql": original.input_value(),
                "lora_sql": adapted.input_value(),
                "base_results": page.locator("#base-results").inner_text(),
                "lora_results": page.locator("#lora-results").inner_text(),
            }
        )
        print(json.dumps(outputs[-1]), flush=True)
        if recording:
            page.wait_for_timeout(6000)
    caption("Selected examples, not a benchmark - see the full-split results in the README")
    if recording:
        page.wait_for_timeout(3000)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=TRACK / "runs/demo")
    parser.add_argument("--assets", type=Path, default=TRACK / "docs/assets")
    parser.add_argument(
        "--rehearse", action="store_true", help="Check the flow without saving media"
    )
    args = parser.parse_args()
    if not args.rehearse:
        # Raw video and receipts stay in ignored runs; refuse to overwrite a previous take.
        args.output.mkdir(parents=True, exist_ok=False)
        args.assets.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=None if args.rehearse else str(args.output / "raw"),
            record_video_size=None if args.rehearse else {"width": 1280, "height": 720},
            color_scheme="light",
            reduced_motion="reduce",
        )
        try:
            page = context.new_page()
            outputs = walkthrough(page, args.url, recording=not args.rehearse)
            if not args.rehearse:
                page.screenshot(path=str(args.output / "demo.png"))
            video = page.video
        finally:
            context.close()
            browser.close()
        raw = None if video is None else Path(video.path())
    if args.rehearse:
        print("Rehearsal passed: all three questions generated SQL and execution results.")
        return
    assert raw is not None
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    movie = args.output / "tinysql-lora.mp4"
    commands = (
        [
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-crf",
            "27",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        [
            "-ss",
            "2",
            "-i",
            str(movie),
            "-filter_complex",
            "fps=3,scale=960:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer",
            "-loop",
            "0",
            str(args.assets / "demo-preview.gif"),
        ],
    )
    for command in commands:
        subprocess.run([ffmpeg, "-y", *command], check=True, capture_output=True)
    (args.output / "recording.json").write_text(
        json.dumps({"url": args.url, "resolution": [1280, 720], "outputs": outputs}, indent=2)
        + "\n"
    )
    print(f"Saved {movie} and {args.assets / 'demo-preview.gif'}")


if __name__ == "__main__":
    main()
