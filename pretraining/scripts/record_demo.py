"""Record the running Gradio app, real predictions, results and reproduction commands."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

TRACK = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--model", type=Path, help="Optionally start and stop a local demo server")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, default=TRACK / "runs/demo")
    parser.add_argument("--assets", type=Path, default=TRACK / "docs/assets")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    movie = args.output / "ticket-transformer.mp4"
    if movie.exists():
        raise FileExistsError("Use a new output directory to preserve the existing recording")
    if args.model is not None:
        command = [
            str(Path(sys.executable).parent / "pretraining"),
            "demo",
            "--model",
            str(args.model),
            "--port",
            str(urlsplit(args.url).port or 7860),
        ]
        if args.report is not None:
            command.extend(["--report", str(args.report)])
        log = (args.output / "server.log").open("w")
        server = subprocess.Popen(
            command, stdout=log, stderr=log, env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}
        )

        def stop_server() -> None:
            server.terminate()
            server.wait(timeout=10)
            log.close()

        atexit.register(stop_server)
        for _ in range(40):
            if server.poll() is not None:
                raise RuntimeError("Demo server failed; inspect server.log")
            try:
                with urllib.request.urlopen(args.url, timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Demo server did not become ready")
    events: list[tuple[float, str]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=str(args.output / "raw"),
            record_video_size={"width": 1280, "height": 800},
            reduced_motion="reduce",
        )
        page = context.new_page()
        start = time.perf_counter()
        page.goto(args.url, wait_until="domcontentloaded")
        box = page.get_by_role("textbox", name="Support ticket", exact=True)
        box.wait_for(timeout=60000)
        page.add_style_tag(
            content="""
            #recording-caption {position:fixed;bottom:0;left:0;right:0;z-index:99999;
                padding:15px 30px;background:rgba(16,36,48,.96);color:#fff;
                font:19px/1.5 system-ui;min-height:58px;box-sizing:border-box;}
            body {padding-bottom:95px!important;}
        """
        )
        page.evaluate(
            "() => {const "
            "e=document.createElement('div');e.id='recording-caption';document.body.appendChild(e);}"
        )

        def caption(text: str) -> None:
            events.append((time.perf_counter() - start, text))
            page.locator("#recording-caption").evaluate(
                "(element,text)=>element.textContent=text", text
            )
            print(text, flush=True)

        def until(seconds: float) -> None:
            remaining = seconds - (time.perf_counter() - start)
            if remaining > 0:
                page.wait_for_timeout(remaining * 1000)

        def route(text: str) -> None:
            box.fill(text)
            page.get_by_role("button", name="Route ticket", exact=True).click()
            page.wait_for_timeout(1400)

        caption(
            "655,872 parameters. Three causal Transformer blocks. Trained from "
            "random weights on synthetic support tickets."
        )
        until(12)
        caption(
            "Live inference: the model generates intent and priority label tokens, "
            "then maps them to JSON."
        )
        route("My invoice is incorrect. There is no rush; next week is fine.")
        page.screenshot(path=str(args.assets / "demo.png"))
        until(30)
        caption(
            "Edit the urgency and run the same model again. These are computed "
            "predictions, not saved example responses."
        )
        route("My invoice is incorrect. Our team is blocked from an essential task today.")
        until(45)
        caption(
            "A real failure: this ticket requests account cancellation, but the "
            "selected model predicts billing_issue."
        )
        route(
            "We use the web app. I want our DataHarbor membership cancelled rather "
            "than renewed. This can be handled within your standard support window."
        )
        until(65)
        caption(
            "Nine training runs, three fixed seeds. Pooled-head variants improve "
            "average scores; classical TF-IDF remains stronger."
        )
        page.get_by_text(
            "Measured comparisons and limitations", exact=True
        ).scroll_into_view_if_needed()
        until(85)
        caption(
            "Training and validation answer loss are plotted separately. All "
            "selection uses validation before final evaluation."
        )
        page.get_by_text("Learning and evaluation figures", exact=True).click()
        page.get_by_text(
            "Training and validation answer loss", exact=True
        ).scroll_into_view_if_needed()
        until(103)
        caption(
            "Reproduce with the pinned CPU environment, deterministic data "
            "generator and the same training command. Synthetic results do not "
            "establish real-ticket accuracy."
        )
        page.get_by_text("Reproduce this model", exact=True).click()
        page.get_by_text("Reproduce this model", exact=True).scroll_into_view_if_needed()
        until(123)
        video = page.video
        assert video is not None
        raw = Path(video.path())
        duration = time.perf_counter() - start
        context.close()
        browser.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
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
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            "14",
            "-t",
            "23",
            "-i",
            str(movie),
            "-filter_complex",
            (
                "fps=3,scale=768:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer"
            ),
            "-loop",
            "0",
            str(args.assets / "demo-preview.gif"),
        ],
        check=True,
        capture_output=True,
    )
    transcript = [
        "# Recorded demo transcript",
        "",
        "Real local Gradio screen recording; captions are burned into the video. No narration.",
        "",
    ]
    subtitles = []
    for index, (begin, text) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else duration

        def stamp(seconds: float) -> str:
            milliseconds = int(seconds * 1000)
            hours, remainder = divmod(milliseconds, 3600000)
            minutes, remainder = divmod(remainder, 60000)
            seconds, fraction = divmod(remainder, 1000)
            return f"{hours:02}:{minutes:02}:{seconds:02},{fraction:03}"

        subtitles.append(f"{index + 1}\n{stamp(begin)} --> {stamp(end)}\n{text}\n")
        transcript.append(f"**{int(begin):02}s:** {text}\n")
    (args.output / "captions.srt").write_text("\n".join(subtitles))
    (args.assets / "demo-transcript.md").write_text("\n".join(transcript) + "\n")
    (args.output / "recording.json").write_text(
        json.dumps(
            {
                "url": args.url,
                "duration_seconds": duration,
                "resolution": [1280, 800],
                "captions": True,
                "narration": False,
                "movie": movie.name,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {movie}", flush=True)


if __name__ == "__main__":
    main()
