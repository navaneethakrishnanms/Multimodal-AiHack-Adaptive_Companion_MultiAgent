"""Post-process the raw demo recording into the final submission video.

- converts webm -> mp4 (H.264)
- re-times the subtitle file to the actual video duration (linear scale)
- burns English subtitles
- prepends a title card and appends an end card
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "demo"
FFMPEG = "/home/maebahesioru/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
FFPROBE = "/home/maebahesioru/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffprobe"
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd[:6]), "...")
        print(r.stderr[-1500:])
        sys.exit(1)


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def parse_srt(text: str) -> list[tuple[int, int, str]]:
    blocks = re.split(r"\n\n+", text.strip())
    out = []
    for b in blocks:
        lines = b.split("\n")
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2]
        end = g[4] * 3600 + g[5] * 60 + g[6]
        out.append((start, end, "\n".join(lines[2:])))
    return out


def fmt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    raw = next(DEMO.glob("video/*.webm"), None)
    if not raw:
        print("no webm found in demo/video/")
        sys.exit(1)
    print("raw:", raw, f"({probe_duration(raw):.1f}s)")

    # 1. convert
    mp4 = DEMO / "demo_main.mp4"
    run([FFMPEG, "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-an", str(mp4)])
    dur = probe_duration(mp4)
    print(f"converted: {dur:.1f}s")

    # 2. re-time subtitles to actual duration
    subs = parse_srt((DEMO / "subtitles.srt").read_text(encoding="utf-8"))
    srt_total = subs[-1][1]
    scale = dur / srt_total
    retimed = []
    for i, (start, end, text) in enumerate(subs, 1):
        retimed.append(f"{i}\n{fmt_ts(start * scale)} --> {fmt_ts(end * scale)}\n{text}\n")
    (DEMO / "subtitles_final.srt").write_text("\n".join(retimed), encoding="utf-8")
    print(f"subtitles re-timed (scale {scale:.2f})")

    # 3. title + end cards
    title_png = DEMO / "title.png"
    end_png = DEMO / "end.png"
    run([FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=0x0f1117:s=1280x800:d=1", "-frames:v", "1",
         "-vf", (f"drawtext=fontfile={FONT}:text='Study Companion':fontsize=64:fontcolor=white:"
                 f"x=(w-text_w)/2:y=300,"
                 f"drawtext=fontfile={FONT}:text='source-grounded, adaptive AI tutor':fontsize=32:"
                 f"fontcolor=0x8b93a7:x=(w-text_w)/2:y=390,"
                 f"drawtext=fontfile={FONT}:text='Multimodal AI Hackathon 2026 - Track D':fontsize=26:"
                 f"fontcolor=0x5b8cff:x=(w-text_w)/2:y=470"),
         str(title_png)])
    run([FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=0x0f1117:s=1280x800:d=1", "-frames:v", "1",
         "-vf", (f"drawtext=fontfile={FONT}:text='github.com/maebahesioru/multimodal-study-companion':"
                 f"fontsize=30:fontcolor=white:x=(w-text_w)/2:y=360,"
                 f"drawtext=fontfile={FONT}:text='Tutoring that stays grounded in your course.':fontsize=28:"
                 f"fontcolor=0x8b93a7:x=(w-text_w)/2:y=430"),
         str(end_png)])

    # 4. assemble: title(4s) + main(subbed) + end(5s)
    title_mp4 = DEMO / "title.mp4"
    end_mp4 = DEMO / "end.mp4"
    run([FFMPEG, "-y", "-loop", "1", "-i", str(title_png), "-t", "4", "-r", "30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "format=yuv420p", str(title_mp4)])
    run([FFMPEG, "-y", "-loop", "1", "-i", str(end_png), "-t", "5", "-r", "30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "format=yuv420p", str(end_mp4)])

    main_subbed = DEMO / "demo_subbed.mp4"
    sub_path = str(DEMO / "subtitles_final.srt").replace(":", "\\:")
    run([FFMPEG, "-y", "-i", str(mp4),
         "-vf", f"subtitles={sub_path}:force_style='FontName=Noto Sans,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=28'",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(main_subbed)])

    concat_txt = DEMO / "concat.txt"
    concat_txt.write_text(
        f"file '{title_mp4}'\nfile '{main_subbed}'\nfile '{end_mp4}'\n", encoding="utf-8"
    )
    final = DEMO / "study_companion_demo.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(final)])
    print("FINAL:", final, f"({probe_duration(final):.1f}s)")


if __name__ == "__main__":
    main()
