import argparse
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_posters(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    posters = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    posters.sort(key=lambda p: p.name.lower())
    return posters


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_bgr(path: Path, bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), bgr)


def overlay_title(bgr: np.ndarray, title: str) -> np.ndarray:
    if not title:
        return bgr
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path
    
    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    
    h, w = bgr.shape[:2]
    
    # Use a truetype font that supports Cyrillic
    font_size = int(max(48, min(80, w / 10.0)))
    try:
        # Common Linux paths for fonts with Cyrillic support
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
        font = None
        for path in font_paths:
            if Path(path).exists():
                font = ImageFont.truetype(path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Get text bbox for centering
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    x = int((w - tw) * 0.5)
    y = int(h * 0.12)
    
    # Draw shadow
    draw.text((x + 3, y + 3), title, font=font, fill=(0, 0, 0))
    # Draw main text
    draw.text((x, y), title, font=font, fill=(255, 255, 255))
    
    # Convert back to BGR
    bgr_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr_out


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(max(0.0, min(1.0, t)))
    return cv2.addWeighted(a, 1.0 - t, b, t, 0.0)


def ease_linear(x: float) -> float:
    return x


def ease_in_quad(x: float) -> float:
    return x * x


def ease_out_quad(x: float) -> float:
    return 1.0 - (1.0 - x) * (1.0 - x)


def ease_in_out_cubic(x: float) -> float:
    if x < 0.5:
        return 4.0 * x * x * x
    return 1.0 - pow(-2.0 * x + 2.0, 3.0) / 2.0


def ease_out_expo(x: float) -> float:
    """Exponential easeOut - fast start, slow finish for premium feel"""
    return 1.0 if x >= 1.0 else 1.0 - pow(2, -10 * x)


def ease_in_expo(x: float) -> float:
    """Exponential easeIn - slow start, explosive finish"""
    return 0.0 if x <= 0.0 else pow(2, 10 * x - 10)


def ease_in_out_back(x: float) -> float:
    """Back easing with anticipation and overshoot for dynamic feel"""
    c1 = 1.70158
    c2 = c1 * 1.525
    if x < 0.5:
        return (pow(2 * x, 2) * ((c2 + 1) * 2 * x - c2)) / 2
    return (pow(2 * x - 2, 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2


EASING = {
    "linear": ease_linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_out_expo": ease_out_expo,
    "ease_in_expo": ease_in_expo,
    "ease_in_out_back": ease_in_out_back,
}


def sample_indices(count_out: int, count_src: int, easing_name: str, reverse: bool = False, pair_duplicates: bool = False) -> list[int]:
    if count_src <= 1:
        return [0] * max(1, count_out)
    ease = EASING.get(easing_name, ease_in_out_cubic)
    out: list[int] = []
    n = max(1, int(count_out))
    for i in range(n):
        t = 0.0 if n == 1 else (i / (n - 1))
        s = float(max(0.0, min(1.0, ease(t))))
        idx = int(round(s * (count_src - 1)))
        if reverse:
            idx = (count_src - 1) - idx
        out.append(idx)
    if pair_duplicates and len(out) > 1:
        paired: list[int] = []
        for idx in out:
            paired.extend([idx, idx])
        return paired
    return out


def load_segment_frames(segment_frames_dir: Path, frame_start: int, frame_end: int) -> list[np.ndarray]:
    frames = []
    for f in range(frame_start, frame_end + 1):
        frames.append(read_bgr(segment_frames_dir / f"frame_{f:04d}.png"))
    return frames


@dataclass
class SegmentPlan:
    stem_unfold: str
    stem_fold: str
    title: str
    unfold_len: int
    fold_len: int
    hold_flat: int
    hold_ball: int
    easing_unfold: str
    easing_fold: str


def run_sim_for_jobs(
    jobs_dir: Path,
    out_dir: Path,
    seed: int,
    frame_start: int,
    frame_end: int,
    samples: int,
    ref_dir: Path,
    match_ref_area: bool,
    ref_time_warp_gamma: float,
):
    cmd = [
        "python",
        str(Path(__file__).with_name("run.py")),
        "--afisha_dir",
        str(jobs_dir),
        "--out_dir",
        str(out_dir),
        "--frame_start",
        str(int(frame_start)),
        "--frame_end",
        str(int(frame_end)),
        "--seed",
        str(int(seed)),
        "--samples",
        str(int(samples)),
        "--ref_dir",
        str(ref_dir),
    ]
    if match_ref_area:
        cmd.append("--match_ref_area")
        cmd.extend(["--ref_time_warp_gamma", str(float(ref_time_warp_gamma))])
    subprocess.check_call(cmd)


def render_motion(
    base: list[np.ndarray],
    length: int,
    easing_name: str,
    reverse: bool,
    frame_interp: bool,
    pair_duplicates: bool,
) -> list[np.ndarray]:
    if length <= 0:
        return []
    if len(base) == 1:
        frames = [base[0]] * int(length)
        return [f for fr in frames for f in ([fr, fr] if pair_duplicates else [fr])]

    ease = EASING.get(easing_name, ease_in_out_cubic)
    out: list[np.ndarray] = []
    n = int(length)
    for i in range(n):
        t = 0.0 if n == 1 else (i / (n - 1))
        s = float(max(0.0, min(1.0, ease(t)))) * (len(base) - 1)
        if reverse:
            s = (len(base) - 1) - s
        if not frame_interp or pair_duplicates:
            idx = int(round(s))
            out.append(base[idx])
            continue
        i0 = int(math.floor(s))
        i1 = min(len(base) - 1, i0 + 1)
        a = float(s - i0)
        out.append(blend(base[i0], base[i1], a))
    if pair_duplicates:
        paired: list[np.ndarray] = []
        for fr in out:
            paired.extend([fr, fr])
        return paired
    return out


def add_audio(video_path: Path, audio_path: Path, audio_start_sec: float, output_path: Path) -> None:
    """Mix audio starting at specified timestamp with video."""
    if not audio_path.exists():
        print(f"Warning: Audio file not found at {audio_path}")
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(float(audio_start_sec)),
        "-i",
        str(audio_path),
        "-i",
        str(video_path),
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afisha_dir", default="afisha")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_posters", type=int, default=4, help="How many posters to show (excluding cover)")
    ap.add_argument("--cover_path", default="", help="Optional cover image; defaults to first poster")
    ap.add_argument("--title", default="Афиши", help="Title text for cover")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--unfold_len", type=int, default=20, help="Output frames for unfold (increased for visible easing)")
    ap.add_argument("--fold_len", type=int, default=24, help="Output frames for fold")
    ap.add_argument("--hold_flat", type=int, default=30, help="Frames to hold fully unfolded poster")
    ap.add_argument("--hold_ball", type=int, default=2, help="Frames to hold crumpled ball")
    ap.add_argument("--cover_hold", type=int, default=12, help="Frames to show cover before crumpling")
    ap.add_argument("--crossfade", type=int, default=0, help="Frames to crossfade between crumpled states (0=disable to avoid ghosting)")

    ap.add_argument("--pair_duplicates", action="store_true", help="Repeat sampled indices (discrete look)")
    ap.add_argument("--frame_interp", action="store_true", default=True, help="Blend between base frames for smoother motion")
    ap.add_argument("--no_frame_interp", action="store_false", dest="frame_interp")
    ap.add_argument("--different_fold", action="store_true", default=False, help="Use a different simulation for folding (disabled for position continuity)")
    ap.add_argument("--same_fold", action="store_false", dest="different_fold")

    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--match_ref_area", action="store_true", default=True)
    ap.add_argument("--ref_time_warp_gamma", type=float, default=2.0)

    ap.add_argument("--audio_path", default="audio/The_xx_-_Intro_47842884.mp3", help="Path to audio file")
    ap.add_argument("--audio_start_sec", type=float, default=81.0, help="Start time in audio file (1:21 = 81 sec)")
    ap.add_argument("--no_audio", action="store_true", help="Skip audio mixing")

    args = ap.parse_args()

    afisha_dir = Path(args.afisha_dir)
    posters = list_posters(afisha_dir)
    if not posters:
        raise SystemExit(f"No posters found in {afisha_dir}")

    cover_path = Path(args.cover_path) if args.cover_path else posters[0]
    if not cover_path.exists():
        cover_path = posters[0]

    # Build a list of N posters to show (cycle if not enough).
    n = max(1, int(args.num_posters))
    show_posters: list[Path] = []
    for i in range(n):
        # Start from index 1 to avoid duplicating cover (posters[0])
        show_posters.append(posters[(i + 1) % len(posters)])

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    jobs_dir = out_dir / "job_posters"
    ensure_dir(jobs_dir)
    # Recreate jobs directory for reproducibility.
    for p in jobs_dir.glob("*"):
        if p.is_file() or p.is_symlink():
            p.unlink()

    # Build job list. Each job gets its own seed via run.py's per-index offset.
    # If --different_fold is enabled, each poster has 2 sims (unfold/fold) so motion differs.
    job_paths: list[tuple[str, Path]] = []
    if args.different_fold:
        job_paths.append(("cover_fold", cover_path))
        for i, p in enumerate(show_posters, start=1):
            job_paths.append((f"poster_{i:02d}_unfold", p))
            job_paths.append((f"poster_{i:02d}_fold", p))
    else:
        job_paths.append(("cover", cover_path))
        for i, p in enumerate(show_posters, start=1):
            job_paths.append((f"poster_{i:02d}", p))

    for stem, src in job_paths:
        dst = jobs_dir / f"{stem}{src.suffix.lower()}"
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    render_dir = out_dir / "render"
    run_sim_for_jobs(
        jobs_dir=jobs_dir,
        out_dir=render_dir,
        seed=int(args.seed),
        frame_start=int(args.frame_start),
        frame_end=int(args.frame_end),
        samples=int(args.samples),
        ref_dir=Path(args.ref_dir),
        match_ref_area=bool(args.match_ref_area),
        ref_time_warp_gamma=float(args.ref_time_warp_gamma),
    )

    rnd = random.Random(int(args.seed))

    # Per-segment variation: easing types and slightly varied durations.
    easing_choices = ["ease_in_out_cubic", "ease_out_quad", "ease_in_quad"]

    segments: list[SegmentPlan] = []
    # MODIFIED: First poster starts fully unfolded (no title, longer hold)
    segments.append(
        SegmentPlan(
            stem_unfold="cover" if not args.different_fold else "cover_fold",
            stem_fold="cover" if not args.different_fold else "cover_fold",
            title="",  # No title overlay for first poster
            unfold_len=0,  # Start fully unfolded
            fold_len=24,  # longer fold for cover to avoid duplicates
            hold_flat=36,  # ~1.5 seconds @ 24fps to show the poster clearly
            hold_ball=int(args.hold_ball),
            easing_unfold="linear",  # 1:1 frame mapping
            easing_fold="ease_out_quad",  # slower end, smoother collapse
        )
    )
    for i in range(1, n + 1):
        if args.different_fold:
            su = f"poster_{i:02d}_unfold"
            sf = f"poster_{i:02d}_fold"
        else:
            su = sf = f"poster_{i:02d}"
        segments.append(
            SegmentPlan(
                stem_unfold=su,
                stem_fold=sf,
                title="",
                unfold_len=int(args.unfold_len) + 8,  # 24-28 frames for visible easing
                fold_len=28,  # Longer fold for smooth motion
                hold_flat=int(args.hold_flat),
                hold_ball=int(args.hold_ball),
                easing_unfold="ease_in_out_back",  # Dramatic overshoot for premium feel
                easing_fold="ease_in_out_back",     # Consistent dramatic easing
            )
        )

    # Assemble the final frame sequence.
    seq_dir = out_dir / "sequence_frames"
    ensure_dir(seq_dir)
    for p in seq_dir.glob("*.png"):
        p.unlink()

    seq: list[np.ndarray] = []

    prev_ball: np.ndarray | None = None
    for seg in segments:
        frames_dir_unfold = render_dir / "posters" / seg.stem_unfold / "frames"
        frames_dir_fold = render_dir / "posters" / seg.stem_fold / "frames"
        base_unfold = load_segment_frames(frames_dir_unfold, int(args.frame_start), int(args.frame_end))
        base_fold = base_unfold if seg.stem_fold == seg.stem_unfold else load_segment_frames(frames_dir_fold, int(args.frame_start), int(args.frame_end))
        idx_ball = 0
        idx_flat = len(base_unfold) - 1
        ball = base_unfold[idx_ball]
        flat = base_unfold[idx_flat]
        flat_fold = base_fold[idx_flat]
        ball_fold = base_fold[idx_ball]

        # Crossfade between crumpled states when swapping posters.
        if prev_ball is not None and int(args.crossfade) > 0:
            c = int(args.crossfade)
            for i in range(c):
                t = 0.0 if c == 1 else (i / (c - 1))
                seq.append(blend(prev_ball, ball, t))

        if seg.title:
            cover_img = overlay_title(flat, seg.title)
            for _ in range(max(1, seg.hold_flat)):
                seq.append(cover_img)
        elif seg.unfold_len > 0:
            # Start each poster segment from a crumpled ball.
            if not seq:
                seq.append(ball)
            # Unfold with easing (non-uniform dynamics).
            seq.extend(
                render_motion(
                    base=base_unfold,
                    length=seg.unfold_len,
                    easing_name=seg.easing_unfold,
                    reverse=False,
                    frame_interp=bool(args.frame_interp),
                    pair_duplicates=bool(args.pair_duplicates),
                )
            )
            for _ in range(max(0, seg.hold_flat)):
                seq.append(flat)

        # Fold down to ball (reverse). If using a separate fold simulation, switch at the flat pose.
        if base_fold is not base_unfold:
            seq.append(flat_fold)
        seq.extend(
            render_motion(
                base=base_fold,
                length=seg.fold_len,
                easing_name=seg.easing_fold,
                reverse=True,
                frame_interp=bool(args.frame_interp),
                pair_duplicates=bool(args.pair_duplicates),
            )
        )
        for _ in range(max(0, seg.hold_ball)):
            seq.append(ball_fold)

        prev_ball = ball_fold

    # Write frames out.
    for i, img in enumerate(seq, start=1):
        write_bgr(seq_dir / f"frame_{i:06d}.png", img)

    # Render MP4.
    out_mp4 = out_dir / "showreel.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(args.fps)),
        "-i",
        str(seq_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)

    # Add audio if requested
    final_mp4 = out_mp4
    if not args.no_audio:
        audio_path = Path(args.audio_path)
        if audio_path.exists():
            final_mp4 = out_dir / "showreel_with_audio.mp4"
            print(f"Adding audio from {audio_path} starting at {args.audio_start_sec}s...")
            add_audio(
                video_path=out_mp4,
                audio_path=audio_path,
                audio_start_sec=float(args.audio_start_sec),
                output_path=final_mp4,
            )
            print(f"Created final video with audio: {final_mp4}")
        else:
            print(f"Warning: Audio file not found at {audio_path}, skipping audio mixing.")

    manifest = {
        "fps": int(args.fps),
        "cover": str(cover_path),
        "posters": [str(p) for p in show_posters],
        "segments": [seg.__dict__ for seg in segments],
        "sequence_frames": len(seq),
        "video_with_audio": str(final_mp4) if not args.no_audio else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
