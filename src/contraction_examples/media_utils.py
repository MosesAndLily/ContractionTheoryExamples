"""Video/GIF writing helpers shared by the example animations.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system install
is required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import matplotlib
from matplotlib import animation


def render_video(
    fig,
    update: Callable[[int], object],
    n_frames: int,
    path: Path,
    fps: int,
    hold_seconds: float = 1.2,
) -> None:
    """Write an H.264 MP4, holding the final frame for ``hold_seconds``."""
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    frames = n_frames + int(hold_seconds * fps)

    def _progress(i, n):
        if i % 60 == 0 or i == n - 1:
            print(f"  frame {i + 1}/{n}")

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="h264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium"],
    )
    anim.save(path, writer=writer, progress_callback=_progress)


def mp4_to_gif(mp4: Path, gif: Path, fps: int = 18, width: int = 880) -> None:
    """Small palette-optimized GIF preview generated from the MP4."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    )
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(mp4), "-filter_complex", vf,
             "-loop", "0", str(gif)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode(errors="replace"))
        raise
