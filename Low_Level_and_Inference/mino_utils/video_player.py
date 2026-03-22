#!/usr/bin/env python3
"""
Entry point for the video player utilities.

This thin wrapper lets you run the Typer CLI without needing to
remember the full module path, e.g.:

    python video_player.py single outputs/2024-07-01/rgb frames/out.mp4 --fps 30
"""

from mino_utils.video_player import app


if __name__ == "__main__":
    app()
