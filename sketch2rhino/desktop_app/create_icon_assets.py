from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _flatten_to_opaque(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    # Use top-left pixel as background to remove any accidental semi-transparency.
    bg_rgb = rgba.getpixel((0, 0))[:3]
    background = Image.new("RGBA", rgba.size, (*bg_rgb, 255))
    background.alpha_composite(rgba)
    return background.convert("RGBA")


def _assert_full_bleed_opaque(image: Image.Image) -> None:
    alpha = image.getchannel("A")
    amin, amax = alpha.getextrema()
    if amin != 255 or amax != 255:
        raise ValueError(
            "Icon source is not fully opaque after flattening. "
            "Please use a full-bleed opaque 1024x1024 image."
        )


def generate_icons(target: str) -> None:
    app_dir = Path(__file__).resolve().parent
    assets_dir = app_dir / "assets"
    source_png = assets_dir / "icon_candidates" / "candidate_a_curve-first.png"
    base_png = assets_dir / "app_icon.png"
    icon_icns = assets_dir / "app_icon.icns"
    icon_ico = assets_dir / "app_icon.ico"

    if not source_png.exists():
        raise FileNotFoundError(f"Missing selected icon source: {source_png}")

    assets_dir.mkdir(parents=True, exist_ok=True)
    image = _flatten_to_opaque(Image.open(source_png))
    _assert_full_bleed_opaque(image)
    image.save(base_png, format="PNG")
    print(f"Generated: {base_png}")

    if target in {"mac", "all"}:
        image.save(
            icon_icns,
            format="ICNS",
            sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
        )
        print(f"Generated: {icon_icns}")

    if target in {"windows", "all"}:
        image.save(
            icon_ico,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        print(f"Generated: {icon_ico}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate icon assets from selected candidate.")
    parser.add_argument(
        "--target",
        choices=["mac", "windows", "all"],
        default="all",
        help="Target icon format set.",
    )
    args = parser.parse_args()
    generate_icons(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
