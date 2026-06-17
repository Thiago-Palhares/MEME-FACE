from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "memes"
SIZE = (720, 720)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, fill: str, size: int) -> None:
    face = font(size)
    bbox = draw.textbbox((0, 0), text, font=face)
    x = (SIZE[0] - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, font=face, fill=fill)


def dog_side_eye() -> Image.Image:
    img = Image.new("RGB", SIZE, "#f2dfbf")
    draw = ImageDraw.Draw(img)
    draw.ellipse((135, 160, 590, 560), fill="#b87432", outline="#6f421d", width=10)
    draw.polygon([(150, 215), (60, 130), (110, 360)], fill="#8f5527")
    draw.polygon([(520, 185), (660, 110), (610, 345)], fill="#8f5527")
    draw.ellipse((230, 285, 325, 360), fill="#fff8ec")
    draw.ellipse((392, 265, 500, 348), fill="#fff8ec")
    draw.ellipse((255, 315, 302, 355), fill="#1b1714")
    draw.ellipse((412, 300, 462, 342), fill="#1b1714")
    draw.ellipse((325, 365, 410, 430), fill="#1b1714")
    draw.arc((290, 410, 450, 500), 5, 165, fill="#3a2112", width=8)
    centered(draw, "side eye", 610, "#3a2112", 54)
    return img


def shrug_emoji() -> Image.Image:
    img = Image.new("RGB", SIZE, "#fff7d6")
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 105, 570, 525), fill="#f5c842", outline="#b6880f", width=8)
    draw.ellipse((260, 250, 325, 300), fill="#1f1f1f")
    draw.ellipse((410, 250, 475, 300), fill="#1f1f1f")
    draw.arc((300, 330, 450, 420), 10, 170, fill="#5f4300", width=10)
    draw.line((130, 430, 45, 350), fill="#f5c842", width=28)
    draw.line((590, 430, 675, 350), fill="#f5c842", width=28)
    centered(draw, "shrug", 610, "#6d4b00", 60)
    return img


def cristiano_smile() -> Image.Image:
    img = Image.new("RGB", SIZE, "#301a1f")
    draw = ImageDraw.Draw(img)
    draw.ellipse((155, 110, 565, 575), fill="#c58c69", outline="#6e3f31", width=8)
    draw.arc((210, 220, 330, 295), 190, 345, fill="#2b1715", width=16)
    draw.arc((395, 220, 515, 295), 190, 345, fill="#2b1715", width=16)
    draw.rounded_rectangle((230, 365, 500, 465), radius=28, fill="#fdf6e8", outline="#5f2e2b", width=8)
    for x in range(270, 480, 45):
        draw.line((x, 368, x, 462), fill="#d8c8b4", width=3)
    draw.arc((220, 330, 510, 510), 0, 180, fill="#5f2e2b", width=10)
    centered(draw, "big smile", 610, "#fdf6e8", 56)
    return img


def npc_stare() -> Image.Image:
    img = Image.new("RGB", SIZE, "#dbe3e7")
    draw = ImageDraw.Draw(img)
    draw.ellipse((175, 105, 545, 570), fill="#f0c7a8", outline="#7d6258", width=8)
    draw.line((245, 275, 320, 275), fill="#202020", width=12)
    draw.line((400, 275, 475, 275), fill="#202020", width=12)
    draw.line((300, 410, 450, 410), fill="#5f3d38", width=10)
    centered(draw, "stare", 610, "#263238", 62)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = {
        "dog_side_eye.png": dog_side_eye(),
        "shrug_emoji.png": shrug_emoji(),
        "cristiano_smile.png": cristiano_smile(),
        "npc_stare.png": npc_stare(),
    }
    for name, image in images.items():
        image.save(OUT_DIR / name)
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
