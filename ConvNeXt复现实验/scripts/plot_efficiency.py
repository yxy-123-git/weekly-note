from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out = Path(__file__).resolve().parent.parent / "figures" / "accuracy_throughput_fp32.png"
im = Image.new("RGB", (1400, 850), "white")
d = ImageDraw.Draw(im)
fonts = Path("C:/Windows/Fonts")
if fonts.is_dir():
    regular = str(fonts / "arial.ttf")
    bold = str(fonts / "arialbd.ttf")
else:
    regular = "DejaVuSans.ttf"
    bold = "DejaVuSans-Bold.ttf"
ft = ImageFont.truetype(bold, 38)
fh = ImageFont.truetype(bold, 27)
fb = ImageFont.truetype(regular, 23)
fs = ImageFont.truetype(regular, 19)
l, r, t, b = 130, 1270, 145, 670
xp = lambda v: l + v / 400 * (r-l)
yp = lambda v: b - (v-80) / 4 * (b-t)

d.text((l, 36), "Accuracy vs. FP32 inference throughput", font=ft, fill="#172238")
d.text((l, 90), "ImageNet-100 | batch 64 | 224 x 224 | A800 GPU", font=fb, fill="#526176")
for v in (0,100,200,300,400):
    x = xp(v)
    d.line((x,t,x,b), fill="#e5eaf0", width=2)
    d.text((x-12,b+15), str(v), font=fs, fill="#3e4b60")
for v in (80,81,82,83,84):
    y = yp(v)
    d.line((l,y,r,y), fill="#e5eaf0", width=2)
    d.text((l-50,y-11), str(v), font=fs, fill="#3e4b60")
d.line((l,t,l,b), fill="#27354b", width=3)
d.line((l,b,r,b), fill="#27354b", width=3)
d.text((560, 730), "FP32 throughput (images/s)", font=fh, fill="#172238")
label = Image.new("RGBA", (230, 50), (255, 255, 255, 0))
ImageDraw.Draw(label).text((0, 0), "Top-1 (%)", font=fh, fill="#172238")
im.paste(label.rotate(90, expand=True), (15, 350), label.rotate(90, expand=True))

for name, throughput, top1, color, dx, dy in [
    ("ConvNeXt-Tiny", 249.4, 82.78, "#2563a6", -50, -81),
    ("ResNet-50", 346.9, 81.64, "#e28125", -118, 28),
]:
    x, y = xp(throughput), yp(top1)
    d.ellipse((x-13,y-13,x+13,y+13), fill=color, outline="white", width=3)
    d.text((x+dx,y+dy), name, font=fh, fill=color)
    d.text((x+dx,y+dy+36), f"{top1:.2f}% | {throughput:.1f} img/s", font=fb, fill="#344154")
d.text((l, 810), "Speed uses synthetic inputs; accuracy comes from prior validation runs.", font=fs, fill="#687689")
im.save(out)
print(out)
