from PIL import Image, ImageDraw, ImageFont
import os

# Font folder
font_root = r"C:\Users\krishiv\Desktop\Instagram\Fonts\Fonts"

# All font files as a list of tuples: (display name, filename)
fonts = [
    ("Attentica UltraLight", "Attentica_4F_UltraLight.ttf"),
    ("FatCow Italic OTF", "FatCow-Italic.otf"),
    ("FatCow Italic TTF", "FatCow-Italic.ttf"),
    ("FatCow OTF", "FatCow.otf"),
    ("FatCow TTF", "FatCow.ttf"),
    ("GlacialIndifference Bold", "GlacialIndifference-Bold.otf"),
    ("GlacialIndifference Regular", "GlacialIndifference-Regular.otf"),
    ("Heathergreen", "Heathergreen.otf"),
    ("Maxwell Bold", "MAXWELL_BOLD.ttf"),
    ("Maxwell Light", "MAXWELL_LIGHT.ttf"),
    ("Maxwell Regular", "MAXWELL_REGULAR.ttf"),
    ("Ostrich Black", "ostrich-black.ttf"),
    ("Ostrich Bold", "ostrich-bold.ttf"),
    ("Ostrich Dashed", "ostrich-dashed.ttf"),
    ("Ostrich Light", "ostrich-light.ttf"),
    ("Ostrich Regular", "ostrich-regular.ttf"),
    ("Ostrich Rounded", "ostrich-rounded.ttf"),
    ("Ostrich Sans Inline Italic", "ostrich-sans-inline-italic.ttf"),
    ("Ostrich Sans Inline Regular", "ostrich-sans-inline-regular.ttf"),
    ("OstrichSans Black", "OstrichSans-Black.otf"),
    ("OstrichSans Bold", "OstrichSans-Bold.otf"),
    ("OstrichSans Light", "OstrichSans-Light.otf"),
    ("OstrichSans Medium", "OstrichSans-Medium.otf"),
    ("OstrichSansDashed Medium", "OstrichSansDashed-Medium.otf"),
    ("OstrichSansInline Italic", "OstrichSansInline-Italic.otf"),
    ("OstrichSansInline Regular", "OstrichSansInline-Regular.otf"),
    ("OstrichSansRounded Medium", "OstrichSansRounded-Medium.otf"),
    ("Oswald Bold", "Oswald-Bold.ttf"),
    ("Oswald Light", "Oswald-Light.ttf"),
    ("Oswald Regular", "Oswald-Regular.ttf"),
    ("PlayfairDisplay Black", "PlayfairDisplay-Black.ttf"),
    ("PlayfairDisplay BlackItalic", "PlayfairDisplay-BlackItalic.ttf"),
    ("PlayfairDisplay Bold", "PlayfairDisplay-Bold.ttf"),
    ("PlayfairDisplay BoldItalic", "PlayfairDisplay-BoldItalic.ttf"),
    ("PlayfairDisplay Italic", "PlayfairDisplay-Italic.ttf"),
    ("PlayfairDisplay Regular", "PlayfairDisplay-Regular.ttf"),
    ("Raleway Bold", "Raleway-Bold.ttf"),
    ("Raleway ExtraBold", "Raleway-ExtraBold.ttf"),
    ("Raleway ExtraLight", "Raleway-ExtraLight.ttf"),
    ("Raleway Heavy", "Raleway-Heavy.ttf"),
    ("Raleway Light", "Raleway-Light.ttf"),
    ("Raleway Medium", "Raleway-Medium.ttf"),
    ("Raleway Regular", "Raleway-Regular.ttf"),
    ("Raleway SemiBold", "Raleway-SemiBold.ttf"),
    ("Raleway Thin", "Raleway-Thin.ttf"),
    ("Sofana", "Sofana.otf"),
    ("WireOne", "WireOne.ttf"),
]

# Create a blank white image
line_height = 60
img_width = 1200
img_height = line_height * len(fonts) + 100
image = Image.new("RGB", (img_width, img_height), "white")
draw = ImageDraw.Draw(image)

# Write each font name using the font itself
y = 20
for name, file in fonts:
    font_path = os.path.join(font_root, file)
    try:
        font = ImageFont.truetype(font_path, 30)
        draw.text((50, y), name, font=font, fill="black")
    except Exception as e:
        # If font fails to load, write with default font
        draw.text((50, y), f"{name} (Error loading font)", fill="red")
    y += line_height

# Save image
output_path = "all_fonts_preview.png"
image.save(output_path)
print(f"✅ Image saved as {output_path}")
