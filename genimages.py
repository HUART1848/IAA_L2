import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter

W = 200
H = 200

def add_noise(img: Image, p: float=0.8):
    pix = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if random.random() < p:
                c = pix[x, y]
                nc = random.randint(0, 127)
                pix[x, y] = (nc + c) // 2

def prepare_source_images(source_folder: str="source", out_folder: str="source_prepared"):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    i = 0
    for file in os.listdir(source_folder):
        in_file = os.path.join(source_folder, file)
        out_file = os.path.join(out_folder, f"{i}.png")
        img = Image.open(in_file)
        img = img.resize((W, H)).rotate(-90).convert("L")
        
        img.save(out_file)
        print(f"{in_file} -> {out_file}")
        i += 1

def generate_lines(source_folder: str, n_per_source: int=10, noise: None | float=0.2) -> list[dict]:
    imgs = []
    i = 0
    for file in tqdm(os.listdir(source_folder)):
        base_file = os.path.join(source_folder, file)

        for _ in range(n_per_source):
            img = Image.open(base_file)
            img_draw = ImageDraw.Draw(img)

            if img.size != (W, H):
                raise ValueError(f"{base_file} is not {W}x{H}")

            x1 = random.randint(0, W)
            x2 = random.randint(0, W)
            y1 = 0
            y2 = random.randint(H // 5, H - H // 8)
            
            img_draw.line([(x1, H - y1), (x2, H - y2)], fill="white", width=random.randint(W//50, W//40))

            if noise is not None and (0.0 < noise < 1.0):
                add_noise(img, p=0.2)

            img = img.filter(ImageFilter.GaussianBlur(1.0))
            
            imgs.append({
                "id": i,
                "a": (x1, y1),
                "b": (x2, y2),
                "img": img
            })

            i += 1

    return imgs

def save_images(images: list[dict], img_folder: str="images", lab_filename: str="labels.csv"):
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    f = open(lab_filename, "w")
    for img in tqdm(images):
        img_name = f"{img['id']}.png"

        data = img["img"]
        data.save(f"{img_folder}/{img_name}")

        f.write(f"{img_name},{img['a'][0]},{img['a'][1]},{img['b'][0]},{img['b'][1]}\n")
    f.close()

def main():
    # On créé des images N&B en 200x200 à partir d'images normales
    print("Preparing base images...")
    prepare_source_images(source_folder="source", out_folder="source_prepared")
    
    print("Drawing lines...")
    imgs = generate_lines("source_prepared", n_per_source=50)

    print("Saving images...")
    save_images(imgs, img_folder="images", lab_filename="labels.csv")

if __name__ == "__main__":
    main()
