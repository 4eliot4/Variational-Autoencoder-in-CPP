
import json
import struct
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
UPSCALE = 4
TRAIN_PER_CLASS = 4500
TEST_PER_CLASS = 500

CATEGORIES = [
    "airplane", "bicycle", "car", "helicopter",
    "sailboat", "train", "truck", "submarine"
]

OUT = Path("data/quickdraw64")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(1337)

BASE = (
    "https://storage.googleapis.com/"
    "quickdraw_dataset/full/simplified/"
)


def render_drawing(strokes):
    canvas_size = SIZE * UPSCALE
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)

    coords = [
        (x, y)
        for xs, ys in strokes
        for x, y in zip(xs, ys)
    ]

    if not coords:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)

    xs, ys = zip(*coords)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin
    height = ymax - ymin

    margin = 0.08 * canvas_size

    scale = (
        (canvas_size - 2 * margin)
        / max(width, height, 1)
    )

    offset_x = (canvas_size - width * scale) / 2
    offset_y = (canvas_size - height * scale) / 2

    line_width = 8

    for stroke_x, stroke_y in strokes:
        points = [
            (
                round((x - xmin) * scale + offset_x),
                round((y - ymin) * scale + offset_y)
            )
            for x, y in zip(stroke_x, stroke_y)
        ]

        if len(points) >= 2:
            draw.line(
                points,
                fill=255,
                width=line_width,
                joint="curve"
            )
        elif len(points) == 1:
            x, y = points[0]
            r = line_width // 2
            draw.ellipse(
                (x-r, y-r, x+r, y+r),
                fill=255
            )

    img = img.resize(
        (SIZE, SIZE),
        Image.Resampling.LANCZOS
    )

    return np.array(img, dtype=np.uint8)


def write_images(path, images):
    with open(path, "wb") as f:
        f.write(
            struct.pack(">IIII", 2051, len(images), SIZE, SIZE)
        )
        f.write(
            np.ascontiguousarray(images, dtype=np.uint8).tobytes()
        )


def write_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(np.asarray(labels, dtype=np.uint8).tobytes())


train_X, train_y = [], []
test_X, test_y = [], []

for label, category in enumerate(CATEGORIES):
    print(f"Downloading and rendering: {category}", flush=True)

    url = BASE + quote(category) + ".ndjson"
    images = []

    # Stream records instead of downloading each whole category.
    with urlopen(url, timeout=120) as response:
        for raw_line in response:
            item = json.loads(raw_line)

            if not item.get("recognized", False):
                continue

            strokes = item.get("drawing", [])
            if not strokes:
                continue

            image = render_drawing(strokes)

            # Reject nearly empty renderings.
            if np.count_nonzero(image > 32) < 20:
                continue

            images.append(image)

            if len(images) == TRAIN_PER_CLASS + TEST_PER_CLASS:
                break

    if len(images) < TRAIN_PER_CLASS + TEST_PER_CLASS:
        raise RuntimeError(
            f"Insufficient drawings for {category}: {len(images)}"
        )

    images = np.stack(images)
    images = images[rng.permutation(len(images))]

    train_X.append(images[:TRAIN_PER_CLASS])
    test_X.append(images[TRAIN_PER_CLASS:])

    train_y.append(
        np.full(TRAIN_PER_CLASS, label, dtype=np.uint8)
    )
    test_y.append(
        np.full(TEST_PER_CLASS, label, dtype=np.uint8)
    )

X_train = np.concatenate(train_X)
X_test = np.concatenate(test_X)
y_train = np.concatenate(train_y)
y_test = np.concatenate(test_y)

p = rng.permutation(len(X_train))
X_train, y_train = X_train[p], y_train[p]

p = rng.permutation(len(X_test))
X_test, y_test = X_test[p], y_test[p]

write_images(OUT / "train-images.idx3-ubyte", X_train)
write_images(OUT / "test-images.idx3-ubyte", X_test)

write_labels(OUT / "train-labels.idx1-ubyte", y_train)
write_labels(OUT / "test-labels.idx1-ubyte", y_test)

(OUT / "classes.txt").write_text(
    "\n".join(CATEGORIES) + "\n"
)

# Preview the inputs before doing any training.
preview = Image.new("L", (4 * SIZE, 4 * SIZE))

for i in range(16):
    tile = Image.fromarray(X_test[i], mode="L")
    preview.paste(
        tile, ((i % 4) * SIZE, (i // 4) * SIZE)
    )

preview.save(OUT / "preview.png")

print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("Saved:", OUT / "preview.png")
