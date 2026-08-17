from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import imagehash


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("data/splits/train.csv")
TEST_CSV = Path("data/splits/test.csv")

OUTPUT_DIR = Path("artifacts/leakage_audit")
OUTPUT_FILE = OUTPUT_DIR / "train_test_near_duplicates.jpg"

HAMMING_THRESHOLD = 2

# Only visualize the strongest matches
MAX_PAIRS = 50

THUMBNAIL_SIZE = (300, 220)

BACKGROUND = "white"
TEXT_COLOR = "black"


# ============================================================
# LOAD DATA
# ============================================================

def load_paths(csv_path):

    df = pd.read_csv(csv_path)

    return df


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_hash(path):

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            return imagehash.phash(image)

    except Exception as e:

        print(f"Could not process {path}: {e}")

        return None


# ============================================================
# BUILD HASHES
# ============================================================

def build_hashes(df, name):

    results = []

    print(f"\nHashing {name} images...")

    for i, row in df.iterrows():

        path = str(row["path"])

        image_hash = calculate_hash(path)

        if image_hash is not None:

            results.append(
                {
                    "path": path,
                    "label": int(row["label"]),
                    "hash": image_hash
                }
            )

        if (i + 1) % 500 == 0:

            print(
                f"  Processed {i + 1}/{len(df)}"
            )

    return results


# ============================================================
# FIND STRONG MATCHES
# ============================================================

def find_matches(train_hashes, test_hashes):

    matches = []

    print("\nComparing Train ↔ Test...")

    for i, train_item in enumerate(train_hashes):

        best_distance = 999
        best_test = None

        for test_item in test_hashes:

            distance = (
                train_item["hash"]
                -
                test_item["hash"]
            )

            if distance < best_distance:

                best_distance = distance
                best_test = test_item

        if (
            best_test is not None
            and best_distance <= HAMMING_THRESHOLD
        ):

            matches.append(
                {
                    "train_path": train_item["path"],
                    "train_label": train_item["label"],
                    "test_path": best_test["path"],
                    "test_label": best_test["label"],
                    "distance": best_distance
                }
            )

        if (i + 1) % 500 == 0:

            print(
                f"  Compared {i + 1}/{len(train_hashes)}"
            )

    # Strongest matches first
    matches.sort(
        key=lambda x: x["distance"]
    )

    return matches[:MAX_PAIRS]


# ============================================================
# OPEN IMAGE
# ============================================================

def prepare_image(path):

    try:

        image = Image.open(path).convert("RGB")

        image.thumbnail(
            THUMBNAIL_SIZE
        )

        canvas = Image.new(
            "RGB",
            THUMBNAIL_SIZE,
            BACKGROUND
        )

        x = (
            THUMBNAIL_SIZE[0]
            -
            image.width
        ) // 2

        y = (
            THUMBNAIL_SIZE[1]
            -
            image.height
        ) // 2

        canvas.paste(
            image,
            (x, y)
        )

        return canvas

    except Exception:

        return Image.new(
            "RGB",
            THUMBNAIL_SIZE,
            "gray"
        )


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(matches):

    if not matches:

        print(
            "\nNo strong near-duplicate pairs found."
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Try a standard font
    try:

        font = ImageFont.truetype(
            "arial.ttf",
            16
        )

    except:

        font = ImageFont.load_default()

    pair_width = (
        THUMBNAIL_SIZE[0] * 2
        +
        40
    )

    pair_height = 300

    columns = 2

    rows = (
        len(matches)
        +
        columns
        -
        1
    ) // columns

    sheet = Image.new(
        "RGB",
        (
            pair_width * columns,
            pair_height * rows
        ),
        BACKGROUND
    )

    draw = ImageDraw.Draw(sheet)

    for index, match in enumerate(matches):

        col = index % columns
        row = index // columns

        x0 = col * pair_width
        y0 = row * pair_height

        train_image = prepare_image(
            match["train_path"]
        )

        test_image = prepare_image(
            match["test_path"]
        )

        sheet.paste(
            train_image,
            (
                x0 + 5,
                y0 + 5
            )
        )

        sheet.paste(
            test_image,
            (
                x0 + 315,
                y0 + 5
            )
        )

        # Short filenames
        train_name = Path(
            match["train_path"]
        ).name

        test_name = Path(
            match["test_path"]
        ).name

        distance = match["distance"]

        train_label = (
            "Fraud"
            if match["train_label"] == 1
            else "Non-Fraud"
        )

        test_label = (
            "Fraud"
            if match["test_label"] == 1
            else "Non-Fraud"
        )

        draw.text(
            (
                x0 + 5,
                y0 + 230
            ),
            f"TRAIN: {train_name} ({train_label})",
            fill=TEXT_COLOR,
            font=font
        )

        draw.text(
            (
                x0 + 315,
                y0 + 230
            ),
            f"TEST: {test_name} ({test_label})",
            fill=TEXT_COLOR,
            font=font
        )

        draw.text(
            (
                x0 + 5,
                y0 + 255
            ),
            f"pHash distance: {distance}",
            fill=TEXT_COLOR,
            font=font
        )

    sheet.save(
        OUTPUT_FILE,
        quality=95
    )

    print()
    print("=" * 60)
    print("CONTACT SHEET CREATED")
    print("=" * 60)

    print(
        f"Pairs visualized: {len(matches)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAIN ↔ TEST NEAR-DUPLICATE VISUAL AUDIT")
    print("=" * 60)

    train_df = load_paths(
        TRAIN_CSV
    )

    test_df = load_paths(
        TEST_CSV
    )

    print(
        f"\nTrain images: {len(train_df)}"
    )

    print(
        f"Test images : {len(test_df)}"
    )

    train_hashes = build_hashes(
        train_df,
        "Train"
    )

    test_hashes = build_hashes(
        test_df,
        "Test"
    )

    matches = find_matches(
        train_hashes,
        test_hashes
    )

    print()
    print(
        f"Strong matches found: "
        f"{len(matches)}"
    )

    create_contact_sheet(
        matches
    )


if __name__ == "__main__":

    main()