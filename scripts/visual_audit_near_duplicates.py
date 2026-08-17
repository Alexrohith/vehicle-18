from pathlib import Path
from collections import defaultdict
import math
import json
import hashlib

import pandas as pd
import imagehash
from PIL import Image, ImageOps, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("data/splits/train.csv")
VAL_CSV = Path("data/splits/val.csv")
TEST_CSV = Path("data/splits/test.csv")

OUTPUT_DIR = Path("artifacts/leakage_visual_audit")

PHASH_THRESHOLD = 4

# Number of cross-split groups to visually inspect
MAX_GROUPS_TO_RENDER = 50

# Maximum images displayed from one group
MAX_IMAGES_PER_GROUP = 8

THUMB_SIZE = (260, 190)

PADDING = 15
LABEL_HEIGHT = 75

BACKGROUND = "white"
TEXT = "black"


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):

        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb

        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra

        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# LOAD SPLITS
# ============================================================

def load_splits():

    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    df = pd.concat(
        [train, val, test],
        ignore_index=True
    )

    # Normalize paths
    df["path"] = df["path"].astype(str)

    return df


# ============================================================
# HASH IMAGES
# ============================================================

def calculate_hashes(df):

    hashes = []

    print("=" * 60)
    print("CALCULATING PERCEPTUAL HASHES")
    print("=" * 60)

    total = len(df)

    for i, row in df.iterrows():

        path = Path(row["path"])

        try:

            with Image.open(path) as img:

                img = img.convert("RGB")

                h = imagehash.phash(img)

            hashes.append(h)

        except Exception as e:

            print(f"\nERROR: {path}")
            print(e)

            hashes.append(None)

        if (i + 1) % 250 == 0 or i + 1 == total:

            print(
                f"Processed {i + 1}/{total}"
            )

    return hashes


# ============================================================
# BUILD GROUPS
# ============================================================

def build_groups(df, hashes):

    print()
    print("=" * 60)
    print("BUILDING NEAR-DUPLICATE GROUPS")
    print("=" * 60)

    valid_indices = [
        i for i, h in enumerate(hashes)
        if h is not None
    ]

    n = len(df)

    uf = UnionFind(n)

    comparisons = 0
    matches = 0

    print()
    print(f"Images to compare : {len(valid_indices)}")
    print(f"pHash threshold   : {PHASH_THRESHOLD}")
    print()

    for pos, i in enumerate(valid_indices):

        hi = hashes[i]

        for j in valid_indices[pos + 1:]:

            hj = hashes[j]

            distance = hi - hj

            comparisons += 1

            if distance <= PHASH_THRESHOLD:

                uf.union(i, j)
                matches += 1

        if (
            (pos + 1) % 250 == 0
            or pos + 1 == len(valid_indices)
        ):

            print(
                f"Compared row {pos + 1}/{len(valid_indices)} "
                f"| Matches: {matches}"
            )

    groups = defaultdict(list)

    for i in valid_indices:

        root = uf.find(i)

        groups[root].append(i)

    duplicate_groups = [
        indices
        for indices in groups.values()
        if len(indices) > 1
    ]

    print()
    print(f"Total comparisons : {comparisons:,}")
    print(f"Near-duplicate links : {matches:,}")
    print(f"Total groups : {len(groups):,}")
    print(
        f"Groups containing >1 image : "
        f"{len(duplicate_groups):,}"
    )

    return duplicate_groups


# ============================================================
# GROUP INFORMATION
# ============================================================

def get_group_info(df, indices):

    rows = df.iloc[indices]

    splits = sorted(
        rows["split"].unique().tolist()
    )

    labels = sorted(
        rows["label"].astype(int).unique().tolist()
    )

    split_set = set(splits)

    cross_split = len(split_set) > 1

    mixed_label = len(labels) > 1

    # Priority:
    # 1. mixed labels
    # 2. train/test
    # 3. train/validation
    # 4. validation/test
    # 5. other

    if mixed_label:
        priority = 0

    elif {"train", "test"}.issubset(split_set):
        priority = 1

    elif {"train", "validation"}.issubset(split_set):
        priority = 2

    elif {"validation", "test"}.issubset(split_set):
        priority = 3

    else:
        priority = 4

    return {
        "indices": indices,
        "size": len(indices),
        "splits": splits,
        "labels": labels,
        "cross_split": cross_split,
        "mixed_label": mixed_label,
        "priority": priority
    }


# ============================================================
# FONT
# ============================================================

def get_font(size):

    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for path in candidates:

        if Path(path).exists():

            try:
                return ImageFont.truetype(
                    path,
                    size
                )
            except:
                pass

    return ImageFont.load_default()


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    df,
    group,
    group_number
):

    indices = group["indices"]

    # Keep the most useful images
    indices = indices[:MAX_IMAGES_PER_GROUP]

    columns = 2

    rows = math.ceil(
        len(indices) / columns
    )

    cell_width = THUMB_SIZE[0]
    cell_height = (
        THUMB_SIZE[1]
        + LABEL_HEIGHT
    )

    sheet_width = (
        columns * cell_width
        + (columns + 1) * PADDING
    )

    sheet_height = (
        rows * cell_height
        + (rows + 1) * PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        BACKGROUND
    )

    draw = ImageDraw.Draw(sheet)

    title_font = get_font(20)
    label_font = get_font(14)

    for position, index in enumerate(indices):

        row = df.iloc[index]

        path = Path(row["path"])

        try:

            with Image.open(path) as img:

                img = img.convert("RGB")

                thumbnail = ImageOps.contain(
                    img,
                    THUMB_SIZE
                )

        except Exception as e:

            print(
                f"Could not open {path}: {e}"
            )

            thumbnail = Image.new(
                "RGB",
                THUMB_SIZE,
                "gray"
            )

        x = (
            PADDING
            + (position % columns) * cell_width
        )

        y = (
            PADDING
            + (position // columns) * cell_height
        )

        # Center image
        paste_x = (
            x
            + (cell_width - thumbnail.width) // 2
        )

        paste_y = (
            y
            + (THUMB_SIZE[1] - thumbnail.height) // 2
        )

        sheet.paste(
            thumbnail,
            (paste_x, paste_y)
        )

        # Border
        draw.rectangle(
            [
                x,
                y,
                x + cell_width,
                y + THUMB_SIZE[1]
            ],
            outline="black",
            width=2
        )

        split = row["split"]
        label = int(row["label"])

        label_text = (
            f"{split.upper()} | "
            f"{'FRAUD' if label == 1 else 'NON-FRAUD'}\n"
            f"{path.name}"
        )

        draw.text(
            (x, y + THUMB_SIZE[1] + 7),
            label_text,
            fill=TEXT,
            font=label_font
        )

    # Header
    title = (
        f"GROUP {group_number} | "
        f"Images={group['size']} | "
        f"Splits={','.join(group['splits'])} | "
        f"Labels={group['labels']}"
    )

    # Put title at top if possible
    # Create a separate title strip
    title_height = 45

    final_sheet = Image.new(
        "RGB",
        (
            sheet.width,
            sheet.height + title_height
        ),
        BACKGROUND
    )

    final_sheet.paste(
        sheet,
        (0, title_height)
    )

    draw2 = ImageDraw.Draw(final_sheet)

    draw2.text(
        (10, 10),
        title,
        fill=TEXT,
        font=title_font
    )

    output_path = (
        OUTPUT_DIR
        / f"group_{group_number:04d}.jpg"
    )

    final_sheet.save(
        output_path,
        quality=95
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("NEAR-DUPLICATE VISUAL AUDIT")
    print("=" * 60)

    print()
    print(
        "This script does NOT modify the dataset."
    )

    print(
        "It creates contact sheets for manual inspection."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING ORIGINAL SPLITS")
    print("=" * 60)

    df = load_splits()

    print()
    print(f"Train       : {(df['split'] == 'train').sum()}")
    print(
        f"Validation  : "
        f"{(df['split'] == 'validation').sum()}"
    )
    print(
        f"Test        : "
        f"{(df['split'] == 'test').sum()}"
    )
    print(f"Total       : {len(df)}")

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    hashes = calculate_hashes(df)

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    duplicate_groups = build_groups(
        df,
        hashes
    )

    # --------------------------------------------------------
    # Group analysis
    # --------------------------------------------------------

    group_infos = []

    for indices in duplicate_groups:

        info = get_group_info(
            df,
            indices
        )

        if info["cross_split"]:

            group_infos.append(info)

    print()
    print("=" * 60)
    print("CROSS-SPLIT GROUP ANALYSIS")
    print("=" * 60)

    print(
        f"Cross-split groups : "
        f"{len(group_infos)}"
    )

    mixed_label_groups = [
        g for g in group_infos
        if g["mixed_label"]
    ]

    train_test_groups = [
        g for g in group_infos
        if {"train", "test"}.issubset(
            set(g["splits"])
        )
    ]

    train_val_groups = [
        g for g in group_infos
        if {"train", "validation"}.issubset(
            set(g["splits"])
        )
    ]

    val_test_groups = [
        g for g in group_infos
        if {"validation", "test"}.issubset(
            set(g["splits"])
        )
    ]

    print(
        f"Mixed-label groups : "
        f"{len(mixed_label_groups)}"
    )

    print(
        f"Train/Test groups  : "
        f"{len(train_test_groups)}"
    )

    print(
        f"Train/Val groups   : "
        f"{len(train_val_groups)}"
    )

    print(
        f"Val/Test groups    : "
        f"{len(val_test_groups)}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    group_infos.sort(
        key=lambda g: (
            g["priority"],
            -g["size"]
        )
    )

    # --------------------------------------------------------
    # Save CSV summary
    # --------------------------------------------------------

    summary_rows = []

    for number, group in enumerate(
        group_infos,
        start=1
    ):

        summary_rows.append({
            "group_number": number,
            "images": group["size"],
            "splits": ",".join(
                group["splits"]
            ),
            "labels": ",".join(
                map(str, group["labels"])
            ),
            "mixed_label": group["mixed_label"]
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "cross_split_groups.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Render contact sheets
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("GENERATING CONTACT SHEETS")
    print("=" * 60)

    selected = group_infos[
        :MAX_GROUPS_TO_RENDER
    ]

    print(
        f"Rendering {len(selected)} groups..."
    )

    rendered = []

    for number, group in enumerate(
        selected,
        start=1
    ):

        path = create_contact_sheet(
            df,
            group,
            number
        )

        rendered.append(str(path))

        print(
            f"[{number}/{len(selected)}] "
            f"{path.name} | "
            f"Splits={','.join(group['splits'])} | "
            f"Labels={group['labels']}"
        )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report = {
        "total_images": len(df),
        "phash_threshold": PHASH_THRESHOLD,
        "duplicate_groups": len(
            duplicate_groups
        ),
        "cross_split_groups": len(
            group_infos
        ),
        "mixed_label_groups": len(
            mixed_label_groups
        ),
        "train_test_groups": len(
            train_test_groups
        ),
        "train_validation_groups": len(
            train_val_groups
        ),
        "validation_test_groups": len(
            val_test_groups
        ),
        "rendered_groups": len(
            rendered
        )
    }

    report_path = (
        OUTPUT_DIR
        / "visual_audit_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VISUAL AUDIT COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Contact sheets saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        f"Summary CSV:"
    )

    print(
        f"  {summary_path}"
    )

    print()
    print(
        "Open the JPG files and inspect the groups."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT train the final model yet."
    )


if __name__ == "__main__":
    main()