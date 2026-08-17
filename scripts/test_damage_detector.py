from pathlib import Path

from PIL import Image

from src.inference.damage_detector import (
    VehicleDamageDetector,
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path("test_images/car1.jpg")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLO VEHICLE DAMAGE TEST")
    print("=" * 60)

    print(
        f"\nImage: {IMAGE_PATH}"
    )

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    # --------------------------------------------------------
    # Load YOLO detector
    # --------------------------------------------------------

    detector = VehicleDamageDetector(
        model_path="models/best.pt",
        confidence=0.25,
        iou=0.45,
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    result = detector.predict(
        image
    )

    # ========================================================
    # OVERALL RESULT
    # ========================================================

    print()

    print(
        f"Damage detected : "
        f"{result['damage_detected']}"
    )

    print(
        f"Damage count    : "
        f"{result['damage_count']}"
    )

    print(
        f"Damage coverage : "
        f"{result['damage_coverage_percentage']:.2f}%"
    )

    print(
        f"Overall severity : "
        f"{result['overall_severity']}"
    )

    print(
        f"Severity score   : "
        f"{result['overall_severity_score']}/100"
    )

    print()
    print("-" * 60)

    # ========================================================
    # INDIVIDUAL DAMAGE DETECTIONS
    # ========================================================

    detections = result.get(
        "detections",
        []
    )

    if not detections:

        print()
        print(
            "No damage regions detected."
        )

    else:

        for index, damage in enumerate(
            detections,
            start=1,
        ):

            print()

            print(
                f"Damage #{index}"
            )

            print(
                f"Type       : "
                f"{damage['damage_type']}"
            )

            print(
                f"Confidence : "
                f"{damage['confidence']:.2%}"
            )

            print(
                f"Box        : "
                f"{damage['bounding_box']}"
            )

            print(
                f"Area       : "
                f"{damage['area_percentage']:.2f}%"
            )

            print(
                f"Severity   : "
                f"{damage['severity']}"
            )

            print(
                f"Score      : "
                f"{damage['severity_score']}/100"
            )

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    output_dir = Path(
        "artifacts/damage_detection"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "car1_annotated.jpg"
    )

    detector.annotate(
        image,
        output_path=output_path,
    )

    print()
    print("-" * 60)

    print(
        f"Annotated image : "
        f"{output_path}"
    )

    print()
    print("=" * 60)
    print("YOLO DAMAGE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
