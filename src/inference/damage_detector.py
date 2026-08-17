# src/inference/damage_detector.py

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


class VehicleDamageDetector:
    """
    YOLO-based vehicle damage detector.

    Detects damage regions using YOLO bounding boxes and
    estimates damage severity using:
        - bounding-box area
        - detection confidence
        - number of detected damage regions

    The severity score is a heuristic score from 0-100.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
    ):

        self.model_path = Path(model_path)

        self.confidence = confidence
        self.iou = iou

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        print(
            f"Loading YOLO damage model: "
            f"{self.model_path}"
        )

        self.model = YOLO(
            str(self.model_path)
        )

        print(
            "YOLO damage model loaded successfully."
        )

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    def _prepare_image(self, image):

        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        if isinstance(
            image,
            (str, Path)
        ):

            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            return str(image_path)

        # ----------------------------------------------------
        # PIL image
        # ----------------------------------------------------

        if isinstance(
            image,
            Image.Image
        ):

            return np.array(
                image.convert("RGB")
            )

        # ----------------------------------------------------
        # NumPy image
        # ----------------------------------------------------

        if isinstance(
            image,
            np.ndarray
        ):

            if image.ndim == 2:

                image = np.stack(
                    [image, image, image],
                    axis=-1
                )

            if (
                image.ndim == 3
                and image.shape[2] == 4
            ):

                image = image[:, :, :3]

            return image

        raise TypeError(
            "Unsupported image type. "
            "Use a file path, PIL Image, "
            "or NumPy array."
        )

    # ========================================================
    # SEVERITY FROM BOUNDING BOX
    # ========================================================

    def _calculate_box_severity(
        self,
        area_percentage,
        confidence,
    ):
        """
        Calculate severity for one detected damage box.

        Area contribution:
            < 2%   -> Low
            2-8%   -> Moderate
            8-20%  -> High
            > 20%  -> Severe
        """

        # ----------------------------------------------------
        # Area score
        # ----------------------------------------------------

        if area_percentage <= 2:

            area_score = (
                area_percentage / 2
            ) * 25

        elif area_percentage <= 8:

            area_score = (
                25
                + (
                    (area_percentage - 2)
                    / 6
                ) * 25
            )

        elif area_percentage <= 20:

            area_score = (
                50
                + (
                    (area_percentage - 8)
                    / 12
                ) * 30
            )

        else:

            area_score = 80

            extra = min(
                area_percentage - 20,
                20
            )

            area_score += (
                extra / 20
            ) * 10

        # ----------------------------------------------------
        # Confidence contribution
        # ----------------------------------------------------

        confidence_score = (
            confidence * 10
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        score = (
            area_score * 0.90
            + confidence_score
        )

        score = min(
            max(score, 0),
            100
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if score < 25:

            severity = "Low"

        elif score < 50:

            severity = "Moderate"

        elif score < 75:

            severity = "High"

        else:

            severity = "Severe"

        return (
            severity,
            round(score, 2)
        )

    # ========================================================
    # DETECTION
    # ========================================================

    def detect(self, image):

        source = self._prepare_image(
            image
        )

        results = self.model.predict(
            source=source,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        if not results:

            return []

        result = results[0]

        if result.boxes is None:

            return []

        boxes = result.boxes

        # ----------------------------------------------------
        # Get image dimensions
        # ----------------------------------------------------

        if isinstance(
            source,
            str
        ):

            with Image.open(
                source
            ) as img:

                image_width, image_height = (
                    img.size
                )

        else:

            image_height, image_width = (
                source.shape[:2]
            )

        image_area = (
            image_width
            * image_height
        )

        detections = []

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------

        names = self.model.names

        # ----------------------------------------------------
        # Process every bounding box
        # ----------------------------------------------------

        for i in range(
            len(boxes)
        ):

            # Bounding box
            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .tolist()
            )

            # Confidence
            confidence = float(
                boxes.conf[i]
                .cpu()
                .item()
            )

            # Class ID
            class_id = int(
                boxes.cls[i]
                .cpu()
                .item()
            )

            # Class name
            if isinstance(
                names,
                dict
            ):

                damage_type = names.get(
                    class_id,
                    f"class_{class_id}"
                )

            else:

                damage_type = names[
                    class_id
                ]

            # ------------------------------------------------
            # Bounding box dimensions
            # ------------------------------------------------

            box_width = max(
                0,
                x2 - x1
            )

            box_height = max(
                0,
                y2 - y1
            )

            box_area = (
                box_width
                * box_height
            )

            # ------------------------------------------------
            # Area percentage
            # ------------------------------------------------

            if image_area > 0:

                area_percentage = (
                    box_area
                    / image_area
                ) * 100

            else:

                area_percentage = 0

            # ------------------------------------------------
            # Individual severity
            # ------------------------------------------------

            severity, severity_score = (
                self._calculate_box_severity(
                    area_percentage,
                    confidence
                )
            )

            # ------------------------------------------------
            # Store detection
            # ------------------------------------------------

            detections.append(
                {
                    "damage_type": str(
                        damage_type
                    ),

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bounding_box": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],

                    "area_percentage": round(
                        area_percentage,
                        2
                    ),

                    "severity": severity,

                    "severity_score": severity_score,
                }
            )

        return detections

    # ========================================================
    # OVERALL DAMAGE COVERAGE
    # ========================================================

    def _calculate_damage_coverage(
        self,
        detections,
        image_width,
        image_height,
    ):
        """
        Calculate the percentage of the image covered by the
        union of all detected damage bounding boxes.

        Overlapping bounding boxes are counted only once.
        """

        if not detections:
            return 0.0

        if image_width <= 0 or image_height <= 0:
            return 0.0

        mask = np.zeros(
            (image_height, image_width),
            dtype=np.uint8,
        )

        for detection in detections:
            x1, y1, x2, y2 = detection["bounding_box"]

            x1 = max(0, min(int(round(x1)), image_width - 1))
            y1 = max(0, min(int(round(y1)), image_height - 1))
            x2 = max(0, min(int(round(x2)), image_width))
            y2 = max(0, min(int(round(y2)), image_height))

            if x2 <= x1 or y2 <= y1:
                continue

            mask[y1:y2, x1:x2] = 1

        damaged_pixels = int(mask.sum())
        total_pixels = image_width * image_height

        return round(
            (damaged_pixels / total_pixels) * 100,
            2,
        )

    # ========================================================
    # COMPLETE PREDICTION
    # ========================================================
    
    def predict(self, image):

        detections = self.detect(
            image
        )

        # Get original image dimensions for overall damage coverage.
        source = self._prepare_image(image)

        if isinstance(source, str):
            with Image.open(source) as img:
                image_width, image_height = img.size
        else:
            image_height, image_width = source.shape[:2]

        damage_coverage = self._calculate_damage_coverage(
            detections,
            image_width,
            image_height,
        )

        # ----------------------------------------------------
        # No damage
        # ----------------------------------------------------

        if not detections:

            return {
                "damage_detected": False,

                "damage_count": 0,

                "damage_coverage_percentage": 0.0,

                "overall_severity":
                    "No Damage Detected",

                "overall_severity_score":
                    0.0,

                "detections": [],
            }

        # ----------------------------------------------------
        # Damage exists
        # ----------------------------------------------------

        severity_scores = [
            detection[
                "severity_score"
            ]
            for detection in detections
        ]

        # Use the most severe detected
        # damage as the overall severity.
        max_score = max(
            severity_scores
        )

        # Add a small multi-damage factor
        # when multiple regions are detected.

        multi_damage_bonus = min(
            (len(detections) - 1)
            * 3,
            15
        )

        overall_score = min(
            max_score
            + multi_damage_bonus,
            100
        )

        # ----------------------------------------------------
        # Overall severity label
        # ----------------------------------------------------

        if overall_score < 25:

            overall_severity = "Low"

        elif overall_score < 50:

            overall_severity = "Moderate"

        elif overall_score < 75:

            overall_severity = "High"

        else:

            overall_severity = "Severe"

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {
            "damage_detected": True,

            "damage_count": len(
                detections
            ),

            "damage_coverage_percentage":
                damage_coverage,

            "overall_severity":
                overall_severity,

            "overall_severity_score":
                round(
                    overall_score,
                    2
                ),

            "detections":
                detections,
        }

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    def annotate(
        self,
        image,
        output_path=None,
    ):

        source = self._prepare_image(
            image
        )

        results = self.model.predict(
            source=source,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        if not results:

            raise RuntimeError(
                "YOLO returned no results."
            )

        result = results[0]

        annotated_image = (
            result.plot()
        )

        # ----------------------------------------------------
        # Save annotated image
        # ----------------------------------------------------

        if output_path is not None:

            output_path = Path(
                output_path
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            import cv2

            cv2.imwrite(
                str(output_path),
                annotated_image
            )

        return annotated_image


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("YOLO VEHICLE DAMAGE DETECTOR")
    print("=" * 60)

    image_path = Path(
        "test_images/car1.jpg"
    )

    detector = VehicleDamageDetector(
        model_path="models/best.pt",
        confidence=0.25,
        iou=0.45,
    )

    result = detector.predict(
        image_path
    )

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

    for i, damage in enumerate(
        result["detections"],
        start=1
    ):

        print(
            f"Damage #{i}"
        )

        print(
            f"  Type       : "
            f"{damage['damage_type']}"
        )

        print(
            f"  Confidence : "
            f"{damage['confidence']:.2%}"
        )

        print(
            f"  Box        : "
            f"{damage['bounding_box']}"
        )

        print(
            f"  Area       : "
            f"{damage['area_percentage']:.2f}%"
        )

        print(
            f"  Severity   : "
            f"{damage['severity']}"
        )

        print(
            f"  Score      : "
            f"{damage['severity_score']}/100"
        )

        print()

    output_path = Path(
        "artifacts/damage_detection/"
        "car1_annotated.jpg"
    )

    detector.annotate(
        image_path,
        output_path=output_path
    )

    print(
        f"Annotated image: "
        f"{output_path}"
    )

    print()
    print("=" * 60)
    print("YOLO DAMAGE TEST COMPLETE")
    print("=" * 60)