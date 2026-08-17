from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from src.models.convnext import ConvNeXtClassifier


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/convnext_final_development_best.pth"
)

OUTPUT_DIR = Path(
    "artifacts/gradcam"
)

IMAGE_SIZE = 224

# LOCKED VALIDATION THRESHOLD
THRESHOLD = 0.30

CLASS_NAMES = {
    0: "Non-Fraud",
    1: "Fraud",
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# LOAD CONVNEXT MODEL
# ============================================================

def load_model():

    print("\nLoading ConvNeXt-Tiny...")

    model = ConvNeXtClassifier(
        num_classes=1,
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    # --------------------------------------------------------
    # Support checkpoint format used by training script
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Grad-CAM requires gradients
    # --------------------------------------------------------

    for param in model.parameters():

        param.requires_grad = True

    model.eval()

    return model


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.hook = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )


    # --------------------------------------------------------
    # FORWARD HOOK
    # --------------------------------------------------------

    def forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

        if output.requires_grad:

            output.retain_grad()


    # --------------------------------------------------------
    # GENERATE CAM
    # --------------------------------------------------------

    def generate(
        self,
        image_tensor,
        predicted_class,
    ):

        with torch.enable_grad():

            self.model.zero_grad(
                set_to_none=True
            )

            output = self.model(
                image_tensor
            )

            # ------------------------------------------------
            # ConvNeXt has ONE binary logit
            # ------------------------------------------------

            logit = output[0, 0]

            # ------------------------------------------------
            # For Fraud:
            # maximize fraud logit
            #
            # For Non-Fraud:
            # maximize negative fraud logit
            # ------------------------------------------------

            if predicted_class == 1:

                target_score = logit

            else:

                target_score = -logit

            target_score.backward()


        # ----------------------------------------------------
        # Verify activation
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )


        if self.activations.grad is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )


        # ----------------------------------------------------
        # Activations
        # ----------------------------------------------------

        activations = (
            self.activations
            .detach()
        )


        gradients = (
            self.activations
            .grad
            .detach()
        )


        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )


        # ----------------------------------------------------
        # Weighted feature maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )


        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = torch.relu(
            cam
        )


        # ----------------------------------------------------
        # Remove dimensions
        # ----------------------------------------------------

        cam = (
            cam
            .squeeze()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()


        return (
            cam,
            output.detach(),
        )


    # --------------------------------------------------------
    # REMOVE HOOK
    # --------------------------------------------------------

    def remove_hooks(self):

        self.hook.remove()


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    output_path=None,
):

    print(
        f"\nLoading image:"
        f"\n{image_path}"
    )


    # ========================================================
    # LOAD ORIGINAL IMAGE
    # ========================================================

    original_image = Image.open(
        image_path
    ).convert("RGB")


    original_array = np.array(
        original_image
    )


    # ========================================================
    # PREPROCESS
    # ========================================================

    image_tensor = transform(
        original_image
    )


    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()


    # ========================================================
    # TARGET LAYER
    # ========================================================

    # Last ConvNeXt feature block.
    #
    # ConvNeXt structure:
    #
    # model.model.features
    #
    # features[-1]  -> final ConvNeXt stage
    # features[-1][-1] -> final ConvNeXt block
    #
    # This retains spatial information required by Grad-CAM.

    target_layer = (
        model.model
        .features[-1][-1]
    )


    print(
        "\nGrad-CAM target layer:"
    )

    print(
        target_layer
    )


    gradcam = GradCAM(
        model,
        target_layer,
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            image_tensor
        )

        # ConvNeXt binary output
        fraud_probability = (
            torch.sigmoid(
                output[0, 0]
            )
            .item()
        )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    predicted_class = int(
        fraud_probability >= THRESHOLD
    )


    predicted_label = (
        CLASS_NAMES[
            predicted_class
        ]
    )


    confidence = (
        fraud_probability
        if predicted_class == 1
        else 1.0 - fraud_probability
    )


    print(
        f"\nPrediction:"
        f" {predicted_label}"
    )


    print(
        f"Fraud Probability:"
        f" {fraud_probability:.4f}"
    )


    print(
        f"Confidence:"
        f" {confidence:.2%}"
    )


    print(
        f"Threshold:"
        f" {THRESHOLD:.2f}"
    )


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    print(
        "\nGenerating ConvNeXt Grad-CAM..."
    )


    heatmap, _ = gradcam.generate(
        image_tensor,
        predicted_class,
    )


    gradcam.remove_hooks()


    # ========================================================
    # RESIZE HEATMAP
    # ========================================================

    heatmap = cv2.resize(
        heatmap,
        (
            original_array.shape[1],
            original_array.shape[0],
        ),
    )


    # ========================================================
    # CONVERT HEATMAP
    # ========================================================

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )


    heatmap_color = (
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )
    )


    heatmap_color = (
        cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB,
        )
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(
        original_array,
        0.60,
        heatmap_color,
        0.40,
        0,
    )


    # ========================================================
    # OUTPUT PATH
    # ========================================================

    if output_path is None:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            OUTPUT_DIR
            /
            f"{Path(image_path).stem}"
            f"_convnext_gradcam.jpg"
        )

    else:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    plt.figure(
        figsize=(15, 5)
    )


    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1,
    )

    plt.imshow(
        original_array
    )

    plt.title(
        "Original Image"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2,
    )

    plt.imshow(
        heatmap,
        cmap="jet",
    )

    plt.title(
        "ConvNeXt Grad-CAM"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3,
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"{predicted_label}\n"
        f"Fraud Probability: "
        f"{fraud_probability:.2%}\n"
        f"Confidence: "
        f"{confidence:.2%}"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()


    print(
        "\nGrad-CAM saved to:"
    )

    print(
        output_path
    )


    return {
        "image": str(
            image_path
        ),

        "prediction": (
            predicted_label
        ),

        "predicted_class": (
            predicted_class
        ),

        "fraud_probability": (
            fraud_probability
        ),

        "confidence": (
            confidence
        ),

        "threshold": (
            THRESHOLD
        ),

        "gradcam_path": (
            str(output_path)
        ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "CONVNEXT-TINY GRAD-CAM"
    )

    print("=" * 60)


    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python -m src.explainability.gradcam '
            '"path/to/image.jpg"'
        )

        sys.exit(1)


    image_path = sys.argv[1]


    result = generate_gradcam(
        image_path
    )


    print(
        "\nPrediction:"
    )

    print(
        result["prediction"]
    )


    print(
        "\nFraud probability:"
    )

    print(
        f'{result["fraud_probability"]:.4f}'
    )


    print(
        "\nConfidence:"
    )

    print(
        f'{result["confidence"]:.2%}'
    )


    print(
        "\nGrad-CAM completed successfully ✅"
    )