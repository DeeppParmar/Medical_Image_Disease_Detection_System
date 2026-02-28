"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utility.
Generates heatmap overlays showing which regions the model focused on.
"""

import base64
import io
import logging

import cv2
import numpy as np

logger = logging.getLogger("MediScan")

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _find_target_layer(model, target_layer_name: str):
    """
    Walk the model's named modules to find the layer matching target_layer_name.
    Supports dotted paths like 'densenet121.features.denseblock4'.
    """
    for name, module in model.named_modules():
        if name == target_layer_name:
            return module
    return None


def generate_gradcam(image_path: str, model, target_layer_name: str, device=None,
                     preprocess_fn=None) -> str | None:
    """
    Generate a Grad-CAM heatmap overlay for the given image using the model.

    Args:
        image_path: Path to the input image.
        model: PyTorch model (must be in eval mode).
        target_layer_name: Dotted path to the target conv layer
                           (e.g. 'densenet121.features.denseblock4').
        device: torch device. Defaults to CPU.
        preprocess_fn: Optional callable(image_path) -> tensor.
                       If None, uses a default ImageNet preprocessing.

    Returns:
        Base64-encoded PNG string of the blended heatmap, or None on failure.
    """
    if not TORCH_AVAILABLE:
        logger.warning("[GRADCAM] PyTorch not available, skipping Grad-CAM")
        return None

    try:
        if device is None:
            device = torch.device('cpu')

        # Find the target layer
        target_layer = _find_target_layer(model, target_layer_name)
        if target_layer is None:
            logger.warning(f"[GRADCAM] Target layer '{target_layer_name}' not found in model")
            return None

        # Storage for activations and gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Preprocess image
            if preprocess_fn is not None:
                input_tensor = preprocess_fn(image_path)
                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = torch.tensor(input_tensor)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
            else:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
                img = Image.open(image_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0)

            input_tensor = input_tensor.to(device)
            model.eval()

            # Enable gradients for this pass
            input_tensor.requires_grad_(False)

            # Forward pass
            output = model(input_tensor)

            # Get top predicted class
            if output.dim() == 1:
                output = output.unsqueeze(0)
            top_class = output.argmax(dim=1).item()

            # Zero gradients
            model.zero_grad()

            # Backward pass on top class
            target_score = output[0, top_class]
            target_score.backward(retain_graph=False)

            if not activations or not gradients:
                logger.warning("[GRADCAM] No activations/gradients captured")
                return None

            # Compute Grad-CAM
            act = activations[0]   # (1, C, H, W)
            grad = gradients[0]    # (1, C, H, W)

            # Weights = Global Average Pooling of gradients
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activations
            cam = torch.sum(weights * act, dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam)

            # Resize to 224x224
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # Normalize to 0-255
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
            cam = (cam * 255).astype(np.uint8)

            # Apply colormap
            heatmap_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

            # Load original image and resize to 224x224
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                # Try with PIL
                pil_img = Image.open(image_path).convert('RGB')
                orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            orig_img = cv2.resize(orig_img, (224, 224))

            # Blend: 60% original + 40% heatmap
            blended = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

            # Encode to base64 PNG
            _, buffer = cv2.imencode('.png', blended)
            b64_string = base64.b64encode(buffer).decode('utf-8')

            logger.info(f"[GRADCAM] Heatmap generated successfully | layer={target_layer_name}")
            return b64_string

        finally:
            # Always remove hooks
            fh.remove()
            bh.remove()

    except Exception as e:
        logger.error(f"[GRADCAM] Failed to generate heatmap: {e}")
        return None
