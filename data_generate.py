import torch
import torchvision.transforms as transform
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# 设置图像和斑点的大小
itera = 1
seed_value = 30
step_size = 0.1
global G0_GSNew


def normalize_to_minus1_1(tensor):
    return tensor * 2.0 - 1.0


transform = transform.Compose([
    transform.Resize((8, 8)),
    transform.ToTensor(),
    normalize_to_minus1_1
])


def gs(image):
    global G0_GSNew

    # Resize and normalize the image
    Amplitude = cv2.resize(image, (8, 8))
    Amplitude = Amplitude / Amplitude.max()  # Normalize to [0, 1] range

    # Generate a random phase

    np.random.seed(seed_value)
    phase = np.random.rand(*Amplitude.shape) * 2 * np.pi
    np.random.seed(None)

    # Create the initial complex amplitude distribution for the GS algorithm
    g0_GS = Amplitude * np.exp(1j * phase)

    # Perform the iterative process
    for n in range(itera):
        G0_GS = fftshift(fft2(g0_GS))  # Fourier transform to frequency domain
        G0_GSNew = G0_GS / np.abs(G0_GS)  # Take the phase value, frequency domain with full 1 amplitude constraint
        g0_GSNew = ifft2(ifftshift(G0_GSNew))  # Inverse Fourier transform back to spatial domain
        g0_GS = Amplitude * (
                g0_GSNew / np.abs(g0_GSNew))  # Directly use the initial amplitude constraint without modification

    phase_image = np.angle(G0_GSNew)  # Extract the phase information
    phase_image_normalized = (phase_image + np.pi) / (2 * np.pi)  # Normalize the phase to [0, 1] range

    phase_image_normalized = torch.from_numpy(phase_image_normalized)
    phase_image_normalized = phase_image_normalized * 2 - 1
    if phase_image_normalized.dtype == torch.float64:
        phase_image_normalized = phase_image_normalized.float()
    return torch.flatten(phase_image_normalized)

