import cv2
import numpy as np
import torch
from codeformer import CodeFormer   
from PIL import Image
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


class VideoEnhancer:
    def __init__(self, mask_path='latentsync/utils/mask.png', var_threshold=50, fidelity_weight=0.9, upscale=1):
        self.mask_path = mask_path
        self.codeformer = CodeFormer(fidelity_weight=fidelity_weight, upscale=upscale)
        self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        self.var_threshold = var_threshold  
        self.upscale = upscale
        self.fidelity_weight = fidelity_weight

    def is_blurred(self, image):
        """Determine if the image is blurred."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.var_threshold   

    def enhance_image(self, input_image):
        frame = tensor2img(input_image)
        mask_resized = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask_indices = np.where(mask_resized == 0)
        if len(mask_indices[0]) == 0:
            combined_frame = frame  
        else:
            y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
            x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
            crop = frame_rgb[y_min:y_max + 1, x_min:x_max + 1]

            # Check if the cropped area is blurred
            if self.is_blurred(crop):
                # Enhance using CodeFormer
                enhanced = self.codeformer.upscale_image(frame_rgb)

                # Convert byte back to an image
                enhanced = np.frombuffer(enhanced, np.uint8)
                enhanced = cv2.imdecode(enhanced, cv2.COLOR_RGB2BGR)

                enhanced_resized = cv2.resize(enhanced, (frame.shape[1], frame.shape[0]))
                # frame_rgb[y_min:y_max + 1, x_min:x_max + 1] = enhanced_resized[y_min:y_max + 1, x_min:x_max + 1] 
        
        return img2tensor(enhanced_resized).to(input_image.device)


# if __name__ =="__main__":
#     # Process single frame
#     enhancer = VideoEnhancer(mask_path='/home/pbhat1/projects/LatentSync/latentsync/utils/mask.png')
#     input_image = cv2.imread('/home/pbhat1/projects/LatentSync/demo.png')  
#     enhanced_image = enhancer.enhance_image(input_image)
#     cv2.imwrite('enhanced_image.png', enhanced_image)
