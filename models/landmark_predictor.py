import sys
import torch
import torchvision
import numpy as np

import openai
import IPython

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
from torch.nn import functional as F
from utils.logger import Logger
from config.landmark_cfg import LandmarkConfig
from .landmark_model import ResNet
from .landmark_model import ResidualBlock
from .landmark_api import get_completion
from .landmark_api import set_open_params
from .landmark_api import print_response

LOGGER = Logger(__file__, log_file="predictor.log")
LOGGER.log.info("Starting Model Serving")

class Predictor:
    def __init__(self, model_name: str, model_weight: str, api_key: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.model_weight = model_weight
        self.api_key = api_key
        self.device = device
        self.create_transform()

    async def predict(self, image):
        pil_img = Image.open(image)

        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        pil_img = np.array(pil_img)[..., :3] # conver to numpy array
        pil_img = torch.tensor(pil_img).permute(2, 0, 1).float() # Conver (H, W, C) to (C, H, W)
        normalized_img = pil_img / 255.0
        transformed_image = torch.unsqueeze(normalized_img, 0) # Thêm số chiều   

        #### gọi model
        model = ResNet(ResidualBlock, LandmarkConfig.N_BLOCKS_LST, LandmarkConfig.N_CLASSES)
        model.load_state_dict(torch.load(self.model_weight, map_location=self.device))
        model.to(self.device)
        model.eval()

        # input, output
        input = transformed_image.to(self.device)
        output = model(input.to(self.device)).cpu()
        
        best_prob, predicted_id, predicted_class, lable_name = self.output2pred(output)

        try: 
            ################# call api
            with open(self.api_key, 'r') as f:
                openai.api_key = f.readline()

            # basic parameters
            params = set_open_params()

            prompt = f"""Giới thiệu địa danh Việt Nam cho khách du lịch:\nYêu cầu: Xin hãy cung cấp một mô tả về lịch sử của {lable_name}"""

            response = get_completion(params, prompt)
            landmark_description = print_response(response)
        except:
            landmark_description = ""

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)

        torch.cuda.empty_cache()

        resp_dict = {
            "best_prob": best_prob,
            "predicted_id": predicted_id, 
            "predicted_class": predicted_class, 
            "lable_name": lable_name,
            "landmark_description": landmark_description
        }

        return resp_dict
    
    def create_transform(self):
        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((LandmarkConfig.IMG_SIZE, LandmarkConfig.IMG_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=LandmarkConfig.NORMALIZE_MEAN, std=LandmarkConfig.NORMALIZE_STD)
        ])

    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1) 
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = LandmarkConfig.ID_TO_LANDMARK_ID[predicted_id]
        lable_name = LandmarkConfig.LANDMARK_ID_TO_LABLE[int(predicted_class)]

        return round(best_prob, 6), predicted_id, predicted_class, lable_name
