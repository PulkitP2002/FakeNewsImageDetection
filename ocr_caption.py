from PIL import Image
import easyocr
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

reader = easyocr.Reader(['en'])
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def extract_text_from_image(image):
    image_np = np.array(image)
    result = reader.readtext(image_np)
    extracted_text = ' '.join([text[1] for text in result])
    return extracted_text

def generate_caption(image: Image.Image) -> str:
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
