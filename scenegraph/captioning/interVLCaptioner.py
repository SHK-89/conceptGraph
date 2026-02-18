import numpy as np
import torch
import torchvision.transforms as T
import transformers.utils.logging
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# -------------------------------------------------------------------
# Basic image transform (InternVL expected format)
# -------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, min_num=4, max_num=4, image_size=448):
    """
    Dynamic tiling as used by InterVL.
    Produces multiple tiles if aspect ratio is wide/high.
    """
    orig_w, orig_h = image.size
    target_w = image_size * 2
    target_h = image_size * 2
    image = image.resize((target_w, target_h))

    tiles = []
    for i in range(2):
        for j in range(2):
            box = (j * image_size, i * image_size,
                   (j + 1) * image_size, (i + 1) * image_size)
            tiles.append(image.crop(box))

    return tiles


# -------------------------------------------------------------------
# InternVL Captioner
# -------------------------------------------------------------------
class InterVLCaptioner:
    """
    Captioner wrapper for InternVL3.5-8B using official .chat() API.

    Supports:
        - Object captioning from RGB crops
        - Pairwise relationship reasoning
    """

    def __init__(self, model_name="OpenGVLab/InternVL3_5-8B", device=None, input_size=448, quiet_mode=False):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading InternVL3.5-8B on device={self.device}")
        self.quiet_mode = quiet_mode

        if quiet_mode:
            transformers.logging.set_verbosity_error()

        #TODO: SHK check if this is needed. It seems that the tokenizer doesn't have a pad token, which causes issues with generation. Setting it to eos token seems to work fine.
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )

        # model
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            # load_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().to(self.device)

        self.transform = build_transform(input_size)
        self.input_size = input_size
        self.generation_config = dict(max_new_tokens=50, do_sample=False)

    # ----------------------------------------------------------------------
    # Preprocess a single crop
    # ----------------------------------------------------------------------
    def preprocess_crop(self, rgb_crop):
        if isinstance(rgb_crop, np.ndarray):
            image = Image.fromarray(rgb_crop.astype(np.uint8))
        else:
            image = rgb_crop

        tiles = dynamic_preprocess(image, image_size=self.input_size)
        pixel_values = [self.transform(t) for t in tiles]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
        return pixel_values

    # ---------------------------------------------------------------
    # Caption a single object crop
    # ---------------------------------------------------------------
    def caption_object_OLD(self, rgb_crop):
        print("Captioning object crop...")
        pv = self.preprocess_crop(rgb_crop).to(self.device).to(torch.bfloat16)

        question = "<image>\nDescribe this object briefly."
        response = self.model.chat(
            self.tokenizer,
            pv,
            question,
            self.generation_config
        )

        return response

    def caption_object(self, rgb_crop):
        print("Captioning object crop...")
        pv = self.preprocess_crop(rgb_crop).to(self.device).to(torch.bfloat16)

        question = "<image>\ngive me the category of the object in this image."
        response = self.model.chat(
            self.tokenizer,
            pv,
            question,
            self.generation_config
        )

        return response

    # ---------------------------------------------------------------
    # Relationship reasoning between two object descriptions
    # ---------------------------------------------------------------
    def relationship(self, descA, descB, geom_hint=None):

        query = f"Object A: {descA}\nObject B: {descB}\n"
        if geom_hint:
            query += f"Spatial hint: {geom_hint}\n"
        query += "Describe the relationship between A and B in one short phrase."

        gen_cfg = dict(max_new_tokens=50, do_sample=False)

        # NOTE: No image needed for relationship reasoning
        response = self.model.chat(
            self.tokenizer,
            None,
            query,
            gen_cfg
        )

        return response.strip()


    # ----------------------------------------------------------------------------
    # relationship reasoning between two image crops
    # ----------------------------------------------------------------------------
    def relationship_between_crops(self, cropA, cropB):

        pvA = self.preprocess_crop(cropA).to(self.device).to(torch.bfloat16)
        pvB = self.preprocess_crop(cropB).to(self.device).to(torch.bfloat16)

        pixel_values = torch.cat((pvA, pvB), dim=0)
        num_patches_list = [pvA.size(0), pvB.size(0)]

        question = (
            "Image A: <image>\n"
            "Image B: <image>\n"
            "Describe the relationship between A and B in one short phrase."
        )

        result = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list
        )
        return result

    def relation_from_paircrop(self, crop_image, centerA, centerB):
        """
        Ask InterVL to infer the relationship between two objects
        inside a *combined* crop containing both of them.
        """
        img = Image.fromarray(crop_image.astype('uint8'))
        pv = self.preprocess_crop(img).to(self.device).to(torch.bfloat16)

        question = f"""
    <image>

    Two objects are present in this image.

    Object A is located at pixel coordinates {centerA}. give the category of the object at A with 1-2 words.
    Object B is located at pixel coordinates {centerB}. give the category of the object at B with 1-2 words.
    what is the plausible and meaningful relation  between them? Answer with a single letter.For instance < object A, hugging, object B >
    IMPORTANT:
    - Do NOT mention "Object A" or "Object B".
    - Do NOT output a full sentence.
    """
        # result in folder 3words:     Give the meaningful relationship between A and B using one short phrase (max 3 words)..
        # short phrase: Give the meaningful relationship between A and B in one short phrase.
        # single word: what is the plausible relation  between them? Answer with a single letter.For instance < adults, hugging, child >
        # question = f"""
        #   <image>
        #
        #   Two objects are present in this image.
        #
        #   Object A is located at pixel coordinates {centerA}.
        #   Object B is located at pixel coordinates {centerB}.
        #
        #   Give the exact meaningful relationship between A and B in one word. for instance
        #   holding, grabbing, sitting on and ignore spatial relation. If there is no meaningful relationship, answer "none".
        #   """

        response = self.model.chat(
            self.tokenizer,
            pv,
            question,
            self.generation_config
        )

        return response
