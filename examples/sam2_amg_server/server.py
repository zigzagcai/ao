import itertools
import uvicorn
import fire
import tempfile
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch._dynamo.config
import torch._inductor.config
from fastapi.responses import Response
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import shutil
from pydantic import BaseModel
import cv2

import matplotlib.pyplot as plt
import numpy as np

# from torch._inductor import config as inductorconfig
# inductorconfig.triton.unique_kernel_names = True
# inductorconfig.coordinate_descent_tuning = True
# inductorconfig.coordinate_descent_check_all_directions = True

# torch.set_float32_matmul_precision('high')


def iou(mask1, mask2):
    assert mask1.dim() == 2
    assert mask2.dim() == 2
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return (intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2)))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    ms = []
    for ann in sorted_anns:
        m = ann['segmentation']
        ms.append(torch.as_tensor(m))
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return torch.stack(ms)


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def main(checkpoint_path,
         baseline=False,
         fast=False,
         furious=False,
         unittest=False,
         benchmark=False,
         profile=None,
         verbose=False,
         points_per_batch=64,
         port=5000,
         host="127.0.0.1",
         dry=False):
    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"Running with fast set to {fast} and furious set to {furious}")
    logging.info(f"Running with port {port} and host {host}")

    if baseline:
        logging.info(f"Importing sam2 from outside of torchao. If this errors, install https://github.com/facebookresearch/sam2")
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.utils.amg import rle_to_mask
    else:
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from torchao._models.sam2.utils.amg import rle_to_mask
    
    device = "cuda"
    from pathlib import Path
    sam2_checkpoint = Path(checkpoint_path) / Path("sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    logging.info(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    logging.info(f"Using {points_per_batch} points_per_batch")
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle")

    if furious:
        torch.set_float32_matmul_precision('high')
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    if fast:
        # TODO: Using CUDA graphs can cause numerical differences?
        mask_generator.predictor.model.image_encoder = torch.compile(
            mask_generator.predictor.model.image_encoder,
            # mode="max-autotune-no-cudagraphs",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        # torch._dynamo.config.capture_dynamic_output_shape_ops = True
        # mask_generator._process_batch = torch.compile(
        #     mask_generator._process_batch,
        #     # mode="max-autotune-no-cudagraphs",
        #     # fullgraph=True,
        #     dynamic=True,
        # )
        mask_generator.predictor._predict = torch.compile(
            mask_generator.predictor._predict,
            # mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )

    example_image = cv2.imread('dog.jpg')
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    with torch.backends.cuda.sdp_kernel(enable_cudnn=True):
        t = time.time()
        logging.info(f"Running one iteration to compile.")
        masks = mask_generator.generate(example_image)
        logging.info(f"First iteration took {time.time() - t}s.")
        if unittest:
            logging.info(f"Running strict comparison to reference mask")
            import json
            ref_masks = json.loads(open("dog_rle.json").read())
            ret_data = {}
            for mask_id in range(len(masks)):
                ret_data[f"mask_{mask_id}"] = masks[mask_id]["segmentation"]
            v0_areas = []
            v1_areas = []
            miou_sum = 0.0
            miou_count = 0
            for k0 in ref_masks:
                assert k0 in ret_data, f"Expected {k0} to be in return data"
                from torchao._models.sam2.utils.amg import area_from_rle
                v0_area = area_from_rle(ref_masks[k0])
                v1_area = area_from_rle(ret_data[k0])
                v0_areas.append(v0_area)
                v1_areas.append(v1_area)
                if v0_area != v1_area:
                    print(f"v0 area {v0_area} doesn't match v1 area {v1_area}")
                v0_mask = torch.from_numpy(rle_to_mask(ref_masks[k0]))
                v1_mask = torch.from_numpy(rle_to_mask(ret_data[k0]))
                if not torch.allclose(v0_mask, v1_mask):
                    miou_sum += iou(v0_mask, v1_mask)
                    miou_count += 1
                    print(f"Masks don't match for key {k0}. IoU is {iou(v0_mask, v1_mask)}")
            if miou_count == 0:
                print("Masks exactly match reference.")
            else:
                print(f"mIoU is {miou_sum / miou_count}")

        if benchmark:
            logging.info(f"Running 3 warumup iterations.")
            for _ in range(3):
                masks = mask_generator.generate(example_image)
            logging.info(f"Running 10 benchmark iterations, then exit.")
            t = time.time()
            for _ in range(10):
                masks = mask_generator.generate(example_image)
            print(f"Benchmark took {(time.time() - t)/10.0}s per iteration.")
            max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
            _, total_memory = torch.cuda.mem_get_info()
            max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
            max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
            print(f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%")

        if profile is not None:
            print(f"Saving profile under {profile}")
            profiler_runner(profile, mask_generator.generate, example_image)

    if dry:
        return

    app = FastAPI()

    # Allow all origins (you can restrict it in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/upload_rle")
    async def upload_rle(image: UploadFile = File(...)):
        # Save the uploaded image to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}")
        with open(temp_file.name, "wb") as b:
            shutil.copyfileobj(image.file, b)

        # Read the image back into memory to send as response
        example_image = cv2.imread(temp_file.name)
        example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
        t = time.time()
        with torch.backends.cuda.sdp_kernel(enable_cudnn=True):
            masks = mask_generator.generate(example_image)
        print(f"Took {time.time() - t} to generate a mask for input image.")
        ret_data = {}
        for mask_id in range(len(masks)):
            ret_data[f"mask_{mask_id}"] = masks[mask_id]["segmentation"]
        return ret_data
    
    @app.post("/upload")
    async def upload_image(image: UploadFile = File(...)):
        # Save the uploaded image to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}")
        with open(temp_file.name, "wb") as b:
            shutil.copyfileobj(image.file, b)
    
        # Read the image back into memory to send as response
        example_image = cv2.imread(temp_file.name)
        example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
        t = time.time()
        with torch.backends.cuda.sdp_kernel(enable_cudnn=True):
            masks = mask_generator.generate(example_image)
        print(f"Took {time.time() - t} to generate a mask for input image.")
        # Save an example
        plt.figure(figsize=(example_image.shape[1]/100., example_image.shape[0]/100.), dpi=100)
        plt.imshow(example_image)
        for i in range(len(masks)):
            masks[i]["segmentation"] = rle_to_mask(masks[i]["segmentation"])
        show_anns(masks)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(temp_file.name, format='png')

        # Read the image back into memory to send as response
        with open(temp_file.name, "rb") as f:
            image_data = f.read()
    
        # Return the image as a StreamingResponse
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    fire.Fire(main)
