import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers import DDIMScheduler
from SD import SD_FASeg
from ptp_utils import concat_images
from FASeg import FASeg


def process_classprompt(class_prompt):
    class_prompt =  class_prompt.replace("tv monitor","tvmonitor")\
                                .replace("potted plant","pottedplant")
    class_prompt =  class_prompt.replace(' ',', ')
    class_prompt =  class_prompt.replace("tvmonitor","tv monitor")\
                                .replace("pottedplant","potted plant")
    return class_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Diffusion")
    parser.add_argument("--work-dir", default='results/context', help="the dir to save the synthetic dataset")
    parser.add_argument("--save-visualize", action="store_true")
    parser.add_argument("--threshold", default=0.5, type=float, help="background threshold")
    parser.add_argument("--sd-path", default="XCLIU/2_rectified_flow_from_sd_1_5")
    args = parser.parse_args()
    return args





def main(args):
    os.makedirs(f"{args.work_dir}/concats", exist_ok=True)
    os.makedirs(f"{args.work_dir}/masks", exist_ok=True)

    with open('dataset/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt','r') as f:
        names = [line.strip() for line in f if line.strip()]
    with open('json/candidate_classes_context.json') as f:
        class_prompts = json.load(f)
    with open("json/caption_context_val_blip1.json") as f:
        prompt_with_blip = json.load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    sd = SD_FASeg.from_pretrained(sd_path,scheduler=scheduler, torch_dtype=torch.float16)
    sd.to("cuda")
    sd.enable_attention_slicing()

    gen_mask = FASeg(sd,args)



    for name in tqdm(names):
        if name in class_prompts and class_prompts[name] != "":
            path = os.path.join('dataset/VOCdevkit/VOC2010/JPEGImages',f"{name}.jpg")

            class_prompt = class_prompts[name]
            class_prompt = process_classprompt(class_prompt)
            prompt = f"{prompt_with_blip[name]} ; {class_prompt}"

            mask = gen_mask.run(path, prompt, class_prompt,name, "context")
            mask.save(os.path.join(f"{args.work_dir}/masks",f"{name}.png"))

            if args.save_visualize:
                label_path = os.path.join('dataset/VOCdevkit/VOC2010/SegmentationClassContextColor',f"{name}.png")
                list_path = [path, label_path,os.path.join(f"{args.work_dir}/masks",name+".png")]
                concat_img = concat_images(list_path)
                concat_img.save(os.path.join(f"{args.work_dir}/concats",name+".png"))
        else:
            path = os.path.join('dataset/VOCdevkit/VOC2010/JPEGImages',f"{name}.jpg")
            
            img = Image.open(path)
            width, height = img.size
            mask = Image.new(mode="L", size=(width, height), color=0)
            mask.save(os.path.join(f"{args.work_dir}/masks",f"{name}.png"))

            if args.save_visualize:
                label_path = os.path.join('dataset/VOCdevkit/VOC2010/SegmentationClassContextColor',f"{name}.png")
                list_path = [path, label_path,os.path.join(f"{args.work_dir}/masks",f"{name}.png")]
                concat_img = concat_images(list_path)
                concat_img.save(os.path.join(f"{args.work_dir}/concats",f"{name}.png"))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    torch.cuda.empty_cache()