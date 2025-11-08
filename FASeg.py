from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, ControlNetModel
import torch.nn.functional as nnf
import numpy as np
import cv2
import math
from tqdm import tqdm
import argparse
import json
import os

from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ptp_utils import init_latent, register_attention_control, AttentionStoreClassPrompts, StoredAttnClassPromptsProcessor, aggregate_attention




class FASeg:
    """
    FA-Seg: A Fast and Accurate Diffusion-Based Method for Open-Vocabulary Segmentation

    This class performs segmentation mask generation using Stable Diffusion models.
    It relies on both cross-attention and self-attention maps to infer fine-grained
    semantic masks from diffusion features.
    """


    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """
        Compute the next latent sample using the DDIM update rule.
        """

        timestep, next_timestep = min(timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    

    @torch.no_grad()
    def get_noise_pred_single(self, latents, t, context):
        """
        Predict noise for a single diffusion step using the UNet.
        """
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=torch.cat((context, context), dim=0))["sample"]
        return noise_pred



    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        """
        Decode latent tensor into an image using the VAE decoder.
        """
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image_path):
        """
        Encode an RGB image into the latent space of the diffusion model.
        Also generates a horizontally flipped version for test-time flipped.
        """
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((self.width, self.height))

        flipped_image = init_image.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.image_processor.preprocess([init_image,flipped_image]).to(dtype=self.model.unet.dtype)
        
        image = image.to(device=self.model.device, dtype=self.model.unet.dtype)
        init_latents = self.model.vae.encode(image).latent_dist.sample(torch.manual_seed(0))
        init_latents = self.model.vae.config.scaling_factor * init_latents
        latent = torch.cat([init_latents], dim=0)

        return latent, init_image

    @torch.no_grad()
    def init_prompt(self, prompt, class_prompt):
        """
        Encode text prompts into embeddings used for conditioning the diffusion model.
        """ 
        context = self.model._encode_prompt(prompt,
                class_prompt,
                device = self.model.device,
                num_images_per_prompt=1)
        self.cond_embeddings, self.class_prompts_embeddings = context.chunk(2)



    def subscribe_attention(self, store = False):
        """
        Register or reset attention processors.

        Args:
            store (bool): If True, attach custom attention storage module
                          to record attention maps during generation.
        """
        if store:
            controller = AttentionStoreClassPrompts(start=0, end=1)
            register_attention_control(self.model, controller, StoredAttnClassPromptsProcessor)
            return controller
        else:
            from diffusers.models.attention_processor import AttnProcessor
            attn_procs = {}
            for name in self.model.unet.attn_processors.keys():
                attn_procs[name] = AttnProcessor()
            self.model.unet.set_attn_processor(attn_procs)
            return None
        
    @torch.no_grad()
    def ddim(self, latent):
        """
        Perform a single DDIM forward step on the latent tensor.
        """
        latent = latent.clone().detach()
        t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - 1]
        noise_pred = self.get_noise_pred_single(latent, t, self.cond_embeddings)
        latent = self.next_step(noise_pred, t, latent)
        return latent


    @torch.no_grad()
    def ddim_inversion(self, image_path):
        """
        Apply DDIM inversion to an input image to retrieve its latent representation.
        """
        latent, gt_image = self.image2latent(image_path)
        ddim_latent = self.ddim(latent)
        return ddim_latent



    @torch.no_grad()
    def reconstruct_image(self,x_t,prompt):
        """
        Reconstruct image from a latent tensor for attention recording.

        This method triggers UNet forward passes and allows collecting attention maps.
        """
        latents = init_latent(x_t, self.model, self.height, self.width, batch_size=2)

        t = self.model.scheduler.timesteps[-self.num_steps:][0]
        with torch.no_grad():
            self.cond_embeddings, self.class_prompts_embeddings = \
                            torch.cat([self.cond_embeddings,self.cond_embeddings]),torch.cat([self.class_prompts_embeddings,self.class_prompts_embeddings])
            context = torch.cat([self.cond_embeddings, self.class_prompts_embeddings])
            latents_input = latents
            noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        #     latents = self.model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        #     image = self.latent2image(latents)
        # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def cross_attn_repare(self,controller,cross_weights):
        """
        Aggregate multi-resolution cross-attention maps into one weighted map.
        """
        cross_attention_8  = aggregate_attention(controller, res=8, is_cross=True).float()
        cross_attention_16 = aggregate_attention(controller, res=16, is_cross=True).float()
        cross_attention_32 = aggregate_attention(controller, res=32, is_cross=True).float()
        cross_attention_64 = aggregate_attention(controller, res=64, is_cross=True).float()


        # Upsample all resolutions to match target resolution
        cross_attention_8 = F.interpolate(cross_attention_8.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="nearest")
        cross_attention_16 = F.interpolate(cross_attention_16.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="nearest")
        cross_attention_32 = F.interpolate(cross_attention_32.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="nearest")
        cross_attention_64 = F.interpolate(cross_attention_64.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="nearest")

        # Combine using predefined weights
        cross_attention = (
            cross_attention_8 * cross_weights[0]
            + cross_attention_16 * cross_weights[1]
            + cross_attention_32 * cross_weights[2]
            + cross_attention_64 * cross_weights[3]
        )
        return cross_attention

    def self_attn_repare(self,controller):
        """
        Process multi-resolution self-attention maps and construct
        affinity matrices representing pixel-level relationships.

        Returns:
            Two lists of affinity matrices (for normal and flipped images).
        """
        self_attention_8  = aggregate_attention(controller, res=8, is_cross=False).float()       
        self_attention_8  = F.interpolate(self_attention_8.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="bicubic")
        self_attention_8  = self_attention_8.reshape(2, 8 , 8 , self.self_res, self.self_res)
        self_attention_8  = torch.repeat_interleave(self_attention_8 , repeats=self.self_res//8,dim=1)
        self_attention_8  = torch.repeat_interleave(self_attention_8 , repeats=self.self_res//8,dim=2)
        self_attention_8  = self_attention_8.reshape(2,self.self_res**2, self.self_res, self.self_res).permute(0, 2, 3, 1)

        self_attention_16 = aggregate_attention(controller, res=16, is_cross=False).float()
        self_attention_16 = F.interpolate(self_attention_16.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="bicubic")
        self_attention_16 = self_attention_16.reshape(2, 16, 16, self.self_res, self.self_res)
        self_attention_16 = torch.repeat_interleave(self_attention_16, repeats=self.self_res//16,dim=1)
        self_attention_16 = torch.repeat_interleave(self_attention_16, repeats=self.self_res//16,dim=2)
        self_attention_16 = self_attention_16.reshape(2,self.self_res**2, self.self_res, self.self_res).permute(0, 2, 3, 1)

        self_attention_32 = aggregate_attention(controller, res=32, is_cross=False).float()
        self_attention_32 = F.interpolate(self_attention_32.permute(0, 3, 1, 2), (self.self_res, self.self_res), mode="bicubic")
        self_attention_32 = self_attention_32.reshape(2, 32, 32, self.self_res, self.self_res)
        self_attention_32 = torch.repeat_interleave(self_attention_32, repeats=self.self_res//32,dim=1)
        self_attention_32 = torch.repeat_interleave(self_attention_32, repeats=self.self_res//32,dim=2)
        self_attention_32 = self_attention_32.reshape(2,self.self_res**2, self.self_res, self.self_res).permute(0, 2, 3, 1)

        self_attention_64 = aggregate_attention(controller, res=self.self_res, is_cross=False).float()

        tau = 2

        affinity_mat_8  = torch.matrix_power(self_attention_8[0].reshape(self.self_res**2, self.self_res**2), tau)
        affinity_mat_16 = torch.matrix_power(self_attention_16[0].reshape(self.self_res**2, self.self_res**2), tau)
        affinity_mat_32 = torch.matrix_power(self_attention_32[0].reshape(self.self_res**2, self.self_res**2), tau)
        affinity_mat_64 = torch.matrix_power(self_attention_64[0].reshape(self.self_res**2, self.self_res**2), tau)

        affinity_mat_8_flip  = torch.matrix_power(self_attention_8[1].reshape(self.self_res**2, self.self_res**2), tau)
        affinity_mat_16_flip = torch.matrix_power(self_attention_16[1].reshape(self.self_res**2, self.self_res**2), tau)
        affinity_mat_32_flip = torch.matrix_power(self_attention_32[1].reshape(self.self_res**2, self.self_res**2), tau)
        affinity_mat_64_flip = torch.matrix_power(self_attention_64[1].reshape(self.self_res**2, self.self_res**2), tau)

        return [[affinity_mat_8,affinity_mat_16,affinity_mat_32,affinity_mat_64],
                [affinity_mat_8_flip,affinity_mat_16_flip,affinity_mat_32_flip,affinity_mat_64_flip]]

    def self_fusion(self, affinity_mat, cross_attn, self_weights):
        """
        Fuse cross- and self-attention to compute per-class segmentation maps.
        """
        affinity_mat_8,affinity_mat_16,affinity_mat_32,affinity_mat_64 = affinity_mat
        
        out_8  = (affinity_mat_8 @ cross_attn.reshape(self.self_res**2, 1)).reshape(self.self_res, self.self_res)
        out_8  = (out_8 - out_8.min()) / (out_8.max() - out_8.min())
        out_16 = (affinity_mat_16 @ cross_attn.reshape(self.self_res**2, 1)).reshape(self.self_res, self.self_res)
        out_16 = (out_16 - out_16.min()) / (out_16.max() - out_16.min())
        out_32 = (affinity_mat_32 @ cross_attn.reshape(self.self_res**2, 1)).reshape(self.self_res, self.self_res)
        out_32 = (out_32 - out_32.min()) / (out_32.max() - out_32.min())
        out_64 = (affinity_mat_64 @ cross_attn.reshape(self.self_res**2, 1)).reshape(self.self_res, self.self_res)
        out_64 = (out_64 - out_64.min()) / (out_64.max() - out_64.min())

        out = out_8 * self_weights[0] + out_16 * self_weights[1] + out_32 * self_weights[2] + out_64 * self_weights[3]

        return out



    def attn2mask(self, prompt, class_prompt, name, controller, dataset):
        """
        Convert stored attention maps into a semantic segmentation mask.
        """
        if dataset == "voc":
            from data.voc import classes, palette, get_indices
        if dataset == "coco":
            from data.coco import classes, palette, get_indices
        if dataset == "context":
            from data.context import classes, palette, get_indices

        indices, labels = get_indices(self.model.tokenizer, classes, prompt, class_prompt)
        batch_indices = indices[0]
        batch_labels = labels[0]

        seg_maps = []
        affinity_mats = self.self_attn_repare(controller)
        for i in range(2):
            affinity_mat = affinity_mats[i]
            cross_attention = self.cross_attn_repare(controller,self.cross_weights)
            outs = []
            for j, index in enumerate(batch_indices):
                if isinstance(index, list):
                    index = [jj + 1 for jj in index]
                    cross_attn = cross_attention[i][index].mean(dim=0)
                elif isinstance(index, int):
                    cross_attn = cross_attention[i][index + 1]
                out = self.self_fusion(affinity_mat, cross_attn, self.self_weights)
                outs.append(out)

            outs = torch.stack(outs)
            outs = F.interpolate(outs.unsqueeze(0), (self.height, self.width), mode="bicubic")[0].cpu().numpy()
            seg_maps.append(outs)

        seg_maps = (seg_maps[0]+seg_maps[1][:,:,::-1])/2

        outs_max = seg_maps.max(axis=0)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        valid = outs_max >= self.mask_threshold
        mask[valid] = (seg_maps.argmax(axis=0) + 1)[valid]
        label = np.array(batch_labels, dtype=np.uint8)
        mask = label[mask]
        mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
        mask.putpalette(palette)

        return mask





    def run(self, image_path, prompt, class_prompt,name, dataset):

        image = Image.open(image_path)
        self.ori_width, self.ori_height = image.size
        self.init_prompt(prompt, class_prompt)
        ddim_latent = self.ddim_inversion(image_path)
        controller = self.subscribe_attention(store = True)
        self.reconstruct_image(ddim_latent,prompt)
        mask = self.attn2mask(prompt, class_prompt,name, controller, dataset)
        controller.reset()
        self.subscribe_attention()
        return mask.resize((self.ori_width, self.ori_height))

    def __init__(self, model, args):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_steps = 1
        self.model = model
        self.model.scheduler.set_timesteps(self.num_steps)
        self.image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(self.model.vae.config.block_out_channels) - 1), do_convert_rgb=True)

        self.cross_weights = [0.15, 0.7, 0.15,0]
        self.self_weights = [0.1,0.1, 0.5, 0.3]
        self.height = self.width = 512
        self.guidance_scale = 7.5
        self.self_res = 64
        self.cross_res = 64
        self.mask_threshold = args.threshold




