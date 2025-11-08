
import abc
import math
from typing import List
import torch
from diffusers.models.attention_processor import Attention
import numpy as np
import torch
from PIL import Image
import cv2
import os


def concat_images(image_paths):
    images = [Image.open(path) for path in image_paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    concatenated_image = Image.new('RGB', (total_width, max_height))
    current_x = 0
    for img in images:
        concatenated_image.paste(img, (current_x, 0))
        current_x += img.width
    return concatenated_image


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def load_512(image_path):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image



def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latents



class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, heads: int, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, heads, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        # if is_cross:
        #     attn = self.forward(attn, heads, is_cross, place_in_unet)
        # else:
        #     attn[h // 2 :] = self.forward(attn[h // 2 :], heads, is_cross, place_in_unet)
        attn = self.forward(attn, heads, is_cross, place_in_unet)
        # print(self.cur_att_layer, self.num_att_layers,"====")
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
            self.cur_step += 1
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStoreClassPrompts(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, heads, is_cross: bool, place_in_unet: str):
        if self.start <= self.cur_step <= self.end:
            if attn.shape[1] <= 64**2:  # avoid memory overhead
                spatial_res = int(math.sqrt(attn.shape[1]))
                attn_store = attn.reshape(-1, heads, spatial_res, spatial_res, attn.shape[-1])
                
                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
                self.step_store[key].append(attn_store)
        return attn

    def between_steps(self):
        if self.start <= self.cur_step <= self.end:
            if self.attention_store is None:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        start = max(0, self.start)
        end = min(self.cur_step, self.end + 1)
        average_attention = {
            key: [item / (end - start) for item in self.attention_store[key]] for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStoreClassPrompts, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = None


    def __init__(self, start=0, end=1000):
        super(AttentionStoreClassPrompts, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = None
        self.start = start
        self.end = end




class StoredAttnClassPromptsProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0,):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        batch_size, sequence_length, _ = hidden_states.shape


        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        is_cross = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        flag_store = True
        if is_cross:
            cross_key = key[attn.heads * batch_size :,]  # get token of the class text
            # print(key.shape, cross_key.shape, attn.heads, batch_size)
            key = key[: attn.heads * batch_size]
            cross_query = query
            value = value[: attn.heads * batch_size]
            
            # print(key.shape,cross_key.shape)
            if cross_key.shape[0]==0:
                flag_store = False
            else:
                cross_attention_probs = attn.get_attention_scores(cross_query, cross_key, None)
                self.attnstore(cross_attention_probs, attn.heads, True, self.place_in_unet)

        
        else:
            if flag_store:
                attention_probs_self = attn.get_attention_scores(query, key, attention_mask)
                self.attnstore(attention_probs_self, attn.heads, False, self.place_in_unet)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states




def aggregate_attention(
    attention_store: AttentionStoreClassPrompts,
    res: int,
    is_cross: bool,
    from_where: List[str] = ["up", "down", "mid"],
    **kwargs,
):
    out = []
    attention_maps = attention_store.get_average_attention(**kwargs)
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[2] == res:
                out.append(item)
    out = torch.cat(out, dim=1).mean(dim=1)
    return out

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image



def register_attention_control(model, controller, processor, **kwargs):
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = processor(attnstore=controller, place_in_unet=place_in_unet, **kwargs)

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count






