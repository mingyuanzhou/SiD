import torch
import diffusers

from diffusers import AutoencoderKL

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    XFormersAttnProcessor,
    LoRAXFormersAttnProcessor,
    LoRAAttnProcessor2_0,
    FusedAttnProcessor2_0,
)

def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
            FusedAttnProcessor2_0,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)


def load_sd_vae(pretrained_vae_model_name_or_path, device): #, weight_dtype):
    # Load the tokenizer
    print(f'pretrained_model_name_or_path: {pretrained_vae_model_name_or_path}')

    #print('tokenizer start')

    vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path)

    # Freeze untrained components
    vae.eval().requires_grad_(False).to(device)
                  
    return vae

    
def vae_sampler_edm2(vae, scale, bias, unet, latents, c, init_sigma_tensor, dtype=torch.float16, train_sampler=True, num_steps=1):
    z = latents
    G = unet

    # Forward pass
    if train_sampler:
        D_x = G(z, init_sigma_tensor, c)
    else:
        with torch.no_grad():
            D_x = G(z, init_sigma_tensor, c)

    # Check if VAE needs upcasting
    needs_upcasting = (getattr(vae, 'dtype', None) == torch.float16) and getattr(vae.config, 'force_upcast', False)
    if needs_upcasting:
        upcast_vae(vae=vae)
        D_x = D_x.to(next(iter(vae.post_quant_conv.parameters())).dtype)

    # Decode images
    images = vae_edm2_decode(vae, scale, bias, D_x).to(torch.float32)
    
    return images


