import math

import torch
from torch import Tensor

try:
  from comfy.ldm.lightricks.model import LTXBaseModel
except:
    from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from ..patch_util import PatchKeys


def ltx_forward_orig(
    self,
    x,
    timestep,
    context,
    attention_mask,
    frame_rate=25,
    guiding_latent=None,
    transformer_options={},
    keyframe_idxs=None,
    **kwargs
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    patches_point = transformer_options.get(PatchKeys.options_key, {})

    transformer_options[PatchKeys.running_net_model] = self

    patches_enter = patches_point.get(PatchKeys.dit_enter, [])
    if patches_enter is not None and len(patches_enter) > 0:
        for patch_enter in patches_enter:
            x, timestep, context, attention_mask, frame_rate, guiding_latent, keyframe_idxs = patch_enter(
                x,
                timestep,
                context,
                attention_mask,
                frame_rate,
                guiding_latent,
                keyframe_idxs,
                transformer_options
            )

    orig_shape = list(x.shape)

    x, latent_coords = self.patchifier.patchify(x)
    pixel_coords = latent_to_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=self.vae_scale_factors,
        causal_fix=self.causal_temporal_positioning,
    )

    if keyframe_idxs is not None:
        pixel_coords[:, :, -keyframe_idxs.shape[2]:] = keyframe_idxs

    fractional_coords = pixel_coords.to(torch.float32)
    fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

    x = self.patchify_proj(x)
    timestep = timestep * 1000.0

    if attention_mask is not None and not torch.is_floating_point(attention_mask):
        attention_mask = (attention_mask - 1).to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(x.dtype).max

    try:
        pe = LTXBaseModel.precompute_freqs_cis(fractional_coords, dim=self.inner_dim, out_dtype=x.dtype)
    except:
        pe = precompute_freqs_cis(fractional_coords, dim=self.inner_dim, out_dtype=x.dtype)

    batch_size = x.shape[0]
    timestep, embedded_timestep = self.adaln_single(
        timestep.flatten(),
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=x.dtype,
    )
    # Second dimension is 1 or number of tokens (if timestep_per_token)
    timestep = timestep.view(batch_size, -1, timestep.shape[-1])
    embedded_timestep = embedded_timestep.view(
        batch_size, -1, embedded_timestep.shape[-1]
    )

    # 2. Blocks
    if self.caption_projection is not None:
        batch_size = x.shape[0]
        context = self.caption_projection(context)
        context = context.view(
            batch_size, -1, x.shape[-1]
        )

    patch_blocks_before = patches_point.get(PatchKeys.dit_blocks_before, [])
    if patch_blocks_before is not None and len(patch_blocks_before) > 0:
        for blocks_before in patch_blocks_before:
            x, context, timestep, ids, pe = blocks_before(img=x, txt=context, vec=timestep, ids=None, pe=pe, transformer_options=transformer_options)

    def double_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={}):
        running_net_model = transformer_options[PatchKeys.running_net_model]
        patch_double_blocks_with_control_replace = patches_point.get(PatchKeys.dit_double_block_with_control_replace)
        for i, block in enumerate(running_net_model.transformer_blocks):
            if patch_double_blocks_with_control_replace is not None:
                img, txt = patch_double_blocks_with_control_replace({'i': i,
                                                                     'block': block,
                                                                     'img': img,
                                                                     'txt': txt,
                                                                     'vec': vec,
                                                                     'pe': pe,
                                                                     'control': control,
                                                                     'attn_mask': attn_mask
                                                                     },
                                                                    {
                                                                        "original_func": double_block_and_control_replace,
                                                                        "transformer_options": transformer_options
                                                                    })
            else:
                img, txt = double_block_and_control_replace(i=i,
                                                            block=block,
                                                            img=img,
                                                            txt=txt,
                                                            vec=vec,
                                                            pe=pe,
                                                            control=control,
                                                            attn_mask=attn_mask,
                                                            transformer_options=transformer_options
                                                            )

        del patch_double_blocks_with_control_replace
        return img, txt

    patch_double_blocks_replace = patches_point.get(PatchKeys.dit_double_blocks_replace)

    if patch_double_blocks_replace is not None:
        x, context = patch_double_blocks_replace({"img": x,
                                                  "txt": context,
                                                  "vec": timestep,
                                                  "pe": pe,
                                                  "control": None,
                                                  "attn_mask": attention_mask,
                                                  },
                                                 {
                                                     "original_blocks": double_blocks_wrap,
                                                     "transformer_options": transformer_options
                                                 })
    else:
        x, context = double_blocks_wrap(img=x,
                                        txt=context,
                                        vec=timestep,
                                        pe=pe,
                                        control=None,
                                        attn_mask=attention_mask,
                                        transformer_options=transformer_options
                                        )

    patches_double_blocks_after = patches_point.get(PatchKeys.dit_double_blocks_after, [])
    if patches_double_blocks_after is not None and len(patches_double_blocks_after) > 0:
        for patch_double_blocks_after in patches_double_blocks_after:
            x, context = patch_double_blocks_after(x, context, transformer_options)

    patch_blocks_transition = patches_point.get(PatchKeys.dit_blocks_transition_replace)

    def blocks_transition_wrap(**kwargs):
        x = kwargs["img"]
        return x

    if patch_blocks_transition is not None:
        x = patch_blocks_transition({"img": x, "txt": context, "vec": timestep, "pe": pe},
                                    {
                                        "original_func": blocks_transition_wrap,
                                        "transformer_options": transformer_options
                                    })
    else:
        x = blocks_transition_wrap(img=x, txt=context)

    patches_single_blocks_before = patches_point.get(PatchKeys.dit_single_blocks_before, [])
    if patches_single_blocks_before is not None and len(patches_single_blocks_before) > 0:
        for patch_single_blocks_before in patches_single_blocks_before:
            x, context = patch_single_blocks_before(x, context, transformer_options)

    def single_blocks_wrap(img, **kwargs):
        return img

    patch_single_blocks_replace = patches_point.get(PatchKeys.dit_single_blocks_replace)

    if patch_single_blocks_replace is not None:
        x, context = patch_single_blocks_replace({"img": x,
                                                  "txt": context,
                                                  "vec": timestep,
                                                  "pe": pe,
                                                  "control": None,
                                                  "attn_mask": attention_mask
                                                  },
                                                 {
                                                     "original_blocks": single_blocks_wrap,
                                                     "transformer_options": transformer_options
                                                 })
    else:
        x = single_blocks_wrap(img=x,
                               txt=context,
                               vec=timestep,
                               pe=pe,
                               control=None,
                               attn_mask=attention_mask,
                               transformer_options=transformer_options
                               )

    patch_blocks_exit = patches_point.get(PatchKeys.dit_blocks_after, [])
    if patch_blocks_exit is not None and len(patch_blocks_exit) > 0:
        for blocks_after in patch_blocks_exit:
            x, context = blocks_after(x, context, transformer_options)

    # 3. Output
    def final_transition_wrap(**kwargs):
        running_net_model = transformer_options[PatchKeys.running_net_model]
        x = kwargs["img"]
        embedded_timestep = kwargs["embedded_timestep"]
        scale_shift_values = (
                running_net_model.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = running_net_model.norm_out(x)
        # Modulation
        x = x * (1 + scale) + shift
        return x

    patch_blocks_after_transition_replace = patches_point.get(PatchKeys.dit_blocks_after_transition_replace)
    if patch_blocks_after_transition_replace is not None:
        x = patch_blocks_after_transition_replace({"img": x, "txt": context, "vec": timestep, "pe": pe, "embedded_timestep": embedded_timestep},
                                                    {
                                                        "original_func": final_transition_wrap,
                                                        "transformer_options": transformer_options
                                                    })
    else:
        x = final_transition_wrap(img=x, embedded_timestep=embedded_timestep)

    patches_final_layer_before = patches_point.get(PatchKeys.dit_final_layer_before, [])
    if patches_final_layer_before is not None and len(patches_final_layer_before) > 0:
        for patch_final_layer_before in patches_final_layer_before:
            x = patch_final_layer_before(img=x, txt=context, transformer_options=transformer_options)

    x = self.proj_out(x)

    x = self.patchifier.unpatchify(
        latents=x,
        output_height=orig_shape[3],
        output_width=orig_shape[4],
        output_num_frames=orig_shape[2],
        out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
    )

    patches_exit = patches_point.get(PatchKeys.dit_exit, [])
    if patches_exit is not None and len(patches_exit) > 0:
        for patch_exit in patches_exit:
            x = patch_exit(x, transformer_options)

    del transformer_options[PatchKeys.running_net_model]

    return x

def double_block_and_control_replace(i, block, img, txt=None, vec=None, pe=None, control=None, attn_mask=None, transformer_options={}):
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    if ("double_block", i) in blocks_replace:
        def block_wrap(args):
            out = {}
            out["img"] = block(x=args["img"],
                               context=args["txt"],
                               timestep=args["vec"],
                               pe=args["pe"],
                               attention_mask=args.get("attention_mask"))
            return out

        out = blocks_replace[("double_block", i)]({"img": img,
                                                   "txt": txt,
                                                   "vec": vec,
                                                   "pe": pe,
                                                   "attention_mask": attn_mask,
                                                   },
                                                  {
                                                      "original_block": block_wrap,
                                                      "transformer_options": transformer_options
                                                  })
        img = out["img"]
    else:
        img = block(x=img, context=txt, timestep=vec, pe=pe, attention_mask=attn_mask)

    return img, txt
