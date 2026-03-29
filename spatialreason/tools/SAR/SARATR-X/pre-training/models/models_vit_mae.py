# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VisionF

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np

class SAR_HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7, kensize=5, img_size=224, patch_size=16):
        super(SAR_HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.img_size = img_size
        self.patch_size = patch_size
        self.k = kensize

        def creat_gauss_kernel(r=1):

            M_13 = np.concatenate([np.ones([r+1, 2*r+1]), np.zeros([r, 2*r+1])], axis=0)
            M_23 = np.concatenate([np.zeros([r, 2 * r + 1]), np.ones([r+1, 2 * r + 1])], axis=0)

            M_11 = np.concatenate([np.ones([2*r+1, r+1]), np.zeros([2*r+1, r])], axis=1)
            M_21 = np.concatenate([np.zeros([2 * r + 1, r]), np.ones([2 * r + 1, r+1])], axis=1)


            return torch.from_numpy((M_13)).float(), torch.from_numpy((M_23)).float(), torch.from_numpy((M_11)).float(), torch.from_numpy((M_21)).float()

        M13, M23, M11, M21 = creat_gauss_kernel(self.k)

        weight_x1 = M11.view(1, 1, self.k*2+1, self.k*2+1)
        weight_x2 = M21.view(1, 1, self.k*2+1, self.k*2+1)

        weight_y1 = M13.view(1, 1, self.k*2+1, self.k*2+1)
        weight_y2 = M23.view(1, 1, self.k*2+1, self.k*2+1)

        self.register_buffer("weight_x1", weight_x1)
        self.register_buffer("weight_x2", weight_x2)
        self.register_buffer("weight_y1", weight_y1)
        self.register_buffer("weight_y2", weight_y2)


    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect") + 1e-2
        gx_1 = F.conv2d(
            x, self.weight_x1, bias=None, stride=1, padding=0, groups=1
        )
        gx_2 = F.conv2d(
            x, self.weight_x2, bias=None, stride=1, padding=0, groups=1
        )
        gy_1 = F.conv2d(
            x, self.weight_y1, bias=None, stride=1, padding=0, groups=1
        )
        gy_2 = F.conv2d(
            x, self.weight_y2, bias=None, stride=1, padding=0, groups=1
        )
        gx_rgb = torch.log((gx_1) / (gx_2))
        gy_rgb = torch.log((gy_1) / (gy_2))
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        # phase = torch.atan2(gx_rgb, gy_rgb)
        # phase = phase / self.pi * self.nbins  # [-9, 9]
        #
        # b, c, h, w = norm_rgb.shape
        # out = torch.zeros(
        #     (b, c, self.nbins, h, w), dtype=torch.float, device=x.device
        # )
        # phase = phase.view(b, c, 1, h, w)
        # norm_rgb = norm_rgb.view(b, c, 1, h, w)

        # plt.subplot(211)
        # plt.imshow(x[0].cpu().squeeze())
        # plt.subplot(212)
        # plt.imshow(norm_rgb[0].cpu().squeeze())
        # plt.show()

        # out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)
        # # b, c, 9, h, w
        #
        # out = out.unfold(3, self.pool, self.pool)
        #
        # out = out.unfold(4, self.pool, self.pool)
        # # b, c, 9, 28, 28, self.pool, self.pool
        # out = out.sum(dim=[-1, -2])
        # # b, c, 9, 28, 28
        # out = torch.nn.functional.normalize(out, p=2, dim=2) # B 1 nbins H W
        # # b, c, 9, 28, 28
        # tmp_hog = out.flatten(1, 2)  # return B C H W
        # # b, 9, 28, 28
        # unfold_size = tmp_hog.shape[-1] // (self.img_size // self.patch_size)
        # # b, 9, 14, 14, 9, 2, 2
        # target = (
        #     tmp_hog.permute(0, 2, 3, 1)
        #         .unfold(1, unfold_size, unfold_size)
        #         .unfold(2, unfold_size, unfold_size)
        #         .flatten(1, 2)
        #         .flatten(2)
        # )

        return norm_rgb

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., stem_layer=PatchEmbed, stem_configs=dict(),
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = stem_layer(
            img_size, patch_size, in_chans, embed_dim, norm_layer=norm_layer, **stem_configs
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.nbins = 9
        self.cell_sz = 8
        # self.hogs = HOGLayerC(nbins=self.nbins,
        #                           pool=self.cell_sz, )
        self.hogs1 = SAR_HOGLayerC(nbins=self.nbins,pool=self.cell_sz,kensize=5,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.hogs2 = SAR_HOGLayerC(nbins=self.nbins,pool=self.cell_sz,kensize=9,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.hogs3 = SAR_HOGLayerC(nbins=self.nbins,pool=self.cell_sz,kensize=13,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.hogs4 = SAR_HOGLayerC(nbins=self.nbins,pool=self.cell_sz,kensize=17,
                                  img_size=self.img_size,patch_size=self.patch_size)


        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, 256*4, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if isinstance(self.patch_embed, ConvPatchEmbed):
            pass
            for m in self.patch_embed.proj[:-1]:
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / m.out_channels))
            for m in self.patch_embed.proj[-1:]:
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(1.0 / m.out_channels))
                    torch.nn.init.zeros_(m.bias)
        else:
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.patch_embed.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def random_masking(self, x, ids_keep):
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward_encoder(self, x, mask_ratio):
        ids_keep, ids_restore, mask = self.masking_id(x.size(0), mask_ratio)

        # embed patches
        x = self.patch_embed(x, mask)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x = self.random_masking(x, ids_keep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        cls_pred, pred = torch.split(x, [1, x.size(1) - 1], dim=1)
        return cls_pred, pred

    def forward_loss(self, imgs, cls_pred, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        num_preds = mask.sum()

                # target = self.patchify(imgs)
        target = torch.cat([self.patchify(self.hogs1(imgs)), self.patchify(self.hogs2(imgs)), self.patchify(self.hogs3(imgs)), self.patchify(self.hogs4(imgs))], dim=-1)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(torch.cat([imgs, imgs, imgs], dim=1), mask_ratio)
        cls_pred, pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, cls_pred, pred, mask)
        return loss, pred, mask


def mae_vit_small_patch16_dec192d6b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=6, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b_cls10crop10(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        cls_loss_weight=1.0, cls_crop_ratio=1.0, **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
