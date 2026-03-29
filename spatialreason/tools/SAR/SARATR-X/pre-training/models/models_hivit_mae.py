import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import trunc_normal_
from .masked_autoencoder import MaskedAutoencoder
from .models_hivit import HiViT, PatchEmbed, PatchMerge, BlockWithRPE
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
import torch.nn.functional as F


class GF(nn.Module):
    def __init__(self, nbins=9, pool=7, kensize=5, img_size=224, patch_size=16):
        super(GF, self).__init__()
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


class HiViTMaskedAutoencoder(MaskedAutoencoder, HiViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, hifeat=False,
                 **kwargs):
        MaskedAutoencoder.__init__(self)
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]
        self.hifeat = hifeat

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features, requires_grad=False)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
            relative_coords[:, :, 0] += Hp - 1 
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale, 
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr), 
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        self.num_features = 7 * embed_dim if self.hifeat else embed_dim
        self.norm = norm_layer(self.num_features)
        # --------------------------------------------------------------------------
        self.img_size = img_size
        self.patch_size = patch_size
        self.nbins = 9
        self.cell_sz = 8
        # self.hogs = HOGLayerC(nbins=self.nbins,
        #                           pool=self.cell_sz, )
        self.sarfeature1 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=9,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature2 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=13,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature3 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=17,
                                  img_size=self.img_size,patch_size=self.patch_size)


        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_patch_size = patch_size
        self.decoder_embed = nn.Linear(self.num_features, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            BlockWithRPE(
                Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale, 
                rpe=False, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, self.decoder_patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, 256*3, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.absolute_pos_embed.shape[-1], Hp, cls_token=False)
        self.absolute_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], Hp, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
    
    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.absolute_pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.absolute_pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.absolute_pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask
    
    def forward_encoder(self, x, mask_ratio):
        ids_keep, ids_restore, mask = self.masking_id(x.size(0), mask_ratio)

        if self.hifeat:
            x = self.forward_features(x, ids_keep=ids_keep, return_hifeat=True)
            h, m, l = x
            B, N, _ = l.shape
            x = torch.cat([h.reshape(B, N, -1), m.reshape(B, N, -1), l], dim=-1)
            x = self.norm(x)
        else:
            x = self.forward_features(x, ids_keep=ids_keep)

        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return None, x

    def forward_loss(self, imgs, cls_pred, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        num_preds = mask.sum()
        # target = self.patchify(imgs)
        target = torch.cat([self.patchify(self.sarfeature1(imgs)), self.patchify(self.sarfeature2(imgs)), self.patchify(self.sarfeature3(imgs))], dim=-1)
        # target = self.patchify(self.hogs2(imgs))
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


def mae_hivit_base_dec512d6b(**kwargs):
    model = HiViTMaskedAutoencoder(
        embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16, hifeat=True,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
