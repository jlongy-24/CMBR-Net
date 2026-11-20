import math
import torch
from torch import nn
import torch.nn.functional as F


def _pick_groups(channels):
    for g in (4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _mask_cues(mask, target_size, use_boundary=True, use_ambiguity=True):
    if mask.shape[2:] != tuple(target_size):
        mask = F.interpolate(mask, size=target_size, mode="bilinear", align_corners=True)

    prob = torch.sigmoid(mask)
    if use_boundary:
        grad_h = F.pad(torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]), (0, 0, 0, 1))
        grad_w = F.pad(torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]), (0, 1, 0, 0))
        boundary = torch.clamp(grad_h + grad_w, 0.0, 1.0)
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
    else:
        boundary = torch.zeros_like(prob)

    if use_ambiguity:
        ambiguity = torch.clamp(1.0 - torch.abs(prob - 0.5) * 2.0, 0.0, 1.0)
    else:
        ambiguity = torch.zeros_like(prob)

    return prob, boundary, ambiguity


class ContextGatedLiteBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Identity() if dim_in == dim_out else nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
        groups = _pick_groups(dim_out)

        self.local = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out, bias=False),
            nn.GroupNorm(groups, dim_out),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=2, dilation=2, groups=dim_out, bias=False),
            nn.GroupNorm(groups, dim_out),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(dim_out * 2, dim_out, kernel_size=1, bias=False)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, groups=groups, bias=True),
            nn.Sigmoid(),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        x = self.proj(x)
        feat = self.fuse(torch.cat([self.local(x), self.context(x)], dim=1))

        avg_map = torch.mean(x, dim=1, keepdim=True)
        grad_h = F.pad(torch.abs(avg_map[:, :, 1:, :] - avg_map[:, :, :-1, :]), (0, 0, 0, 1))
        grad_w = F.pad(torch.abs(avg_map[:, :, :, 1:] - avg_map[:, :, :, :-1]), (0, 1, 0, 0))
        edge = self.edge_gate(grad_h + grad_w)

        feat = feat * self.channel_gate(feat) * (1.0 + edge)
        return x + self.res_scale * feat


class MaskCalibratedGroupAggregationBridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=(1, 2, 5, 7),
                 use_boundary_prompt=True, use_uncertainty_prompt=True):
        super().__init__()
        if dim_xl % 4 != 0:
            raise ValueError(f"dim_xl must be divisible by 4, got {dim_xl}")

        self.use_boundary_prompt = use_boundary_prompt
        self.use_uncertainty_prompt = use_uncertainty_prompt
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, kernel_size=1)
        self.mask_gate = nn.Sequential(
            nn.Conv2d(3, dim_xl, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_pick_groups(dim_xl), dim_xl),
            nn.GELU(),
            nn.Conv2d(dim_xl, dim_xl, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        branch_channels = (dim_xl // 4) * 3
        self.g0 = self._make_branch(branch_channels, k_size, d_list[0])
        self.g1 = self._make_branch(branch_channels, k_size, d_list[1])
        self.g2 = self._make_branch(branch_channels, k_size, d_list[2])
        self.g3 = self._make_branch(branch_channels, k_size, d_list[3])
        self.tail_conv = nn.Sequential(
            LayerNorm(branch_channels * 4, data_format="channels_first"),
            nn.Conv2d(branch_channels * 4, dim_xl, kernel_size=1),
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.5)

    @staticmethod
    def _make_branch(channels, k_size, dilation):
        pad = (k_size + (k_size - 1) * (dilation - 1)) // 2
        return nn.Sequential(
            LayerNorm(channels, data_format="channels_first"),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=pad, dilation=dilation, groups=channels),
            nn.GELU(),
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=(xl.size(2), xl.size(3)), mode="bilinear", align_corners=True)

        cues = torch.cat(
            _mask_cues(mask, xl.shape[2:], self.use_boundary_prompt, self.use_uncertainty_prompt),
            dim=1,
        )
        gate = self.mask_gate(cues)
        xh = xh * (1.0 + gate)
        xl_gated = xl * (1.0 + gate)

        xh_chunks = torch.chunk(xh, 4, dim=1)
        xl_chunks = torch.chunk(xl_gated, 4, dim=1)
        g_chunks = torch.chunk(gate, 4, dim=1)

        x0 = self.g0(torch.cat((xh_chunks[0], xl_chunks[0], g_chunks[0]), dim=1))
        x1 = self.g1(torch.cat((xh_chunks[1], xl_chunks[1], g_chunks[1]), dim=1))
        x2 = self.g2(torch.cat((xh_chunks[2], xl_chunks[2], g_chunks[2]), dim=1))
        x3 = self.g3(torch.cat((xh_chunks[3], xl_chunks[3], g_chunks[3]), dim=1))
        out = self.tail_conv(torch.cat((x0, x1, x2, x3), dim=1))
        return xl + self.res_scale * out


class ProgressiveMaskRefinement(nn.Module):
    def __init__(self, feat_channels, hidden_channels=4,
                 use_boundary_prompt=True, use_uncertainty_prompt=True):
        super().__init__()
        self.use_boundary_prompt = use_boundary_prompt
        self.use_uncertainty_prompt = use_uncertainty_prompt
        self.feature_hint = nn.Conv2d(feat_channels, 1, kernel_size=1, bias=False)
        self.delta = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.25)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.delta[-1]._skip_cmbr_init = True

    def forward(self, feat, current_mask, prev_mask=None):
        if prev_mask is None:
            prev_mask = current_mask
        prob, boundary, ambiguity = _mask_cues(
            prev_mask, current_mask.shape[2:4], self.use_boundary_prompt, self.use_uncertainty_prompt
        )
        hint = self.feature_hint(feat)
        refine_in = torch.cat([current_mask, prob, boundary, ambiguity, hint], dim=1)
        return current_mask + self.res_scale * self.delta(refine_in)


class CMBRNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=(8, 12, 16, 24, 32, 44),
                 use_cglite=True, bridge=True, gt_ds=True, mask_refine=True,
                 init_weights=True, use_boundary_prompt=True, use_uncertainty_prompt=True):
        super().__init__()
        self.bridge = bridge
        self.gt_ds = gt_ds
        self.mask_refine = mask_refine

        cue_opts = dict(
            use_boundary_prompt=use_boundary_prompt,
            use_uncertainty_prompt=use_uncertainty_prompt,
        )

        def _make_deep_block(cin, cout):
            if use_cglite:
                return ContextGatedLiteBlock(cin, cout)
            return nn.Conv2d(cin, cout, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.encoder4 = nn.Sequential(_make_deep_block(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(_make_deep_block(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(_make_deep_block(c_list[4], c_list[5]))

        if bridge:
            self.GAB1 = MaskCalibratedGroupAggregationBridge(c_list[1], c_list[0], **cue_opts)
            self.GAB2 = MaskCalibratedGroupAggregationBridge(c_list[2], c_list[1], **cue_opts)
            self.GAB3 = MaskCalibratedGroupAggregationBridge(c_list[3], c_list[2], **cue_opts)
            self.GAB4 = MaskCalibratedGroupAggregationBridge(c_list[4], c_list[3], **cue_opts)
            self.GAB5 = MaskCalibratedGroupAggregationBridge(c_list[5], c_list[4], **cue_opts)

        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            if mask_refine:
                self.mask_refine5 = ProgressiveMaskRefinement(c_list[4], **cue_opts)
                self.mask_refine4 = ProgressiveMaskRefinement(c_list[3], **cue_opts)
                self.mask_refine3 = ProgressiveMaskRefinement(c_list[2], **cue_opts)
                self.mask_refine2 = ProgressiveMaskRefinement(c_list[1], **cue_opts)
                self.mask_refine1 = ProgressiveMaskRefinement(c_list[0], **cue_opts)

        self.decoder1 = nn.Sequential(_make_deep_block(c_list[5], c_list[4]))
        self.decoder2 = nn.Sequential(_make_deep_block(c_list[4], c_list[3]))
        self.decoder3 = nn.Sequential(_make_deep_block(c_list[3], c_list[2]))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1))

        self.ebn1 = nn.GroupNorm(_pick_groups(c_list[0]), c_list[0])
        self.ebn2 = nn.GroupNorm(_pick_groups(c_list[1]), c_list[1])
        self.ebn3 = nn.GroupNorm(_pick_groups(c_list[2]), c_list[2])
        self.ebn4 = nn.GroupNorm(_pick_groups(c_list[3]), c_list[3])
        self.ebn5 = nn.GroupNorm(_pick_groups(c_list[4]), c_list[4])
        self.dbn1 = nn.GroupNorm(_pick_groups(c_list[4]), c_list[4])
        self.dbn2 = nn.GroupNorm(_pick_groups(c_list[3]), c_list[3])
        self.dbn3 = nn.GroupNorm(_pick_groups(c_list[2]), c_list[2])
        self.dbn4 = nn.GroupNorm(_pick_groups(c_list[1]), c_list[1])
        self.dbn5 = nn.GroupNorm(_pick_groups(c_list[0]), c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if getattr(m, "_skip_cmbr_init", False):
            return
        if isinstance(m, nn.Conv2d):
            fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.out_channels) // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        def _zero_mask(feat):
            return torch.zeros((feat.size(0), 1, feat.size(2), feat.size(3)), device=feat.device, dtype=feat.dtype)

        t1 = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t2 = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t3 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(t3)), 2, 2))
        t5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(t4)), 2, 2))
        t6 = F.gelu(self.encoder6(t5))

        out5 = F.gelu(self.dbn1(self.decoder1(t6)))
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            if self.mask_refine:
                gt_pre5 = self.mask_refine5(out5, gt_pre5)
            mask5 = gt_pre5
            if self.bridge:
                t5 = self.GAB5(t6, t5, mask5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode="bilinear", align_corners=True)
        elif self.bridge:
            t5 = self.GAB5(t6, t5, _zero_mask(t5))
        out5 = out5 + t5

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            if self.mask_refine:
                gt_pre4 = self.mask_refine4(out4, gt_pre4, mask5)
            mask4 = gt_pre4
            if self.bridge:
                t4 = self.GAB4(t5, t4, mask4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode="bilinear", align_corners=True)
        elif self.bridge:
            t4 = self.GAB4(t5, t4, _zero_mask(t4))
        out4 = out4 + t4

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            if self.mask_refine:
                gt_pre3 = self.mask_refine3(out3, gt_pre3, mask4)
            mask3 = gt_pre3
            if self.bridge:
                t3 = self.GAB3(t4, t3, mask3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode="bilinear", align_corners=True)
        elif self.bridge:
            t3 = self.GAB3(t4, t3, _zero_mask(t3))
        out3 = out3 + t3

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            if self.mask_refine:
                gt_pre2 = self.mask_refine2(out2, gt_pre2, mask3)
            mask2 = gt_pre2
            if self.bridge:
                t2 = self.GAB2(t3, t2, mask2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode="bilinear", align_corners=True)
        elif self.bridge:
            t2 = self.GAB2(t3, t2, _zero_mask(t2))
        out2 = out2 + t2

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            if self.mask_refine:
                gt_pre1 = self.mask_refine1(out1, gt_pre1, mask2)
            mask1 = gt_pre1
            if self.bridge:
                t1 = self.GAB1(t2, t1, mask1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode="bilinear", align_corners=True)
        elif self.bridge:
            t1 = self.GAB1(t2, t1, _zero_mask(t1))
        out1 = out1 + t1

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode="bilinear", align_corners=True)

        if self.gt_ds:
            aux = (
                torch.sigmoid(gt_pre5),
                torch.sigmoid(gt_pre4),
                torch.sigmoid(gt_pre3),
                torch.sigmoid(gt_pre2),
                torch.sigmoid(gt_pre1),
            )
            return aux, torch.sigmoid(out0)
        return torch.sigmoid(out0)


if __name__ == "__main__":
    model = CMBRNet()
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 256, 256)
    aux_preds, out = model(x)
    print(f"CMBR-Net params: {params:,} ({params / 1e3:.2f}K)")
    print(f"Output shape: {tuple(out.shape)}, Aux count: {len(aux_preds)}")
