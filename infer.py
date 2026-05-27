import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


# ======================
# VAE blocks
# ======================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),

            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.layer(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self, ch=64, latent=4):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=3, padding=1),

            ResBlock(ch),
            ResBlock(ch),
            Downsample(ch, ch * 2),

            ResBlock(ch * 2),
            ResBlock(ch * 2),
            Downsample(ch * 2, ch * 4),

            ResBlock(ch * 4),
            ResBlock(ch * 4),
            Downsample(ch * 4, ch * 4),

            ResBlock(ch * 4),
            ResBlock(ch * 4),
        )

        self.mu = nn.Conv2d(ch * 4, latent, kernel_size=3, padding=1)
        self.logv = nn.Conv2d(ch * 4, latent, kernel_size=3, padding=1)

        self.dec = nn.Sequential(
            nn.Conv2d(latent, ch * 4, kernel_size=3, padding=1),

            ResBlock(ch * 4),
            ResBlock(ch * 4),
            Upsample(ch * 4, ch * 4),

            ResBlock(ch * 4),
            ResBlock(ch * 4),
            Upsample(ch * 4, ch * 2),

            ResBlock(ch * 2),
            ResBlock(ch * 2),
            Upsample(ch * 2, ch),

            ResBlock(ch),
            ResBlock(ch),

            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logv = self.logv(h)
        return mu, logv

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logv = self.encode(x)
        std = torch.exp(logv * 0.5)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z)
        return recon, mu, logv, z


# ======================
# UNet blocks
# ======================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb_scale))

        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        return emb


class Attention(nn.Module):
    def __init__(self, ch, heads=8, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.att = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape

        residual = x

        x = self.norm(x)
        x = x.flatten(2).permute(0, 2, 1)

        x, _ = self.att(x, x, x, need_weights=False)

        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return residual + x


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )

        self.norm1 = nn.GroupNorm(GROUPS, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(GROUPS, out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return (h + self.res_conv(x)) / math.sqrt(2)


class UNET(nn.Module):
    def __init__(
        self,
        latent=4,
        base_dim=96,
        time_dim=384,
        num_blocks=2,
        level_mul=(1, 2, 4, 4),
        level_att=(False, True, True, True),
        self_condition=True,
    ):
        super().__init__()

        self.self_condition = self_condition

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        dims = [base_dim * i for i in level_mul]

        in_latent_ch = latent * 2 if self_condition else latent
        self.inp = nn.Conv2d(in_latent_ch, dims[0], kernel_size=3, padding=1)

        self.downblocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, out_ch in enumerate(dims):
            blocks = nn.ModuleList(
                [Block(out_ch, out_ch, time_dim) for _ in range(num_blocks)]
            )

            attn = Attention(out_ch) if level_att[i] else nn.Identity()

            self.downblocks.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "attn": attn,
                    }
                )
            )

            if i != len(dims) - 1:
                self.downsamples.append(Downsample(out_ch, dims[i + 1]))

        self.mid_block1 = Block(dims[-1], dims[-1], time_dim)
        self.mid_attn = Attention(dims[-1])
        self.mid_block2 = Block(dims[-1], dims[-1], time_dim)

        self.upblocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        rev_dims = list(reversed(dims))
        rev_attn = list(reversed(level_att))

        for i in range(len(rev_dims) - 1):
            in_ch = rev_dims[i]
            skip_ch = rev_dims[i + 1]
            out_ch = rev_dims[i + 1]

            self.upsamples.append(Upsample(in_ch, out_ch))

            blocks = nn.ModuleList(
                [Block(out_ch + skip_ch, out_ch, time_dim)]
                + [Block(out_ch, out_ch, time_dim) for _ in range(num_blocks - 1)]
            )

            use_attn = rev_attn[i + 1]
            attn = Attention(out_ch) if use_attn else nn.Identity()

            self.upblocks.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "attn": attn,
                    }
                )
            )

        self.out = nn.Conv2d(base_dim, latent, kernel_size=3, padding=1)

    def apply_blocks(self, blocks, x, t_emb):
        for block in blocks:
            x = block(x, t_emb)
        return x

    def forward(self, x, t, x_self_cond=None):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)

            x = torch.cat([x_self_cond, x], dim=1)

        emb = self.time_emb(t)

        x = self.inp(x)

        skips = []

        for i, down in enumerate(self.downblocks):
            x = self.apply_blocks(down["blocks"], x, emb)
            x = down["attn"](x)

            skips.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        skips = skips[:-1]

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        skips = list(reversed(skips))

        for i, up in enumerate(self.upblocks):
            x = self.upsamples[i](x)

            skip = skips[i]

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

            x = torch.cat([x, skip], dim=1)

            x = self.apply_blocks(up["blocks"], x, emb)
            x = up["attn"](x)

        return self.out(x)


# ======================
# Utils
# ======================

def load_state_dict(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "ema_model"]:
            if key in ckpt:
                ckpt = ckpt[key]
                break

    if isinstance(ckpt, dict):
        ckpt = {
            k.replace("module.", ""): v
            for k, v in ckpt.items()
        }

    model.load_state_dict(ckpt, strict=True)
    return model


@torch.no_grad()
def sample(
    unet,
    vae,
    n_samples,
    latent_mean,
    latent_std,
    device,
    t_steps=228,
    latent_dim=4,
    latent_size=32,
    beta_start=1e-4,
    beta_end=0.035,
    clamp_x0=3.0,
):
    unet.eval()
    vae.eval()

    betas = torch.linspace(beta_start, beta_end, t_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    z = torch.randn(n_samples, latent_dim, latent_size, latent_size, device=device)

    x_self_cond = None

    for i in reversed(range(t_steps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alpha_bars[i]

        if i > 0:
            alpha_bar_prev = alpha_bars[i - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        pred_noise = unet(z, t, x_self_cond)

        x0_pred = (
            z - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        ) / torch.sqrt(alpha_bar_t)

        x0_pred = x0_pred.clamp(-clamp_x0, clamp_x0)

        # self-conditioning
        x_self_cond = x0_pred.detach()

        coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)

        mean = coef1 * x0_pred + coef2 * z

        if i > 0:
            var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(z)
            z = mean + torch.sqrt(var) * noise
        else:
            z = mean

    z = z * latent_std + latent_mean

    images = vae.decode(z)
    images = (images.clamp(-1, 1) + 1) / 2

    return images


# ======================
# config
# ======================

VAE_PATH = "vae_WW.pth"
UNET_PATH = "ema_unet_L.pth"
STATS_PATH = "latent_stats.pth"

OUT_PATH = "samples.png"

N_SAMPLES = 16
NROW = 4

VAE_BASE_DIM = 64
UNET_BASE_DIM = 96
LATENT_DIM = 4
TIME_DIM = 384
GROUPS = 8

T_STEPS = 228
LATENT_SIZE = 32

CLAMP_X0 = 3.0
SEED = 42


def load_latent_stats(path, device):
    if path is None or not os.path.exists(path):
        print("latent_stats.pth not found, using mean=0 and std=1")
        mean = torch.tensor(0.0, device=device)
        std = torch.tensor(1.0, device=device)
        return mean, std

    stats = torch.load(path, map_location=device)

    mean = stats["mean"]
    std = stats["std"]

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=device)
    else:
        mean = mean.to(device)

    if not torch.is_tensor(std):
        std = torch.tensor(std, device=device)
    else:
        std = std.to(device)

    print(f"LATENT_MEAN = {mean}")
    print(f"LATENT_STD  = {std}")

    return mean, std


def main():
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vae = VAE(
        ch=VAE_BASE_DIM,
        latent=LATENT_DIM,
    ).to(device)

    unet = UNET(
        latent=LATENT_DIM,
        base_dim=UNET_BASE_DIM,
        time_dim=TIME_DIM,
        num_blocks=2,
        level_mul=(1, 2, 4, 4),
        level_att=(False, True, True, True),
        self_condition=True,
    ).to(device)

    load_state_dict(vae, VAE_PATH, device)
    load_state_dict(unet, UNET_PATH, device)

    latent_mean, latent_std = load_latent_stats(STATS_PATH, device)

    images = sample(
        unet=unet,
        vae=vae,
        n_samples=N_SAMPLES,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=device,
        t_steps=T_STEPS,
        latent_dim=LATENT_DIM,
        latent_size=LATENT_SIZE,
        clamp_x0=CLAMP_X0,
    )

    save_image(images, OUT_PATH, nrow=NROW)
    print(f"Saved samples to: {OUT_PATH}")


if __name__ == "__main__":
    main()