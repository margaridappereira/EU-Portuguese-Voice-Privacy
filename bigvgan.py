# Copyright (c) 2024 NVIDIA CORPORATION.
# Licensed under the MIT license.
# This file vendors a minimal BigVGAN inference model.

import json
import os
import math
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from huggingface_hub import hf_hub_download


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))

def load_checkpoint(filepath: str, device: torch.device) -> dict:
    if not filepath:
        raise ValueError("checkpoint path must be provided")
    return torch.load(filepath, map_location=device)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.sin(x * alpha).pow(2)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "sinc"):
        return torch.sinc(x)
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter_ = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter_ = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class AMPBlock1(nn.Module):
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(len(self.convs1) + len(self.convs2))
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(len(self.convs1) + len(self.convs2))
                ]
            )
        else:
            raise NotImplementedError("activation incorrectly specified")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(nn.Module):
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(len(self.convs))
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                    for _ in range(len(self.convs))
                ]
            )
        else:
            raise NotImplementedError("activation incorrectly specified")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act in zip(self.convs, self.activations):
            xt = act(x)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class BigVGAN(nn.Module):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        if use_cuda_kernel:
            raise ValueError("CUDA kernel support is not included in the vendored model.")
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        in_channels = h.get("input_channels", h.num_mels_f)
        self.conv_pre = weight_norm(
            Conv1d(in_channels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Incorrect resblock class specified: {h.resblock}")

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock_class(h, ch, k, d, activation=h.activation))

        if h.activation == "snake":
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
        elif h.activation == "snakebeta":
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
        else:
            raise NotImplementedError("activation incorrectly specified")

        self.activation_post = Activation1d(activation=activation_post)
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def remove_weight_norm(self) -> None:
        try:
            for layers in self.ups:
                for layer in layers:
                    remove_weight_norm(layer)
            for layer in self.resblocks:
                layer.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            pass

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str = None,
        cache_dir: str = '.cache/huggingface/hub',
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model