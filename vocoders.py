import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from bigvgan import AttrDict, BigVGAN as _BigVGAN
from bigvgan import load_checkpoint


def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
	if not prefix:
		return state_dict
	if not any(k.startswith(prefix) for k in state_dict.keys()):
		return state_dict
	return {
		(k[len(prefix):] if k.startswith(prefix) else k): v
		for k, v in state_dict.items()
	}

class BigVGANVocoder(nn.Module):
	def __init__(
		self,
		checkpoint_path: Union[str, Path] = None,
		config_path: Optional[Union[str, Path]] = None,
		model_id: Optional[str] = None,
		device: Optional[Union[str, torch.device]] = None,
		use_cuda_kernel: bool = False,
	):
		super().__init__()
		
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = torch.device(device)

		if model_id is not None:
			self.generator = _BigVGAN.from_pretrained(model_id=model_id, 
											 use_cuda_kernel=use_cuda_kernel).to(device)
			self.h = self.generator.h
			self.h.use_cuda_kernel = use_cuda_kernel
		else:		
			checkpoint_path = Path(checkpoint_path)
			if config_path is None:
				config_path = checkpoint_path.parent / "config.json"
			config_path = Path(config_path)

			with open(config_path, "r", encoding="utf-8") as f:
				config = json.load(f)
			self.h = AttrDict(config)
			self.h.use_cuda_kernel = use_cuda_kernel

			self.generator = _BigVGAN(self.h, use_cuda_kernel=use_cuda_kernel).to(
			self.device
			)
			state_dict = load_checkpoint(str(checkpoint_path), self.device)
			generator_state = state_dict["generator"]
			try:
				self.generator.load_state_dict(generator_state)
			except RuntimeError as exc:
				# Compatibility fix for checkpoints saved from torch.compile wrappers.
				sanitized_state = _strip_prefix_from_state_dict(generator_state, "_orig_mod.")
				if sanitized_state is generator_state:
					raise
				self.generator.load_state_dict(sanitized_state)
		self.generator.eval()
		self.generator.remove_weight_norm()

	@torch.no_grad()
	def forward(self, mel: torch.Tensor) -> torch.Tensor:
		mel = mel.to(self.device)
		return self.generator(mel.transpose(1,2))

	@classmethod
	def from_pretrained(
		cls,
		model_id: str,
		use_cuda_kernel: bool = False,
	):
		return cls(model_id=model_id, use_cuda_kernel=use_cuda_kernel)

