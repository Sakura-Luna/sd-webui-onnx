"""
ONNX Runtime UNet
"""
import os
import torch
import numpy as np
import onnxruntime as ort

from modules import script_callbacks, sd_unet, devices, paths_internal

ort.set_default_logger_severity(3)


class OrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[ORT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return OrtUnet(self.filename)


class OrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.engine = None

    def forward(self, x, timesteps, context, *args, **kwargs):
        def to_numpy(tensor):
            return np.half(tensor.detach().cpu()) if tensor.requires_grad else np.half(tensor.cpu())

        ort_inputs = {
            'sample': to_numpy(x), 'timestep': to_numpy(timesteps), 'encoder_hidden_states': to_numpy(context)}
        ort_outs = self.engine.run(None, ort_inputs)

        return torch.Tensor(ort_outs[0]).to(devices.device)

    def activate(self):
        print("Loading models into ORT session...")
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        ort_sess = ort.InferenceSession(self.filename, providers=['DmlExecutionProvider'], sess_options=options)

        self.engine = ort_sess

    def deactivate(self):
        self.engine = None
        devices.torch_gc()


def list_unets(model_list):
    ort_dir = os.path.join(paths_internal.models_path, 'Unet-onnx')
    if not os.path.exists(ort_dir):
        os.makedirs(ort_dir)
    files = list(filter(lambda x: x.lower().endswith('.onnx'), os.listdir(ort_dir)))

    for f in sorted(files):
        filename = os.path.join(ort_dir, f)
        opt = OrtUnetOption(filename, f)
        model_list.append(opt)


script_callbacks.on_list_unets(list_unets)
