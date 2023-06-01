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
        self.label = f'[ORT] {name}'
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return OrtUnet(self.filename)


class OrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.model = None
        self.shape = None
        self.engine = None

    @staticmethod
    def clip(tensor, shape):
        tmp = [tensor.shape[i + 2] - shape[i] for i in range(2)]
        if sum(tmp) != 0:
            tmp.extend([i // 2 for i in tmp])
            tmp[:2] = [tmp[i] if sum(tmp[i::2]) != 0 else -shape[i] for i in range(2)]
            return tensor[:, :, tmp[2]:tmp[2] - tmp[0], tmp[3]:tmp[3] - tmp[1]]
        return tensor

    @staticmethod
    def pad(tensor):
        tmp = [i % 8 for i in tensor.shape[2:]]
        if sum(tmp) != 0:
            tmp = [8 - i if i != 0 else 0 for i in tmp]
            tmp.extend([i // 2 for i in tmp])
            return torch.nn.functional.pad(tensor, (tmp[3], tmp[1] - tmp[3], tmp[2], tmp[0] - tmp[2]), 'reflect')
        return tensor

    def forward(self, x, timesteps, context, *args, **kwargs):
        def to_numpy(tensor):
            return tensor.half().cpu().numpy()

        shape = x.shape[2:]
        x = self.pad(x)
        if self.shape is None or self.shape != x.shape[2:]:
            self.engine = None
            devices.torch_gc()

        if self.engine is None:
            options = ort.SessionOptions()
            options.enable_mem_pattern = False

            names = dict(zip(
                ['unet_sample_batch', 'unet_sample_channels', 'unet_sample_height', 'unet_sample_width'], x.shape))
            names['unet_time_batch'] = timesteps.shape[0]
            names.update(zip(
                ['unet_hidden_batch', 'unet_hidden_sequence'], context.shape[:2]))
            for k, v in names.items():
                options.add_free_dimension_override_by_name(k, v)
            self.shape = x.shape[2:]
            self.engine = ort.InferenceSession(self.model, providers=['DmlExecutionProvider'], sess_options=options)

        ort_inputs = {
            'sample': to_numpy(x), 'timestep': to_numpy(timesteps), 'encoder_hidden_states': to_numpy(context)}
        ort_outs = torch.Tensor(self.engine.run(None, ort_inputs)[0]).to(devices.device)

        return self.clip(ort_outs, shape)

    def activate(self):
        print('Loading models ...')
        self.model = open(self.filename, 'rb').read()

    def deactivate(self):
        self.model = None
        self.shape = None
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
