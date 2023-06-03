"""
ONNX Runtime UNet
"""
import os
import torch
import psutil
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
        self.cache = True
        self.shape = []
        self.engine = []

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

    def __make_engine(self, x, timesteps, context):
        def mul(t1, t2):
            return t1[0] * t2[1] == t1[1] * t2[0]

        shape = x.shape[2:]
        if shape in self.shape:
            return self.engine[self.shape.index(shape)]
        devices.torch_gc()
        if not self.cache or (devices.device == 'cuda' and get_size(torch.cuda.mem_get_info()[0]) < 4):
            self.shape.clear()
            self.engine.clear()
        elif len(self.shape) > 1:
            idx = 1 if mul(shape, self.shape[0]) else 0
            del self.shape[idx]
            del self.engine[idx]

        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False

        names = dict(zip(
            ['unet_sample_batch', 'unet_sample_channels', 'unet_sample_height', 'unet_sample_width'], x.shape))
        names['unet_time_batch'] = timesteps.shape[0]
        names |= zip(['unet_hidden_batch', 'unet_hidden_sequence'], context.shape[:2])
        for k, v in names.items():
            options.add_free_dimension_override_by_name(k, v)
        self.shape.append(shape)
        self.engine.append(
            ort.InferenceSession(self.filename, providers=['DmlExecutionProvider'], sess_options=options))
        return self.engine[-1]

    def forward(self, x, timesteps, context, *args, **kwargs):
        def to_numpy(tensor):
            return tensor.half().cpu().numpy()

        shape = x.shape[2:]
        x = self.pad(x)
        sess = self.__make_engine(x, timesteps, context)
        ort_inputs = {
            'sample': to_numpy(x), 'timestep': to_numpy(timesteps), 'encoder_hidden_states': to_numpy(context)}
        binding = sess.io_binding()
        binding.bind_output('out_sample')
        for k, v in ort_inputs.items():
            binding.bind_cpu_input(k, v)

        sess.run_with_iobinding(binding)
        del ort_inputs
        ort_outs = torch.tensor(binding.copy_outputs_to_cpu()[0], device=x.device)

        return self.clip(ort_outs, shape)

    def activate(self):
        cond = [all([devices.device == 'cuda', get_size(torch.cuda.mem_get_info()[1]) < 15]),
                all([devices.device == 'cpu', get_size(psutil.virtual_memory()[4]) < 10])]
        if any(cond):
            self.cache = False

    def deactivate(self):
        self.cache = True
        self.shape = []
        self.engine = []
        devices.torch_gc()


def get_size(ram):
    return round(ram / 1073741824, 2)


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
