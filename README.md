# ONNX Runtime support for WebUI

Provides the Unet calculation function of the Direct-ML backend for WebUI. Tested on Windows.

## Usage

1. Use [Olive](https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion) to convert the model to ONNX.
2. Install this extension and move the Unet model to the `models/Unet-onnx` directory.
3. (Optional) Update Nvidia display driver to 532 or AMD display driver to 23.5.2.
4. Make sure the WebUI works on the `dev` branch, select the model that contains `[ORT]` in the settings.
5. Start generate.

## Notice

1. The performance of the Direct-ML backend is affected by many factors and is relatively unstable. It is recommended to adjust the power plan.
2. Due to the additional hardware resources required, it is not recommended to run on devices with lower hardware specifications.
3. In terms of low sampling steps, the speed is similar to the original version. As the number of steps increases, the performance difference will gradually become obvious.
4. The extension lacks extensive testing, please file an issue if you encounter problems.