# ONNX support for WebUI

Provides the Unet calculation function of the Direct-ML backend for WebUI. Tested on Windows.

## Usage

1. Use [Olive](https://github.com/microsoft/olive) to convert the model to ONNX.
2. Install extension.
3. (Optional) Update the Nvidia display driver to 532 or higher.
4. Make sure the WebUI works on the `dev` branch, select the model that contains `[ORT]` in the settings.
5. Generate.

## Notice

The extension currently only implements basic functions and may encounter many problems.