import launch

if launch.is_installed('onnxruntime-gpu'):
    launch.run_pip('uninstall -y onnxruntime-gpu')

if not launch.skip_install and not launch.is_installed('onnxruntime-directml'):
    launch.run_pip("install onnxruntime-directml==1.15.0", 'requirement for SD-ONNX')
