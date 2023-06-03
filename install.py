import launch

if not launch.is_installed('onnxruntime-directml'):
    launch.run_pip('install onnxruntime-directml==1.15.0', 'requirement for SD-ONNX')

if not launch.is_installed('psutil'):
    launch.run_pip('install psutil')
