# ONNX and TensorRT Model Deployment

**Mức đã trao đổi:** **5/5**

### Interviewer có thể hỏi thêm

- What is ONNX?
  + Onnx is standard format file for representing deep learning model. It serve inference and deploy model. It was generated from pt or pth. 
- Why convert a PyTorch model to ONNX?
  + We convert a Pytorch model to ONNX mainly for deployment. ONNX provides a framework-independent representation of the model. And then, I can use it to convert ncnn and run on Vulkan GPU to deploy model without tensorrt. In the other hand, I just convert TensorRT to deploy on CuDNN of CUDA framework to infer on Jetson. 
- What is TensorRT?
  + TensorRT is the file model run on Jetson or Nvidia GPU specificly, so it can use cuDNN to run on Nvidia GPU more faster. 
- What happens when TensorRT builds an engine?
  + When TensorRT builds an engine, it analyzes the model graph and optimizes it for the target NVIDIA GPU. It performs optimizations such as layer fusion, constant folding, precision selection like FP16 or INT8, memory planning, and kernel or tactic selection for each operation.
- What is the difference between FP32, FP16 and INT8?
  + FP32, FP16, and INT8 are different numerical precisions used to represent model weights and activations. FP32 uses 32-bit floating-point numbers, so it provides the highest numerical precision but requires more memory and computation. FP16 uses 16-bit floating-point numbers. It reduces memory usage and usually improves inference speed on GPUs, while keeping accuracy very closed to FP32 for many deep learning models. INT8 uses 8-bit integer values. It provides the highest speed and lowest memory usage among the three, but it has much lower numerical precision, so quantization is required and accuracy can drop if the model is not calibrated properly.
- Why can INT8 be faster?
  + INT8 can be faster because it uses 8-bit integer arithmetic instead of 16-bit or 32-bit floating-point arithmetic. This reduces memory usage and memory bandwidth, and modern GPUs such as NVIDIA GPUs have specialized hardware that can perform INT8 operations with very high throughput.
- What is the process convert pt to onnx ?
  + Step 1 : 
    + First of all, I will load model pytorch as : torch.load.
    + Second, I eval model to take a input size, output size
    + Third, I send dummy random input into model and export it as torch.onnx.export().
    + Finally, the model has been generated with torch.onnx.export
  + End then, I will convert onnx to engine as trtexec
    + trtexec \ --onnx=model.onnx \ --saveEngine=model.engine
  + If I have a quantization in process convert engine : 
    + First of all, you need to calibration dataset to model engine could read how about int8 in images. So, I will prepare calibration dataset and write calibrator to TensorRT has been seen image. 
      + import tensorrt as trt
        import pycuda.driver as cuda
        import cv2
        import numpy as np
        import os
        class INT8Calibrator(trt.IInt8EntropyCalibrator2):

            def __init__(
                self,
                image_paths,
                input_shape,
                cache_file="calibration.cache"
            ):
                super().__init__()

                self.image_paths = image_paths
                self.input_shape = input_shape
                self.cache_file = cache_file
                self.index = 0

                # N,C,H,W
                n, c, h, w = input_shape

                self.device_input = cuda.mem_alloc(
                    n * c * h * w * np.float32().nbytes
                )

            def get_batch_size(self):
                return self.input_shape[0]

            def get_batch(self, names):

                if self.index >= len(self.image_paths):
                    return None

                path = self.image_paths[self.index]
                self.index += 1

                image = cv2.imread(path)

                _, _, h, w = self.input_shape

                image = cv2.resize(image, (w, h))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = image.astype(np.float32) / 255.0

                # HWC -> CHW
                image = image.transpose(2, 0, 1)

                # NCHW
                image = np.expand_dims(image, axis=0)

                image = np.ascontiguousarray(image)

                cuda.memcpy_htod(
                    self.device_input,
                    image
                )

                return [int(self.device_input)]

            def read_calibration_cache(self):

                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        return f.read()

                return None

            def write_calibration_cache(self, cache):

                with open(self.cache_file, "wb") as f:
                    f.write(cache)
    + Ater that, I have calibration cache on INT8. So, the second of all, I will start the first build with calibrator. I use library trt instead of trtexec. In trt library, they have trt.OnnxParse to read model Onnx, and trt.Builder to build model with my config INT8. In my config fuction, I will write config INT8 and address calibrator. 
      + import tensorrt as trt
        import glob

        LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(LOGGER)

        network_flags = (
            1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )
        )

        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(
            network,
            LOGGER
        )

        config = builder.create_builder_config()
        Parse ONNX: 
        with open("model.onnx", "rb") as f:
        if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))

        raise RuntimeError("ONNX parse failed")
        config.set_flag(
        trt.BuilderFlag.INT8
        )
    + If you need to build after, you should use trtexec that have --calib.
- What is INT8 calibration?
    + INT8 calibration use to with library tensorrt, not trtexec, it use for quantization in convert model from onnx to engine.
    + It make model see what is the differen between image fp32 and image int8 
- What is a TensorRT optimization profile?
  + Without quantization, I have a prune as library torch : torch.nn.utils.prune as prune. And I will choose layer and the number of amout to prune as : 
    + prune.l1_unstructured(
      model.conv1,
      name="weight",
      amount=0.3
      ) 
- What is dynamic shape?
  + When you convert from onnx to engine. Normally, you encoutered static shape more than dynamic shape. With static shape, onnx allow you a input shape. For example, only shape input 1x3x224x224 to model, so when you convert it, you don't need address for model about shape. trtexec will read input for you. But, if onnx have many input shape, it accepts 1x3x224x224, 1x3x640x640, ..., you must address engine about input shape as minShape, optShape, maxShape with trtexec. And if you want generate onnx dynamic, in pt to onnx process, you attach --dynamic-axes {0, 2, 3}, so it fit with n,c,w,h
- How do you validate that ONNX/TensorRT output matches PyTorch output?
  + I use onnx checker : onnx.checker.check_model(onnx_model)
- Why can ONNX Runtime sometimes be slower than PyTorch?
  + Because onnx runtime is alway save all model, it is smaller than pt. However, if I save model in pth, a file to save only weight, not full model, not architecture model, architecture I get in source code to load weight. At this moment, it is faster than onnx runtime.


### Project-based follow-up

Interviewer có thể đào trực tiếp:

- You mentioned StreamPETR. How did you convert or deploy it?
- Which operators caused problems?
- Why did INT8 improve performance?
- How did you measure FPS?
- Was the reported FPS model-only or end-to-end?
- Did you include image decoding, preprocessing and postprocessing?
- Why could PyTorch appear faster than ONNX in some experiments?
- How did you verify accuracy after optimization?

---
In mention book !!!
