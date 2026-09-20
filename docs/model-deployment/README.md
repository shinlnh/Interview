# ONNX and TensorRT Model Deployment

**Mức đã trao đổi:** **5/5**

### Interviewer có thể hỏi thêm

- What is ONNX?
- Why convert a PyTorch model to ONNX?
- What is TensorRT?
- What happens when TensorRT builds an engine?
- What is the difference between FP32, FP16 and INT8?
- Why can INT8 be faster?
- What is INT8 calibration?
- What is a TensorRT optimization profile?
- What is dynamic shape?
- What happens if an ONNX operator is not supported by TensorRT?
- What is a TensorRT plugin?
- How do you validate that ONNX/TensorRT output matches PyTorch output?
- Why can ONNX Runtime sometimes be slower than PyTorch?
- Why can TensorRT be faster on NVIDIA hardware?
- What causes numerical differences after FP16 or INT8 conversion?
- What is latency vs throughput?
- How would you benchmark inference correctly?
- Does higher FPS always mean lower latency?
- How do preprocessing and postprocessing affect end-to-end FPS?
- How would you deploy a model on Jetson Xavier?

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
