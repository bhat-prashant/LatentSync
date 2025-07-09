<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>


## 📖 Introduction
This repo contains optimization efforts, in terms of both quality and speed, to improve LatentSync infernece. The major updates are as follows:

- Profiler `cProfile` added. The four major performance bottlenecks include:
  - [Affine transformation](https://github.com/bytedance/LatentSync/blob/main/latentsync/pipelines/lipsync_pipeline.py#L252)
  - [FaceAnalysis](https://github.com/bytedance/LatentSync/blob/main/latentsync/utils/face_detector.py#L10)
  - [Denoising steps](https://github.com/bytedance/LatentSync/blob/main/latentsync/pipelines/lipsync_pipeline.py#L426)
  - [Restoring faces in synced video](https://github.com/bytedance/LatentSync/blob/main/latentsync/pipelines/lipsync_pipeline.py#L266)

Both Affine transformation and Restoring faces in synced video are processes that entail certain operation on each frame within a video, independent of the rest of the frames. Therefore, the lack of inter-dependency enables us to parallelize the execution across frames. Therefore, both these processes are sped up through multi-threading. As models and several other objects are shared, multi-threading is an ideal choice as it is lightweight and devoid of any inter-process communication. 

FaceAnalysis is another important component within LatentSync that enables detecting faces using a FaceDetector. Essentially, the FaceDetector class utilizes the `InsightFace` library to detect faces in image frames and applies filters based on size, aspect ratio, and detection score. It identifies the largest valid face and retrieves its 2D landmarks, calculating a bounding box that encapsulates the detected face. The class returns the adjusted bounding box coordinates and the landmarks for further processing. Although the processing of frames in Affine transformation (including FaceAnalysis) is already parallelized as above, the detection is still cumbersome. As noted in [InsightFace](https://github.com/deepinsight/insightface/tree/master/model_zoo), the perfromance difference between large model (buffalo_l) and small model (buffalo_s) is minimal while difference in the number of parameters is huge. Considering that we have single-face videos covering almost entirety of the screen space, smaller model (buffalo_s) is chosen as a default for FaceAnalysis. 

With regard to optimizing the denoising steps, it can be done in several ways:
- DataParallel or Distributed Data Parallel modeling instead of plain pytorch models. 
- Increasing the batch size (Currently batch size is set to 1) --> Requires substantive efforts as LatentSync Assumes `batch_size=1` in majority of the places. 
- Decreasing number of denoising steps at the expense of quality. 
- Infernece with lower precision or mixed precision --> LatentSync uses fp16 if supported by the hardware. 
- Layer pruning --> Layers with redundant information can be pruned after exhaustive validation.
- Quantization --> Models within LatentSync (VAE, U-Net and Audio Encoder) can be quantized to reduce their footprint and increase perfromance. 
- Inference with lower image resolution 

Due to limited GPU credits and time, I haven't had a chance to dig deeper into the aforementioned aspects. Coupled with extensive hyper-parameter tuning, the above optimizations can yield substantial improvements in terms of speed and efficiency. Last, but not the least, inference optimization can also be achieved through hardware-aware inference engines such as ONNX, TensorRT or NVIDIA Triton. 



## 🎬 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="32%"><b>Original video</b></td>
        <td width="32%"><b>Lip-synced video</b></td>
        <td width="32%"><b>Optimized Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/b778e3c3-ba25-455d-bdf3-d89db0aa75f4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/ac791682-1541-4e6a-aa11-edd9427b977e controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/ac791682-1541-4e6a-aa11-edd9427b977e controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/fb4dc4c1-cc98-43dd-a211-1ff8f843fcfa controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/7c6ca513-d068-4aa9-8a82-4dfd9063ac4e controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/7c6ca513-d068-4aa9-8a82-4dfd9063ac4e controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0756acef-2f43-4b66-90ba-6dc1d1216904 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/663ff13d-d716-4a35-8faa-9dcfe955e6a5 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/663ff13d-d716-4a35-8faa-9dcfe955e6a5 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
</table>


