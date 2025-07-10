<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>


## 📖 Improvements and roadmap  

This repo contains optimization efforts (as a part of VidLab7 take-home test), in terms of both quality and speed, to improve LatentSync infernece. The major updates are as follows:

- Profiler `cProfile` added to this repo. The four major performance bottlenecks found were:
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

#### Super-Resolution 

While LatentSync often yields clearer faces than pixel-domain diffusion, very high-quality frames can still appear slightly blurry or distorted under constrained settings. The official implementation notes that more sampling steps produce sharper images (at the cost of slower inference). Tuning the classifier-free guidance similarly shows trade-offs: increasing the guidance scale improves lipsync accuracy but may introduce visual jitter or artifacts . In practice, addressing this requires more compute or model capacity. For example, using more diffusion steps or upgrading to a larger latent model (e.g. SDXL) and running at higher resolution (512×512) can sharpen details, albeit with higher VRAM usage. Post-processing (e.g. super-resolution or sharpening filters) can also help. 

[VideoEnhancer](https://github.com/bhat-prashant/LatentSync/blob/main/latentsync/utils/enhancer.py) within this repo does exactly the same: upscales those syncronized, processed frames which are blurred in the regions that overlaps with the mask (mouth region). This is still Proof-of-Concept implementation aimed at showcasing the usefulness of such an approach in LatentSync. Further optimizations are very necessary to integrate this fully into inference pipeline. 


## 🎬 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="24%"><b>Original video</b></td>
        <td width="24%"><b>Lip-synced video</b></td>
        <td width="24%"><b>Optimized Lip-synced video</b></td>
        <td width="24%"><b>Optimized, super-resolution video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo1_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo1_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo1_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo1_video.mp4 controls preload></video>
    </td>
  </tr>

  <tr>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo2_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo2_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo2_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo2_video.mp4 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/bhat-prashant/LatentSync/blob/main/assets/demo3_video.mp4 controls preload></video>
    </td>
  </tr>
</table>


## How to setup and run infernece?

Follow the same instructions as in [ByteDance LatentSync](https://github.com/bytedance/LatentSync)