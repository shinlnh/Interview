# Generative AI: Diffusion, Flow Matching, and VLMs

### 2.1 “Are you willing to optimize the denoising strategy of diffusion models for faster sampling?”

**Mức đã trao đổi:** **3/5**

Mức 3 hợp lý nếu em hiểu diffusion/sampling nhưng chưa trực tiếp tối ưu sampler.

### Interviewer có thể hỏi thêm

- What is a diffusion model?
- What is the forward diffusion process?
- What is the reverse process?
- What is denoising?
- What does sampling mean in a diffusion model?
- Why does diffusion sampling take many steps?
- Why is sampling speed important in robotics?
- What is DDPM?
- What is DDIM?
- What is a noise scheduler?
- What is the difference between training steps and inference sampling steps?
- Can we reduce the number of denoising steps?
- What trade-off happens when reducing sampling steps?
- What are DPM-Solver, distillation or consistency models?
- What is classifier-free guidance?
- How can diffusion be used to generate robot actions instead of images?
- What is an action chunk?
- Is action horizon the same as number of denoising steps?
- How would you benchmark a faster sampler?

### Important distinction

**Action horizon ≠ denoising steps.**

Ví dụ action horizon = 40 nghĩa là model có thể sinh một chunk gồm 40 action steps. Nó không có nghĩa là model denoise 40 lần.

---

## 4. SOTA Knowledge

### 4.1 “Are you familiar with LLaVA models?”

**Mức đã trao đổi:** **3/5**

### Interviewer có thể hỏi thêm

- What is LLaVA?
- What does VLM stand for?
- What are the main components of LLaVA?
- How are image features connected to a language model?
- What is a vision encoder?
- Why is CLIP commonly used as a vision encoder?
- What is a projector / multimodal projection layer?
- What is visual instruction tuning?
- LLaVA vs CLIP?
- VLM vs LLM?
- Can LLaVA output robot actions directly?
- How would you use a VLM for robotic failure detection?
- How would you force a VLM to produce structured JSON output?
- Why might a VLM hallucinate?
- How can you evaluate a VLM-based detector?
- How would you reduce VLM latency in a robot?

### Connection to em's recovery project

Pipeline em có thể mô tả:

`Robot/VLA execution → observation → VLM failure detector → recovery selector → rollback subtask → continue`

Interviewer có thể hỏi:

- Why use a VLM for failure detection instead of a classifier?
- Why use the same VLM backbone for detector and selector?
- Why separate failure detection and recovery selection?
- How do you prevent the recovery selector from choosing an invalid previous subtask?
- How do you evaluate recovery success?
- What happens if the VLM itself makes a wrong decision?

---

### 4.2 “Are you familiar with Latent Stable Diffusion models?”

**Mức đã trao đổi:** **3/5**

### Interviewer có thể hỏi thêm

- What is Stable Diffusion?
- What does latent diffusion mean?
- Why perform diffusion in latent space instead of pixel space?
- What is a VAE?
- What is the role of the encoder?
- What is the role of the decoder?
- What is the denoising network?
- What is a U-Net?
- What is conditioning?
- How is text conditioning injected into the model?
- What is cross-attention?
- What is classifier-free guidance?
- What is a noise scheduler?
- What is the difference between training and sampling?
- Why is latent diffusion computationally cheaper than pixel-space diffusion?
- What is the difference between DDPM and Stable Diffusion?
- What is the relationship between Stable Diffusion and diffusion policies in robotics?

---

## 5. Diffusion vs Flow Matching vs RL — rất dễ bị hỏi vì GR00T

> Xem bản đồ lịch sử và phân loại đầy đủ bằng tiếng Việt tại [3.7 — Từ Markov Chain đến các nhánh Reinforcement Learning, VLA và π0.5](../ml-math/07-reinforcement-learning-vla.md).

### Diffusion

Forward:

```math
x_0 \rightarrow x_1 \rightarrow ... \rightarrow x_T
```

Thêm noise dần.

Reverse:

```math
x_T \rightarrow ... \rightarrow x_1 \rightarrow x_0
```

Model học denoise.

Typical noise-prediction objective:

```math
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
```

```math
L_{diff}
=
\|
\epsilon-\epsilon_\theta(x_t,t)
\|^2
```

### Flow Matching

Một cách đơn giản:

```math
x_t=(1-t)x_0+tx_1
```

Target velocity:

```math
v^*=x_1-x_0
```

Loss:

```math
L_{FM}
=
\|
v_\theta(x_t,t)-v^*
\|^2
```

### Interviewer có thể hỏi

- Diffusion vs flow matching?
- Both can use MSE, so how do you tell them apart?
- What does the model predict in diffusion?
- What does the model predict in flow matching?
- Why can flow matching use fewer integration steps?
- What is an ODE?
- How is an ODE solver used during sampling?
- Is PPO a diffusion algorithm?
- Can RL and diffusion/flow matching be used in the same robotics stack?
- Where does PPO usually appear in humanoid robotics?
- What is a low-level controller vs a high-level VLA policy?

### Important

**PPO** là thuật toán **Reinforcement Learning**.
**Diffusion / Flow Matching** mô tả cách model học/generate continuous outputs.

Không nên nói:

> “PPO is another type of diffusion.”

---
