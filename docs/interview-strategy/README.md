# Physical Robotics Interview Strategy

> Tổng hợp từ các câu hỏi trong form và phần mình đã trao đổi hôm qua.
> **Thang điểm của form:**
>
> - **1/5 — Impossible to do**
> - **2/5 — Basic awareness**
> - **3/5 — Understand the concept**
> - **4/5 — Able to implement with guidance**
> - **5/5 — Easy to do without document reference**
>
> **Lưu ý:** Với vài câu trong lịch sử chat, mức điểm được lưu lại dưới dạng khoảng như `3–4` thay vì một con số cuối cùng. Anh giữ nguyên khoảng đó và đánh dấu là **chưa xác nhận chính xác**, thay vì tự bịa lại số sao em đã bấm.

---

# 8. Star-rating summary

| Topic | Rating remembered from discussion |
|---|---:|
| Replicate architecture from paper | **4/5** |
| Redesign / modify SOTA model | **3–4/5** ⚠️ |
| ONNX / TensorRT optimization | **5/5** |
| FSDP / DDP | **2–3/5** ⚠️ |
| Big-O | **4/5** |
| Heap / QuickSort | **3–4/5** ⚠️ |
| Design Patterns | **3/5** |
| OOP | **4/5** |
| Optimize diffusion denoising for faster sampling | **3/5** |
| Adam pseudocode | **3–4/5; discussion leaned 4** ⚠️ |
| Masked Language Modeling | **3/5** |
| Matrix decomposition | **3/5** |
| Maximum Likelihood Estimation | **3/5** |
| Markov Chain | **3/5** |
| Derive MSE from Gaussian MLE | **3/5** |
| LLaVA | **3/5** |
| Latent Stable Diffusion | **3/5** |

**⚠️ = lịch sử hiện còn lưu không cho anh một con số cuối cùng đủ chắc chắn; anh không tự đổi thành một mức cụ thể.**

---

# 9. Những câu nên luyện trả lời bằng miệng trước interview

1. **Tell me about the tracking algorithm you developed.**
2. **What was your exact contribution?**
3. **Why did you use a Kalman Filter?**
4. **Why did you use the Fourier domain?**
5. **Explain Big-O and give me an O(log n) example.**
6. **Explain Adam without looking at documentation.**
7. **What is Maximum Likelihood Estimation?**
8. **Why does Gaussian MLE lead to MSE?**
9. **What is the Markov property?**
10. **Markov Chain vs MDP vs POMDP?**
11. **What is a diffusion model?**
12. **Why is diffusion sampling slow?**
13. **Diffusion vs flow matching?**
14. **What is LLaVA?**
15. **What is latent diffusion?**
16. **What is DDP and how is it different from FSDP?**
17. **How do you deploy a PyTorch model with ONNX/TensorRT?**
18. **Why humanoid robotics?**
19. **Why Physical Robotics?**
20. **Explain your VLA recovery project.**
21. **Why do you use a VLM for failure detection?**
22. **What happens when the recovery VLM is wrong?**
23. **What is Isaac Sim vs Isaac Lab?**
24. **How do you fuse camera and radar data?**
25. **What happens when perception and mapping/sensors disagree?**

---

# 10. Rule khi interviewer đào theo số sao

Form định nghĩa số sao rất rõ, nên interviewer có thể dùng rating của em để chọn độ sâu câu hỏi.

### Nếu em chọn 3/5

Họ chủ yếu kỳ vọng:

- hiểu concept;
- giải thích được thuật ngữ;
- biết input/output;
- biết dùng trong trường hợp nào;
- chưa nhất thiết code from scratch.

### Nếu em chọn 4/5

Họ có thể kỳ vọng:

- giải thích concept;
- viết pseudocode;
- implement khi có reference/guidance;
- debug cơ bản;
- biết trade-off.

### Nếu em chọn 5/5

Họ có thể kỳ vọng:

- trả lời không cần tài liệu;
- tự implement;
- tự debug;
- giải thích edge cases;
- giải thích optimization;
- đưa ra kinh nghiệm thực tế.

Vì vậy **TensorRT/ONNX 5/5** là phần nên ôn kỹ nhất theo kinh nghiệm thực tế của em.

---

# 11. Quick interview structure

Khi bị hỏi về một kỹ thuật, có thể trả lời theo 5 bước:

1. **Definition** — Nó là gì?
2. **Purpose** — Nó giải quyết vấn đề gì?
3. **Mechanism** — Nó hoạt động thế nào?
4. **Trade-off** — Điểm mạnh/yếu?
5. **My experience** — Em đã dùng nó ở đâu?

Ví dụ với Kalman Filter:

> A Kalman Filter is a recursive state estimator. I use it to estimate a target state from noisy observations. It has a prediction step based on the motion model and a correction step using the measurement. The uncertainty is represented by covariance matrices. In my tracking project, I used Kalman prediction to estimate the target position and support tracking when the visual tracker became unreliable.

Không cần học thuộc nguyên câu; chỉ cần nhớ cấu trúc.

---

# 12. Priority review order

Nếu thời gian ngắn, nên ôn theo thứ tự:

**Tier A — liên quan trực tiếp CV + rating cao**

- TensorRT / ONNX
- Tracking algorithm
- Kalman Filter
- Fourier domain
- Big-O
- OOP
- Camera + Radar
- Isaac Sim / Isaac Lab
- VLA recovery project

**Tier B — form hỏi trực tiếp**

- Adam
- MLE
- MSE from Gaussian MLE
- Markov Chain
- Diffusion
- Flow Matching
- LLaVA
- Latent Diffusion

**Tier C — mức em chưa mạnh**

- DDP
- FSDP
- Matrix decomposition
- MLM
- Design patterns
- Heap / QuickSort

Mục tiêu không phải biến tất cả thành 5/5 trước interview. Mục tiêu là **rating nào đã khai thì giải thích được đúng với mức đó**.
