# 3.7 Cây Machine Learning: Imitation Learning, Reinforcement Learning và VLA

[← Mục lục Machine Learning and Mathematics](./README.md)

> Phần này mô tả bức tranh của lĩnh vực tính đến **tháng 9/2026**. Thứ tự các mục bên dưới đi **đúng theo thứ tự của cây thuật toán**: thấy một node trong cây thì có thể kéo xuống đúng mục mang cùng số để đọc phân tích.

## 1. Cây tổng quát và bản đồ mục lục

```text
Machine Learning
│
├── 2. Supervised Learning
│   ├── 2.1 Classification
│   ├── 2.2 Regression
│   └── 2.3 Sequence prediction
│
├── 3. Unsupervised / Self-supervised Learning
│   ├── 3.1 Representation learning
│   ├── 3.2 Contrastive learning
│   └── 3.3 Masked / next-token / vision-language pre-training
│
├── 4. Imitation Learning — IL
│   ├── 4.1 Behavioral Cloning — BC
│   │   ├── 4.1.1 One-step action prediction
│   │   ├── 4.1.2 Autoregressive action tokens
│   │   ├── 4.1.3 Action chunking
│   │   ├── 4.1.4 Diffusion policy
│   │   └── 4.1.5 Flow-matching policy
│   ├── 4.2 Interactive IL
│   │   ├── 4.2.1 DAgger
│   │   └── 4.2.2 Human intervention / correction
│   ├── 4.3 Inverse Reinforcement Learning — IRL
│   └── 4.4 Generative Adversarial Imitation Learning — GAIL
│
└── 5. Reinforcement Learning — RL
    ├── 5.1 MDP, value, Bellman và Temporal Difference
    ├── 5.2 Value-based
    │   ├── 5.2.1 SARSA
    │   ├── 5.2.2 Q-learning
    │   ├── 5.2.3 DQN
    │   └── 5.2.4 Double / Dueling / PER / Distributional DQN
    ├── 5.3 Policy-based
    │   ├── 5.3.1 REINFORCE
    │   ├── 5.3.2 TRPO
    │   └── 5.3.3 GRPO
    ├── 5.4 Actor–Critic
    │   ├── 5.4.1 A2C / A3C
    │   ├── 5.4.2 PPO
    │   ├── 5.4.3 DDPG
    │   ├── 5.4.4 TD3
    │   └── 5.4.5 SAC
    ├── 5.5 Model-based RL
    │   ├── 5.5.1 Dyna
    │   ├── 5.5.2 Model Predictive Control — MPC
    │   ├── 5.5.3 Dreamer
    │   ├── 5.5.4 MuZero
    │   └── 5.5.5 World-model RL cho VLA
    └── 5.6 Offline RL
        ├── 5.6.1 BCQ
        ├── 5.6.2 CQL
        ├── 5.6.3 IQL
        └── 5.6.4 Decision Transformer

6. Các trục phân loại chồng lấp
7. VLA nằm ở đâu và đã phát triển thế nào?
8. Câu trả lời phỏng vấn
```

### Quy tắc đọc cây

`Value-based`, `Policy-based` và `Actor–Critic` trả lời câu hỏi **agent học đối tượng nào**. `Model-based`, `Offline`, `On-policy` hay `Off-policy` lại là các trục khác, nên một thuật toán có thể mang nhiều nhãn.

Ví dụ:

- SARSA: value-based, model-free, on-policy.
- Q-learning và DQN: value-based, model-free, off-policy.
- PPO: policy-gradient về objective, nhưng implementation thông thường là on-policy Actor–Critic.
- SAC: off-policy Actor–Critic, model-free, maximum-entropy RL.
- WMPO: model-based VLA RL, đồng thời dùng on-policy GRPO để update policy.

---

## 2. Supervised Learning

Supervised Learning học mapping từ input sang target đã biết:

```math
\mathcal L_{\mathrm{sup}}(\theta)
=\frac{1}{N}\sum_{i=1}^{N}\ell\!\left(f_\theta(x_i),y_i\right)
```

### 2.1 Classification

> **Paper nên đọc:** [ImageNet Classification with Deep Convolutional Neural Networks — AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).

Classification dự đoán một nhãn rời rạc. Với softmax classifier, loss thường là cross-entropy:

```math
\mathcal L_{\mathrm{CE}}
=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i)
```

Ví dụ: ảnh chứa mèo hay chó, trạng thái robot là thành công hay thất bại, vật thể thuộc loại nào.

### 2.2 Regression

> **Đọc thêm:** [Derive MSE from Gaussian Maximum Likelihood](./06-gaussian-mle-to-mse.md).

Regression dự đoán giá trị liên tục. Nếu giả định target có Gaussian noise với variance cố định, maximum likelihood dẫn đến MSE:

```math
\mathcal L_{\mathrm{MSE}}
=\frac{1}{N}\sum_{i=1}^{N}
\left\|y_i-f_\theta(x_i)\right\|_2^2
```

Ví dụ: dự đoán vị trí, vận tốc, torque hoặc một continuous robot action.

### 2.3 Sequence prediction

> **Paper nên đọc:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) và [RT-2](https://arxiv.org/abs/2307.15818).

Model dự đoán chuỗi token hoặc action theo phân rã autoregressive:

```math
p_\theta(y_{1:T}\mid x)
=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{\lt t})
```

Language modeling và action-token prediction có cùng hình thức toán học. Nếu target action đến từ expert demonstrations, quá trình học robot policy vẫn là Behavioral Cloning dù architecture là Transformer.

---

## 3. Unsupervised / Self-supervised Learning

Nhánh này học representation hoặc tạo target từ chính dữ liệu, không cần người gắn nhãn thủ công cho từng sample.

### 3.1 Representation learning

> **Paper nên đọc:** [A Simple Framework for Contrastive Learning of Visual Representations — SimCLR](https://arxiv.org/abs/2002.05709).

Mục tiêu là học embedding giữ lại thông tin hữu ích. Representation tốt có thể được fine-tune cho classification, retrieval hoặc robot control với ít dữ liệu downstream hơn.

### 3.2 Contrastive learning

> **Paper nên đọc:** [CLIP](https://arxiv.org/abs/2103.00020).

Contrastive learning kéo positive pair lại gần và đẩy negative pair ra xa trong embedding space. CLIP dùng image-text pair để học không gian biểu diễn chung, cung cấp semantic visual knowledge cho nhiều VLM/VLA backbone.

### 3.3 Masked, next-token và vision-language pre-training

> **Paper nên đọc:** [BERT](https://arxiv.org/abs/1810.04805), [GPT-3](https://arxiv.org/abs/2005.14165) và [PaLI](https://arxiv.org/abs/2209.06794).

- **Masked modeling:** che một phần input rồi dự đoán phần bị che.
- **Next-token prediction:** dự đoán token tiếp theo từ prefix.
- **Vision-language pre-training:** học joint understanding từ ảnh và text.

Trong VLA, nhánh này cung cấp backbone hiểu vật thể, ngôn ngữ và quan hệ không gian trước khi model học action từ robot demonstrations.

---

## 4. Imitation Learning — IL

Imitation Learning học behavior từ expert demonstrations thay vì bắt đầu bằng trial-and-error từ reward.

### 4.1 Behavioral Cloning — BC

> **Paper nên đọc:** [DAgger](https://proceedings.mlr.press/v15/ross11a.html), paper dùng supervised imitation/BC làm điểm xuất phát và phân tích compounding error.

Cho expert dataset:

```math
\mathcal D_E=\left\{(o_t,l,a_t)\right\}
```

BC tối đa hóa likelihood của expert action:

```math
\mathcal L_{\mathrm{BC}}(\theta)
=-\mathbb E_{(o,l,a)\sim\mathcal D_E}
\left[\log\pi_\theta(a\mid o,l)\right]
```

**Ưu điểm:** ổn định, tận dụng teleoperation data, không cần reward.

**Hạn chế:** khi policy mắc lỗi, nó có thể đi vào state chưa từng có trong expert data; lỗi tiếp tục tích lũy thành **covariate shift/compounding error**.

#### 4.1.1 One-step action prediction

> **Paper nên đọc:** [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702).

Policy dự đoán một action cho observation hiện tại:

```math
\hat a_t=f_\theta(o_t,l)
```

Với continuous action và Gaussian variance cố định, loss thường là MSE hoặc L1. Cách này nhanh nhưng dễ tạo action trung bình khi demonstrations có nhiều mode hợp lệ.

#### 4.1.2 Autoregressive action tokens

> **Paper nên đọc:** [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246) và [FAST](https://arxiv.org/abs/2501.09747).

Action được lượng tử hóa thành token rồi dự đoán như language:

```math
p_\theta(a_{1:K}\mid o,l)
=\prod_{k=1}^{K}p_\theta(a_k\mid o,l,a_{\lt k})
```

Ưu điểm là tái sử dụng decoder và cross-entropy training của LLM/VLM. Hạn chế là quantization error và autoregressive latency.

#### 4.1.3 Action chunking

> **Paper nên đọc:** [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — ACT](https://arxiv.org/abs/2304.13705).

Policy dự đoán một đoạn action thay vì một bước:

```math
A_t=\left(a_t,a_{t+1},\ldots,a_{t+H-1}\right)
```

Chunking giảm effective horizon, biểu diễn motion có cấu trúc và giảm số lần gọi model lớn. Nếu chunk quá dài, policy lại kém reactive vì thực thi gần open-loop.

#### 4.1.4 Diffusion policy

> **Paper nên đọc:** [Diffusion Policy](https://arxiv.org/abs/2303.04137) và [Octo](https://arxiv.org/abs/2405.12213).

Diffusion policy học khử noise để sinh action trajectory đa mode:

```math
\mathcal L_{\mathrm{diff}}
=\mathbb E_{A,\epsilon,k}
\left[
\left\|\epsilon-\epsilon_\theta(A^{(k)},o,k)\right\|_2^2
\right]
```

Diffusion là **cách model action distribution**, không phải RL. Nếu data là expert demonstrations và không có reward optimization, nó vẫn nằm dưới BC.

#### 4.1.5 Flow-matching policy

> **Paper nên đọc:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) và [π0](https://arxiv.org/abs/2410.24164).

Flow matching học vector field biến noise thành continuous action chunk:

```math
\mathcal L_{\mathrm{FM}}
=\mathbb E_{A,\epsilon,\tau}
\left[
\left\|v_\theta(A^\tau,o,\tau)-u(A,\epsilon)\right\|_2^2
\right]
```

π0 kết hợp pretrained VLM với action expert sinh action chunks bằng flow matching. π0 gốc vì vậy là VLA + IL/BC + generative action modeling, không phải một RL algorithm.

### 4.2 Interactive Imitation Learning

Interactive IL thu thập thêm label/correction tại những state mà learner thực sự gặp khi rollout.

#### 4.2.1 DAgger

> **Paper gốc:** [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html).

```text
train BC
   ↓
rollout learner
   ↓
expert gắn action đúng tại state learner đã gặp
   ↓
aggregate dataset và train lại
```

DAgger trực tiếp giảm mismatch giữa expert-state distribution và learner-state distribution. Đổi lại, nó cần expert liên tục cung cấp label.

#### 4.2.2 Human intervention / correction

> **Paper nên đọc:** [RT-H: Action Hierarchies Using Language](https://arxiv.org/abs/2403.01823) và [Learning from Interventions](https://arxiv.org/abs/1808.01818).

Human can thiệp khi robot sắp thất bại và cung cấp corrective action hoặc language motion. Những đoạn này vừa bảo vệ robot, vừa tạo dữ liệu recovery mà demonstrations hoàn hảo thường thiếu.

### 4.3 Inverse Reinforcement Learning — IRL

> **Paper gốc:** [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf).

IRL không học policy trực tiếp. Nó suy ra reward function giải thích expert behavior, rồi dùng planning hoặc RL để tìm policy:

```text
expert trajectories → infer reward → optimize policy under learned reward
```

Ưu điểm là reward có thể transfer sang dynamics mới. Hạn chế là nhiều reward khác nhau có thể giải thích cùng một behavior.

### 4.4 Generative Adversarial Imitation Learning — GAIL

> **Paper gốc:** [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476).

GAIL dùng discriminator phân biệt expert transition với learner transition. Policy học để occupancy distribution của nó giống expert:

```math
\min_\pi\max_D
\mathbb E_{\pi_E}[\log D(s,a)]
+\mathbb E_\pi[\log(1-D(s,a))]
-\lambda\mathcal H(\pi)
```

Nó tránh phải khôi phục reward tường minh như IRL, nhưng training adversarial có thể bất ổn và vẫn cần environment interaction.

---

## 5. Reinforcement Learning — RL

### 5.1 MDP, value, Bellman và Temporal Difference

> **Tài liệu nền tảng:** [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).

MDP được viết:

```math
\left(\mathcal S,\mathcal A,P,R,\gamma\right)
```

Policy tối đa hóa expected discounted return:

```math
J(\pi)=\mathbb E_{\tau\sim\pi}
\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\right]
```

State-value, action-value và advantage:

```math
V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
```

```math
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]
```

```math
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
```

Temporal Difference học sau mỗi transition bằng bootstrapping:

```math
V(S_t)\leftarrow V(S_t)
+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right]
```

SARSA, Q-learning, DQN và phần lớn Actor–Critic đều dùng TD target ở một hình thức nào đó.

### 5.2 Value-based RL

Value-based methods học $V$ hoặc $Q$, sau đó suy ra action từ value. Chúng đặc biệt tự nhiên với discrete action spaces.

#### 5.2.1 SARSA

> **Technical report gốc:** [On-Line Q-Learning Using Connectionist Systems](https://www.cs.utexas.edu/~shivaram/readings/b2hd-RummeryNiranjan1994.html).

Tên SARSA đến từ tuple:

```text
State → Action → Reward → next State → next Action
  S_t     A_t      R        S_{t+1}       A_{t+1}
```

Update:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha\left[
R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
\right]
```

Điểm quyết định là $A_{t+1}$ được sample từ chính behavior policy hiện tại, chẳng hạn epsilon-greedy. Vì học value của policy đang hành động, SARSA là **on-policy**.

**Trực giác:** nếu exploration policy đôi khi đi ngẫu nhiên, SARSA tính cả rủi ro đó vào target. Trong Cliff Walking, SARSA thường học đường an toàn hơn vì nó biết epsilon-greedy vẫn có thể bước nhầm xuống vực.

**Ưu điểm:** update đơn giản, phù hợp khi behavior lúc deploy vẫn có exploration hoặc noise.

**Hạn chế:** policy cải thiện cùng dữ liệu nên có thể học chậm; tabular SARSA không scale trực tiếp tới observation lớn.

#### 5.2.2 Q-learning

> **Paper gốc:** [Q-learning — Watkins và Dayan, 1992](https://doi.org/10.1007/BF00992698).

Update:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha\left[
R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)
\right]
```

Q-learning dùng action tốt nhất theo current Q trong target, bất kể behavior policy đã chọn action nào để thu data. Vì vậy nó là **off-policy**.

**Trực giác:** behavior có thể epsilon-greedy để khám phá, nhưng target giả sử tương lai sẽ greedy.

**Ưu điểm:** học trực tiếp optimal action-value; dữ liệu có thể đến từ policy khác.

**Hạn chế:** toán tử `max` dễ gây overestimation; bảng Q không scale khi state/action space lớn.

##### SARSA và Q-learning khác nhau ở đúng một target

| Thuộc tính | SARSA | Q-learning |
|---|---|---|
| Target | Action thực sự được behavior policy chọn | Action có Q lớn nhất |
| Policy type | On-policy | Off-policy |
| Học value của | Policy đang chạy | Greedy target policy |
| Khi exploration có rủi ro | Thường thận trọng hơn | Có thể học đường sát nguy hiểm hơn |

```math
\mathrm{SARSA:}\quad
y_t=r_t+\gamma Q(s_{t+1},a_{t+1})
```

```math
\mathrm{Q\text{-}learning:}\quad
y_t=r_t+\gamma\max_aQ(s_{t+1},a)
```

#### 5.2.3 DQN — Deep Q-Network

> **Paper gốc:** [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236).

DQN thay bảng Q bằng neural network:

```math
Q_\theta(s,a)\approx Q^*(s,a)
```

Target và loss:

```math
y_t=r_t+\gamma(1-d_t)
\max_{a'}Q_{\bar\theta}(s_{t+1},a')
```

```math
\mathcal L_{\mathrm{DQN}}(\theta)
=\mathbb E_{(s,a,r,s',d)\sim\mathcal B}
\left[\left(y-Q_\theta(s,a)\right)^2\right]
```

DQN ổn định deep Q-learning bằng:

1. **Replay buffer:** phá tương quan theo thời gian và reuse transitions.
2. **Target network:** giữ target tương đối ổn định trong nhiều update.
3. **Epsilon-greedy:** cân bằng exploitation và exploration.

**Điểm mạnh:** học từ raw observation và reuse data nhờ off-policy replay.

**Hạn chế:** action phải rời rạc hoặc đủ nhỏ để tính `max`; dễ overestimate Q và nhạy với reward scale/training stability.

#### 5.2.4 Double DQN, Dueling DQN, PER và Distributional DQN

> **Paper:** [Double DQN](https://arxiv.org/abs/1509.06461) · [Dueling DQN](https://arxiv.org/abs/1511.06581) · [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) · [Distributional DQN/C51](https://arxiv.org/abs/1707.06887) · [Rainbow](https://arxiv.org/abs/1710.02298).

- **Double DQN:** online network chọn action, target network đánh giá action; giảm overestimation.
- **Dueling DQN:** tách state value và action advantage rồi kết hợp thành Q.
- **PER:** sample transition có TD error lớn thường xuyên hơn, kèm importance correction.
- **Distributional DQN:** học distribution của return thay vì chỉ expected return.
- **Rainbow:** kết hợp nhiều cải tiến DQN trong một agent.

### 5.3 Policy-based RL

Policy-based methods tối ưu trực tiếp policy parameters thay vì bắt buộc suy action từ argmax Q.

#### 5.3.1 REINFORCE

> **Paper gốc:** [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696).

```math
\nabla_\theta J(\theta)
\approx
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\left(G_t-b(s_t)\right)
```

Return tốt hơn baseline làm tăng xác suất action; return kém hơn làm giảm xác suất. Baseline giảm variance nhưng không nhất thiết là một learned critic.

**Phân loại:** model-free, on-policy, Monte Carlo policy gradient.

**Ưu điểm:** đơn giản, dùng được với stochastic policy và action rời rạc/liên tục.

**Hạn chế:** variance cao, phải đợi return và không reuse old data hiệu quả.

#### 5.3.2 TRPO — Trust Region Policy Optimization

> **Paper gốc:** [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html).

TRPO tối đa hóa surrogate objective nhưng giới hạn KL divergence giữa policy cũ và mới:

```math
\max_\theta
\mathbb E_t\left[
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\hat A_t
\right]
```

```math
\mathrm{subject\ to}\quad
\mathbb E_t\left[
D_{\mathrm{KL}}\!\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\,\|\,
\pi_\theta(\cdot\mid s_t)
\right)
\right]\le\delta
```

Trust region ngăn policy update quá xa gây performance collapse. Hạn chế là cần second-order approximation, conjugate gradient và line search; PPO ra đời để đạt mục tiêu gần giống bằng first-order optimization đơn giản hơn.

#### 5.3.3 GRPO — Group Relative Policy Optimization

> **Paper gốc:** [DeepSeekMath](https://arxiv.org/abs/2402.03300); ứng dụng cho VLA world-model RL: [WMPO](https://arxiv.org/abs/2511.09515).

GRPO sample một nhóm candidate cho cùng context rồi chuẩn hóa reward trong nhóm:

```math
\hat A_i=
\frac{R_i-\mathrm{mean}(R_1,\ldots,R_G)}
{\mathrm{std}(R_1,\ldots,R_G)+\varepsilon}
```

Policy update thường dùng clipped ratio và KL regularization. So với PPO điển hình, GRPO không bắt buộc train một value critic riêng.

**Ưu điểm:** giảm memory dành cho value model, tận dụng relative comparison.

**Hạn chế:** cần nhiều candidate cho cùng context và reward đủ đáng tin cậy.

### 5.4 Actor–Critic

Actor–Critic học đồng thời:

- **Actor:** policy $\pi_\theta(a\mid s)$.
- **Critic:** $V_\phi(s)$ hoặc $Q_\phi(s,a)$ để đánh giá action và giảm variance.

#### 5.4.1 A2C và A3C

> **Paper gốc:** [Asynchronous Methods for Deep Reinforcement Learning — A3C](https://arxiv.org/abs/1602.01783). A2C là biến thể synchronous phổ biến của cùng họ.

TD advantage:

```math
\hat A_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)
```

```math
\mathcal L_{\mathrm{actor}}
=-\mathbb E_t
\left[\log\pi_\theta(a_t\mid s_t)\hat A_t\right]
```

```math
\mathcal L_{\mathrm{critic}}
=\mathbb E_t\left[
\left(r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)\right)^2
\right]
```

- **A3C:** nhiều worker update shared parameters bất đồng bộ.
- **A2C:** rollout song song nhưng gom gradient/update đồng bộ.

Cả hai là on-policy. A2C dễ vectorize; A3C dùng asynchronous workers để decorrelate experience.

#### 5.4.2 PPO — Proximal Policy Optimization

> **Paper gốc:** [PPO](https://arxiv.org/abs/1707.06347) và [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

Policy ratio:

```math
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
```

Clipped surrogate:

```math
L^{\mathrm{clip}}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\mathrm{clip}\!\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right]
```

Implementation thường dùng actor loss, value loss và entropy bonus. Clipping làm update first-order đơn giản hơn TRPO.

**Phân loại:** on-policy policy-gradient, thường implement bằng Actor–Critic.

**Ưu điểm:** ổn định, dễ implement, dùng được với discrete/continuous action.

**Hạn chế:** sample efficiency thấp hơn off-policy methods; old data nhanh hết hiệu lực.

#### 5.4.3 DDPG — Deep Deterministic Policy Gradient

> **Paper gốc:** [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971).

DDPG học deterministic actor:

```math
a=\mu_\theta(s)
```

Actor update theo gradient của critic:

```math
\nabla_\theta J(\theta)
\approx
\mathbb E_s\left[
\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)
\right]
```

DDPG là off-policy Actor–Critic cho continuous control, dùng replay buffer và target networks. Nó sample-efficient nhưng dễ bất ổn vì critic overestimation và exploration noise.

#### 5.4.4 TD3 — Twin Delayed DDPG

> **Paper gốc:** [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html).

TD3 sửa ba điểm của DDPG:

1. Twin critics và lấy minimum.
2. Delayed actor update.
3. Target policy smoothing.

```math
y=r+\gamma(1-d)
\min_{i\in\{1,2\}}
Q_{\bar\phi_i}
\left(s',\mu_{\bar\theta}(s')+\epsilon\right)
```

**Phân loại:** model-free, off-policy, deterministic Actor–Critic, continuous action.

**Điểm mạnh:** ổn định hơn DDPG và reuse replay data.

**Hạn chế:** cần external exploration noise và vẫn phụ thuộc chất lượng critic.

#### 5.4.5 SAC — Soft Actor-Critic

> **Paper gốc:** [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html).

SAC tối đa hóa reward và entropy:

```math
J(\pi)=\mathbb E_\pi\left[
\sum_t\gamma^t
\left(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\right)
\right]
```

Soft Q target:

```math
y=r+\gamma(1-d)\left[
\min_iQ_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right]
```

SAC dùng stochastic actor, twin critics, replay buffer và thường tự tune temperature $\alpha$.

**Phân loại:** model-free, off-policy, stochastic Actor–Critic, maximum-entropy RL.

**Điểm mạnh:** sample-efficient và exploration tự nhiên.

**Hạn chế:** implementation phức tạp hơn; robot thật vẫn cần safety/reset/reward đáng tin cậy.

### 5.5 Model-based RL

Model-based RL học hoặc dùng dynamics model để planning hay tạo imagined experience.

#### 5.5.1 Dyna

> **Paper gốc:** [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](https://mlanthology.org/icml/1990/sutton1990icml-integrated/).

Dyna kết hợp real experience và simulated experience từ learned model:

```text
real transition → update value/policy
                → update dynamics model
                → sample imagined transitions
                → update value/policy thêm nhiều lần
```

Điểm cốt lõi là cùng một learning update có thể dùng cho cả real và model-generated transition. Model sai sẽ tạo model bias.

#### 5.5.2 Model Predictive Control — MPC

> **Paper tổng quan:** [A Survey of Industrial Model Predictive Control Technology](https://doi.org/10.1016/S0967-0661%2802%2900186-7).

MPC tối ưu một action sequence trong horizon hữu hạn, chỉ thực thi action đầu rồi quan sát và lập kế hoạch lại:

```math
a_{t:t+H-1}^*
=\arg\max_{a_{t:t+H-1}}
\sum_{k=0}^{H-1}\gamma^k\hat r_{t+k}
```

Receding-horizon replanning giúp sửa sai model sau mỗi observation, nhưng planning online có thể tốn compute.

#### 5.5.3 Dreamer

> **Paper nên đọc:** [Dreamer](https://arxiv.org/abs/1912.01603) và [DreamerV3](https://arxiv.org/abs/2301.04104).

Dreamer học latent world model từ pixels rồi train actor và critic trên imagined latent trajectories. Nó không cần reconstruct mọi chi tiết thế giới thật hoàn hảo; representation cần đủ tốt cho reward, continuation và control.

**Điểm mạnh:** data efficiency cao nhờ imagination.

**Hạn chế:** world-model error tích lũy có thể khiến actor khai thác imagined dynamics sai.

#### 5.5.4 MuZero

> **Paper gốc:** [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265).

MuZero không học trực tiếp raw environment dynamics. Nó học representation, dynamics và prediction functions để dự đoán những đại lượng cần cho planning: reward, value và policy; sau đó dùng tree search.

**Điểm mạnh:** planning mạnh mà không cần biết luật/dynamics thật.

**Hạn chế:** search và training system phức tạp, compute lớn.

#### 5.5.5 World-model RL cho VLA

> **Paper:** [WMPO](https://arxiv.org/abs/2511.09515) và [VLA-MBPO](https://arxiv.org/abs/2603.20607).

World model tạo imagined robot rollouts để giảm chi phí và rủi ro interaction thật. WMPO dùng pixel world model cùng on-policy GRPO; VLA-MBPO dùng chunk-level branched rollouts để hạn chế compounding model error.

Đây là node **Model-based RL**. GRPO chỉ là optimizer policy nằm trên một trục khác.

### 5.6 Offline RL

Offline RL học từ static dataset có reward mà không tiếp tục tương tác environment:

```math
\mathcal D=\left\{(s_t,a_t,r_t,s_{t+1})\right\}
```

Thách thức cốt lõi là extrapolation/distribution error: critic có thể đánh giá quá cao action ngoài support của dataset.

#### 5.6.1 BCQ — Batch-Constrained Q-learning

> **Paper gốc:** [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900).

BCQ học generative model của behavior actions rồi chỉ tối ưu trong vùng action gần dataset. Ý tưởng là tránh chọn action có Q cao giả tạo nhưng chưa từng được quan sát.

**Ưu điểm:** trực tiếp kiểm soát out-of-distribution action.

**Hạn chế:** phụ thuộc chất lượng behavior model và constraint.

#### 5.6.2 CQL — Conservative Q-Learning

> **Paper gốc:** [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).

CQL thêm penalty làm Q của action ngoài dataset thấp hơn:

```math
\mathcal L_{\mathrm{CQL}}
=\mathcal L_{\mathrm{Bellman}}
+\alpha\left(
\mathbb E_{s}\left[\log\sum_a\exp Q(s,a)\right]
-\mathbb E_{(s,a)\sim\mathcal D}[Q(s,a)]
\right)
```

**Ưu điểm:** conservative value giảm overestimation.

**Hạn chế:** quá conservative có thể bỏ lỡ action tốt và cần tune penalty.

#### 5.6.3 IQL — Implicit Q-Learning

> **Paper gốc:** [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).

IQL không explicit maximize Q trên unseen actions. Nó dùng expectile regression để học value từ dataset actions, rồi advantage-weighted regression để extract policy.

```math
\mathcal L_V
=\mathbb E_{(s,a)\sim\mathcal D}
\left[L_\tau^2\!\left(Q(s,a)-V(s)\right)\right]
```

**Ưu điểm:** tránh query critic tại OOD action trong training.

**Hạn chế:** expectile và temperature ảnh hưởng mạnh trade-off giữa imitation và improvement.

#### 5.6.4 Decision Transformer

> **Paper gốc:** [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).

Decision Transformer biểu diễn trajectory thành sequence:

```text
return-to-go, state, action, return-to-go, state, action, ...
```

Model dự đoán action có điều kiện theo desired return và history. Training objective gần supervised sequence modeling, nhưng use case là offline RL/control.

**Ưu điểm:** tận dụng Transformer và không cần Bellman backup.

**Hạn chế:** phụ thuộc coverage/chất lượng dataset và desired-return conditioning; không tự extrapolate tốt ra ngoài data.

### 5.7 Bảng so sánh toàn bộ nhánh RL

| Thuật toán | Nhánh chính | On/Off-policy | Model | Action thường gặp | Điểm nhớ nhất |
|---|---|---|---|---|---|
| SARSA | Value-based | On-policy | Model-free | Discrete | Target dùng action thực sự kế tiếp |
| Q-learning | Value-based | Off-policy | Model-free | Discrete | Target dùng max Q |
| DQN | Value-based | Off-policy | Model-free | Discrete | Q-network + replay + target network |
| REINFORCE | Policy-based | On-policy | Model-free | Cả hai | Monte Carlo policy gradient |
| TRPO | Policy-based | On-policy | Model-free | Cả hai | KL trust region |
| GRPO | Policy-based | On-policy | Model-free hoặc world-model rollout | Cả hai | Group-relative advantage |
| A2C/A3C | Actor–Critic | On-policy | Model-free | Cả hai | TD advantage; sync/async workers |
| PPO | Actor–Critic | On-policy | Model-free | Cả hai | Clipped policy ratio |
| DDPG | Actor–Critic | Off-policy | Model-free | Continuous | Deterministic actor |
| TD3 | Actor–Critic | Off-policy | Model-free | Continuous | Twin critics + delayed actor + smoothing |
| SAC | Actor–Critic | Off-policy | Model-free | Continuous | Stochastic actor + entropy |
| Dyna | Model-based | Tùy implementation | Learned model | Cả hai | Real + imagined updates |
| MPC | Planning/model-based | Không cố định | Known/learned model | Thường continuous | Replan mỗi bước |
| Dreamer | Actor–Critic + model-based | Off-policy data | Latent world model | Cả hai | Learn in imagination |
| MuZero | Planning/model-based | Self-generated data | Learned planning model | Discrete/structured | Tree search với learned reward/value/policy |
| BCQ/CQL/IQL | Offline RL | Off-policy dataset | Model-free | Cả hai | Kiểm soát OOD action/value |
| Decision Transformer | Offline sequence modeling | Offline dataset | Không dynamics model | Cả hai | Condition action on return-to-go |

---

## 6. Các trục phân loại chồng lấp

Phần 5 đi theo **nhánh thuật toán chính**. Bảng này giải thích vì sao một thuật toán vẫn có nhiều nhãn bổ sung.

| Trục | Các nhóm | Câu hỏi |
|---|---|---|
| Đối tượng học | Value-based / Policy-based / Actor–Critic | Học Q, học policy hay cả hai? |
| Dynamics | Model-free / Model-based | Có học hoặc dùng transition model không? |
| Nguồn data | Online / Offline | Agent có tiếp tục tương tác environment không? |
| Policy sinh data | On-policy / Off-policy | Data có do policy hiện tại tạo không? |
| Action space | Discrete / Continuous / Hybrid | Agent phải chọn loại action nào? |
| Quan sát | MDP / POMDP | Thấy state đầy đủ hay cần history/memory? |
| Objective | Expected return / Entropy / Constraint / Safety | Tối ưu reward thuần hay có regularization/ràng buộc? |

Vì vậy không nên hỏi “SAC thuộc đúng một nhánh nào?”. Câu đầy đủ là: SAC thuộc Actor–Critic theo đối tượng học, off-policy theo data-policy relation, model-free theo dynamics và maximum-entropy theo objective.

---

## 7. VLA nằm ở đâu và đã phát triển thế nào?

### 7.1 VLA là model/policy class, không phải một RL algorithm

> **Paper nên đọc:** [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246) và [π0](https://arxiv.org/abs/2410.24164).

VLA nhận vision, language, thường thêm proprioception/history, rồi sinh robot action:

```math
\pi_\theta
\left(a_{t:t+H-1}\mid I_{\le t},q_{\le t},l\right)
```

```text
Self-supervised / supervised VLM pre-training
                         │
Robot demonstrations ───┼──→ IL / Behavioral Cloning
                         │
                         ▼
                    VLA policy
                         │
                         ├── deploy trực tiếp
                         ├── fine-tune bằng demonstrations
                         └── post-train bằng offline/online RL
```

```math
\boxed{
\mathrm{VLA}
=\mathrm{VLM\ pretraining}
+\mathrm{robot\ IL}
+\mathrm{optional\ RL}
}
```

### 7.2 Mỗi kỹ thuật VLA nằm ở node nào trong cây?

| Kỹ thuật/model | Node trong cây | Lý do |
|---|---|---|
| [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246) | 4.1.2 Autoregressive action tokens | Học token expert action bằng supervised BC |
| [ACT](https://arxiv.org/abs/2304.13705) | 4.1.3 Action chunking | Dự đoán action sequence từ demonstrations |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137), [Octo](https://arxiv.org/abs/2405.12213) | 4.1.4 Diffusion policy | Denoising action distribution từ expert data |
| [π0](https://arxiv.org/abs/2410.24164) | 4.1.5 Flow matching | Flow-based action expert học từ robot data |
| [π0.5](https://arxiv.org/abs/2504.16054) | 3.3 + 4.1.2 + 4.1.5 | Heterogeneous co-training, FAST tokens, flow action head |
| [RT-H](https://arxiv.org/abs/2403.01823) | 4.2.2 Human correction | Language-motion hierarchy và interventions |
| [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759) | 4 IL + 5.6 Offline RL + online experience | Demonstrations, rollouts, corrections, advantage conditioning |
| [RL Token](https://www.pi.website/research/rlt) | 5.4 Off-policy Actor–Critic | Actor–critic nhỏ chỉnh action chunk của frozen VLA |
| [EXIMO](https://arxiv.org/abs/2608.19891) | 4 IL → 5.4 residual off-policy RL | Explore → imitate → optimize |
| [WMPO](https://arxiv.org/abs/2511.09515) | 5.5.5 + 5.3.3 | World model + on-policy GRPO |
| [VLA-MBPO](https://arxiv.org/abs/2603.20607) | 5.5.5 | World-model branched rollout cho VLA |

### 7.3 Timeline phát triển VLA

#### Giai đoạn 1 — imitation policy biểu diễn action tốt hơn

- [ACT](https://arxiv.org/abs/2304.13705): action chunking + CVAE cho bimanual manipulation.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137): conditional diffusion cho multimodal action trajectories.

Đây chủ yếu là IL/BC, chưa phải RL.

#### Giai đoạn 2 — generalist policy và action token

- [RT-1](https://arxiv.org/abs/2212.06817): Transformer policy học multi-task real-robot demonstrations.
- [RT-2](https://arxiv.org/abs/2307.15818): co-fine-tune VLM trên web data và robot data; action được biểu diễn như text token.

Điểm mới là transfer semantic knowledge từ VLM, nhưng robot action vẫn chủ yếu học bằng supervised imitation.

#### Giai đoạn 3 — open và cross-embodiment VLA

- [Octo](https://arxiv.org/abs/2405.12213): generalist Transformer policy với diffusion action head, train trên Open X-Embodiment.
- [OpenVLA](https://arxiv.org/abs/2406.09246): open-source VLA 7B, train trên 970k robot demonstrations bằng discretized action tokens.

#### Giai đoạn 4 — continuous generative VLA

- [π0](https://arxiv.org/abs/2410.24164): PaliGemma backbone + flow-matching action expert + high-frequency action chunks.
- [π0.5](https://arxiv.org/abs/2504.16054): heterogeneous co-training, semantic subtask prediction, [FAST](https://arxiv.org/abs/2501.09747) tokens ở pre-training và flow matching ở post-training.

Hai model này vẫn lấy IL/BC làm lõi; diffusion/flow là action modeling, không phải RL.

#### Giai đoạn 5 — VLA học từ outcome và experience

- [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759): demonstrations + autonomous rollouts + expert corrections + advantage-conditioned RL.
- [RL Token](https://www.pi.website/research/rlt): off-policy actor–critic nhỏ học nhanh từ real-robot experience.
- [EXIMO](https://arxiv.org/abs/2608.19891): VLM-guided exploration, imitation rồi residual off-policy RL.
- [WMPO](https://arxiv.org/abs/2511.09515) và [VLA-MBPO](https://arxiv.org/abs/2603.20607): dùng world model để giảm real-world interaction.

#### Giai đoạn 6 — steerability, memory và embodied reasoning

- [π0.7](https://www.pi.website/blog/pi07): diverse context conditioning, subgoal images và metadata để steer strategy.
- [Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) và [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/): embodied reasoning, hierarchical execution, whole-body control và on-device adaptation.

Các điểm này mở rộng architecture/system design; không đủ cơ sở để mặc định mọi phiên bản đều dùng một RL algorithm cụ thể.

### 7.4 Cây VLA sau khi nối lại với cây Machine Learning

```text
VLA — lớp hội tụ của nhiều nhánh
│
├── Backbone knowledge
│   └── 2–3. Supervised + Self-supervised VLM/video pre-training
│
├── Robot action learning
│   └── 4. Imitation Learning
│       ├── 4.1.2 Action tokens: RT-1, RT-2, OpenVLA, FAST
│       ├── 4.1.3 Action chunks: ACT
│       ├── 4.1.4 Diffusion: Diffusion Policy, Octo
│       ├── 4.1.5 Flow matching: π0, π0.5
│       └── 4.2 Interactive correction: DAgger, RT-H
│
├── Experience-based improvement
│   └── 5. Reinforcement Learning
│       ├── 5.6 Offline/advantage-conditioned RL: RECAP
│       ├── 5.4 Off-policy Actor–Critic: RL Token
│       ├── 5.4 Residual off-policy RL: EXIMO
│       └── 5.5 World-model RL: WMPO, VLA-MBPO
│
└── System extensions
    ├── Hierarchical planning / embodied reasoning
    ├── Memory cho POMDP và long-horizon task
    ├── Cross-embodiment adaptation
    └── Real-time/on-device inference
```

---

## 8. Cách trả lời phỏng vấn

### “SARSA khác Q-learning ở đâu?”

> SARSA là on-policy: target dùng action tiếp theo do behavior policy thực sự chọn. Q-learning là off-policy: target dùng action có Q lớn nhất, dù behavior đang exploration. Vì vậy SARSA thường phản ánh rủi ro exploration vào value, còn Q-learning học greedy target policy.

### “DQN có phải một thuật toán khác Q-learning không?”

> DQN là deep function-approximation version của Q-learning. Phần cốt lõi vẫn là Bellman optimality target có `max`; DQN bổ sung neural Q-network, replay buffer và target network để scale và ổn định training.

### “PPO thuộc Policy-based hay Actor–Critic?”

> PPO là policy-gradient method về objective và thường được implement bằng Actor–Critic với policy actor cùng value critic. Hai nhãn này không loại trừ nhau.

### “VLA dùng IL hay RL?”

> Phần lớn VLA nền tảng học robot action ban đầu bằng IL/Behavioral Cloning trên demonstrations. Token, diffusion và flow matching chỉ là cách sinh action. Các thế hệ mới bổ sung offline/online/model-based RL để học từ outcome và experience, như RECAP, RL Token, EXIMO, WMPO.

### “Tại sao VLA không dùng RL từ đầu?”

> Real-robot interaction đắt, chậm và có rủi ro; reward cho task dài cũng khó thiết kế. IL tạo skill prior từ demonstrations trước, sau đó RL cải thiện precision, recovery và throughput hiệu quả hơn.

---

## 9. Bản đồ ghi nhớ cuối cùng

```text
Có target label/action
└── Supervised Learning / Behavioral Cloning

Tự tạo target từ dữ liệu
└── Self-supervised Learning

Học từ demonstrations
└── Imitation Learning
    ├── BC: one-step / token / chunk / diffusion / flow
    ├── DAgger / intervention
    └── IRL / GAIL

Học từ reward
└── Reinforcement Learning
    ├── Value-based
    │   ├── SARSA: on-policy target
    │   ├── Q-learning: off-policy max target
    │   └── DQN: neural Q-learning
    ├── Policy-based
    │   ├── REINFORCE
    │   ├── TRPO
    │   └── GRPO
    ├── Actor–Critic
    │   ├── On-policy: A2C / A3C / PPO
    │   └── Off-policy: DDPG / TD3 / SAC
    ├── Model-based: Dyna / MPC / Dreamer / MuZero
    └── Offline: BCQ / CQL / IQL / Decision Transformer

Vision + language → robot action
└── VLA
    ├── backbone từ supervised/self-supervised pre-training
    ├── action prior chủ yếu từ IL/BC
    ├── action decoder: token/diffusion/flow
    └── optional RL post-training
```

---

## Nguồn chính theo đúng nhánh cây

### Imitation Learning

- [DAgger](https://proceedings.mlr.press/v15/ross11a.html).
- [Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf).
- [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476).
- [ACT](https://arxiv.org/abs/2304.13705).
- [Diffusion Policy](https://arxiv.org/abs/2303.04137).
- [Flow Matching](https://arxiv.org/abs/2210.02747).

### Value-based RL

- [SARSA technical report](https://www.cs.utexas.edu/~shivaram/readings/b2hd-RummeryNiranjan1994.html).
- [Q-learning](https://doi.org/10.1007/BF00992698).
- [DQN](https://www.nature.com/articles/nature14236).
- [Double DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), [PER](https://arxiv.org/abs/1511.05952), [C51](https://arxiv.org/abs/1707.06887), [Rainbow](https://arxiv.org/abs/1710.02298).

### Policy-based và Actor–Critic

- [REINFORCE](https://doi.org/10.1007/BF00992696).
- [TRPO](https://proceedings.mlr.press/v37/schulman15.html).
- [A3C](https://arxiv.org/abs/1602.01783).
- [PPO](https://arxiv.org/abs/1707.06347).
- [DDPG](https://arxiv.org/abs/1509.02971).
- [TD3](https://proceedings.mlr.press/v80/fujimoto18a.html).
- [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html).
- [GRPO](https://arxiv.org/abs/2402.03300).

### Model-based và Offline RL

- [Dyna](https://mlanthology.org/icml/1990/sutton1990icml-integrated/).
- [DreamerV3](https://arxiv.org/abs/2301.04104).
- [MuZero](https://arxiv.org/abs/1911.08265).
- [BCQ](https://arxiv.org/abs/1812.02900), [CQL](https://arxiv.org/abs/2006.04779), [IQL](https://arxiv.org/abs/2110.06169), [Decision Transformer](https://arxiv.org/abs/2106.01345).

### Robot policy và VLA

- [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [RT-H](https://arxiv.org/abs/2403.01823).
- [Octo](https://arxiv.org/abs/2405.12213), [OpenVLA](https://arxiv.org/abs/2406.09246).
- [π0](https://arxiv.org/abs/2410.24164), [π0.5](https://arxiv.org/abs/2504.16054), [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759), [π0.7](https://www.pi.website/blog/pi07).
- [RL Token](https://www.pi.website/research/rlt), [EXIMO](https://arxiv.org/abs/2608.19891), [WMPO](https://arxiv.org/abs/2511.09515), [VLA-MBPO](https://arxiv.org/abs/2603.20607).
