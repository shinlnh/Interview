# 3.7 Cây Machine Learning: Imitation Learning, Reinforcement Learning và VLA

[← Mục lục Machine Learning and Mathematics](./README.md)

> Phần này mô tả bức tranh của lĩnh vực tính đến **tháng 9/2026**. Tên thuật toán tiếng Anh được giữ lại để tiện đọc paper. Cây bên dưới là bản mở rộng từ sơ đồ gốc, đồng thời phân biệt rõ **learning paradigm**, **RL algorithm**, **policy architecture** và **action representation**.

## 1. Cây tổng quát

> **Paper định hướng theo cây:** [DAgger — Imitation Learning](https://proceedings.mlr.press/v15/ross11a.html), [DQN — Value-based RL](https://www.nature.com/articles/nature14236), [REINFORCE — Policy Gradient](https://doi.org/10.1007/BF00992696), [A3C — Actor–Critic](https://arxiv.org/abs/1602.01783) và [RT-2 — Vision-Language-Action](https://arxiv.org/abs/2307.15818).

```text
Machine Learning
│
├── Supervised Learning
│   ├── Classification
│   ├── Regression
│   └── Sequence prediction / representation fine-tuning
│
├── Unsupervised / Self-supervised Learning
│   ├── Representation learning
│   ├── Contrastive learning
│   └── Masked / next-token / vision-language pre-training
│
├── Imitation Learning — IL
│   ├── Behavioral Cloning — BC
│   │   ├── One-step action prediction
│   │   ├── Autoregressive action tokens
│   │   ├── Action chunking
│   │   ├── Diffusion policy
│   │   └── Flow-matching policy
│   ├── Interactive IL
│   │   ├── DAgger
│   │   └── Human intervention / correction
│   └── Learning objective from demonstrations
│       ├── Inverse Reinforcement Learning — IRL
│       └── Generative Adversarial Imitation Learning — GAIL
│
└── Reinforcement Learning — RL
    ├── Value-based
    │   ├── SARSA / Q-learning
    │   └── DQN and its extensions
    ├── Policy-based
    │   ├── REINFORCE
    │   ├── TRPO
    │   └── GRPO
    ├── Actor–Critic
    │   ├── On-policy: A2C / A3C / PPO
    │   └── Off-policy: DDPG / TD3 / SAC
    ├── Model-based RL
    │   ├── Dyna / MPC
    │   └── Dreamer / MuZero / world-model RL
    └── Offline RL
        ├── BCQ / CQL / IQL
        └── Sequence-modeling approaches
```

### VLA nằm ở đâu trong cây?

> **Paper nên đọc:** [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246) và [π0](https://arxiv.org/abs/2410.24164).

**VLA — Vision-Language-Action — không phải một nhánh học loại trừ lẫn nhau với IL hoặc RL.** VLA là một loại multimodal robot policy/foundation model. Quá trình tạo ra một VLA thường lấy thành phần từ nhiều nhánh:

```text
Self-supervised / supervised pre-training
image + text + video trên Internet
                 │
                 ▼
        VLM/LLM backbone có kiến thức ngữ nghĩa
                 │
                 ├───────────────┐
                 │               │
Robot demonstrations            │
        │                        │
        ▼                        │
Imitation Learning / BC          │
        │                        │
        └───────────────┬────────┘
                        ▼
         Vision-Language-Action policy
                        │
                        ├── dùng trực tiếp
                        ├── fine-tune bằng demonstration
                        └── post-train bằng offline/online RL
```

Nói ngắn gọn:

```math
\boxed{\mathrm{VLA}=\mathrm{VLM\ pretraining}+\mathrm{robot\ imitation\ learning}+\mathrm{optional\ RL}}
```

`Autoregressive`, `action chunking`, `diffusion` và `flow matching` mô tả **cách policy biểu diễn hoặc sinh action**. Chúng không tự động biến policy thành RL. Nếu model học action bằng cách khớp demonstrations mà không tối đa hóa reward, nó vẫn chủ yếu thuộc Behavioral Cloning.

---

## 2. Ba learning paradigm cần phân biệt

### 2.1 Supervised Learning

> **Paper nên đọc:** [ImageNet Classification with Deep Convolutional Neural Networks — AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), một ví dụ kinh điển của supervised classification bằng deep learning.

Mỗi input có target label rõ ràng. Classification dự đoán nhãn rời rạc; regression dự đoán giá trị liên tục.

```math
\mathcal L_{\mathrm{sup}}(\theta)
=\frac{1}{N}\sum_{i=1}^{N}\ell\!\left(f_\theta(x_i),y_i\right)
```

Behavioral Cloning có hình thức gần supervised learning: observation và instruction là input, còn expert action là target. Điểm khác là dữ liệu đến từ trajectory tuần tự và prediction của policy sẽ làm thay đổi observation mà nó gặp về sau.

### 2.2 Unsupervised và Self-supervised Learning

> **Paper nên đọc:** [BERT](https://arxiv.org/abs/1810.04805) cho masked language modeling và [CLIP](https://arxiv.org/abs/2103.00020) cho image-text contrastive pre-training.

Self-supervised learning tạo supervision từ chính dữ liệu, chẳng hạn che token rồi dự đoán token bị che, dự đoán token kế tiếp hoặc contrast hai view của cùng một ảnh.

Trong VLA, nhánh này cung cấp backbone có hiểu biết về vật thể, ngôn ngữ và quan hệ không gian trước khi model học điều khiển robot. Đây là lý do RT-2, OpenVLA, π0 và các model tương tự không bắt đầu hoàn toàn từ số không.

### 2.3 Imitation Learning và Reinforcement Learning

> **Paper đối chiếu:** [DAgger](https://proceedings.mlr.press/v15/ross11a.html) đại diện cho imitation learning có expert feedback; [DQN](https://www.nature.com/articles/nature14236) đại diện cho reinforcement learning từ reward và interaction.

| Câu hỏi | Imitation Learning | Reinforcement Learning |
|---|---|---|
| Tín hiệu học chính | Expert action hoặc expert trajectory | Reward/return từ outcome |
| Mục tiêu trực tiếp | Bắt chước expert | Tối đa hóa expected return |
| Có cần trial-and-error? | Không bắt buộc | Thường có, trực tiếp hoặc qua dataset/world model |
| Có thể học từ static dataset? | Có | Có với offline RL |
| Có thể dùng online interaction? | Có với DAgger/intervention | Có với online RL |
| Lỗi thường gặp | Covariate shift, bắt chước cả lỗi trong data | Reward design, exploration, sample cost, instability |

---

## 3. Imitation Learning — nhánh chính của phần lớn VLA nền tảng

### 3.1 Behavioral Cloning — BC

> **Paper nên đọc:** [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html). Paper này mô tả supervised imitation/Behavioral Cloning làm baseline và phân tích vấn đề compounding error.

Cho demonstration dataset:

```math
\mathcal D_E=\left\{(o_t,l,a_t)\right\}
```

trong đó quan sát là $o_t$, language instruction là $l$, và expert action là $a_t$. BC học policy bằng maximum likelihood:

```math
\mathcal L_{\mathrm{BC}}(\theta)
=-\mathbb E_{(o,l,a)\sim\mathcal D_E}
\left[\log \pi_\theta(a\mid o,l)\right]
```

Nếu action liên tục và policy chỉ dự đoán một mean với Gaussian variance cố định, loss thường rút về MSE:

```math
\mathcal L_{\mathrm{BC}}(\theta)
=\mathbb E_{(o,l,a)\sim\mathcal D_E}
\left[\left\|a-f_\theta(o,l)\right\|_2^2\right]
```

**Ưu điểm:** đơn giản, ổn định, tận dụng tốt teleoperation data và không cần thiết kế reward.

**Nhược điểm:** model chỉ được huấn luyện trên state do expert tạo ra. Khi tự chạy, một lỗi nhỏ có thể đưa robot sang state lạ, làm lỗi tiếp tục tích lũy. Đây là **covariate shift** hay **compounding error**.

### 3.2 Autoregressive action tokenization

> **Paper nên đọc:** [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818) và [OpenVLA](https://arxiv.org/abs/2406.09246).

Action liên tục được lượng tử hóa thành token, rồi model dự đoán chuỗi action giống next-token prediction:

```math
p_\theta(a_{1:K}\mid o,l)
=\prod_{k=1}^{K}p_\theta(a_k\mid o,l,a_{\lt k})
```

RT-1, RT-2 và OpenVLA là các ví dụ nổi bật của hướng action-token prediction. Đây vẫn là BC/supervised action prediction nếu target là action từ demonstrations.

### 3.3 Action chunking

> **Paper nên đọc:** [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — ACT](https://arxiv.org/abs/2304.13705).

Thay vì dự đoán một action, policy dự đoán cả đoạn:

```math
A_t=\left(a_t,a_{t+1},\ldots,a_{t+H-1}\right)
```

Action chunking giúp biểu diễn motion có cấu trúc theo thời gian, giảm số lần phải gọi model lớn và thường làm hành động mượt hơn. ACT và π0 đều dùng action chunk, nhưng dùng cách modeling khác nhau.

### 3.4 Diffusion Policy

> **Paper nên đọc:** [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) và [Octo](https://arxiv.org/abs/2405.12213) cho diffusion action head trong generalist robot policy.

Diffusion policy học khử noise để biến noise thành action trajectory có điều kiện theo observation. Nó phù hợp với action distribution đa mode: cùng một task có thể có nhiều trajectory hợp lệ.

Một dạng diffusion objective đơn giản là:

```math
\mathcal L_{\mathrm{diff}}
=\mathbb E_{A,\epsilon,k}
\left[\left\|\epsilon-\epsilon_\theta(A^{(k)},o,k)\right\|_2^2\right]
```

Nếu training data vẫn là expert demonstrations, Diffusion Policy thuộc nhánh **BC → generative action modeling**, không phải RL.

### 3.5 Flow-matching policy

> **Paper nên đọc:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) cho nền tảng toán học và [π0](https://arxiv.org/abs/2410.24164) cho ứng dụng vào VLA action expert.

Flow matching học vector field để vận chuyển một sample noise thành action chunk. π0 kết hợp pretrained VLM với một **action expert** được huấn luyện bằng flow matching.

```math
\mathcal L_{\mathrm{FM}}
=\mathbb E_{A,\epsilon,\tau}
\left[\left\|v_\theta(A^\tau,o,\tau)-u(A,\epsilon)\right\|_2^2\right]
```

Flow matching liên quan đến generative modeling và diffusion, nhưng không phải một RL algorithm. Classification đúng của π0 gốc là **VLA + pretrained VLM + imitation learning + flow-matching action generation**.

### 3.6 Interactive IL: DAgger và human correction

> **Paper nên đọc:** [DAgger](https://proceedings.mlr.press/v15/ross11a.html) và [RT-H: Action Hierarchies Using Language](https://arxiv.org/abs/2403.01823) cho language intervention/correction trong robot policy.

DAgger cho policy chạy, hỏi expert action tại chính những state policy gặp, rồi gộp dữ liệu mới vào dataset:

```text
train BC → rollout learner → expert relabels visited states
         → aggregate dataset → train again
```

Nó giảm distribution mismatch giữa expert data và learner rollout. Với robot thật, expert có thể can thiệp khi policy sắp thất bại; những đoạn correction vừa tăng an toàn vừa tạo dữ liệu recovery quan trọng.

### 3.7 IRL và GAIL

> **Paper nên đọc:** [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) và [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476).

- **Inverse Reinforcement Learning — IRL:** suy ra reward function giải thích expert behavior, rồi dùng RL để tối ưu reward đó.
- **GAIL:** dùng discriminator để phân biệt expert trajectory với learner trajectory; policy học để làm hai occupancy distribution giống nhau.

Hai hướng này nằm giữa imitation và RL vì demonstration cung cấp tín hiệu, nhưng policy improvement thường cần optimization theo reward/discriminator.

---

## 4. Nền tảng Reinforcement Learning

### 4.1 Từ Markov Chain đến MDP và POMDP

> **Paper nên đọc:** [Planning and Acting in Partially Observable Stochastic Domains](https://doi.org/10.1016/S0004-3702%2898%2900023-X) cho POMDP; phần MDP/RL nền tảng có trong [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).

```text
Markov Chain
state + transition
      │ thêm reward
      ▼
Markov Reward Process — MRP
      │ thêm action và quyền quyết định
      ▼
Markov Decision Process — MDP
      │ state thật bị ẩn, chỉ thấy observation
      ▼
Partially Observable MDP — POMDP
```

MDP thường được viết:

```math
\left(\mathcal S,\mathcal A,P,R,\gamma\right)
```

Policy chọn action theo state:

```math
\pi(a\mid s)=P(A_t=a\mid S_t=s)
```

và tối đa hóa expected discounted return:

```math
J(\pi)=\mathbb E_{\tau\sim\pi}
\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\right]
```

Robot thực tế thường gần POMDP vì camera bị occlusion, sensor có noise và state vật lý không được quan sát đầy đủ. Policy khi đó cần history, recurrent state hoặc memory:

```math
\pi(a_t\mid o_{\le t},a_{\lt t})
```

### 4.2 Value function, Q-function và advantage

> **Paper nên đọc:** [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

```math
V^\pi(s)=\mathbb E_\pi\left[G_t\mid S_t=s\right]
```

```math
Q^\pi(s,a)=\mathbb E_\pi\left[G_t\mid S_t=s,A_t=a\right]
```

```math
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
```

Advantage trả lời: action này tốt hơn mức trung bình của policy tại state hiện tại bao nhiêu?

### 4.3 Các trục phân loại RL chồng lấp nhau

> **Paper tổng quan:** [Reinforcement Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10166) và [A Survey on Offline Reinforcement Learning](https://arxiv.org/abs/2203.01387).

| Trục | Nhóm | Ý nghĩa |
|---|---|---|
| Đối tượng học | Value-based / Policy-based / Actor–Critic | Học value, policy hay cả hai |
| Dynamics | Model-free / Model-based | Có học hoặc dùng transition model không |
| Nguồn dữ liệu | Online / Offline | Có tiếp tục tương tác môi trường không |
| Policy sinh dữ liệu | On-policy / Off-policy | Dữ liệu có do policy hiện tại tạo không |
| Action | Discrete / Continuous / Hybrid | Không gian hành động của task |
| Quan sát | MDP / POMDP | Thấy state đầy đủ hay chỉ observation/history |

Một thuật toán có thể nằm trên nhiều trục. SAC chẳng hạn là model-free, off-policy, Actor–Critic, maximum-entropy RL và thường dùng cho continuous control.

---

## 5. Chi tiết các thuật toán trong cây

### 5.1 DQN — Deep Q-Network

> **Paper gốc:** [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236).

**Nhánh:** Value-based → model-free → off-policy → discrete action.

DQN dùng neural network để xấp xỉ action-value function:

```math
Q_\theta(s,a)\approx Q^*(s,a)
```

Target Bellman của một transition là:

```math
y_t=r_t+\gamma(1-d_t)\max_{a'}Q_{\bar\theta}(s_{t+1},a')
```

Loss:

```math
\mathcal L_{\mathrm{DQN}}(\theta)
=\mathbb E_{(s,a,r,s',d)\sim\mathcal B}
\left[\left(y-Q_\theta(s,a)\right)^2\right]
```

DQN ổn định deep Q-learning nhờ hai cơ chế chính:

1. **Replay buffer:** lưu transition rồi sample minibatch, giảm tương quan theo thời gian và cho phép tái sử dụng dữ liệu.
2. **Target network:** dùng tham số cũ $\bar\theta$ để tạo target, tránh target thay đổi quá nhanh cùng online network.

Khi rollout, agent thường dùng epsilon-greedy: phần lớn chọn action có Q cao nhất, đôi lúc chọn ngẫu nhiên để exploration.

**Điểm mạnh:** hiệu quả với discrete action, có thể reuse experience.

**Hạn chế:** `max` dễ gây overestimation; khó áp dụng trực tiếp cho continuous action vì không thể liệt kê mọi action. Double DQN, Dueling DQN, Prioritized Replay và Distributional DQN là các mở rộng phổ biến.

### 5.2 REINFORCE

> **Paper gốc:** [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696).

**Nhánh:** Policy-based → model-free → on-policy → Monte Carlo policy gradient.

REINFORCE tối ưu trực tiếp stochastic policy. Với return $G_t$, gradient estimator là:

```math
\nabla_\theta J(\theta)
\approx
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\left(G_t-b(s_t)\right)
```

Trực giác: nếu return tốt hơn baseline, tăng xác suất action đã chọn; nếu kém hơn, giảm xác suất đó. Baseline không làm gradient bị bias nếu không phụ thuộc vào action, nhưng giúp giảm variance.

**Điểm mạnh:** đơn giản, áp dụng được cho stochastic policy và cả action rời rạc lẫn liên tục.

**Hạn chế:** phải đợi return, variance cao, on-policy nên không reuse data cũ hiệu quả. Actor–Critic ra đời để thay Monte Carlo return bằng critic có bootstrap, đổi một phần variance lấy bias.

### 5.3 A2C và A3C

> **Paper gốc:** [Asynchronous Methods for Deep Reinforcement Learning — A3C](https://arxiv.org/abs/1602.01783). A2C là biến thể synchronous được dùng phổ biến của cùng họ Advantage Actor–Critic, không có một paper khai sinh riêng được thống nhất như A3C.

**Nhánh:** Actor–Critic → model-free → on-policy.

- **Actor** là policy $\pi_\theta(a\mid s)$.
- **Critic** estimate $V_\phi(s)$ hoặc $Q_\phi(s,a)$.

One-step TD advantage thường dùng:

```math
\hat A_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)
```

Actor loss:

```math
\mathcal L_{\mathrm{actor}}
=-\mathbb E_t
\left[\log\pi_\theta(a_t\mid s_t)\hat A_t\right]
```

Critic loss:

```math
\mathcal L_{\mathrm{critic}}
=\mathbb E_t
\left[\left(r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)\right)^2\right]
```

Khác nhau chính:

- **A3C — Asynchronous Advantage Actor–Critic:** nhiều worker chạy environment và update shared parameters bất đồng bộ.
- **A2C — Advantage Actor–Critic:** thường thu thập rollout song song nhưng gom gradient/update đồng bộ.

A3C dùng diversity giữa các worker để decorrelate experience. A2C dễ vectorize trên accelerator và dễ tái lập hơn. Cả hai vẫn là on-policy và nhạy với advantage estimation.

### 5.4 PPO — Proximal Policy Optimization

> **Paper gốc:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Phần GAE dùng trong PPO nên đọc thêm [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

**Nhánh thực dụng:** Actor–Critic → model-free → on-policy. Về mặt objective, PPO là một policy-gradient method.

Policy ratio:

```math
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
```

Clipped surrogate objective:

```math
L^{\mathrm{clip}}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\mathrm{clip}\!\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right]
```

Nếu policy mới thay đổi quá mạnh, ratio bị clip nên update không còn được hưởng lợi từ việc đi xa hơn. Implementation thường cộng thêm value loss và entropy bonus:

```math
\mathcal L_{\mathrm{PPO}}
=-L^{\mathrm{clip}}
+c_v\mathcal L_V
-c_e\mathcal H(\pi_\theta)
```

Generalized Advantage Estimation — GAE thường được dùng để cân bằng bias và variance:

```math
\hat A_t^{\mathrm{GAE}}
=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}
```

**Điểm mạnh:** khá ổn định, dễ implement, dùng được cho discrete và continuous action.

**Hạn chế:** chủ yếu on-policy; dù mỗi batch được update nhiều epoch, dữ liệu cũ nhanh hết hiệu lực. Robot thật vì vậy có thể tốn nhiều interaction hơn off-policy methods.

### 5.5 DDPG — cầu nối đến TD3

> **Paper gốc:** [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971).

**Nhánh:** Actor–Critic → model-free → off-policy → deterministic continuous control.

DDPG học deterministic actor:

```math
a=\mu_\theta(s)
```

và critic $Q_\phi(s,a)$. Actor được update để chọn action mà critic đánh giá cao:

```math
\nabla_\theta J(\theta)
\approx
\mathbb E_s\left[
\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)
\right]
```

DDPG dùng replay buffer và target networks nhưng dễ bất ổn do critic overestimation. TD3 trực tiếp xử lý các vấn đề này.

### 5.6 TD3 — Twin Delayed DDPG

> **Paper gốc:** [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html).

**Nhánh:** Actor–Critic → model-free → off-policy → deterministic continuous control.

TD3 thêm ba cơ chế vào DDPG:

1. **Twin critics:** học hai Q-network và lấy minimum để giảm overestimation.
2. **Delayed policy update:** critic update thường xuyên hơn actor.
3. **Target policy smoothing:** thêm noise nhỏ vào target action để policy không khai thác những đỉnh Q giả, quá hẹp.

Target:

```math
y=r+\gamma(1-d)
\min_{i\in\{1,2\}}
Q_{\bar\phi_i}
\left(s',\mu_{\bar\theta}(s')+\epsilon\right)
```

với clipped noise:

```math
\epsilon\sim
\mathrm{clip}\!\left(\mathcal N(0,\sigma^2),-c,c\right)
```

**Điểm mạnh:** sample-efficient hơn on-policy RL, hợp continuous control và thường ổn định hơn DDPG.

**Hạn chế:** deterministic policy cần exploration noise bên ngoài; vẫn phụ thuộc mạnh vào độ chính xác của critic và hyperparameters.

### 5.7 SAC — Soft Actor-Critic

> **Paper gốc:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html).

**Nhánh:** Actor–Critic → model-free → off-policy → stochastic continuous control → maximum-entropy RL.

SAC tối đa hóa cả reward và entropy:

```math
J(\pi)=
\mathbb E_\pi\left[
\sum_t\gamma^t
\left(r_t+\alpha\mathcal H\!\left(\pi(\cdot\mid s_t)\right)\right)
\right]
```

Soft Q target thường dùng twin critics:

```math
y=r+\gamma(1-d)
\left[
\min_{i\in\{1,2\}}Q_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right]
```

Actor objective:

```math
J_\pi(\theta)=
\mathbb E_{s\sim\mathcal B,\,a\sim\pi_\theta}
\left[
\alpha\log\pi_\theta(a\mid s)
-\min_iQ_{\phi_i}(s,a)
\right]
```

Temperature $\alpha$ điều khiển trade-off giữa reward và entropy, thường có thể tự động điều chỉnh theo target entropy.

**Điểm mạnh:** reuse replay data, exploration tự nhiên nhờ stochastic policy, mạnh và khá ổn định cho continuous control.

**Hạn chế:** actor, hai critics và temperature làm implementation phức tạp hơn; real-robot RL vẫn cần safety guard, reset strategy và reward đáng tin cậy.

### 5.8 GRPO — Group Relative Policy Optimization

> **Paper gốc:** [DeepSeekMath](https://arxiv.org/abs/2402.03300), paper giới thiệu GRPO; ứng dụng world-model GRPO cho VLA được trình bày trong [WMPO](https://arxiv.org/abs/2511.09515).

**Nhánh:** Policy-based → model-free → on-policy. Khi rollout nằm trong learned world model, toàn hệ thống lại thuộc model-based RL.

GRPO sample một nhóm trajectory/candidate cho cùng context, rồi chuẩn hóa reward trong nhóm để tạo relative advantage mà không bắt buộc học value critic riêng:

```math
\hat A_i=
\frac{R_i-\mathrm{mean}(R_1,\ldots,R_G)}
{\mathrm{std}(R_1,\ldots,R_G)+\varepsilon}
```

Policy update thường dùng clipped ratio gần PPO và thêm regularization để policy không trôi quá xa reference policy. GRPO xuất phát nổi bật trong reasoning model post-training; với VLA, WMPO dùng world-model rollout cùng on-policy GRPO để giảm interaction trên robot thật.

**Điểm mạnh:** không cần train một value model lớn, tận dụng so sánh tương đối giữa nhiều rollout.

**Hạn chế:** cần nhiều candidate/rollout cho cùng context và reward đủ tin cậy; chất lượng world model sẽ giới hạn policy nếu training trên imagined experience.

### 5.9 So sánh nhanh

> **Paper tra cứu theo bảng:** [DQN](https://www.nature.com/articles/nature14236) · [REINFORCE](https://doi.org/10.1007/BF00992696) · [A3C](https://arxiv.org/abs/1602.01783) · [PPO](https://arxiv.org/abs/1707.06347) · [GRPO](https://arxiv.org/abs/2402.03300) · [TD3](https://proceedings.mlr.press/v80/fujimoto18a.html) · [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html).

| Thuật toán | Học gì? | On/Off-policy | Action | Điểm nhớ nhất |
|---|---|---|---|---|
| DQN | Q-network | Off-policy | Discrete | Replay buffer + target network |
| REINFORCE | Stochastic policy | On-policy | Cả hai | Monte Carlo policy gradient, variance cao |
| A2C/A3C | Actor + value critic | On-policy | Cả hai | TD advantage; sync so với async workers |
| PPO | Actor + value critic | On-policy | Cả hai | Clipped policy ratio |
| GRPO | Policy + group-relative advantage | On-policy | Cả hai | So sánh reward trong nhóm, không bắt buộc value critic |
| TD3 | Deterministic actor + twin Q | Off-policy | Continuous | Twin critics + delayed actor + smoothing |
| SAC | Stochastic actor + twin soft Q | Off-policy | Thường continuous | Reward + entropy, sample-efficient |

> PPO nằm dưới Actor–Critic trong cây vì implementation chuẩn dùng actor và value critic. Tuy nhiên, cũng đúng khi gọi PPO là policy-gradient algorithm. Các taxonomy này chồng lấp, không loại trừ nhau.

---

## 6. Các nhánh cần thêm để mô tả VLA hiện đại

Cây gốc của em đúng cho phần nhập môn, nhưng VLA hiện đại cần thêm các nhánh sau.

### 6.1 Offline RL

> **Paper nên đọc:** [BCQ](https://arxiv.org/abs/1812.02900), [CQL](https://arxiv.org/abs/2006.04779), [IQL](https://arxiv.org/abs/2110.06169) và [Decision Transformer](https://arxiv.org/abs/2106.01345).

Offline RL học từ static transition dataset có reward:

```math
\mathcal D=\left\{(s_t,a_t,r_t,s_{t+1})\right\}
```

Khác BC, offline RL không bắt buộc bắt chước mọi action. Nó có thể ưu tiên action có outcome tốt hơn. Thách thức lớn nhất là **distribution shift**: critic có thể đánh giá quá cao action ngoài dataset.

- **BCQ:** giới hạn action gần support của dataset.
- **CQL:** học Q-value bảo thủ để giảm overestimation với unseen actions.
- **IQL:** tránh explicit maximization trên out-of-distribution actions và dùng expectile value regression.
- **Decision Transformer:** biến trajectory thành sequence modeling có điều kiện theo return; thường được xếp vào offline RL theo use case, dù objective gần supervised sequence modeling.

RECAP/π*0.6 dùng offline RL trong pre-training rồi tiếp tục thu thập on-robot experience.

### 6.2 Model-based RL và world models

> **Paper nên đọc:** [DreamerV3](https://arxiv.org/abs/2301.04104), [MuZero](https://arxiv.org/abs/1911.08265), [WMPO](https://arxiv.org/abs/2511.09515) và [VLA-MBPO](https://arxiv.org/abs/2603.20607).

Model-based RL học hoặc dùng dynamics model để dự đoán tương lai, planning hoặc tạo imagined rollout. Với VLA, world model có thể giảm số trial tốn kém trên robot thật.

```math
\hat s_{t+1},\hat r_t=f_\psi(s_t,a_t)
```

Các hướng VLA gần đây như WMPO hoặc VLA-MBPO đưa world-model rollout vào policy optimization. WMPO kết hợp world model với on-policy GRPO. Chúng nằm dưới **RL → Model-based RL**, không phải một loại BC mới.

### 6.3 Hierarchical và language-conditioned control

> **Paper nên đọc:** [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction](https://doi.org/10.1016/S0004-3702%2899%2900052-1), [RT-H](https://arxiv.org/abs/2403.01823) và [π0.5](https://arxiv.org/abs/2504.16054).

Task dài thường được tách thành:

```text
high-level planner / embodied reasoning
             │ sinh subgoal hoặc language plan
             ▼
low-level VLA policy
             │ sinh action chunk
             ▼
robot controller
```

Planner có thể là VLM/LLM, learned high-level policy hoặc search/planning module. Low-level VLA có thể vẫn được học bằng BC. Vì vậy có planner không đồng nghĩa với dùng RL.

### 6.4 Hybrid IL → RL

> **Paper nên đọc:** [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759), [RL Token](https://www.pi.website/download/rlt.pdf) và [EXIMO](https://arxiv.org/abs/2608.19891).

Đây là pipeline ngày càng quan trọng trong robotics:

```text
large demonstration dataset
          │
          ▼
Behavioral Cloning pre-training
          │ broad skill prior
          ▼
offline RL / preference or advantage conditioning
          │
          ▼
online RL + interventions + replay
          │ task-specific precision and recovery
          ▼
deployment policy
```

IL giải quyết exploration ban đầu và cung cấp skill prior; RL học từ success/failure để vượt qua chất lượng của việc bắt chước thuần túy.

---

## 7. VLA là gì và đã phát triển như thế nào?

### 7.1 Định nghĩa

> **Paper nên đọc:** [RT-2](https://arxiv.org/abs/2307.15818) cho định nghĩa VLA theo action token, [OpenVLA](https://arxiv.org/abs/2406.09246) cho open-source VLA và [π0](https://arxiv.org/abs/2410.24164) cho continuous flow-based VLA.

VLA nhận vision, language và thường cả proprioception/history để sinh robot action:

```math
\pi_\theta
\left(a_{t:t+H-1}\mid I_{\le t},q_{\le t},l\right)
```

trong đó:

- $I$ là ảnh hoặc video quan sát.
- $q$ là robot state/proprioception.
- $l$ là language instruction.
- $a_{t:t+H-1}$ là một action hoặc action chunk.

VLA là **model/policy class**. BC, offline RL, PPO, SAC hay residual RL là **cách huấn luyện/cải thiện policy**. Autoregressive token, diffusion và flow matching là **cách biểu diễn action distribution**.

### 7.2 Dòng phát triển chính

#### Giai đoạn 1 — imitation policy và action chunking

> **Paper:** [ACT](https://arxiv.org/abs/2304.13705) · [Diffusion Policy](https://arxiv.org/abs/2303.04137).

Các robot policy ban đầu chủ yếu học từ teleoperation demonstrations. ACT cho thấy Transformer có thể dự đoán action chunk cho bimanual manipulation. Diffusion Policy dùng conditional diffusion để biểu diễn action trajectory đa mode. Cả hai chủ yếu thuộc IL/BC.

#### Giai đoạn 2 — generalist robot policy và action token

> **Paper:** [RT-1](https://arxiv.org/abs/2212.06817) · [RT-2](https://arxiv.org/abs/2307.15818).

**[RT-1 (2022)](https://arxiv.org/abs/2212.06817)** học nhiều task từ large real-robot demonstration dataset. Model token hóa action và dự đoán action token bằng supervised imitation learning.

**[RT-2 (2023)](https://arxiv.org/abs/2307.15818)** biến pretrained VLM thành VLA bằng co-fine-tuning trên web-scale vision-language data và robot data. Robot action được biểu diễn như token text. Đây vẫn chủ yếu là supervised/BC, không phải PPO hay SAC.

#### Giai đoạn 3 — open cross-embodiment VLA

> **Paper:** [Octo](https://arxiv.org/abs/2405.12213) · [OpenVLA](https://arxiv.org/abs/2406.09246).

**[Octo (2024)](https://arxiv.org/abs/2405.12213)** train Transformer policy trên Open X-Embodiment để dễ fine-tune sang sensor, robot và action space mới. Octo dùng diffusion action head cho continuous action.

**[OpenVLA (2024)](https://arxiv.org/abs/2406.09246)** kết hợp pretrained visual encoders và LLM rồi train trên 970k robot demonstrations. Model dự đoán discretized action tokens, hỗ trợ fine-tune bằng LoRA và có thể phục vụ ở dạng quantized. Nhánh chính vẫn là VLM pretraining + BC.

#### Giai đoạn 4 — generative continuous-action VLA

> **Paper:** [π0](https://arxiv.org/abs/2410.24164) · [π0.5](https://arxiv.org/abs/2504.16054) · [FAST](https://arxiv.org/abs/2501.09747).

**[π0 (2024)](https://arxiv.org/abs/2410.24164)** dùng pretrained PaliGemma VLM và action expert sinh high-frequency continuous action chunks bằng flow matching. Model được train trên hơn 10.000 giờ robot data đa embodiment. π0 gốc là imitation learning/generative policy, chưa phải RL algorithm.

**[π0.5 (2025)](https://arxiv.org/abs/2504.16054)** mở rộng π0 bằng heterogeneous co-training: web data, multi-robot data, object detection, high-level semantic subtask prediction và low-level actions. Giai đoạn pre-training biểu diễn robot action bằng [FAST](https://arxiv.org/abs/2501.09747) discrete tokens; giai đoạn post-training dùng flow matching cho continuous action. Điểm chính là open-world generalization và knowledge transfer; training cốt lõi vẫn là supervised/IL co-training.

#### Giai đoạn 5 — VLA học từ experience bằng RL

> **Paper:** [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759) · [RL Token](https://www.pi.website/download/rlt.pdf) · [EXIMO](https://arxiv.org/abs/2608.19891) · [WMPO](https://arxiv.org/abs/2511.09515) · [VLA-MBPO](https://arxiv.org/abs/2603.20607).

**[RECAP và π*0.6 (2025)](https://arxiv.org/abs/2511.14759)** đưa RL vào rõ ràng. RECAP dùng demonstrations, on-policy rollouts và expert corrections; generalist VLA được pre-train bằng offline RL rồi chuyên biệt hóa bằng on-robot data collection. Advantage conditioning giúp policy phân biệt behavior tốt và kém thay vì bắt chước tất cả như BC.

**[RL Token (2026)](https://www.pi.website/download/rlt.pdf)** thêm một compact representation làm interface giữa frozen VLA và actor–critic nhỏ. Actor–critic được train bằng sample-efficient off-policy online RL để chỉnh action chunk của VLA trong các task cần độ chính xác cao.

**[EXIMO (2026)](https://arxiv.org/abs/2608.19891)** dùng pipeline `explore → imitate → optimize`: VLM planner hỗ trợ thu thập dữ liệu cho task mới, VLA imitate orchestrated data, rồi residual off-policy RL tối ưu tiếp. Đây là ví dụ rất rõ của hybrid IL → RL.

**World-model VLA RL (2025–2026)** như [WMPO](https://arxiv.org/abs/2511.09515) và [VLA-MBPO](https://arxiv.org/abs/2603.20607) dùng learned world model để tạo rollout hoặc policy optimization mà giảm interaction trực tiếp với robot thật. WMPO tối ưu policy bằng on-policy GRPO trên imagined trajectories. Đây là nhánh **model-based RL**, còn optimizer GRPO thuộc policy-based/on-policy.

#### Giai đoạn 6 — steerability, memory và embodied reasoning

> **Paper/technical report:** [π0.7](https://www.pi.website/download/pi07.pdf) · [Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) · [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/).

**[π0.7 (2026)](https://www.pi.website/download/pi07.pdf)** tập trung vào diverse context conditioning: language, subgoal images và episode metadata giúp steer strategy và compositional generalization. Model có thể đạt khả năng gần một số specialist đã RL-fine-tune, nhưng điểm mới chính của π0.7 là conditioning/data composition, không nên mặc định gắn toàn bộ model này vào RL.

**[Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) / [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) (2025–2026)** tách embodied reasoning/planning và low-level VLA execution rõ hơn, đồng thời mở rộng sang whole-body control, multi-robot collaboration và on-device adaptation. Đây là mở rộng về multimodal reasoning, hierarchy và deployment; public model cards không đủ để kết luận mọi phiên bản được train bằng một RL algorithm cụ thể.

### 7.3 Bảng phân loại từng mốc VLA

| Mốc | Tín hiệu học policy chính | Action modeling | Có RL không? | Đặt ở đâu trong cây? |
|---|---|---|---|---|
| [ACT](https://arxiv.org/abs/2304.13705) | Expert demonstrations | Action chunking, CVAE-style latent | Không phải thành phần chính | IL → BC → Action chunking |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Expert demonstrations | Conditional diffusion | Không | IL → BC → Diffusion policy |
| [RT-1](https://arxiv.org/abs/2212.06817) | Robot demonstrations | Discrete action tokens | Không | IL → BC → Autoregressive tokens |
| [RT-2](https://arxiv.org/abs/2307.15818) | Web/VLM data + robot demonstrations | Action tokens như text | Không phải thành phần chính | Self-supervised pretraining + IL/BC |
| [Octo](https://arxiv.org/abs/2405.12213) | Cross-embodiment trajectories | Diffusion action head | Không phải thành phần chính | IL → BC → Diffusion policy |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | 970k robot demonstrations | Discretized action tokens | Không | IL → BC → Autoregressive tokens |
| [π0](https://arxiv.org/abs/2410.24164) | Multi-robot demonstrations | Action chunks + flow matching | Không trong bản gốc | IL → BC → Flow matching |
| [π0.5](https://arxiv.org/abs/2504.16054) | Heterogeneous co-training + demonstrations | FAST tokens khi pre-train; flow matching khi post-train; semantic outputs | Không phải điểm chính | Self-supervised/supervised + IL/BC |
| [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759) | Demonstrations + rollouts + corrections + outcomes | VLA action policy + advantage conditioning | Có: offline và online/on-robot RL | IL + Offline RL + Online RL |
| [RL Token](https://www.pi.website/download/rlt.pdf) | VLA prior + real-robot experience | Small actor–critic edits action chunks | Có: off-policy online RL | Actor–Critic → Off-policy |
| [π0.7](https://www.pi.website/download/pi07.pdf) | Diverse prompts, subgoal images, metadata, robot/non-robot data | Steerable VLA policy | Không nên mặc định là RL | IL/foundation policy + context conditioning |
| [EXIMO](https://arxiv.org/abs/2608.19891) | Planned exploration data + imitation + reward experience | Residual policy refinement | Có: residual off-policy RL | Hybrid IL → Actor–Critic/off-policy RL |
| [WMPO](https://arxiv.org/abs/2511.09515) / [VLA-MBPO](https://arxiv.org/abs/2603.20607) | Imagined/world-model rollouts + reward | World-model policy optimization; WMPO dùng GRPO | Có | Model-based RL + Policy-based/on-policy |

### 7.4 Cây mở rộng dành riêng cho VLA

> **Paper theo từng nhánh:** action token — [RT-2](https://arxiv.org/abs/2307.15818); action chunk — [ACT](https://arxiv.org/abs/2304.13705); diffusion — [Diffusion Policy](https://arxiv.org/abs/2303.04137); flow matching — [π0](https://arxiv.org/abs/2410.24164); offline/online RL — [RECAP](https://arxiv.org/abs/2511.14759), [RL Token](https://www.pi.website/download/rlt.pdf); model-based RL — [WMPO](https://arxiv.org/abs/2511.09515).

```text
Robot Learning / VLA — lớp hội tụ, không phải một learning paradigm độc lập
│
├── Backbone knowledge
│   ├── Supervised learning
│   └── Self-supervised VLM/LLM/video pre-training
│
├── Robot action learning
│   └── Imitation Learning
│       ├── Behavioral Cloning
│       │   ├── Autoregressive action tokens: RT-1, RT-2, OpenVLA, FAST/π0.5 pre-training
│       │   ├── Action chunking: ACT
│       │   ├── Diffusion action head: Diffusion Policy, Octo
│       │   └── Flow-matching action head: π0, π0.5
│       └── Interactive corrections / interventions
│
├── Experience-based improvement
│   └── Reinforcement Learning
│       ├── Offline RL: RECAP pre-training, CQL/IQL-style ideas
│       ├── Online off-policy Actor–Critic: RL Token
│       ├── Residual off-policy RL: EXIMO optimize stage
│       └── Model-based RL
│           ├── WMPO: world model + on-policy GRPO
│           └── VLA-MBPO: world-model branched rollout
│
└── System-level extensions
    ├── Hierarchical language planning / embodied reasoning
    ├── Memory for partial observability and long-horizon tasks
    ├── Cross-embodiment training and adaptation
    ├── Human corrections, preferences and safety constraints
    └── Real-time/on-device inference and action execution
```

Điểm quan trọng là bốn nhánh con cuối của `Robot Learning / VLA` không cùng một loại taxonomy: backbone, learning signal, action decoder và system design là các trục khác nhau. Cây này vẽ chúng cạnh nhau để theo dõi pipeline, không tuyên bố chúng loại trừ lẫn nhau.

---

## 8. Cách trả lời phỏng vấn

### “VLA dùng IL hay RL?”

> **Paper đối chiếu:** [π0 — imitation + flow matching](https://arxiv.org/abs/2410.24164), [π*0.6 / RECAP — VLA + RL](https://arxiv.org/abs/2511.14759) và [RL Token — online actor–critic](https://www.pi.website/download/rlt.pdf).

> Phần lớn VLA nền tảng ban đầu học robot action chủ yếu bằng imitation learning/Behavioral Cloning trên demonstrations, kết hợp backbone đã self-supervised hoặc supervised pre-train trên image-text data. Action có thể được sinh autoregressively, bằng diffusion hoặc flow matching; đây là action modeling, không phải RL. Các thế hệ mới bổ sung offline/online RL để học từ success, failure và on-robot experience, ví dụ RECAP/π*0.6, RL Token và EXIMO. Vì vậy câu đúng là VLA có pipeline hybrid, nhưng không phải mọi VLA đều dùng RL.

### “PPO thuộc Policy-based hay Actor–Critic?”

> **Paper đối chiếu:** [PPO](https://arxiv.org/abs/1707.06347) và [A3C](https://arxiv.org/abs/1602.01783).

> PPO là policy-gradient method về objective và thường được implement theo Actor–Critic với một policy actor cùng value critic. Vì taxonomy chồng lấp, cả hai cách gọi đều đúng nếu nói rõ tiêu chí phân loại.

### “SAC khác TD3 ở đâu?”

> **Paper đối chiếu:** [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html) và [TD3](https://proceedings.mlr.press/v80/fujimoto18a.html).

> Cả hai đều là off-policy Actor–Critic cho continuous control và đều thường dùng twin critics. TD3 có deterministic actor, external exploration noise, target policy smoothing và delayed actor updates. SAC có stochastic policy và tối ưu maximum-entropy objective; entropy trực tiếp khuyến khích exploration.

### “Tại sao VLA không chỉ dùng RL từ đầu?”

> **Paper đối chiếu:** [RT-1](https://arxiv.org/abs/2212.06817) và [π0](https://arxiv.org/abs/2410.24164) cho large-scale imitation pre-training; [RECAP](https://arxiv.org/abs/2511.14759) cho RL sau khi đã có VLA prior.

> Real-robot interaction đắt, chậm, nguy hiểm và reward cho task dài khó thiết kế. Demonstrations giúp policy có skill prior tốt mà không cần tự khám phá từ đầu. Sau đó RL phù hợp để cải thiện recovery, throughput, precision và outcome vượt quá việc bắt chước expert.

---

## 9. Bản đồ ghi nhớ cuối cùng

> **Paper tra cứu nhanh:** [DAgger/IL](https://proceedings.mlr.press/v15/ross11a.html) · [DQN](https://www.nature.com/articles/nature14236) · [REINFORCE](https://doi.org/10.1007/BF00992696) · [PPO](https://arxiv.org/abs/1707.06347) · [TD3](https://proceedings.mlr.press/v80/fujimoto18a.html) · [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) · [RT-2/VLA](https://arxiv.org/abs/2307.15818) · [π0](https://arxiv.org/abs/2410.24164) · [RECAP](https://arxiv.org/abs/2511.14759).

```text
Nếu target là label/action có sẵn
        └── Supervised Learning / Behavioral Cloning

Nếu target được tạo từ chính dữ liệu
        └── Self-supervised Learning

Nếu học bằng demonstrations
        └── Imitation Learning
            ├── BC
            ├── DAgger / intervention
            └── IRL / GAIL

Nếu tối đa hóa reward qua experience
        └── Reinforcement Learning
            ├── DQN: value-based, discrete, off-policy
            ├── REINFORCE: policy-based, on-policy
            ├── A2C/A3C/PPO: on-policy Actor–Critic
            ├── GRPO: policy-based, group-relative, on-policy
            └── TD3/SAC: off-policy Actor–Critic, continuous

Nếu input là vision + language và output là robot action
        └── VLA policy
            ├── thường pre-train bằng image-text self-supervision
            ├── thường học action ban đầu bằng IL/BC
            ├── có thể dùng token/diffusion/flow để sinh action
            └── có thể post-train bằng offline/online/model-based RL
```

---

## Nguồn chính

### RL algorithms

- Sutton và Barto, [Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html).
- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning — REINFORCE](https://doi.org/10.1007/BF00992696).
- Mnih và cộng sự, [Human-level control through deep reinforcement learning — DQN](https://www.nature.com/articles/nature14236).
- Mnih và cộng sự, [Asynchronous Methods for Deep Reinforcement Learning — A3C](https://arxiv.org/abs/1602.01783).
- Schulman và cộng sự, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman và cộng sự, [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Lillicrap và cộng sự, [Continuous Control with Deep Reinforcement Learning — DDPG](https://arxiv.org/abs/1509.02971).
- Fujimoto và cộng sự, [Addressing Function Approximation Error in Actor-Critic Methods — TD3](https://proceedings.mlr.press/v80/fujimoto18a.html).
- Haarnoja và cộng sự, [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html).
- Shao và cộng sự, [DeepSeekMath — Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300).
- Ross và cộng sự, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning — DAgger](https://proceedings.mlr.press/v15/ross11a.html).
- Ho và Ermon, [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476).
- Fujimoto và cộng sự, [Off-Policy Deep Reinforcement Learning without Exploration — BCQ](https://arxiv.org/abs/1812.02900).
- Kumar và cộng sự, [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).
- Kostrikov và cộng sự, [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
- Chen và cộng sự, [Decision Transformer](https://arxiv.org/abs/2106.01345).
- Hafner và cộng sự, [Mastering Diverse Domains through World Models — DreamerV3](https://arxiv.org/abs/2301.04104).
- Schrittwieser và cộng sự, [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model — MuZero](https://arxiv.org/abs/1911.08265).

### Robot policy và VLA

- Zhao và cộng sự, [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — ACT](https://arxiv.org/abs/2304.13705).
- Chi và cộng sự, [Diffusion Policy](https://arxiv.org/abs/2303.04137).
- Lipman và cộng sự, [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Brohan và cộng sự, [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817).
- Brohan và cộng sự, [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818).
- Belkhale và cộng sự, [RT-H: Action Hierarchies Using Language](https://arxiv.org/abs/2403.01823).
- Octo Model Team, [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213).
- Kim và cộng sự, [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246).
- Black và cộng sự, [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164).
- Pertsch và cộng sự, [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747).
- Physical Intelligence, [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054).
- Physical Intelligence, [π*0.6: A VLA That Learns From Experience — RECAP](https://arxiv.org/abs/2511.14759).
- Xu và cộng sự, [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://www.pi.website/download/rlt.pdf).
- Physical Intelligence, [π0.7: A Steerable Generalist Robotic Foundation Model](https://www.pi.website/download/pi07.pdf).
- Sukhija và cộng sự, [EXIMO: VLM Guided Exploration of VLA Policies](https://arxiv.org/abs/2608.19891).
- Zhu và cộng sự, [WMPO: World Model-based Policy Optimization for VLA Models](https://arxiv.org/abs/2511.09515).
- Zhang và cộng sự, [VLA-MBPO: Towards Practical World Model-based RL for VLA Models](https://arxiv.org/abs/2603.20607).
- Google DeepMind, [Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/).
- Google DeepMind, [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/).
