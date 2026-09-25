# 3.5 Markov Chain

[← Mục lục Machine Learning and Mathematics](./README.md)

**Mức đã trao đổi:** **3/5**

Em đang học phần này nên mức 3 là phù hợp với định nghĩa của form: **understand the concept**.

### Kiến thức cần nhớ

#### 1. What is a Markov Chain?

**Markov Chain — chuỗi Markov** là mô hình xác suất mô tả một hệ thống chuyển đổi giữa các trạng thái theo từng bước thời gian. Một Markov Chain hữu hạn thường gồm:

- Tập trạng thái $S$.
- Phân phối trạng thái ban đầu $\pi_0$.
- Xác suất chuyển trạng thái $P(s'\mid s)$.

Ví dụ, thời tiết có hai trạng thái `Nắng` và `Mưa`. Mỗi ngày, hệ thống chuyển sang trạng thái của ngày tiếp theo theo một xác suất nhất định.

#### 2. What is the Markov property?

Tính chất Markov nói rằng khi đã biết trạng thái hiện tại, trạng thái tiếp theo không còn phụ thuộc trực tiếp vào toàn bộ lịch sử:

```math
P(S_{t+1}\mid S_t,S_{t-1},\ldots,S_0) = P(S_{t+1}\mid S_t)
```

Có thể viết ngắn gọn:

```text
Future ⟂ Past | Present
```

Điều này không có nghĩa quá khứ luôn vô dụng. Nó có nghĩa state hiện tại phải chứa đủ thông tin liên quan từ quá khứ để dự đoán bước tiếp theo. Nếu một state không chứa đủ thông tin này thì cách định nghĩa state chưa thỏa mãn Markov property.

#### 3. What is a state?

**State — trạng thái** là biểu diễn tình trạng hiện tại của hệ thống tại một thời điểm.

Ví dụ:

- Trong mô hình thời tiết: `Nắng` hoặc `Mưa`.
- Trong robot: vị trí, hướng quay và vận tốc.
- Trong game: vị trí nhân vật, lượng máu và các vật phẩm đang có.

Một state tốt phải chứa đủ thông tin để phân phối của state tiếp theo chỉ phụ thuộc vào state hiện tại và, trong bài toán điều khiển, action hiện tại.

#### 4. What is a transition probability?

**Transition probability — xác suất chuyển trạng thái** là xác suất hệ thống đi từ state $i$ ở thời điểm hiện tại sang state $j$ ở bước tiếp theo:

```math
P_{ij} = P(S_{t+1}=j\mid S_t=i)
```

Ví dụ:

```math
P(\text{Mưa ngày mai}\mid\text{Nắng hôm nay})=0.3
```

Trong một **time-homogeneous Markov Chain**, xác suất này không thay đổi theo thời gian. Nếu xác suất phụ thuộc vào $t$, ta có một time-inhomogeneous Markov Chain.

#### 5. What is a transition matrix?

**Transition matrix** chứa xác suất chuyển đổi giữa mọi cặp trạng thái. Với thứ tự state là `[Nắng, Mưa]`, ta có thể có:

```math
P= \begin{bmatrix} 0.7 & 0.3\\ 0.4 & 0.6 \end{bmatrix}
```

| State hiện tại / State tiếp theo | Nắng | Mưa |
|---|---:|---:|
| Nắng | $0.7$ | $0.3$ |
| Mưa | $0.4$ | $0.6$ |

Phần tử ở hàng $i$, cột $j$ chính là $P_{ij}=P(S_{t+1}=j\mid S_t=i)$.

#### 6. Why does each row of the transition matrix sum to 1?

Khi state hiện tại $i$ đã được xác định, hệ thống phải chuyển tới một trong các state tiếp theo có thể có. Vì các khả năng này loại trừ lẫn nhau và bao phủ toàn bộ không gian state:

```math
\sum_j P_{ij}=1
```

Ví dụ, nếu ngày mai chỉ có thể nắng hoặc mưa thì:

```math
P(\text{Nắng}\mid\text{Nắng}) +P(\text{Mưa}\mid\text{Nắng}) =0.7+0.3=1
```

Ma trận có các phần tử không âm và mỗi hàng có tổng bằng $1$ được gọi là **row-stochastic matrix**.

#### 7. How do you compute the state distribution after n steps?

Nếu $\pi_t$ là row vector biểu diễn phân phối state tại thời điểm $t$, sau một bước:

```math
\pi_{t+1}=\pi_tP
```

Sau $n$ bước:

```math
\boxed{\pi_{t+n}=\pi_tP^n}
```

Ví dụ, nếu hôm nay chắc chắn nắng:

```math
\pi_0= \begin{bmatrix} 1 & 0 \end{bmatrix}
```

thì phân phối của ngày mai là:

```math
\pi_1 = \pi_0P = \begin{bmatrix} 0.7 & 0.3 \end{bmatrix}
```

Nếu dùng column vector cho phân phối thì phải nhất quán với convention tương ứng, ví dụ $\pi_{t+1}=P^T\pi_t$ khi $P$ vẫn là row-stochastic.

#### 8. What is a stationary distribution?

**Stationary distribution — phân phối dừng** là một phân phối $\pi^*$ không thay đổi sau một bước chuyển:

```math
\boxed{\pi^*=\pi^*P}
```

với:

```math
\pi_i^*\geq 0, \qquad \sum_i\pi_i^*=1
```

Với transition matrix thời tiết ở trên:

```math
\pi^*= \begin{bmatrix} \frac{4}{7} & \frac{3}{7} \end{bmatrix} \approx \begin{bmatrix} 0.571 & 0.429 \end{bmatrix}
```

Nếu hệ thống bắt đầu từ phân phối này thì các bước sau vẫn có cùng phân phối. Stationary distribution tồn tại không tự động có nghĩa mọi phân phối ban đầu đều hội tụ về nó; tính duy nhất và hội tụ còn phụ thuộc vào các tính chất như irreducibility và aperiodicity.

#### 9. What is an absorbing state?

**Absorbing state — trạng thái hấp thụ** là state mà sau khi đi vào, hệ thống không thể rời khỏi:

```math
P_{ii}=1
```

Do tổng mỗi hàng bằng $1$, điều này cũng kéo theo:

```math
P_{ij}=0 \quad\text{với mọi }j\neq i
```

Ví dụ, trong mô hình vòng đời tài khoản, nếu `Đã đóng vĩnh viễn` là absorbing state thì một tài khoản đã vào state này sẽ luôn ở đó trong các bước tiếp theo.

#### 10. How can transition probabilities be estimated from data?

Ta đếm số lần quan sát được mỗi chuyển đổi. Gọi $N_{ij}$ là số lần chuyển từ state $i$ sang state $j$, ước lượng là:

```math
\hat P_{ij} = \frac{N_{ij}}{\sum_kN_{ik}}
```

Ví dụ, trong dữ liệu có $70$ lần `Nắng → Nắng` và $30$ lần `Nắng → Mưa`:

```math
\hat P_{\text{Nắng,Nắng}}=\frac{70}{100}=0.7
```

```math
\hat P_{\text{Nắng,Mưa}}=\frac{30}{100}=0.3
```

Nếu dữ liệu ít, có thể dùng Laplace smoothing hoặc Bayesian estimation để tránh kết luận rằng một transition chưa từng xuất hiện phải có xác suất đúng bằng zero.

#### 11. How is MLE related to a Markov Chain?

Markov Chain và MLE có hai vai trò khác nhau:

```math
\boxed{\text{Markov Chain}=\text{probabilistic model}}
```

```math
\boxed{\text{MLE}=\text{method for estimating model parameters from data}}
```

Markov Chain định nghĩa cấu trúc xác suất $P(X_{t+1}\mid X_t)$. Khi transition matrix chưa biết, ta quan sát các trajectory rồi dùng MLE để ước lượng các phần tử của ma trận đó.

##### Ví dụ chi tiết với hai states

Giả sử Markov Chain có hai states $A$ và $B$, với transition matrix:

```math
P=\begin{bmatrix}p_{AA}&p_{AB}\\p_{BA}&p_{BB}\end{bmatrix}
```

Mỗi hàng là một probability distribution nên:

```math
p_{AA}+p_{AB}=1,\qquad p_{BA}+p_{BB}=1
```

Ta quan sát trajectory:

```math
A\rightarrow A\rightarrow B\rightarrow A\rightarrow B\rightarrow B
```

Các transition counts là:

| State hiện tại | Transition quan sát được | Tổng số lần rời state |
|---|---|---:|
| $A$ | $N_{AA}=1$, $N_{AB}=2$ | $3$ |
| $B$ | $N_{BA}=1$, $N_{BB}=1$ | $2$ |

MLE cho hàng bắt đầu từ $A$ là:

```math
\hat p_{AA}=\frac13,\qquad \hat p_{AB}=\frac23
```

MLE cho hàng bắt đầu từ $B$ là:

```math
\hat p_{BA}=\frac12,\qquad \hat p_{BB}=\frac12
```

Vì vậy, transition matrix ước lượng được là:

```math
\boxed{\hat P=\begin{bmatrix}\frac13&\frac23\\\frac12&\frac12\end{bmatrix}}
```

##### Tại sao transition counts cho đúng nghiệm MLE?

Do Markov property:

```math
P(X_{t+1}\mid X_t,X_{t-1},\ldots,X_0)=P(X_{t+1}\mid X_t)
```

xác suất của một trajectory $X_0,X_1,\ldots,X_T$ được factorize thành:

```math
P(X_0,X_1,\ldots,X_T)=P(X_0)\prod_{t=0}^{T-1}P(X_{t+1}\mid X_t)
```

Với trajectory trong ví dụ:

```math
L(P)=P(X_0=A)\,p_{AA}p_{AB}p_{BA}p_{AB}p_{BB}
```

Nếu initial distribution đã biết và không phụ thuộc vào transition parameters, $P(X_0=A)$ là constant đối với bài toán tối ưu. Khi đó:

```math
L(P)\propto p_{AA}p_{AB}^{2}p_{BA}p_{BB}
```

MLE tìm transition matrix làm likelihood này lớn nhất:

```math
\hat P=\arg\max_P L(P)
```

Tổng quát, nếu $N_{ij}$ là số lần quan sát transition $i\rightarrow j$, likelihood của transition parameters có thể nhóm lại thành:

```math
L(P)\propto\prod_i\prod_jP_{ij}^{N_{ij}}
```

Log-likelihood là:

```math
\log L(P)=\sum_i\sum_jN_{ij}\log P_{ij}
```

Tối đa hóa log-likelihood với các ràng buộc $\sum_jP_{ij}=1$ cho từng hàng cho nghiệm:

```math
\boxed{\hat P_{ij}^{\mathrm{MLE}}=\frac{N_{ij}}{\sum_kN_{ik}}}
```

Trong đó:

- $N_{ij}$ là số lần quan sát transition $i\rightarrow j$.
- $\sum_kN_{ik}$ là tổng số lần trajectory rời state $i$.

Nói ngắn gọn:

```math
\boxed{\hat P(j\mid i)=\frac{\text{number of observed }i\rightarrow j\text{ transitions}}{\text{total number of transitions leaving }i}}
```

Đây là MLE vì các relative frequencies trên chính là giá trị làm trajectory likelihood lớn nhất, không chỉ là một quy tắc đếm theo trực giác.

##### Nối sang transition dynamics trong Reinforcement Learning

Trong MDP, transition dynamics còn phụ thuộc vào action:

```math
P(s'\mid s,a)
```

Nếu dynamics chưa biết nhưng dataset chứa các transition tuples $(s,a,s')$, ta có thể ước lượng bằng:

```math
\boxed{\hat P(s'\mid s,a)=\frac{N(s,a,s')}{N(s,a)}}
```

Đây vẫn là cùng một nguyên lý MLE, chỉ khác là mỗi distribution được conditioned trên cả state $s$ và action $a$.

Mạch cần nhớ là:

```text
Markov assumption
       ↓
trajectory likelihood
       ↓
maximize with MLE
       ↓
estimated transition matrix P
```

Quan hệ này tương tự phần Gaussian và MSE:

```math
\text{Gaussian model}+\text{MLE}\Longrightarrow\text{MSE}
```

```math
\text{Markov model}+\text{MLE}\Longrightarrow\text{estimated transition matrix }\hat P
```

MLE là cùng một nguyên lý học tham số; probability model khác nhau sẽ dẫn tới parameters và objective cụ thể khác nhau.

#### 12. Markov Chain vs MDP?

| Thành phần | Markov Chain | Markov Decision Process — MDP |
|---|---|---|
| State | Có | Có |
| Transition | $P(s'\mid s)$ | $P(s'\mid s,a)$ |
| Action | Không | Có |
| Reward | Không | Có |
| Policy | Không cần | $\pi(a\mid s)$ |
| Mục tiêu điều khiển | Không | Tối đa hóa expected return |

Markov Chain mô tả diễn biến ngẫu nhiên của một hệ thống không có quyết định của agent. MDP mô tả một bài toán ra quyết định tuần tự, trong đó action của agent ảnh hưởng tới state tiếp theo và reward nhận được.

#### 13. What happens when we add actions?

Khi thêm action, transition model trở thành:

```math
P(S_{t+1}=s'\mid S_t=s,A_t=a)
```

Cùng một state có thể dẫn tới các phân phối state tiếp theo khác nhau tùy action. Agent cần một **policy** để chọn action:

```math
\pi(a\mid s)=P(A_t=a\mid S_t=s)
```

Nếu mới có state, action và transition nhưng chưa có reward, mô hình thường được gọi là **controlled Markov process**. Ta có thể điều khiển hệ thống nhưng chưa có tiêu chí để nói policy nào tốt hơn.

#### 14. What happens when we add rewards?

Reward $R(s,a,s')$ cho biết mức độ tốt hoặc xấu của một transition. Khi có states, actions, transitions và rewards, ta có một **Markov Decision Process**:

```math
(S,A,P,R,\gamma)
```

Mục tiêu thường là tìm policy tối đa hóa discounted return kỳ vọng:

```math
\boxed{ J(\pi) = \mathbb E_{\pi} \left[ \sum_{t=0}^{\infty}\gamma^tR_{t+1} \right] }
```

Trong đó $0\leq\gamma\lt 1$ là discount factor. Giá trị $\gamma$ nhỏ ưu tiên reward gần; giá trị lớn coi trọng kết quả dài hạn hơn.

Nếu policy $\pi$ được cố định, action được lấy theo policy và MDP tạo ra một Markov Chain trên các state với transition probability:

```math
P_{\pi}(s'\mid s) = \sum_a\pi(a\mid s)P(s'\mid s,a)
```

#### 15. What is a POMDP?

**Partially Observable Markov Decision Process — POMDP** là MDP mà agent không quan sát trực tiếp được state thật $S_t$. Thay vào đó, agent nhận observation $O_t$ được tạo ra từ state thông qua observation model, ví dụ:

```math
P(O_t=o\mid S_t=s)
```

Một POMDP thường được mô tả bởi:

```math
(S,A,P,R,\Omega,O,\gamma)
```

Trong đó $\Omega$ là tập observations và $O$ là observation model. Vì một observation có thể tương ứng với nhiều state thật khác nhau, agent dùng toàn bộ lịch sử action và observation để duy trì **belief state**:

```math
b_t(s) = P(S_t=s\mid o_{1:t},a_{0:t-1})
```

Belief state là một phân phối xác suất thể hiện agent đang tin state thật là mỗi khả năng với xác suất bao nhiêu.

#### 16. Why are robotics problems often partially observable?

Trong robotics, robot hiếm khi biết chính xác toàn bộ state của bản thân và môi trường vì:

- Camera, LiDAR, radar, IMU và GPS đều có noise và sai số đo.
- Vật thể có thể bị che khuất hoặc nằm ngoài field of view của sensor.
- Sensor chỉ quan sát một phần của môi trường tại mỗi thời điểm.
- Một ảnh đơn lẻ thường không cho biết trực tiếp velocity, depth hoặc intention của vật thể.
- Calibration và time synchronization giữa các sensor không hoàn hảo.
- Môi trường động có thể thay đổi trong khi robot đang quan sát và hành động.

Do đó, robot phải kết hợp observation hiện tại, lịch sử measurements, actions đã thực hiện và motion model để ước lượng hidden state. Kalman Filter, Particle Filter, Bayesian filtering và SLAM đều liên quan tới việc suy luận state từ các quan sát không đầy đủ hoặc có noise.

### Connection to Reinforcement Learning

```text
Markov Chain:        state + transition
                          │
                    thêm actions
                          ▼
Controlled process: state + action + transition
                          │
                    thêm rewards
                          ▼
MDP:                 state + action + transition + reward
                          │
              không quan sát trực tiếp state
                          ▼
POMDP:               MDP + observation model + belief state
```

---
