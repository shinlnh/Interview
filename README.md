# Machine Learning and Mathematics

### 3.1 Write or explain Adam optimizer pseudocode

**Mức đã trao đổi:** **3–4/5**, trong trao đổi anh nghiêng về **4/5** — **chưa xác nhận chính xác em đã bấm**

### Công thức phải nhớ

Gradient:

\[
g_t = \nabla_\theta L
\]

First moment:

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\]

Second moment:

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

Bias correction:

\[
\hat m_t = \frac{m_t}{1-\beta_1^t}
\]

\[
\hat v_t = \frac{v_t}{1-\beta_2^t}
\]

Update:

\[
\theta_t =
\theta_{t-1}
-
\alpha
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

### Interviewer có thể hỏi thêm

- What is Adam?
- Why does Adam use first and second moments?
- What does momentum do?
- Why do we need bias correction?
- What are beta1 and beta2?
- What does epsilon do?
- Adam vs SGD?
- Adam vs SGD with momentum?
- Why can SGD sometimes generalize better?
- What happens if learning rate is too high?
- What is AdamW?
- What is weight decay?
- What is the difference between L2 regularization and decoupled weight decay?

---

### 3.2 Masked Language Modeling — MLM

**Mức đã trao đổi:** **3/5**

### Interviewer có thể hỏi thêm

- What is Masked Language Modeling?
- Which famous model uses MLM?
- How is BERT trained?
- What happens when a token is masked?
- What is the training target?
- Which loss is commonly used?
- Why is cross-entropy used?
- What is the difference between MLM and autoregressive language modeling?
- BERT vs GPT training objective?
- Why can BERT use information from both left and right context?
- Why is MLM not naturally autoregressive?
- What is a tokenizer?
- BPE vs Unigram tokenizer?
- Is likelihood used in tokenizer training?

---

### 3.3 Matrix decomposition

**Mức đã trao đổi:** **3/5**

### Interviewer có thể hỏi thêm

- What is matrix decomposition?
- What is SVD?
- Write the SVD equation.
- What do U, Sigma and V represent?
- What is matrix rank?
- What is low-rank approximation?
- Why can low-rank approximation reduce model size?
- How is SVD related to PCA?
- What is eigenvalue decomposition?
- SVD vs eigenvalue decomposition?
- Can every matrix be decomposed with SVD?
- How can low-rank methods be used in neural networks?
- What is LoRA and how is it related to low-rank matrices?

### Công thức

\[
W = U \Sigma V^T
\]

Low-rank approximation:

\[
W \approx U_k \Sigma_k V_k^T
\]

---

### 3.4 Maximum Likelihood Estimation — MLE

**Mức đã trao đổi:** **3/5**

### Core idea

\[
\theta^* = \arg\max_\theta P(D \mid \theta)
\]

Nếu samples độc lập:

\[
L(\theta)
=
\prod_i
p(y_i \mid x_i,\theta)
\]

Thường chuyển sang Negative Log-Likelihood:

\[
\min_\theta -\log P(D\mid\theta)
\]

### Interviewer có thể hỏi thêm

- What is likelihood?
- What is the difference between probability and likelihood?
- What does theta represent?
- What does argmax mean?
- Why do we multiply probabilities across samples?
- Why do we take the logarithm of likelihood?
- Why does log convert multiplication into addition?
- Why is negative log-likelihood minimized?
- What is the MLE of the probability of heads if a coin gives 7 heads and 3 tails?
- What is MAP estimation?
- MLE vs MAP?
- How is MLE related to cross-entropy?
- How is likelihood used in LLM training?
- Why does autoregressive language modeling multiply conditional token probabilities?

### Coin example

7 heads, 3 tails:

\[
L(p)=p^7(1-p)^3
\]

\[
\log L(p)
=
7\log p + 3\log(1-p)
\]

Maximum occurs at:

\[
p=0.7
\]

---

### 3.5 Markov Chain

**Mức đã trao đổi:** **3/5**

Em đang học phần này nên mức 3 là phù hợp với định nghĩa của form: **understand the concept**.

### Markov property

\[
P(X_{t+1}\mid X_t,X_{t-1},...,X_0)
=
P(X_{t+1}\mid X_t)
\]

Nói đơn giản:

`Future ⟂ Past | Present`

### Transition probability

\[
P_{ij}
=
P(X_{t+1}=j \mid X_t=i)
\]

Distribution update:

\[
\pi_{t+1}=\pi_tP
\]

\[
\pi_{t+n}=\pi_tP^n
\]

### Interviewer có thể hỏi thêm

- What is a Markov Chain?
- What is the Markov property?
- What is a state?
- What is a transition probability?
- What is a transition matrix?
- Why does each row of the transition matrix sum to 1?
- How do you compute the state distribution after n steps?
- What is a stationary distribution?
- What is an absorbing state?
- How can transition probabilities be estimated from data?
- How is MLE related to a Markov Chain?
- Markov Chain vs MDP?
- What happens when we add actions?
- What happens when we add rewards?
- What is a POMDP?
- Why are robotics problems often partially observable?

### Connection to RL

Markov Chain:

\[
P(S_{t+1}\mid S_t)
\]

With action:

\[
P(S_{t+1}\mid S_t,A_t)
\]

Add reward → **MDP**

\[
(S,A,P,R,\gamma)
\]

---

### 3.6 Derive MSE from Gaussian Maximum Likelihood

**Mức đã trao đổi:** **3/5**

### Derivation

Giả sử:

\[
y_i=f_\theta(x_i)+\epsilon_i
\]

với

\[
\epsilon_i\sim\mathcal N(0,\sigma^2)
\]

thì:

\[
p(y_i\mid x_i,\theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}
\right)
\]

Likelihood của toàn bộ dataset:

\[
L(\theta)
=
\prod_i
p(y_i\mid x_i,\theta)
\]

Log-likelihood:

\[
\log L(\theta)
=
C
-
\frac{1}{2\sigma^2}
\sum_i
(y_i-\hat y_i)^2
\]

Nếu \(\sigma^2\) cố định:

\[
\arg\max_\theta \log L(\theta)
\equiv
\arg\min_\theta
\sum_i(y_i-\hat y_i)^2
\]

Chia cho \(N\):

\[
MSE
=
\frac{1}{N}
\sum_i(y_i-\hat y_i)^2
\]

### Interviewer có thể hỏi thêm

- Why does Gaussian noise lead to MSE?
- Where does the exponential disappear after taking log?
- Why can constants be ignored during optimization?
- What assumption do we make about the residual error?
- What if variance is not constant?
- What loss would another probability distribution lead to?
- Why is MSE sensitive to outliers?
- MSE vs MAE?
- What is heteroscedastic regression?
- Why is minimizing SSE equivalent to minimizing MSE for a fixed dataset size?

---
