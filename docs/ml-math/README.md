# Machine Learning and Mathematics

### 3.1 Write or explain Adam optimizer pseudocode

**Mức đã trao đổi:** **3–4/5**, trong trao đổi anh nghiêng về **4/5** — **chưa xác nhận chính xác em đã bấm**

### Công thức phải nhớ

#### Phần 1 — Loss function và sự phát triển của loss trong deep learning

##### 1. Loss function dùng để làm gì?

Với model $f_\theta$, prediction $\hat y_i=f_\theta(x_i)$ và dataset có $N$ samples, training objective thường có dạng:

```math
J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell(\hat y_i,y_i) + \lambda\Omega(\theta)
```

Trong đó:

- $\ell(\hat y_i,y_i)$ đo sai số dự đoán trên một sample.
- $\Omega(\theta)$ là regularization, ví dụ L1 hoặc L2.
- $\lambda$ kiểm soát mức regularization.
- Optimizer tìm $\theta$ làm $J(\theta)$ nhỏ nhất.
- Backpropagation tính gradient $\nabla_\theta J$; optimizer dùng gradient đó để cập nhật parameters.

> Loss function định nghĩa model cần học điều gì; optimizer định nghĩa model học bằng cách nào.

##### 2. Regression: MSE → MAE → Huber loss

**Mean Squared Error — MSE:**

```math
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
```

MSE xuất hiện tự nhiên khi giả định residual tuân theo Gaussian distribution. Nó trơn, dễ tối ưu và phạt rất mạnh lỗi lớn, nhưng vì bình phương sai số nên nhạy với outlier.

**Mean Absolute Error — MAE:**

```math
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|
```

MAE bền vững hơn với outlier và liên quan đến giả định Laplace distribution. Hạn chế của nó là không khả vi tại zero và gradient có độ lớn gần như không đổi.

**Huber loss** kết hợp ưu điểm của MSE và MAE. Đặt residual $r=y-\hat y$:

```math
\ell_\delta(r) = \begin{cases} \frac{1}{2}r^2, & |r|\leq\delta, \\ \delta\left(|r|-\frac{1}{2}\delta\right), & |r|\gt\delta. \end{cases}
```

Sai số nhỏ được xử lý giống MSE để tối ưu mượt; sai số lớn được xử lý gần giống MAE để giảm ảnh hưởng của outlier.

**Bounding-box regression trong YOLO — IoU-family loss:**

Các detector ban đầu có thể regression trực tiếp từng tọa độ $(x,y,w,h)$ bằng squared error. Cách này có hai hạn chế:

- Sai số của từng tọa độ được tối ưu độc lập, trong khi chất lượng detection được đánh giá bằng độ overlap của toàn bộ box.
- Cùng một coordinate error có ảnh hưởng khác nhau đối với box nhỏ và box lớn.

Vì vậy, nhiều phiên bản YOLO hiện đại sử dụng loss dựa trên **Intersection over Union — IoU**. Với predicted box $B$ và ground-truth box $B^{gt}$:

```math
\mathrm{IoU}(B,B^{gt}) = \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}
```

```math
\ell_{\text{IoU}} = 1-\mathrm{IoU}(B,B^{gt})
```

IoU loss trực tiếp tối ưu metric overlap, không phụ thuộc tuyệt đối vào scale của box. Tuy nhiên, nếu hai box không giao nhau thì $\mathrm{IoU}=0$, khiến loss khó cung cấp tín hiệu hữu ích về hướng di chuyển hai box lại gần nhau.

**Generalized IoU — GIoU** thêm penalty cho vùng trống trong enclosing box nhỏ nhất $C$ chứa cả hai box:

```math
\mathrm{GIoU} = \mathrm{IoU} - \frac{|C\setminus(B\cup B^{gt})|}{|C|}
```

```math
\ell_{\text{GIoU}}=1-\mathrm{GIoU}
```

GIoU vẫn tạo được learning signal khi hai box không overlap, nhưng convergence có thể chậm khi enclosing box không thay đổi nhiều.

**Distance IoU — DIoU** thêm khoảng cách giữa hai tâm box:

```math
\ell_{\text{DIoU}} = 1-\mathrm{IoU} + \frac{\rho^2(b,b^{gt})}{c^2}
```

Trong đó $b$ và $b^{gt}$ là tâm của hai box, $\rho$ là Euclidean distance giữa hai tâm, còn $c$ là đường chéo của enclosing box $C$. DIoU khuyến khích predicted box vừa overlap tốt vừa có tâm gần ground truth.

**Complete IoU — CIoU** bổ sung thêm độ tương đồng về aspect ratio:

```math
\ell_{\text{CIoU}} = 1-\mathrm{IoU} + \frac{\rho^2(b,b^{gt})}{c^2} + \alpha v
```

và:

```math
v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2
```

```math
\alpha = \frac{v}{1-\mathrm{IoU}+v}
```

CIoU đồng thời xem xét ba yếu tố:

1. Độ overlap giữa hai box.
2. Khoảng cách giữa hai tâm.
3. Sự khác nhau về aspect ratio.

Do đó, CIoU thường phù hợp hơn coordinate-wise MSE cho bounding-box regression. Tùy phiên bản và implementation, YOLO có thể dùng GIoU, DIoU, CIoU hoặc một biến thể IoU mới hơn.

Tổng training loss của một YOLO detector thường có dạng:

```math
\mathcal{L}_{\text{YOLO}} = \lambda_{\text{box}}\mathcal{L}_{\text{box}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{obj}}\mathcal{L}_{\text{obj}} + \lambda_{\text{DFL}}\mathcal{L}_{\text{DFL}}
```

Trong đó $\mathcal{L}_{\text{box}}$ có thể là một IoU-family loss. Classification loss, objectness loss và Distribution Focal Loss — DFL có được sử dụng hay không phụ thuộc vào kiến trúc YOLO cụ thể.

##### 3. Classification: 0–1 loss → Binary Cross-Entropy → Softmax Cross-Entropy

**0–1 loss** chỉ kiểm tra dự đoán đúng hay sai:

```math
\ell_{0/1}(\hat y,y)=\mathbf{1}[\hat y\neq y]
```

Loss này phản ánh trực tiếp accuracy nhưng không khả vi, nên không phù hợp với gradient descent.

Với binary classification, model dự đoán probability $p=P(y=1\mid x)$. Ta dùng **Binary Cross-Entropy — BCE**:

```math
\ell_{\text{BCE}} = -\left[y\log p+(1-y)\log(1-p)\right]
```

Với multi-class classification, logits $z_k$ được chuyển thành probability bằng softmax:

```math
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
```

Sau đó dùng **Cross-Entropy — CE**:

```math
\ell_{\text{CE}} = -\sum_{k=1}^{K}y_k\log p_k = -\log p_y
```

Cross-entropy cũng chính là negative log-likelihood của categorical distribution. Nó không chỉ quan tâm class đúng hay sai mà còn phạt mức độ tự tin sai của model.

##### 4. Loss hiện đại phát triển từ các hạn chế cụ thể

**Class imbalance — Weighted Cross-Entropy:**

Gán trọng số lớn hơn cho class hiếm:

```math
\ell = -\sum_{k=1}^{K}w_k y_k\log p_k
```

**Nhiều easy examples — Focal loss:**

```math
\mathrm{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log p_t
```

Factor $(1-p_t)^\gamma$ giảm ảnh hưởng của easy examples và khiến model tập trung hơn vào hard examples. Focal loss thường được dùng trong object detection có foreground/background imbalance.

**Model quá tự tin — Label smoothing:**

```math
y'_k = (1-\varepsilon)y_k+\frac{\varepsilon}{K}
```

Label smoothing thay one-hot target bằng target mềm hơn, giúp regularization và thường cải thiện calibration.

**Segmentation — Dice loss:**

```math
\ell_{\text{Dice}} = 1- \frac{2\sum_i p_i y_i+\varepsilon} {\sum_i p_i+\sum_i y_i+\varepsilon}
```

Dice loss tối ưu độ overlap và hữu ích khi foreground nhỏ hơn background rất nhiều. Trong thực tế, segmentation thường kết hợp BCE/CE với Dice hoặc IoU loss.

**Metric learning — Contrastive loss và Triplet loss:**

Các loss này không chỉ dự đoán class mà còn tổ chức embedding space: samples giống nhau nằm gần nhau, samples khác nhau nằm xa nhau. Chúng thường được dùng trong face recognition, retrieval và re-identification.

**Sequence modeling và LLM — Token-level Negative Log-Likelihood:**

```math
\ell_{\text{NLL}} = -\sum_{t=1}^{T} \log p_\theta(y_t\mid y_{\lt t},x)
```

Đây là cross-entropy áp dụng tại từng token. Với speech hoặc sequence chưa có alignment rõ ràng, CTC loss có thể tổng hợp probability của nhiều alignment hợp lệ.

> Sự phát triển của loss không có nghĩa loss mới thay thế hoàn toàn loss cũ. Mỗi loss giải quyết một giả định dữ liệu, một task hoặc một failure mode khác nhau.

#### Phần 2 — Từ Gradient Descent cổ điển đến Adam

##### 1. Gradient và Gradient Descent cổ điển

Gradient cho biết hướng làm objective tăng nhanh nhất:

```math
g_t=\nabla_\theta J(\theta_{t-1})
```

Vì cần minimize objective, parameters được cập nhật theo hướng ngược gradient:

```math
\theta_t=\theta_{t-1}-\alpha g_t
```

Trong đó $\alpha$ là learning rate.

**Batch Gradient Descent** tính gradient trên toàn bộ $N$ samples:

```math
g_t = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta\ell_i(\theta_{t-1})
```

Gradient ổn định và chính xác, nhưng mỗi update rất tốn thời gian và memory khi dataset lớn.

##### 2. Stochastic Gradient Descent — SGD

SGD đúng nghĩa lấy một sample ngẫu nhiên $i_t$ cho mỗi update:

```math
g_t=\nabla_\theta\ell_{i_t}(\theta_{t-1})
```

```math
\theta_t=\theta_{t-1}-\alpha g_t
```

Mỗi bước rất rẻ và noise trong gradient có thể giúp thoát saddle point hoặc sharp region. Tuy nhiên, đường tối ưu dao động mạnh và khó tận dụng GPU hiệu quả.

##### 3. Mini-batch SGD

Mini-batch SGD lấy một batch $B_t$ gồm $b$ samples:

```math
g_t = \frac{1}{b}\sum_{i\in B_t} \nabla_\theta\ell_i(\theta_{t-1})
```

```math
\theta_t=\theta_{t-1}-\alpha g_t
```

Nó cân bằng giữa hai phía:

- Ít noise hơn SGD một sample.
- Rẻ hơn Batch Gradient Descent.
- Vectorization tốt trên GPU/TPU.
- Đây là cách train tiêu chuẩn trong deep learning; thuật ngữ “SGD” trong thực tế thường ám chỉ mini-batch SGD.

##### 4. Momentum

SGD có thể dao động mạnh theo những hướng có curvature lớn và tiến chậm theo hướng gradient ổn định. Momentum dùng exponential moving average của gradient:

```math
m_t = \beta m_{t-1}+(1-\beta)g_t
```

```math
\theta_t = \theta_{t-1}-\alpha m_t
```

Momentum tích lũy vận tốc theo hướng gradient nhất quán và triệt bớt dao động đổi dấu. Giá trị phổ biến là $\beta=0.9$.

**Nesterov momentum** nhìn trước tại vị trí dự kiến rồi mới tính gradient, nhờ đó có thể điều chỉnh sớm hơn khi sắp đi quá xa:

```math
g_t = \nabla_\theta J(\theta_{t-1}-\alpha\beta m_{t-1})
```

##### 5. AdaGrad

Momentum thích nghi theo **hướng**, nhưng vẫn dùng cùng một learning rate cho mọi parameter. AdaGrad tạo learning rate riêng cho từng parameter bằng tổng bình phương gradient:

```math
s_t=s_{t-1}+g_t\odot g_t
```

```math
\theta_t = \theta_{t-1} - \alpha\frac{g_t}{\sqrt{s_t}+\epsilon}
```

Phép toán được thực hiện element-wise. Parameter có gradient lớn thường xuyên sẽ nhận bước update nhỏ hơn; parameter hiếm khi có gradient sẽ nhận bước lớn hơn. AdaGrad phù hợp với sparse features, nhưng $s_t$ chỉ tăng nên effective learning rate có thể giảm đến mức model gần như ngừng học.

##### 6. RMSProp

RMSProp sửa nhược điểm tích lũy vô hạn của AdaGrad bằng exponential moving average của squared gradient:

```math
v_t = \rho v_{t-1}+(1-\rho)g_t\odot g_t
```

```math
\theta_t = \theta_{t-1} - \alpha\frac{g_t}{\sqrt{v_t}+\epsilon}
```

Do gradient cũ dần bị quên, effective learning rate không liên tục giảm như AdaGrad. RMSProp thích hợp với objective non-stationary và từng được dùng nhiều cho RNN.

##### 7. Adam — Adaptive Moment Estimation

Adam kết hợp:

- **Momentum:** exponential moving average của gradient, hay first moment.
- **RMSProp:** exponential moving average của squared gradient, hay second raw moment.

Gradient tại step $t$:

```math
g_t=\nabla_\theta J(\theta_{t-1})
```

First moment:

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
```

Second raw moment:

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t
```

Vì $m_0=v_0=0$, các moving averages ban đầu bị bias về zero. Adam dùng bias correction:

```math
\begin{aligned} \hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \\ \hat{v}_t &= \frac{v_t}{1-\beta_2^t}. \end{aligned}
```

Update:

```math
\theta_t = \theta_{t-1} - \alpha \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

Default values thường dùng:

```math
\beta_1=0.9,\qquad \beta_2=0.999,\qquad \epsilon=10^{-8}
```

Adam pseudocode:

```text
initialize parameters θ
initialize m = 0, v = 0, t = 0

for each mini-batch:
    t = t + 1
    g = gradient of objective with respect to θ
    m = β₁m + (1 - β₁)g
    v = β₂v + (1 - β₂)(g ⊙ g)
    m_hat = m / (1 - β₁ᵗ)
    v_hat = v / (1 - β₂ᵗ)
    θ = θ - α m_hat / (sqrt(v_hat) + ε)
```

##### 8. AdamW

Adam với L2 term bên trong loss làm regularization bị scale bởi adaptive denominator. AdamW tách weight decay khỏi gradient update:

```math
\theta_t = (1-\alpha\lambda)\theta_{t-1} - \alpha \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

AdamW thường là lựa chọn mặc định tốt hơn Adam khi train Transformer và nhiều modern deep network.

##### 9. So sánh nhanh

| Optimizer | Gradient source | Adaptive learning rate | Momentum | Điểm mạnh | Hạn chế chính |
|---|---|---:|---:|---|---|
| Batch GD | Toàn bộ dataset | No | No | Gradient ổn định | Mỗi update rất đắt |
| SGD | Một sample | No | No | Update rẻ, noise có thể hữu ích | Dao động mạnh |
| Mini-batch SGD | Một batch | No | Optional | Hiệu quả trên GPU, thường generalize tốt | Cần tune learning-rate schedule |
| Momentum | Một batch | No | Yes | Nhanh hơn theo hướng nhất quán | Một learning rate cho mọi parameter |
| AdaGrad | Một batch | Yes | No | Tốt cho sparse features | Learning rate giảm liên tục |
| RMSProp | Một batch | Yes | No | Không bị decay vô hạn như AdaGrad | Không có first-moment momentum chuẩn |
| Adam | Một batch | Yes | Yes | Hội tụ nhanh, ít phải tune ban đầu | Có thể generalize kém hơn tuned SGD |
| AdamW | Một batch | Yes | Yes | Decoupled weight decay | Vẫn cần tune learning rate và decay |

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

  Masked Language Modeling, or MLM, is a self-supervised pretraining objective. The model receives a sequence in which some tokens have been hidden or replaced and learns to recover the original tokens from their surrounding context.

  Because the prediction can use tokens on both the left and the right, MLM helps the model learn bidirectional contextual representations.

- Which famous model uses MLM?

  **BERT** is the best-known model pretrained with MLM. RoBERTa, ALBERT, and many other encoder-only Transformer models also use variants of this objective.

- How is BERT trained?

  The original BERT is first pretrained on unlabeled text using two objectives:

  - **Masked Language Modeling:** Predict selected original tokens from their context.
  - **Next Sentence Prediction:** Predict whether one sentence follows another in the source text.

  For MLM, BERT selects 15% of the input tokens. Among those selected tokens, 80% are replaced by `[MASK]`, 10% by a random token, and 10% are left unchanged. After pretraining, the model is fine-tuned on a downstream task using labeled data.

- What happens when a token is masked?

  The token's original identity is hidden from the input, usually by replacing it with `[MASK]`. BERT processes the entire sequence with bidirectional self-attention and uses the final hidden state at that position to predict a probability distribution over the vocabulary.

  The token is not assigned a score of zero. Its contextual representation is computed from the visible tokens on both sides. Also, because of BERT's 80/10/10 replacement rule, a selected token is not always literally replaced by `[MASK]`.

- What is the training target?

  The target is the **original token ID at each selected position**. Positions that were not selected for MLM do not contribute to the MLM loss. In original BERT, Next Sentence Prediction has a separate binary target, but it is not part of the token-level MLM target.

- Which loss is commonly used?

  MLM commonly uses **cross-entropy loss over the vocabulary**, evaluated only at the selected token positions. If $\mathcal{M}$ is the set of selected positions, the loss is:

```math
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta\!\left(x_i \mid \tilde{x}\right)
```

  Here, $x_i$ is the original token and $\tilde{x}$ is the corrupted input sequence. In code, unselected positions are typically ignored with a special label such as `-100`.

- Why is cross-entropy used?

  Predicting a masked token is a multi-class classification problem: the model produces one logit for every token in the vocabulary, while the original token provides the correct class. Cross-entropy converts these logits through softmax and minimizes the negative log-probability of the correct token.

  It is differentiable, strongly penalizes confident wrong predictions, and is equivalent to maximum likelihood for a categorical target. Original BERT also uses cross-entropy for NSP, but that is a separate binary classification loss.

- What is the difference between MLM and autoregressive language modeling?

  **MLM** corrupts selected input tokens and predicts their original values using context from both sides. It normally computes loss only at the selected positions and can predict several positions in parallel.

  **Autoregressive language modeling** factorizes a sequence from left to right:

```math
p(x_1,\ldots,x_T) = \prod_{t=1}^{T}p(x_t\mid x_{\lt t})
```

  It uses a causal attention mask so position $t$ cannot access future tokens. During generation, each predicted token is appended to the context before predicting the next one. The `[MASK]` corruption used by MLM and the causal attention mask used by an autoregressive model therefore serve different purposes.

- BERT vs GPT training objective?

  The original **BERT** is an encoder model pretrained with MLM and NSP. Its objective is to recover masked tokens from bidirectional context and learn representations that can be fine-tuned for understanding tasks such as classification or question answering.

  **GPT** is a decoder-only model pretrained with causal next-token prediction. It maximizes the probability of each token conditioned only on earlier tokens, which directly trains it to generate text from left to right.

  Modern BERT-like models may omit NSP, and modern GPT-like models may add instruction tuning or preference optimization, but their core pretraining objectives remain bidirectional masked-token prediction versus causal next-token prediction.

- Why can BERT use information from both left and right context?

  BERT uses a Transformer encoder without a causal triangular mask. Except for padding restrictions, every position can attend to tokens before and after it through self-attention. When a selected token is hidden, its representation can therefore combine evidence from both directions while predicting the original token.

- Why is MLM not naturally autoregressive?

  MLM does not learn the left-to-right factorization required for autoregressive generation. It predicts selected positions from a corrupted sequence, often predicts several positions in parallel, and may use future context that would not exist during left-to-right generation.

  It also introduces special `[MASK]` inputs that normally do not appear in generated text. An MLM can generate through iterative masking and resampling, but this requires a separate procedure and is not its native training behavior.

- What is a tokenizer?

  A tokenizer converts raw text into a sequence of token IDs that a model can process. A typical tokenizer normalizes text, segments it into words or subwords, maps those pieces to a fixed vocabulary, and inserts special tokens such as `[CLS]`, `[SEP]`, or `[MASK]` when required.

  Subword or byte-level tokenization limits out-of-vocabulary problems because an unfamiliar word can be represented by smaller known units. Decoding performs the reverse mapping from token IDs back to text.

- BPE vs Unigram tokenizer?

  **Byte Pair Encoding — BPE** starts with small symbols, such as characters or bytes, and repeatedly merges the most frequent adjacent pair. The learned merge rules are then applied to segment new text. BPE is simple and fast, but its greedy merge process does not explicitly model the probability of every possible segmentation.

  **Unigram tokenization** starts with a large candidate vocabulary and assigns a probability to each token. It repeatedly removes tokens whose removal hurts the corpus likelihood the least. At inference, it can use the Viterbi algorithm to select the highest-probability segmentation or sample alternative segmentations.

  In short, BPE builds the vocabulary by adding frequent merges, whereas Unigram begins with many candidates and prunes them using a probabilistic objective.

- Is likelihood used in tokenizer training?

  It depends on the tokenizer algorithm. A **Unigram language-model tokenizer** explicitly optimizes corpus likelihood by assigning probabilities to tokens and considering possible segmentations:

```math
p(x) = \sum_{s\in\mathcal{S}(x)} \prod_{u\in s}p(u)
```

  Here, $\mathcal{S}(x)$ is the set of valid segmentations of string $x$. By contrast, standard BPE greedily chooses frequent pair merges rather than directly maximizing this likelihood. Therefore, likelihood is central to Unigram training but is not a universal objective for every tokenizer.

---

### 3.3 Matrix decomposition

**Mức đã trao đổi:** **3/5**

### Kiến Thức Cần Nhớ

#### 1. Hạng của ma trận

**Định nghĩa trực quan:** Hạng của ma trận cho biết phép biến đổi do ma trận đó tạo ra còn giữ được bao nhiêu **hướng độc lập**. Nói cách khác, hạng đo lượng thông tin độc lập có trong các hàng hoặc các cột của ma trận.

Với ma trận $A\in\mathbb{R}^{m\times n}$:

```math
\mathrm{rank}(A) = \text{số cột độc lập tuyến tính của }A = \text{số hàng độc lập tuyến tính của }A
```

Ta luôn có:

```math
0\leq \mathrm{rank}(A)\leq \min(m,n)
```

Ví dụ:

```math
A= \begin{bmatrix} 1&0\\ 0&1 \end{bmatrix}
```

Hai cột $(1,0)^T$ và $(0,1)^T$ chỉ hai hướng độc lập trong mặt phẳng, nên $\mathrm{rank}(A)=2$. Phép biến đổi này vẫn giữ được không gian hai chiều.

Ngược lại:

```math
B= \begin{bmatrix} 1&2\\ 2&4 \end{bmatrix}
```

Cột thứ hai bằng hai lần cột thứ nhất, nên ma trận chỉ có một hướng độc lập và $\mathrm{rank}(B)=1$. Với mọi vector $(x,y)^T$:

```math
B \begin{bmatrix} x\\y \end{bmatrix} = \begin{bmatrix} x+2y\\ 2x+4y \end{bmatrix} = (x+2y) \begin{bmatrix} 1\\2 \end{bmatrix}
```

Mọi điểm trong mặt phẳng sau phép biến đổi đều bị ép lên cùng một đường thẳng. Vì một chiều đã bị mất nên ta không thể khôi phục duy nhất $(x,y)$ từ kết quả.

> Hạng càng thấp thì ma trận càng có nhiều thông tin trùng lặp hoặc càng làm mất nhiều hướng. Hạng đầy đủ nghĩa là ma trận giữ được số hướng độc lập lớn nhất có thể.

#### 2. Định thức của ma trận

**Định nghĩa trực quan:** Định thức chỉ được định nghĩa cho ma trận vuông và cho biết phép biến đổi tuyến tính làm **diện tích hoặc thể tích có hướng** thay đổi bao nhiêu lần.

- Trong không gian hai chiều, $|\det(A)|$ là hệ số co giãn diện tích.
- Trong không gian ba chiều, $|\det(A)|$ là hệ số co giãn thể tích.
- $\det(A)\gt 0$: phép biến đổi giữ nguyên chiều định hướng.
- $\det(A)\lt 0$: phép biến đổi đảo chiều định hướng, giống như phản chiếu qua gương.
- $\det(A)=0$: diện tích hoặc thể tích bị ép về $0$, nghĩa là ít nhất một chiều đã bị mất.

Với ma trận $2\times2$:

```math
A= \begin{bmatrix} a&b\\ c&d \end{bmatrix}, \qquad \det(A)=ad-bc
```

Ví dụ:

```math
C= \begin{bmatrix} 2&0\\ 0&3 \end{bmatrix}, \qquad \det(C)=2\cdot3=6
```

$C$ kéo dài một hướng lên $2$ lần và hướng còn lại lên $3$ lần. Vì vậy, hình vuông đơn vị có diện tích $1$ sẽ trở thành hình chữ nhật có diện tích $6$.

Với ma trận $B$ ở trên:

```math
\det(B)=1\cdot4-2\cdot2=0
```

Mặt phẳng bị ép xuống một đường thẳng nên diện tích sau biến đổi bằng $0$. Đây cũng là lý do $B$ có hạng nhỏ hơn $2$.

> Hạng trả lời câu hỏi “còn bao nhiêu hướng độc lập?”, còn định thức trả lời “diện tích hoặc thể tích có hướng bị thay đổi bao nhiêu lần?”. Định thức bằng $0$ cho biết có hướng đã bị mất, nhưng không cho biết chính xác còn lại bao nhiêu hướng; muốn biết điều đó phải dùng hạng.

#### 3. Ma trận khả nghịch

**Định nghĩa trực quan:** Ma trận khả nghịch biểu diễn một phép biến đổi có thể **đảo ngược hoàn toàn**. Nếu $A$ biến $x$ thành $y$, thì $A^{-1}$ có thể biến $y$ trở lại đúng $x$:

```math
y=Ax \qquad\Longrightarrow\qquad x=A^{-1}y
```

Ma trận nghịch đảo thỏa mãn:

```math
AA^{-1}=A^{-1}A=I
```

Với ma trận vuông $A\in\mathbb{R}^{n\times n}$, các phát biểu sau là tương đương:

```math
A\text{ khả nghịch} \Longleftrightarrow \mathrm{rank}(A)=n \Longleftrightarrow \det(A)\neq0
```

Ví dụ, với ma trận $C$:

```math
C^{-1}= \begin{bmatrix} \frac{1}{2}&0\\ 0&\frac{1}{3} \end{bmatrix}
```

$C$ kéo dài hai hướng lên $2$ và $3$ lần; $C^{-1}$ co chúng lại còn $\frac{1}{2}$ và $\frac{1}{3}$, nên khôi phục được vector ban đầu.

Ma trận $B$ không khả nghịch vì nó ép cả mặt phẳng xuống một đường thẳng. Nhiều điểm đầu vào khác nhau có thể cho cùng một đầu ra, nên không thể xác định duy nhất điểm ban đầu.

> Ma trận khả nghịch không làm mất thông tin. Ma trận không khả nghịch làm mất ít nhất một hướng nên không thể đảo ngược duy nhất.

#### 4. Ứng dụng trong đại số tuyến tính

- **Giải hệ phương trình:** Với hệ $Ax=b$, nếu $A$ vuông và khả nghịch thì hệ có nghiệm duy nhất $x=A^{-1}b$. Trong tính toán thực tế, thường giải trực tiếp hệ tuyến tính thay vì tính $A^{-1}$ vì ổn định và hiệu quả hơn.
- **Kiểm tra thông tin độc lập:** Hạng cho biết số phương trình hoặc số đặc trưng thực sự độc lập; nhờ đó phát hiện các hàng, cột hay biến bị trùng lặp.
- **Xác định số nghiệm:** Hạng giúp biết một hệ phương trình có vô nghiệm, một nghiệm hay vô số nghiệm.
- **Đo sự thay đổi hình học:** Định thức mô tả mức co giãn diện tích, thể tích và sự đảo chiều của phép biến đổi.
- **Đổi biến:** Khi đổi hệ tọa độ hoặc đổi biến trong tích phân, trị tuyệt đối của định thức Jacobi cho biết phần tử diện tích hoặc thể tích thay đổi thế nào.

#### 5. Ứng dụng trong AI và học máy

- **Nén mô hình:** Nếu ma trận trọng số $W$ có hạng thấp hoặc gần hạng thấp, có thể xấp xỉ $W\approx U_k\Sigma_kV_k^T$. Việc lưu các ma trận nhỏ này cần ít tham số và phép tính hơn lưu toàn bộ $W$.
- **LoRA:** Thay vì cập nhật trực tiếp một ma trận trọng số lớn, LoRA học một thay đổi hạng thấp $\Delta W=BA$. Cách này giúp tinh chỉnh mô hình lớn với ít tham số hơn.
- **PCA và giảm chiều:** PCA dùng các hướng chứa nhiều biến thiên nhất của dữ liệu. Hạng cho biết dữ liệu thực sự trải trên bao nhiêu chiều độc lập; các hướng ít quan trọng có thể được loại bỏ để giảm nhiễu và giảm kích thước.
- **Phát hiện đặc trưng dư thừa:** Ma trận dữ liệu hoặc ma trận hiệp phương sai có hạng thấp cho thấy một số đặc trưng có thể được tạo từ các đặc trưng khác.
- **Phân phối Gaussian:** Định thức của ma trận hiệp phương sai xuất hiện trong mật độ Gaussian và biểu diễn độ lớn vùng không gian mà dữ liệu phân bố. Định thức bằng $0$ nghĩa là covariance suy biến và phân phối bị ép vào một không gian thấp chiều hơn.
- **Mô hình luồng chuẩn hóa:** Các mô hình này cần phép biến đổi khả nghịch để đi qua lại giữa dữ liệu và biến ẩn. Định thức Jacobi được dùng để điều chỉnh mật độ xác suất khi không gian bị co giãn.

Tóm lại:

| Khái niệm | Câu hỏi trực quan | Điều cần nhớ |
|---|---|---|
| Hạng | Còn bao nhiêu hướng độc lập? | Hạng thấp nghĩa là có hướng hoặc thông tin bị trùng lặp hay mất đi |
| Định thức | Diện tích hoặc thể tích có hướng đổi bao nhiêu lần? | $\det(A)=0$ nghĩa là phép biến đổi làm mất ít nhất một chiều |
| Khả nghịch | Có khôi phục duy nhất đầu vào từ đầu ra không? | Với ma trận vuông: khả nghịch $\Leftrightarrow$ hạng đầy đủ $\Leftrightarrow\det(A)\neq0$ |

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

```math
W = U \Sigma V^T
```

Low-rank approximation:

```math
W \approx U_k \Sigma_k V_k^T
```

---

### 3.4 Maximum Likelihood Estimation — MLE

**Mức đã trao đổi:** **3/5**

### Core idea

```math
\theta^* = \arg\max_\theta P(D \mid \theta)
```

Nếu samples độc lập:

```math
L(\theta) = \prod_i p(y_i \mid x_i,\theta)
```

Thường chuyển sang Negative Log-Likelihood:

```math
\min_\theta -\log P(D\mid\theta)
```

### Kiến Thức Cần Nhớ

#### 1. Ví dụ MLE chi tiết: tung đồng xu

Giả sử có một đồng xu nhưng chưa biết xác suất xuất hiện mặt ngửa. Gọi:

```math
p=P(\text{ngửa}), \qquad 1-p=P(\text{sấp})
```

Ở đây:

- **Dữ liệu** $D$ là kết quả các lần tung đồng xu.
- **Tham số cần tìm** $\theta$ chính là $p$.
- MLE chọn giá trị $p$ làm cho dữ liệu đã quan sát có khả năng xuất hiện cao nhất.

Ta tung đồng xu độc lập $10$ lần và thu được $7$ mặt ngửa, $3$ mặt sấp. Xác suất của một thứ tự cụ thể, ví dụ `NNNSNNNSNS`, là:

```math
P(D\mid p) = p^7(1-p)^3
```

Khi xem biểu thức này như một hàm của tham số chưa biết $p$, ta gọi nó là **hàm hợp lý**:

```math
L(p)=p^7(1-p)^3
```

Nếu dữ liệu chỉ ghi nhận “có $7$ lần ngửa trong $10$ lần tung” mà không quan tâm thứ tự, xác suất sẽ là phân phối nhị thức:

```math
P(H=7\mid p) = \binom{10}{7}p^7(1-p)^3
```

Hệ số $\binom{10}{7}$ không phụ thuộc vào $p$, nên có nhân thêm hệ số này hay không thì giá trị $p$ làm likelihood lớn nhất vẫn giống nhau.

Một số giá trị thử:

| $p$ giả định | $L(p)$ | $\log L(p)$ |
|---:|---:|---:|
| $0.5$ | $0.0009766$ | $-6.9315$ |
| $0.6$ | $0.0017916$ | $-6.3247$ |
| $0.7$ | $0.0022236$ | $-6.1086$ |
| $0.8$ | $0.0016777$ | $-6.3903$ |

Trong các giá trị trên, $p=0.7$ cho likelihood lớn nhất. Để tìm chính xác thay vì thử từng giá trị, ta lấy log:

```math
\ell(p) = \log L(p) = 7\log p+3\log(1-p)
```

Log là hàm đồng biến nên giá trị làm $L(p)$ lớn nhất cũng làm $\ell(p)$ lớn nhất. Lấy đạo hàm:

```math
\frac{d\ell}{dp} = \frac{7}{p}-\frac{3}{1-p}
```

Cho đạo hàm bằng $0$:

```math
\frac{7}{p}-\frac{3}{1-p}=0
```

```math
7(1-p)=3p
```

```math
\hat p_{\text{MLE}}=\frac{7}{10}=0.7
```

Đạo hàm bậc hai là:

```math
\frac{d^2\ell}{dp^2} = -\frac{7}{p^2}-\frac{3}{(1-p)^2}\lt 0
```

Do đó $p=0.7$ thực sự là điểm cực đại. Kết luận tổng quát: nếu có $h$ lần ngửa trong tổng số $N$ lần tung thì:

```math
\hat p_{\text{MLE}}=\frac{h}{N}
```

**Trực giác:** ta chọn xác suất của mô hình khớp với tần suất quan sát được trong dữ liệu. MLE không nói rằng xác suất thật chắc chắn bằng $0.7$; nó chỉ nói rằng trong họ mô hình Bernoulli, $p=0.7$ giải thích dữ liệu hiện có tốt nhất.

#### 2. Probability và likelihood khác nhau ở đâu?

Cùng một biểu thức $p^7(1-p)^3$, nhưng câu hỏi khác nhau:

- **Probability:** đã biết $p$, hỏi dữ liệu nào có thể xuất hiện và xác suất của dữ liệu là bao nhiêu.
- **Likelihood:** đã quan sát cố định dữ liệu $D$, hỏi giá trị $p$ nào giải thích dữ liệu đó tốt nhất.

Ta không nói “xác suất của $p$” trong MLE. Ta đang so sánh mức độ phù hợp của các giá trị $p$ đối với cùng một dữ liệu.

#### 3. Từ đồng xu sang mô hình token

Đồng xu có hai kết quả là ngửa và sấp. Một mô hình token có nhiều kết quả, mỗi kết quả là một token trong từ vựng $V$. Gọi $\theta_u$ là xác suất của token $u$:

```math
\theta_u=P(u), \qquad \sum_{u\in V}\theta_u=1
```

Nếu sau khi tách từ, corpus trở thành chuỗi token $t_1,t_2,\ldots,t_N$ và tạm giả sử các token độc lập, likelihood là:

```math
L(\theta) = P(D\mid\theta,V) = \prod_{i=1}^{N}\theta_{t_i} = \prod_{u\in V}\theta_u^{c_u}
```

Trong đó $c_u$ là số lần token $u$ xuất hiện. Log-likelihood là:

```math
\ell(\theta) = \sum_{u\in V}c_u\log\theta_u
```

Với từ vựng và cách tách token đã cố định, nghiệm MLE là:

```math
\hat\theta_u = \frac{c_u}{N}
```

Đây chính là phiên bản nhiều loại kết quả của ví dụ đồng xu: xác suất MLE của mỗi token bằng tần suất tương đối của token đó trong corpus.

#### 4. Ánh xạ MLE sang bài toán học từ vựng WordPiece

WordPiece bắt đầu với một từ vựng nhỏ, thường chứa các ký tự cơ bản, sau đó dần thêm các mảnh từ dài hơn. Về mặt ý tưởng, ở mỗi bước nó tìm mảnh từ mới làm tăng likelihood của corpus dưới mô hình ngôn ngữ nhiều nhất.

Bài toán có thể nhìn thành hai tầng:

1. **Ước lượng tham số:** với một từ vựng $V$ và cách tách token cố định, dùng MLE để tìm các xác suất token $\hat\theta_u=c_u/N$.
2. **Chọn cấu trúc từ vựng:** thử thêm hoặc ghép một mảnh từ, tách lại corpus, ước lượng lại $\theta$, rồi đánh giá likelihood mới.

Viết cô đọng:

```math
V^* \approx \underset{V\text{ được mở rộng dần}}{\arg\max} \left[ \max_{\theta} \log P(D\mid V,\theta) \right]
```

Dấu $\approx$ nhấn mạnh rằng WordPiece xây từ vựng theo kiểu tham lam từng bước, không duyệt tất cả từ vựng có thể có để tìm nghiệm tối ưu toàn cục.

##### Ví dụ corpus nhỏ

Giả sử corpus gồm:

```text
“học” xuất hiện 6 lần
“hỏi” xuất hiện 2 lần
“đọc” xuất hiện 3 lần
```

Ban đầu, ta minh họa cách tách theo ký tự với `##` đánh dấu mảnh nằm giữa từ:

```text
học  → [h, ##ọ, ##c]
hỏi  → [h, ##ỏ, ##i]
đọc  → [đ, ##ọ, ##c]
```

Bảng đếm token là:

| Token | Số lần xuất hiện |
|---|---:|
| `h` | $8$ |
| `##ọ` | $9$ |
| `##c` | $9$ |
| `##ỏ` | $2$ |
| `##i` | $2$ |
| `đ` | $3$ |

Tổng cộng có $N=33$ token. Theo MLE:

```math
P(h)=\frac{8}{33}, \quad P(\text{\#\#ọ})=\frac{9}{33}, \quad P(\text{\#\#c})=\frac{9}{33}, \quad\ldots
```

Log-likelihood cực đại của cách tách hiện tại là:

```math
\ell_{\text{cũ}} = 8\log\frac{8}{33} +9\log\frac{9}{33} +9\log\frac{9}{33} +2\log\frac{2}{33} +2\log\frac{2}{33} +3\log\frac{3}{33} \approx -53.13
```

Bây giờ thử ghép `##ọ` và `##c` thành token `##ọc`:

```text
học  → [h, ##ọc]
hỏi  → [h, ##ỏ, ##i]
đọc  → [đ, ##ọc]
```

Các count mới là:

| Token | Số lần xuất hiện |
|---|---:|
| `h` | $8$ |
| `##ọc` | $9$ |
| `##ỏ` | $2$ |
| `##i` | $2$ |
| `đ` | $3$ |

Chuỗi corpus lúc này chỉ còn $N'=24$ token. Ước lượng lại xác suất bằng MLE, ta được:

```math
\ell_{\text{mới}} = 8\log\frac{8}{24} +9\log\frac{9}{24} +2\log\frac{2}{24} +2\log\frac{2}{24} +3\log\frac{3}{24} \approx -33.79
```

Mức cải thiện trong ví dụ unigram đơn giản này là:

```math
\Delta\ell = \ell_{\text{mới}}-\ell_{\text{cũ}} \approx 19.34
```

Log-likelihood mới lớn hơn vì $-33.79\gt -53.13$. Điều đó cho thấy `##ọc` là một mảnh từ hữu ích theo mô hình minh họa này: nó xuất hiện lặp lại, làm cách biểu diễn corpus ngắn hơn và giúp mô hình gán xác suất cao hơn cho dữ liệu đã thấy.

#### 5. Điểm ghép thường dùng để giải thích WordPiece

Một cách tái dựng WordPiece thường gặp không tính lại toàn bộ likelihood cho mọi ứng viên vì làm như vậy rất tốn kém. Thay vào đó, nó chấm điểm cặp token kề nhau bằng:

```math
\mathrm{score}(a,b) = \frac{\mathrm{freq}(a,b)} {\mathrm{freq}(a)\mathrm{freq}(b)}
```

Điểm này đo mức độ hai token **đặc biệt thích đi cùng nhau**, thay vì chỉ nhìn số lần cặp xuất hiện như BPE.

Trong corpus trên:

```math
\mathrm{score}(\text{\#\#ọ},\text{\#\#c}) = \frac{9}{9\cdot9} = \frac{1}{9}
```

```math
\mathrm{score}(\text{\#\#ỏ},\text{\#\#i}) = \frac{2}{2\cdot2} = \frac{1}{2}
```

Dù `##ỏ ##i` chỉ xuất hiện $2$ lần, hai token này luôn đi cùng nhau trong corpus nhỏ, nên điểm liên kết của chúng cao. Vì vậy, quy tắc điểm ghép này có thể chọn `##ỏi` trước `##ọc`.

> Cần phân biệt: công thức score trên là cách giải thích hoặc tái dựng phổ biến cho tiêu chí ghép của WordPiece; nó không đồng nhất với phép tính lại chính xác toàn bộ corpus likelihood trong ví dụ ở trên. Chi tiết đầy đủ của thuật toán huấn luyện WordPiece gốc không được công bố cùng mã nguồn, nên các thư viện có thể dùng tiêu chí gần đúng khác nhau.

#### 6. MLE thông thường và likelihood trong WordPiece giống nhau ở đâu?

| Điểm giống | Giải thích |
|---|---|
| Đều có dữ liệu quan sát cố định | Đồng xu dùng chuỗi ngửa/sấp; WordPiece dùng corpus văn bản |
| Đều xây một mô hình xác suất | Đồng xu có $p$ và $1-p$; mô hình token có $\theta_u$ cho từng token |
| Đều ưu tiên cách giải thích dữ liệu tốt nhất | Chọn tham số hoặc phép ghép làm likelihood lớn hơn |
| Đều dùng log-likelihood | Tích xác suất trở thành tổng, dễ tính và tránh underflow |
| Tần suất đóng vai trò trung tâm | Với mô hình cố định, nghiệm MLE chính là tần suất tương đối |

#### 7. Chúng khác nhau ở đâu?

| MLE đồng xu | Học từ vựng WordPiece |
|---|---|
| Không gian kết quả `{ngửa, sấp}` đã cố định | Chính tập token và cách chia văn bản cũng đang được thay đổi |
| Chỉ tối ưu tham số liên tục $p$ | Vừa ước lượng xác suất, vừa ra quyết định rời rạc nên ghép token nào |
| Có nghiệm đóng $\hat p=h/N$ | Việc tìm từ vựng tối ưu toàn cục rất lớn, nên thường xây từ vựng tham lam |
| Mỗi quan sát đã có nhãn rõ ràng là ngửa hay sấp | Một từ có thể có nhiều cách phân mảnh tiềm năng |
| Kết quả là một xác suất | Kết quả huấn luyện WordPiece chủ yếu là một từ vựng và quy tắc tách token |

#### 8. Ba loại likelihood dễ bị nhầm

1. **Likelihood khi học từ vựng WordPiece:** dùng corpus để quyết định mảnh từ nào nên có trong vocabulary.
2. **Likelihood khi huấn luyện BERT hoặc mô hình ngôn ngữ:** vocabulary đã cố định; model học tham số neural để tăng xác suất của token đúng trong ngữ cảnh. Đây là bài toán cross-entropy hoặc negative log-likelihood khác.
3. **Unigram tokenizer:** duy trì xác suất cho các mảnh từ và có thể xét nhiều cách phân đoạn; thường dùng thuật toán kiểu EM rồi loại dần các mảnh làm giảm likelihood ít nhất. Nó không phải chính là quy trình ghép tham lam của WordPiece.

Sau khi từ vựng WordPiece đã được học xong, bước tokenize một từ mới thường dùng chiến lược **khớp mảnh dài nhất từ trái sang phải**. Bước suy luận này là một quy tắc tìm kiếm xác định, không phải mỗi lần tokenize lại giải một bài toán MLE.

### Interviewer có thể hỏi thêm

- What is likelihood?

  **Likelihood** đo mức độ một giá trị tham số $\theta$ giải thích tốt dữ liệu $D$ đã quan sát như thế nào. Nó được định nghĩa bởi:

```math
L(\theta;D)=P(D\mid\theta)
```

  Khi nói về likelihood, dữ liệu $D$ được giữ cố định và $\theta$ là biến ta thay đổi. MLE chọn giá trị $\theta$ làm likelihood lớn nhất:

```math
\hat\theta_{\text{MLE}} = \arg\max_\theta L(\theta;D)
```

  Ví dụ, sau khi quan sát $7$ lần ngửa và $3$ lần sấp, ta so sánh các giá trị $p=0.5,0.6,0.7,\ldots$ để xem giá trị nào giải thích dữ liệu tốt nhất.

- What is the difference between probability and likelihood?

  Hai khái niệm có thể dùng cùng biểu thức $P(D\mid\theta)$ nhưng giữ cố định những đại lượng khác nhau:

  - **Probability:** cố định tham số $\theta$, xem dữ liệu $D$ là biến. Câu hỏi là: “Với mô hình này, dữ liệu nào có thể xuất hiện và xác suất là bao nhiêu?”.
  - **Likelihood:** cố định dữ liệu đã quan sát $D$, xem tham số $\theta$ là biến. Câu hỏi là: “Tham số nào giải thích dữ liệu này tốt nhất?”.

  Probability phải tạo thành một phân phối hợp lệ trên mọi dữ liệu có thể xảy ra:

```math
\sum_D P(D\mid\theta)=1
```

  Ngược lại, likelihood không phải phân phối xác suất trên $\theta$ và không bắt buộc có tổng bằng $1$ theo $\theta$. Muốn có phân phối xác suất của tham số sau khi thấy dữ liệu, ta cần Bayesian posterior $P(\theta\mid D)$.

- What does theta represent?

  $\theta$ đại diện cho toàn bộ **tham số chưa biết của mô hình** mà quá trình huấn luyện cần ước lượng từ dữ liệu.

  - Với đồng xu, $\theta=p$, là xác suất xuất hiện mặt ngửa.
  - Với linear regression, $\theta$ có thể gồm vector trọng số $w$, bias $b$ và phương sai nhiễu $\sigma^2$.
  - Với neural network hoặc LLM, $\theta$ là toàn bộ weight và bias của mô hình, có thể lên đến hàng tỷ tham số.

  $\theta$ không phải dữ liệu đầu vào $x$ hay nhãn $y$; nó là những giá trị bên trong mô hình được optimizer cập nhật trong quá trình training.

- What does argmax mean?

  $\arg\max$ trả về **đầu vào làm hàm đạt giá trị lớn nhất**, không phải bản thân giá trị lớn nhất.

  Ví dụ:

```math
f(p)=p^7(1-p)^3
```

```math
\arg\max_{0\leq p\leq1}f(p)=0.7
```

  Trong khi đó:

```math
\max_{0\leq p\leq1}f(p) = f(0.7) \approx0.0022236
```

  Vì vậy, $\hat\theta=\arg\max_\theta L(\theta)$ có nghĩa là tìm **bộ tham số** tốt nhất, không phải chỉ tìm giá trị likelihood lớn nhất.

- Why do we multiply probabilities across samples?

  Nếu các sample độc lập có điều kiện khi biết $\theta$, joint probability của toàn bộ dataset bằng tích xác suất của từng sample:

```math
P(D\mid\theta) = P(D_1,D_2,\ldots,D_N\mid\theta) = \prod_{i=1}^{N}P(D_i\mid\theta)
```

  Trong supervised learning, với $D_i=(x_i,y_i)$ và coi $x_i$ là đầu vào đã cho:

```math
L(\theta) = \prod_{i=1}^{N}p_\theta(y_i\mid x_i)
```

  Ta nhân vì mô hình cần giải thích **đồng thời tất cả sample**. Chỉ cần gán xác suất rất thấp cho một sample thì likelihood chung cũng giảm mạnh.

  Nếu các sample không độc lập thì không được tùy ý nhân các marginal probability. Khi đó phải mô hình hóa joint distribution hoặc dùng chain rule với các conditional probability thích hợp.

- Why do we take the logarithm of likelihood?

  Ta dùng log-likelihood vì bốn lý do chính:

  1. Log là hàm đồng biến, nên không thay đổi vị trí nghiệm tối ưu:

```math
\arg\max_\theta L(\theta) = \arg\max_\theta\log L(\theta)
```

  2. Log biến tích thành tổng, làm công thức và đạo hàm đơn giản hơn:

```math
\log\prod_i p_i = \sum_i\log p_i
```

  3. Tổng log-probability dễ tính theo mini-batch và dễ phân rã contribution của từng sample hoặc token.
  4. Tích hàng triệu xác suất nhỏ có thể bị làm tròn về $0$ trên máy tính; cộng log-probability giúp tránh numerical underflow.

- Why does log convert multiplication into addition?

  Đây là tính chất cơ bản của logarithm:

```math
\log(ab)=\log a+\log b
```

  Có thể thấy từ định nghĩa số mũ. Nếu $a=e^x$ và $b=e^y$ thì:

```math
ab=e^xe^y=e^{x+y}
```

  Do đó:

```math
\log(ab)=x+y=\log a+\log b
```

  Tương tự, lũy thừa được đưa xuống thành hệ số:

```math
\log(a^n)=n\log a
```

  Vì vậy:

```math
\log\left[p^7(1-p)^3\right] = 7\log p+3\log(1-p)
```

- Why is negative log-likelihood minimized?

  MLE yêu cầu tối đa hóa likelihood. Vì log đồng biến, ta có thể tối đa hóa log-likelihood. Nhân thêm dấu âm sẽ đổi bài toán cực đại thành cực tiểu:

```math
\arg\max_\theta L(\theta) = \arg\max_\theta\log L(\theta) = \arg\min_\theta\left[-\log L(\theta)\right]
```

  Các thư viện machine learning và optimizer thường được thiết kế để **minimize loss**, nên ta dùng Negative Log-Likelihood — NLL làm loss. NLL nhỏ nghĩa là mô hình gán xác suất cao cho dữ liệu đúng; NLL lớn nghĩa là mô hình cho rằng dữ liệu đúng khó xảy ra.

- What is the MLE of the probability of heads if a coin gives 7 heads and 3 tails?

  Gọi $p$ là xác suất xuất hiện mặt ngửa. Likelihood là:

```math
L(p)=p^7(1-p)^3
```

  Log-likelihood:

```math
\ell(p)=7\log p+3\log(1-p)
```

  Cho đạo hàm bằng $0$:

```math
\frac{d\ell}{dp} = \frac{7}{p}-\frac{3}{1-p} =0
```

  Suy ra:

```math
\hat p_{\text{MLE}} = \frac{7}{7+3} =0.7
```

  Tổng quát, nếu có $h$ lần ngửa trong $N$ lần tung thì $\hat p_{\text{MLE}}=h/N$.

- What is MAP estimation?

  **Maximum A Posteriori — MAP** chọn tham số có posterior probability lớn nhất sau khi kết hợp dữ liệu với niềm tin có trước về tham số:

```math
\hat\theta_{\text{MAP}} = \arg\max_\theta P(\theta\mid D)
```

  Theo định lý Bayes:

```math
P(\theta\mid D) = \frac{P(D\mid\theta)P(\theta)}{P(D)}
```

  Vì $P(D)$ không phụ thuộc vào $\theta$:

```math
\hat\theta_{\text{MAP}} = \arg\max_\theta \left[ \log P(D\mid\theta)+\log P(\theta) \right]
```

  $P(\theta)$ là prior, giúp đưa kiến thức hoặc giả định có trước vào quá trình ước lượng. Ví dụ, Gaussian prior trên weight dẫn đến một penalty có dạng L2.

- MLE vs MAP?

  | MLE | MAP |
  |---|---|
  | Tối đa hóa $P(D\mid\theta)$ | Tối đa hóa $P(\theta\mid D)$ |
  | Chỉ sử dụng dữ liệu quan sát | Kết hợp likelihood với prior $P(\theta)$ |
  | Không trực tiếp ưu tiên giá trị tham số nào | Có thể ưu tiên các tham số hợp lý theo kiến thức có trước |
  | Dễ overfit hơn khi dữ liệu ít | Prior có thể regularize khi dữ liệu ít |
  | Tương đương MAP nếu prior là uniform | Khi dữ liệu rất lớn, ảnh hưởng của prior thường giảm |

  Dưới dạng bài toán cực tiểu:

```math
\text{MLE:} \qquad \min_\theta -\log P(D\mid\theta)
```

```math
\text{MAP:} \qquad \min_\theta \left[ -\log P(D\mid\theta)-\log P(\theta) \right]
```

  Có thể hiểu MAP là MLE cộng thêm một regularization term đến từ prior.

- How is MLE related to cross-entropy?

  Với bài toán phân loại, model dự đoán phân phối $p_\theta(k\mid x)$ trên các class. Nếu target $y$ là one-hot vector, cross-entropy của một sample là:

```math
\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K}y_k\log p_\theta(k\mid x)
```

  Vì chỉ class đúng $y$ có giá trị $1$, biểu thức rút gọn thành:

```math
\mathcal{L}_{\text{CE}} = -\log p_\theta(y\mid x)
```

  Với toàn bộ dataset:

```math
\sum_{i=1}^{N}\mathcal{L}_{\text{CE}}^{(i)} = -\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i) = -\log\prod_{i=1}^{N}p_\theta(y_i\mid x_i)
```

  Vế phải chính là negative conditional log-likelihood. Vì vậy, với one-hot target, **minimize cross-entropy tương đương với maximum likelihood estimation**. Nếu dùng soft target hoặc label smoothing, cross-entropy vẫn hợp lệ nhưng không còn đơn giản là NLL của một class duy nhất.

- How is likelihood used in LLM training?

  Với chuỗi token $x_1,x_2,\ldots,x_T$, LLM tự hồi quy mô hình hóa xác suất của cả chuỗi bằng:

```math
p_\theta(x_1,\ldots,x_T) = \prod_{t=1}^{T}p_\theta(x_t\mid x_{\lt t})
```

  MLE tìm tham số $\theta$ làm xác suất của các chuỗi trong training corpus lớn nhất. Trong thực tế, model minimize token-level negative log-likelihood:

```math
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t\mid x_{\lt t})
```

  Đây cũng là cross-entropy giữa phân phối dự đoán trên vocabulary và token đúng tại mỗi vị trí. Khi training, model thường dùng **teacher forcing**: để dự đoán $x_t$, context $x_{\lt t}$ lấy từ chuỗi thật thay vì các token model tự sinh trước đó.

  Mục tiêu này dạy model tăng xác suất cho token thực sự xuất hiện trong ngữ cảnh của corpus. Nó không trực tiếp dạy model “hiểu” hay “nói thật”; các khả năng đó xuất hiện gián tiếp từ việc học cấu trúc thống kê của dữ liệu và các giai đoạn alignment bổ sung.

- Why does autoregressive language modeling multiply conditional token probabilities?

  Vì **chain rule của xác suất** phân rã joint probability của một chuỗi thành tích các conditional probability:

```math
P(x_1,x_2,\ldots,x_T) = P(x_1) P(x_2\mid x_1) P(x_3\mid x_1,x_2) \cdots P(x_T\mid x_{\lt T})
```

  Viết gọn:

```math
P(x_{1:T}) = \prod_{t=1}^{T}P(x_t\mid x_{\lt t})
```

  Phép phân rã này **không giả định các token độc lập**. Ngược lại, xác suất của mỗi token phụ thuộc vào toàn bộ token đứng trước nó. Ta nhân vì xác suất của cả câu phải bao gồm việc token thứ nhất xuất hiện, rồi token thứ hai xuất hiện trong context thứ nhất, và tiếp tục như vậy đến cuối chuỗi.

  Khi lấy log, sequence log-likelihood trở thành tổng token log-likelihood:

```math
\log P(x_{1:T}) = \sum_{t=1}^{T}\log P(x_t\mid x_{\lt t})
```

  Nhờ đó, training loss có thể được tính cho tất cả vị trí token song song mặc dù mô hình vẫn tuân theo factorization từ trái sang phải.

---

### 3.5 Markov Chain

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

### 3.6 Derive MSE from Gaussian Maximum Likelihood

**Mức đã trao đổi:** **3/5**

### Mạch suy luận đầy đủ: MLE → NLL → MSE

Giả sử model dự đoán:

```math
\hat y_i=f_\theta(x_i)
```

và residual tuân theo Gaussian:

```math
\epsilon_i=y_i-\hat y_i\sim\mathcal N(0,\sigma^2)
```

Khi đó conditional probability của một sample là:

```math
p(y_i\mid x_i,\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-\hat y_i)^2}{2\sigma^2}\right)
```

Model cần chọn $\theta$ sao cho dữ liệu đã quan sát có xác suất lớn nhất. Đây chính là Maximum Likelihood Estimation:

```math
\hat\theta_{\mathrm{MLE}}=\arg\max_\theta\prod_{i=1}^{N}p(y_i\mid x_i,\theta)
```

Lấy negative log biến tích thành tổng:

```math
-\log L(\theta)=\frac{N}{2}\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\sum_{i=1}^{N}(y_i-\hat y_i)^2
```

Nếu $\sigma^2$ cố định, term đầu là constant và $1/(2\sigma^2)$ là positive constant. Do đó:

```math
\arg\max_\theta L(\theta)=\arg\min_\theta\sum_{i=1}^{N}(y_i-\hat y_i)^2
```

Biểu thức bên phải là Sum of Squared Errors — SSE. Chia cho dataset size $N$ cho Mean Squared Error — MSE mà không làm thay đổi nghiệm tối ưu:

```math
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
```

Mạch cần nhớ là:

```text
Gaussian noise assumption
           ↓
     likelihood model
           ↓
 maximize likelihood — MLE
           ↓
  minimize negative log-likelihood — NLL
           ↓
      minimize SSE/MSE
```

Vì vậy, câu “Gaussian dẫn tới MSE” là cách nói rút gọn. Chính xác hơn:

```math
\boxed{\text{Gaussian noise}+\text{Maximum Likelihood}\Longrightarrow\text{MSE}}
```

### Interviewer có thể hỏi thêm

#### 1. Why does Gaussian noise lead to MSE?

Giả sử:

```math
y_i=f_\theta(x_i)+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma^2)
```

Gaussian likelihood của một sample là:

```math
p(y_i\mid x_i,\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right)
```

MLE chọn $\theta$ làm likelihood của toàn bộ dữ liệu lớn nhất. Negative log-likelihood tương ứng là:

```math
-\log L(\theta)=\text{constant}+\frac{1}{2\sigma^2}\sum_i(y_i-\hat y_i)^2
```

Khi $\sigma^2$ cố định, minimize NLL tương đương minimize SSE. MSE chỉ là SSE chia cho positive constant $N$, nên có cùng nghiệm tối ưu:

```math
\boxed{\text{Gaussian noise assumption}+\text{MLE}\Longrightarrow\text{MSE}}
```

#### 2. Where does the exponential disappear after taking log?

Viết Gaussian dưới dạng:

```math
p(y_i\mid x_i,\theta)=C\exp\left(-\frac{(y_i-\hat y_i)^2}{2\sigma^2}\right)
```

Sau khi lấy log:

```math
\log p(y_i\mid x_i,\theta)=\log C-\frac{(y_i-\hat y_i)^2}{2\sigma^2}
```

Exponential không bị bỏ tùy ý mà bị triệt tiêu bởi identity:

```math
\boxed{\log(\exp z)=z}
```

Log cũng biến tích likelihood của nhiều samples thành tổng, giúp việc tính toán và tối ưu dễ hơn:

```math
\log\prod_i p_i=\sum_i\log p_i
```

#### 3. Why can constants be ignored during optimization?

Nếu objective có dạng:

```math
J(\theta)=aF(\theta)+b,\qquad a\gt 0
```

trong đó $a$ và $b$ không phụ thuộc vào $\theta$, thì:

```math
\arg\min_\theta J(\theta)=\arg\min_\theta F(\theta)
```

Cộng cùng một constant vào mọi giá trị hoặc nhân mọi giá trị với cùng một positive constant không thay đổi vị trí minimum.

Trong Gaussian NLL, nếu $\sigma^2$ cố định thì $N\log(2\pi\sigma^2)/2$ và $1/(2\sigma^2)$ không ảnh hưởng tới nghiệm tối ưu theo $\theta$. Nếu $\sigma^2$ cũng là parameter cần học thì chúng không còn là constants và không được bỏ.

#### 4. What assumption do we make about the residual error?

Residual là:

```math
\epsilon_i=y_i-\hat y_i
```

Để suy ra MSE từ Gaussian MLE, ta thường giả định:

```math
\epsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2)
```

Điều này bao gồm các giả định:

- Residual có mean bằng zero.
- Residual tuân theo Gaussian distribution.
- Các residual độc lập khi đã biết input.
- Mọi samples có cùng variance $\sigma^2$.
- Variance không phụ thuộc vào input $x_i$.
- Model dự đoán conditional mean $\hat y_i=\mathbb E[Y\mid X=x_i]$.

Nếu residual có systematic bias hoặc variance thay đổi theo input thì giả định này không còn phù hợp hoàn toàn.

#### 5. What if variance is not constant?

Nếu variance phụ thuộc vào sample hoặc input:

```math
\epsilon_i\sim\mathcal N(0,\sigma_i^2)
```

thì Gaussian NLL trở thành:

```math
\mathcal L=\sum_i\left[\frac12\log(2\pi\sigma_i^2)+\frac{(y_i-\mu_i)^2}{2\sigma_i^2}\right]
```

Bỏ constant $\log(2\pi)/2$:

```math
\boxed{\mathcal L=\sum_i\left[\frac12\log\sigma_i^2+\frac{(y_i-\mu_i)^2}{2\sigma_i^2}\right]}
```

Sample có uncertainty lớn nhận trọng số nhỏ hơn thông qua $1/\sigma_i^2$. Term $\log\sigma_i^2$ ngăn model tăng variance vô hạn để giảm squared-error penalty.

Nếu $\sigma_i^2$ đã biết, bài toán trở thành weighted least squares:

```math
\mathcal L\propto\sum_i\frac{(y_i-\hat y_i)^2}{\sigma_i^2}
```

Nếu variance chưa biết, model có thể học đồng thời $\mu_\theta(x)$ và $\log\sigma_\theta^2(x)$.

#### 6. What loss would another probability distribution lead to?

Mỗi giả định phân phối dẫn tới một negative log-likelihood khác nhau:

| Distribution | Loss từ negative log-likelihood |
|---|---|
| Gaussian | MSE/L2 loss |
| Laplace | MAE/L1 loss |
| Bernoulli | Binary Cross-Entropy |
| Categorical | Multi-class Cross-Entropy |
| Poisson | Poisson NLL |
| Student's t | Robust heavy-tailed loss |

Ví dụ, với Laplace noise:

```math
p(\epsilon)=\frac{1}{2b}\exp\left(-\frac{|\epsilon|}{b}\right)
```

Negative log-likelihood là:

```math
-\log p(\epsilon)=\log(2b)+\frac{|\epsilon|}{b}
```

Nếu $b$ cố định, MLE tương đương minimize MAE:

```math
\boxed{\text{Laplace noise}+\text{MLE}\Longrightarrow\text{MAE}}
```

Quy tắc tổng quát là:

```math
\boxed{\text{probability model}\Longrightarrow\text{NLL tương ứng}\Longrightarrow\text{loss function}}
```

#### 7. Why is MSE sensitive to outliers?

MSE bình phương residual:

```math
\mathrm{MSE}=\frac1N\sum_i r_i^2
```

Nếu residual tăng gấp $10$ lần thì contribution vào loss tăng gấp $100$ lần. Gradient của squared error cũng tăng theo độ lớn của residual:

```math
\frac{\partial r^2}{\partial\hat y}=-2r
```

Vì vậy, một số ít outliers có thể tạo loss và gradient rất lớn, khiến model thay đổi mạnh để cố fit chúng. MSE phù hợp với Gaussian noise có tail tương đối mỏng nhưng kém robust khi dữ liệu có extreme errors hoặc heavy-tailed noise.

#### 8. MSE vs MAE?

| Đặc điểm | MSE | MAE |
|---|---|---|
| Công thức | $\frac1N\sum_i r_i^2$ | $\frac1N\sum_i\lvert r_i\rvert$ |
| Noise assumption | Gaussian | Laplace |
| Conditional estimate | Mean | Median |
| Outlier sensitivity | Cao | Thấp hơn |
| Gradient magnitude | Tăng theo $\lvert r\rvert$ | Gần như constant |
| Smooth tại zero | Có | Không |
| Cách phạt lỗi lớn | Quadratic | Linear |

Với MSE:

```math
\frac{\partial r^2}{\partial\hat y}=-2r
```

Residual càng lớn thì gradient càng lớn. MSE tối ưu mượt nhưng nhạy với outliers.

Với MAE:

```math
\frac{\partial|r|}{\partial\hat y}=-\mathrm{sign}(r)
```

Gradient có độ lớn gần như không đổi, nên outlier ít chi phối hơn. Tuy nhiên, MAE không khả vi tại residual bằng zero và có thể khó tối ưu hơn. Huber loss là lựa chọn trung gian: quadratic với lỗi nhỏ và gần linear với lỗi lớn.

#### 9. What is heteroscedastic regression?

**Heteroscedastic regression** là regression trong đó noise variance thay đổi theo input:

```math
Y\mid X=x\sim\mathcal N\left(\mu_\theta(x),\sigma_\theta^2(x)\right)
```

Ví dụ, depth prediction ở vùng ảnh rõ có thể có uncertainty thấp, còn vùng bị che khuất có uncertainty cao. Model thường dự đoán hai outputs:

```math
\mu_\theta(x),\qquad s_\theta(x)=\log\sigma_\theta^2(x)
```

Gaussian NLL của một sample có thể viết ổn định dưới dạng:

```math
\boxed{\mathcal L_i=\frac12\left[s_i+(y_i-\mu_i)^2e^{-s_i}\right]}
```

Trong đó $\mu_i$ là prediction, còn $s_i$ biểu diễn uncertainty. Model không chỉ dự đoán giá trị mà còn học mức độ không chắc chắn của prediction đó.

#### 10. Why is minimizing SSE equivalent to minimizing MSE for a fixed dataset size?

SSE và MSE lần lượt là:

```math
\mathrm{SSE}(\theta)=\sum_{i=1}^{N}(y_i-\hat y_i)^2
```

```math
\mathrm{MSE}(\theta)=\frac1N\mathrm{SSE}(\theta)
```

Với dataset cố định, $N$ là positive constant. Positive scaling không làm thay đổi vị trí minimum:

```math
\boxed{\arg\min_\theta\mathrm{SSE}(\theta)=\arg\min_\theta\mathrm{MSE}(\theta)}
```

Gradient chỉ khác nhau về scale:

```math
\nabla_\theta\mathrm{MSE}=\frac1N\nabla_\theta\mathrm{SSE}
```

Hai objective có cùng optimum nhưng giá trị loss và độ lớn gradient khác nhau, nên learning rate phù hợp có thể khác. Nếu batch size thay đổi hoặc objective có thêm regularization term, các thành phần phải được scale nhất quán.

---

### 3.7 Từ Markov Chain đến các nhánh Reinforcement Learning, VLA và π0.5

> Phần này mô tả bức tranh của lĩnh vực tính đến **tháng 9/2026**. Tên thuật toán và thuật ngữ tiếng Anh được giữ lại để tiện đọc paper, còn phần giải thích được viết bằng tiếng Việt.

### Câu trả lời quan trọng nhất

**Reinforcement Learning không có một số lượng “nhánh” cố định được toàn ngành thống nhất.** Một thuật toán có thể đồng thời thuộc nhiều nhóm khác nhau.

Ví dụ, **SAC** đồng thời là:

- Model-free RL.
- Off-policy RL.
- Actor–Critic.
- Deep RL.
- Maximum-entropy RL.
- Thường được dùng trong online RL, nhưng cũng có biến thể offline.

Vì vậy, không nên đếm `model-based`, `offline`, `actor–critic`, `multi-agent` và `safe RL` như các nhánh loại trừ lẫn nhau. Chúng là các **trục phân loại khác nhau**.

Và điểm dễ nhầm nhất:

```math
\boxed{\text{VLA không phải là một nhánh RL thuần túy}}
```

VLA — Vision-Language-Action — là một loại **robot policy/foundation model** ánh xạ quan sát và instruction thành hành động. VLA có thể được huấn luyện bằng imitation learning, supervised learning, generative modeling, offline RL hoặc online RL.

**π0.5** là một VLA/robotic foundation model, không phải một thuật toán RL như Q-learning, PPO hay SAC. Bản π0.5 chủ yếu học từ heterogeneous data và robot demonstrations bằng supervised co-training. Các thế hệ như **π*0.6** mới đưa RL từ trải nghiệm robot vào pipeline một cách rõ ràng.

---

### 1. Chuỗi nền tảng: Markov Chain → MRP → MDP/POMDP → Planning hoặc RL

```text
Markov Chain
state + transition
      │
      │ thêm reward
      ▼
Markov Reward Process — MRP
state + transition + reward + discount
      │
      │ thêm action và quyền ra quyết định
      ▼
Markov Decision Process — MDP
state + action + transition + reward + discount
      │
      ├── quan sát đầy đủ state → MDP
      │
      └── state thật bị ẩn → POMDP
                             observation + belief/history/memory

MDP hoặc POMDP
      ├── biết model và giải được → Dynamic Programming / Planning
      │
      └── model chưa biết hoặc bài toán quá lớn
                             ▼
                  Reinforcement Learning
```

#### 1.1 Markov Chain — môi trường tự chuyển trạng thái

Markov Chain chỉ mô tả dynamics:

```math
P(S_{t+1}\mid S_t)
```

Không có action để agent lựa chọn và chưa cần có reward.

#### 1.2 Markov Reward Process — thêm reward

MRP thường được viết:

```math
\left(S,P,R,\gamma\right)
```

Mỗi transition hoặc state có reward, nhưng agent vẫn chưa có action để điều khiển hệ thống.

#### 1.3 Markov Decision Process — thêm action

MDP thường được viết:

```math
\left(S,A,P,R,\gamma\right)
```

Dynamics trở thành:

```math
P(S_{t+1}=s'\mid S_t=s,A_t=a)
```

Agent dùng policy:

```math
\pi(a\mid s)=P(A_t=a\mid S_t=s)
```

để tối đa hóa expected return:

```math
J(\pi)=\mathbb E_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\right]
```

#### 1.4 Reinforcement Learning — học cách giải MDP từ dữ liệu

MDP là **mô hình toán của bài toán**. RL là **họ phương pháp học policy/value/model** khi agent phải dùng interaction hoặc dataset thay vì được cho sẵn lời giải.

```math
\boxed{\text{MDP}=\text{problem formulation},\qquad \text{RL}=\text{solution methods}}
```

Nếu transition và reward model đã biết, ta có thể dùng Dynamic Programming hoặc planning mà không cần học từ trial-and-error. Nếu model chưa biết, agent phải estimate value, policy hoặc dynamics từ experience — đây là bối cảnh điển hình của RL.

#### 1.5 POMDP — state thật bị ẩn

Trong robotics, agent thường chỉ nhận observation $O_t$ thay vì state thật $S_t$:

```math
O_t\sim P(O_t\mid S_t)
```

Policy khi đó có thể phụ thuộc vào history, memory hoặc belief state:

```math
\pi(a_t\mid o_{\leq t},a_{\lt t})
```

Camera bị occlusion, sensor có noise và velocity không được quan sát trực tiếp khiến phần lớn bài toán robot thực tế gần với POMDP hơn MDP fully observable.

#### 1.6 Bandit — trường hợp quyết định một bước

Multi-armed bandit hoặc contextual bandit có thể xem là bài toán decision-making đơn giản hơn MDP: agent chọn action và nhận reward nhưng không cần mô hình hóa chuỗi state transitions dài hạn. Bandit quan trọng trong recommendation và exploration, nhưng không phải nguồn duy nhất hay toàn bộ của RL tuần tự.

---

### 2. RL hiện đại được phân loại theo những trục nào?

| Trục phân loại | Các nhóm thường gặp | Câu hỏi cần trả lời |
|---|---|---|
| Có dùng dynamics model không? | Model-free / Model-based | Agent có học hoặc dùng $P(s'\mid s,a)$ không? |
| Học đối tượng nào? | Value-based / Policy-based / Actor–Critic | Agent học $Q$, học policy trực tiếp hay học cả hai? |
| Data đến từ đâu? | Online / Offline | Agent có tiếp tục tương tác với environment không? |
| Data do policy nào tạo? | On-policy / Off-policy | Data có phải do chính policy hiện tại sinh ra không? |
| State có quan sát đầy đủ không? | MDP / POMDP / Memory-based RL | Agent thấy state thật hay chỉ thấy observation/history? |
| Có bao nhiêu agent? | Single-agent / Multi-agent RL | Một hay nhiều agent cùng hợp tác hoặc cạnh tranh? |
| Task có cấu trúc gì? | Flat / Hierarchical / Goal-conditioned | Có subgoal, option hoặc skill hierarchy không? |
| Reward đến từ đâu? | Environment / Human preference / Learned reward / Intrinsic reward | Tín hiệu đánh giá được định nghĩa bằng cách nào? |
| Objective có ràng buộc không? | Standard / Safe / Constrained / Risk-sensitive RL | Chỉ tối đa hóa return hay còn phải giữ an toàn? |
| Khả năng thích nghi | Single-task / Multi-task / Meta-RL | Học một task, nhiều task hay học cách thích nghi nhanh? |

Một model có thể nằm tại một ô trên **mỗi trục**. Vì vậy, câu hỏi “RL có bao nhiêu nhánh?” nên được trả lời là:

> Không có một con số chuẩn. Cách đúng là nêu các trục phân loại, sau đó đặt từng thuật toán vào nhiều trục tương ứng.

---

### 3. Các họ thuật toán cốt lõi

#### 3.1 Dynamic Programming — biết đầy đủ model

Dynamic Programming dùng Bellman equations khi transition và reward model đã biết.

Bellman expectation equation:

```math
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+\gamma V^\pi(s')\right]
```

Các thuật toán tiêu biểu:

- Policy Evaluation.
- Policy Iteration.
- Value Iteration.

Dynamic Programming là nền tảng của RL, nhưng thường được xem là planning vì không cần học model từ experience.

#### 3.2 Monte Carlo methods — học từ return đầy đủ

Monte Carlo đợi episode kết thúc rồi dùng observed return:

```math
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
```

Ưu điểm là không cần dynamics model và target không bootstrap. Hạn chế là variance cao và thường cần episodic tasks.

> Monte Carlo method trong RL không phải Markov Chain; hai khái niệm chỉ vô tình cùng viết tắt là MC.

#### 3.3 Temporal-Difference Learning — học từng bước và bootstrap

TD kết hợp ý tưởng sampling của Monte Carlo với Bellman bootstrap của Dynamic Programming:

```math
V(S_t)\leftarrow V(S_t)+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right]
```

TD error là:

```math
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
```

Các thuật toán như SARSA, Q-learning và nhiều Actor–Critic method đều sử dụng TD learning.

#### 3.4 Value-based RL — học giá trị rồi suy ra action

Agent học $V(s)$ hoặc $Q(s,a)$. Với optimal action-value function:

```math
\pi(s)=\arg\max_aQ(s,a)
```

Q-learning update:

```math
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]
```

Ví dụ:

- Q-learning.
- SARSA.
- DQN.
- Double DQN, Dueling DQN, C51, Rainbow.

Value-based methods tự nhiên với discrete actions; continuous actions làm phép $\arg\max_aQ(s,a)$ khó hơn.

#### 3.5 Policy-based RL — tối ưu policy trực tiếp

Policy được parameterize trực tiếp:

```math
\pi_\theta(a\mid s)
```

Policy Gradient tối đa hóa:

```math
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[R(\tau)\right]
```

với gradient dạng REINFORCE:

```math
\nabla_\theta J(\theta)=\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right]
```

Policy-based methods phù hợp với stochastic policy và continuous actions, nhưng gradient có thể có variance cao.

Ví dụ:

- REINFORCE/VPG.
- TRPO.
- PPO.

#### 3.6 Actor–Critic — policy và value học cùng nhau

Actor sinh action; Critic đánh giá action hoặc state:

```text
Actor:  πθ(a|s)  → chọn hành động
Critic: Vφ(s) hoặc Qφ(s,a) → đánh giá hành động
```

Critic giúp giảm variance hoặc cung cấp learning signal cho Actor.

Ví dụ:

- A2C/A3C.
- DDPG.
- TD3.
- SAC.
- PPO implementations thường dùng Actor–Critic architecture, dù PPO được xếp trong policy optimization.

#### 3.7 Model-based RL — học hoặc dùng world model

Model-based RL dùng dynamics/reward model để planning hoặc tạo imagined rollouts:

```math
\hat P_\phi(s'\mid s,a),\qquad \hat R_\phi(s,a)
```

Model có thể được cho sẵn hoặc học từ data. Sau đó agent có thể dùng search, Model Predictive Control hoặc policy learning trong imagined trajectories.

Ví dụ:

- Dyna.
- PILCO.
- PETS/MBPO.
- Dreamer.
- MuZero.

Model-based thường sample-efficient hơn nhưng có nguy cơ model bias: prediction error tích lũy khi rollout dài.

---

### 4. Các hướng hiện đại giao nhau với những họ cốt lõi

#### 4.1 Online RL và Offline RL

**Online RL:** agent vừa thu thập experience vừa update policy. Data distribution thay đổi cùng policy.

**Offline RL:** agent chỉ học từ dataset cố định, không được thử action mới trong environment khi training.

Offline RL khó hơn behavior cloning vì nó vẫn muốn tối ưu reward và chọn action tốt hơn behavior policy, nhưng phải tránh extrapolation sang actions nằm ngoài dataset.

Ví dụ offline RL:

- BCQ.
- CQL.
- IQL.
- Decision Transformer — mô hình hóa trajectory như sequence có điều kiện trên return; thường được đặt cạnh offline RL dù không dùng Bellman backup truyền thống.

#### 4.2 On-policy và Off-policy

**On-policy:** update bằng data từ policy hiện tại hoặc rất gần policy hiện tại. Ví dụ SARSA, REINFORCE, PPO.

**Off-policy:** có thể học từ data của policy khác hoặc từ replay buffer. Ví dụ Q-learning, DQN, DDPG, TD3, SAC.

On-policy thường ổn định hơn nhưng tốn samples. Off-policy tái sử dụng data tốt hơn nhưng có thể kém ổn định do distribution mismatch.

#### 4.3 Imitation Learning — họ hàng gần, không phải RL thuần túy

Behavior Cloning học trực tiếp từ demonstrations:

```math
\mathcal L_{\mathrm{BC}}(\theta)=-\sum_t\log\pi_\theta(a_t^{\mathrm{expert}}\mid o_t)
```

Không có reward maximization hoặc trial-and-error nên Behavior Cloning thường được xếp vào **Imitation Learning**, không phải RL strict.

Các hướng liên quan:

- Behavior Cloning — supervised imitation.
- DAgger — interactive imitation, thu thập correction từ expert.
- Inverse RL — suy ra reward từ demonstrations, sau đó giải RL.
- GAIL — adversarial imitation.

Đây là nhánh cực kỳ quan trọng để hiểu VLA vì phần lớn VLA ban đầu học action từ robot demonstrations.

#### 4.4 Preference-based RL và RLHF

Thay vì tự viết reward, con người so sánh các outputs hoặc trajectories. Preference data có thể dùng để học reward model, sau đó policy được fine-tune bằng RL.

```text
human comparisons
       ↓
learned reward model
       ↓
RL policy optimization
```

RLHF là một trường hợp của preference-based learning. Tuy nhiên, các phương pháp như DPO tối ưu trực tiếp preference objective và không phải RL theo nghĩa tương tác với MDP.

#### 4.5 Hierarchical RL

Agent ra quyết định ở nhiều temporal scales:

```text
high-level policy: chọn goal/subtask/skill
                         ↓
low-level policy: sinh motor actions
```

Khái niệm tiêu biểu gồm options, skills và subgoals. Một VLA có high-level subtask predictor chưa tự động trở thành Hierarchical RL; chỉ gọi là Hierarchical RL khi hierarchy được học hoặc tối ưu bằng RL objective/reward.

#### 4.6 Multi-Agent RL

Nhiều agents cùng tương tác trong một environment:

- Cooperative — hợp tác.
- Competitive — cạnh tranh.
- Mixed — vừa hợp tác vừa cạnh tranh.

Khó khăn chính là non-stationarity: policy của các agent khác cũng thay đổi trong lúc một agent đang học.

#### 4.7 Goal-conditioned, Multi-task và Meta-RL

Goal-conditioned policy:

```math
\pi(a\mid s,g)
```

Multi-task RL học nhiều task trong một policy. Meta-RL học cách thích nghi nhanh sang task mới từ ít experience.

Các hướng này gần mục tiêu generalist robotics, nhưng không đồng nghĩa với VLA: VLA thêm language/vision foundation-model pretraining và action generation ở quy mô lớn.

#### 4.8 Safe, Constrained và Risk-sensitive RL

Ngoài reward, agent phải thỏa safety cost hoặc constraint:

```math
\max_\pi J_R(\pi)\qquad\text{subject to}\qquad J_C(\pi)\leq d
```

Đây là hướng đặc biệt quan trọng trong robotics vì exploration sai có thể làm hỏng robot hoặc gây nguy hiểm cho con người.

#### 4.9 Distributional RL

RL truyền thống học expected return $Q(s,a)$. Distributional RL học cả random return distribution:

```math
Z(s,a)\overset{D}{=}R+\gamma Z(S',A')
```

Điều này biểu diễn uncertainty và hình dạng của return tốt hơn một giá trị trung bình duy nhất. Ví dụ gồm C51, QR-DQN và IQN.

#### 4.10 Exploration và Intrinsic Motivation

Khi reward hiếm, agent cần chủ động khám phá. Intrinsic reward có thể dựa trên novelty, curiosity, prediction error hoặc information gain.

#### 4.11 Sim-to-Real và Robot RL

Robot RL thường train trong simulation rồi chuyển sang real world bằng:

- Domain randomization.
- System identification.
- Dynamics adaptation.
- Residual RL.
- Fine-tuning bằng real-world data.

Sim-to-real là một deployment strategy trong robot learning, không phải một họ thuật toán RL loại trừ với model-free/model-based.

#### 4.12 Transfer, Continual Learning và Curriculum trong RL

- **Transfer RL:** dùng knowledge từ task/domain cũ cho task mới.
- **Continual/Lifelong RL:** tiếp tục học qua nhiều task mà hạn chế catastrophic forgetting.
- **Curriculum Learning:** sắp xếp task từ dễ đến khó để quá trình học ổn định hơn.
- **Self-play:** tạo đối thủ và curriculum tự động; đặc biệt quan trọng trong game và Multi-Agent RL.

Đây là các chiến lược về cách phân phối task và experience phát triển theo thời gian; chúng có thể kết hợp với model-free, model-based, online hoặc offline RL.

---

### 5. Bản đồ nhanh các thuật toán và model thường gặp

| Thuật toán/model | Phân loại đúng | Ghi nhớ |
|---|---|---|
| Q-learning | Model-free, off-policy, value-based, TD | Học bảng hoặc hàm $Q$ |
| SARSA | Model-free, on-policy, value-based, TD | Target dùng action thật của policy hiện tại |
| DQN | Model-free, off-policy, value-based, Deep RL | Q-learning với neural network và replay buffer |
| REINFORCE | Model-free, on-policy, policy gradient | Dùng Monte Carlo return, variance cao |
| PPO | Model-free, on-policy, policy optimization/Actor–Critic | Update bị giới hạn bằng clipped surrogate |
| DDPG/TD3 | Model-free, off-policy, Actor–Critic | Continuous deterministic control |
| SAC | Model-free, off-policy, Actor–Critic, maximum entropy | Continuous control, sample reuse tốt |
| Dreamer/MuZero | Model-based Deep RL | Học model/latent dynamics để planning hoặc imagination |
| CQL/IQL | Offline RL | Học từ dataset cố định, hạn chế out-of-distribution actions |
| Decision Transformer | Offline sequence modeling | Condition trên desired return, không phải Bellman RL truyền thống |
| Behavior Cloning | Imitation learning, supervised learning | Bắt chước demonstrations, không tối ưu reward trực tiếp |
| Diffusion Policy/ACT | Generative hoặc sequence imitation policy | Robot policy học từ demonstrations; không tự động là RL |
| RT-2/OpenVLA/π0/π0.5 | VLA/robot foundation model | Chủ yếu pretraining + imitation/supervised co-training |
| π*0.6/RECAP | VLA + offline/online RL | Học thêm từ experience, reward và expert corrections |
| RL Token — RLT | VLA + efficient online off-policy Actor–Critic | Dùng representation của VLA để RL một actor/critic nhỏ |

---

### 6. Lịch sử phát triển từ Markov đến RL hiện đại

| Giai đoạn | Bước phát triển | Ý nghĩa |
|---|---|---|
| 1906 | Markov Chain | Mô hình hóa chuyển trạng thái chỉ phụ thuộc hiện tại |
| 1950s–1960s | MDP, Bellman equation, Dynamic Programming | Đặt nền tảng toán học cho sequential decision-making |
| 1980s | Temporal-Difference learning, Actor–Critic sơ khai | Học value từng bước mà không cần biết dynamics |
| 1989–1992 | Q-learning và REINFORCE | Hình thành value-based và policy-gradient model-free RL |
| 1990s–2000s | SARSA, function approximation, Dyna, policy search, robot RL | Đi từ bảng nhỏ sang state/action liên tục |
| 2013–2015 | DQN/Deep RL | Neural network học action-value trực tiếp từ pixels |
| 2015–2018 | TRPO, DDPG, PPO, TD3, SAC | Policy optimization và continuous-control Actor–Critic trưởng thành |
| 2016–2020 | AlphaGo, AlphaZero, MuZero, world models | Kết hợp deep networks, search, self-play và model-based reasoning |
| 2017–nay | Preference-based RL/RLHF | Reward được học từ đánh giá hoặc preference của con người |
| 2018–nay | Offline RL | Học policy từ dataset cố định cho các domain khó exploration trực tiếp |
| 2021 | Decision Transformer và sequence modeling | Biểu diễn decision-making như bài toán modeling trajectory |
| 2022 | RT-1 | Transformer policy được scale bằng multi-task robot demonstrations |
| 2023 | RT-2 và Diffusion Policy | VLM được chuyển thành VLA; generative action policy phát triển mạnh |
| 2024 | OpenVLA và π0 | Open-source VLA; VLM kết hợp action expert/flow matching |
| 2025 | π0.5 | Heterogeneous co-training và high-level semantics cho open-world generalization |
| 2025 | π*0.6/RECAP | VLA được pretrain bằng offline RL rồi cải thiện bằng real-world experience và corrections |
| 2026 | RL Token và π0.7 | RL thích nghi nhanh trên representation của VLA; generalist VLA có compositional generalization mạnh hơn |

Deep RL không thay MDP bằng một bài toán khác. Nó thay bảng $V/Q/\pi$ bằng neural network để xử lý image, language và state spaces lớn.

```math
\boxed{\text{Deep RL}=\text{RL objective}+\text{deep function approximation}}
```

---

### 7. VLA không chỉ phát triển từ RL

Nếu chỉ vẽ `Markov Chain → RL → VLA` thì thiếu ba dòng lịch sử lớn. VLA là điểm hội tụ của nhiều hướng:

```text
Markov Chain → MDP/POMDP → RL/Planning ───────────────┐
                                                       │
Robot demonstrations → Imitation/Behavior Cloning ────┤
                                                       ├→ VLA robot policy
Transformer → LLM/VLM → semantic world knowledge ─────┤
                                                       │
Autoregressive/Diffusion/Flow Matching action models ──┘
```

RL đóng góp framework state–action–reward và khả năng cải thiện từ experience. Nhưng nhiều VLA hiện đại ban đầu được train chủ yếu bằng supervised imitation từ demonstrations, không phải trial-and-error RL.

---

### 8. VLA là gì?

VLA — Vision-Language-Action model — nhận:

- Vision: camera images/video.
- Language: task instruction hoặc subtask.
- Robot state: proprioception, joint state, gripper state.
- Có thể thêm history hoặc memory.

và sinh:

- Discrete action tokens.
- Continuous actions.
- Action chunks gồm nhiều control steps.
- Hoặc high-level plan kết hợp low-level actions.

Có thể viết policy VLA tổng quát là:

```math
\pi_\theta\left(a_{t:t+H}\mid o_{\leq t},\ell,q_t\right)
```

Trong đó:

- $o_{\leq t}$ là image/observation history.
- $\ell$ là language instruction.
- $q_t$ là proprioceptive state.
- $a_{t:t+H}$ là action chunk.

#### VLA khác VLM ở đâu?

```text
VLM: image + text → text/semantic output
VLA: image + text + robot state → robot action
```

Một VLM hiểu “cái cốc nào màu đỏ”; VLA phải biến hiểu biết đó thành trajectory điều khiển robot tiếp cận, grasp và di chuyển cái cốc.

#### VLA có bắt buộc dùng reward không?

Không. Một VLA có thể train bằng Behavior Cloning:

```math
\mathcal L_{\mathrm{VLA}}=-\sum_t\log\pi_\theta(a_t^{\mathrm{expert}}\mid o_t,\ell)
```

hoặc bằng diffusion/flow-matching objective trên continuous action chunks. Những objective này học phân phối action trong demonstrations và không tự động trở thành RL.

#### Diffusion/Flow Matching có phải RL không?

Không. Chúng là cách parameterize và train action generator. Một flow-matching action expert có thể nằm trong policy của một RL agent, nhưng flow-matching loss tự nó không tối đa hóa expected reward.

---

### 9. π0.5 thuộc nhánh nào?

Tên đúng là **π0.5**, đọc là “pi zero point five”. Có thể đặt nó vào bản đồ như sau:

| Trục | Phân loại của π0.5 |
|---|---|
| Lĩnh vực | Robot learning / Embodied AI |
| Loại model | Vision-Language-Action model |
| Vai trò | Generalist robot policy / robotic foundation model |
| Input | Images, language, robot state và heterogeneous prompts/examples |
| Output | High-level semantic subtask và low-level continuous actions/action chunks |
| Action model | Kế thừa họ π0 với action expert dựa trên flow matching |
| Training chính | Supervised co-training/imitation-style learning trên heterogeneous robot, semantic và web data; action expert dùng flow matching |
| Có phải thuật toán RL không? | Không |
| Có thể kết hợp RL không? | Có; các thế hệ π*0.6 và RLT thể hiện rõ bước này |

#### Điều π0.5 cố giải quyết

π0 và nhiều VLA trước đó mạnh trong các environment gần training distribution. π0.5 tập trung vào **open-world generalization**: thực hiện long-horizon manipulation trong ngôi nhà mới, với object và arrangement mới.

Ý tưởng chính là co-training trên heterogeneous data:

- Data từ nhiều robot embodiments.
- Robot demonstrations và low-level actions.
- High-level semantic subtask prediction.
- Verbal instructions.
- Object detection và image-language data.
- Web data để giữ semantic knowledge của VLM.

Pipeline khái niệm:

```text
instruction: “dọn phòng ngủ”
               ↓
VLM/VLA semantic reasoning
               ↓
subtask: “nhặt chiếc gối”
               ↓
flow-matching action expert
               ↓
continuous action chunk cho robot
```

Đây là **hierarchical architecture/inference**, nhưng chưa nên gọi là Hierarchical RL nếu các tầng không được tối ưu bằng reward-based RL.

#### Vì sao π0.5 vẫn liên quan đến MDP/POMDP?

Khi deploy, π0.5 vẫn đóng vai trò policy trong một partially observable control problem:

```math
\pi_\theta(a_t\mid o_{\leq t},\ell)
```

Robot nhận observation, phát action, environment chuyển state rồi trả observation mới. Cấu trúc interaction vẫn là MDP/POMDP. Điểm khác là cách policy được học ban đầu chủ yếu là imitation/co-training, không phải RL reward maximization.

---

### 10. Lịch sử ngắn của họ π và vị trí của RL

#### π0 — 2024: VLA generalist với flow matching

π0 đặt một flow-matching action expert lên pretrained VLM và train trên data từ nhiều robot platforms. Mục tiêu là generalist manipulation, dexterity và action chunk generation.

```text
pretrained VLM + multi-robot demonstrations + flow action expert → π0
```

π0 chủ yếu thuộc imitation/foundation-policy path, không phải một RL algorithm.

#### π0.5 — tháng 4/2025: open-world generalization

π0.5 thêm heterogeneous co-training, high-level semantic prediction và knowledge transfer từ web/multiple robots để generalize sang nhà và object chưa thấy.

```text
π0 + heterogeneous co-training + semantic subtasks + web knowledge → π0.5
```

Base π0.5 vẫn chủ yếu dùng supervised learning.

#### π*0.6 — tháng 11/2025: VLA học từ experience bằng RL

RECAP — RL with Experience and Corrections via Advantage-conditioned Policies — kết hợp:

- Generalist VLA pretraining bằng offline RL.
- Demonstrations.
- On-policy robot experience.
- Expert teleoperation corrections.
- Reward/outcome feedback.

```text
base VLA
   ↓ offline RL pretraining
advantage-conditioned π*0.6
   ↓ demonstrations + autonomous rollouts + corrections + rewards
task-specialized, more reliable policy
```

Đây mới là ví dụ rõ của:

```math
\boxed{\text{VLA}+\text{Offline RL}+\text{Online real-world RL}}
```

#### RL Token — tháng 3/2026: RL nhanh trên representation của VLA

RLT không fine-tune toàn bộ VLA. VLA sinh một compact **RL token**, sau đó một actor và critic nhỏ học bằng sample-efficient off-policy RL để chỉnh action của VLA cho các phase đòi hỏi độ chính xác cao.

```text
frozen VLA representation
          ↓
       RL token
          ↓
small actor + critic + replay buffer
          ↓
refined action close to VLA prior
```

Đây là cầu nối trực tiếp giữa VLA foundation policy và online Actor–Critic.

#### π0.7 — tháng 4/2026: steerable generalist và compositional generalization

π0.7 tập trung vào một unified model có thể nhận language coaching, visual subgoals và nhiều control modalities, thực hiện nhiều dexterous tasks mà không cần fine-tune riêng cho từng task, đồng thời tái kết hợp skills cho task mới.

π0.7 cho thấy hướng phát triển không chỉ là “thêm RL”, mà là kết hợp:

- Scale và data diversity.
- Multimodal prompting.
- Cross-embodiment transfer.
- Compositional generalization.
- Kỹ năng và độ tin cậy đã được cải thiện từ các thế hệ dùng RL.

---

### 11. VLA và RL có thể nằm chung trong một robotics stack như thế nào?

```text
Language instruction
        ↓
VLM/VLA high-level reasoning
        ↓
VLA action-chunk policy — pretrained bằng demonstrations
        ↓
RL residual / actor-critic adaptation — cải thiện reward, speed, precision
        ↓
low-level controller — torque/position control
        ↓
robot + environment
        ↓
observation, reward, success/failure, human correction
        └───────────────────────────────────────────────┘
```

Các tầng có vai trò khác nhau:

| Tầng | Vai trò | Cách học thường gặp |
|---|---|---|
| VLM/high-level planner | Hiểu instruction, chọn subtask | Web pretraining, instruction tuning, supervised learning |
| VLA policy | Biến vision/language thành action chunks | Behavior cloning, token prediction, diffusion/flow matching |
| RL adaptation | Cải thiện success, reward, speed, recovery | Offline RL, online RL, Actor–Critic, preference/reward feedback |
| Low-level controller | Theo dõi pose/velocity/torque target | Classical control, MPC hoặc RL locomotion/control |

Không nên đồng nhất toàn bộ stack với RL. Chỉ những component được tối ưu từ reward/return hoặc interaction objective mới là RL theo nghĩa chặt.

---

### 12. Những câu phân biệt dễ bị hỏi trong phỏng vấn

#### RL có bắt đầu từ Markov Chain không?

Nói chính xác hơn, nền tảng toán học phát triển theo chuỗi:

```math
\text{Markov Chain}\rightarrow\text{MRP}\rightarrow\text{MDP/POMDP}\rightarrow\text{RL methods}
```

Markov Chain cung cấp transition structure; MDP thêm actions và rewards; RL học cách giải MDP khi model hoặc optimal policy chưa biết.

#### RL có bao nhiêu nhánh?

Không có số lượng chuẩn vì các nhóm chồng lấp. Hãy phân loại theo các trục: model-based/model-free, value/policy/actor–critic, online/offline, on/off-policy, single/multi-agent, fully/partially observable, flat/hierarchical và standard/safe RL.

#### VLA có phải là một nhánh RL không?

Không. VLA là một policy architecture/foundation-model paradigm cho robotics. Nó có thể được train bằng imitation learning và sau đó fine-tune bằng RL.

#### π0.5 có được huấn luyện bằng RL không?

π0.5 chủ yếu dùng supervised heterogeneous co-training trên robot demonstrations, semantic tasks và web data. Nó không phải một RL algorithm. π*0.6/RECAP là bước trong họ π đưa offline và online RL vào rõ ràng hơn.

#### Flow matching có phải là thuật toán RL không?

Không. Flow matching là generative modeling objective dùng để học continuous action distribution/vector field. Nó có thể parameterize policy, nhưng chỉ trở thành một phần của RL system khi policy được tối ưu theo reward/return.

#### Behavior Cloning khác Offline RL như thế nào?

Behavior Cloning bắt chước action trong dataset:

```math
\max_\theta\sum_t\log\pi_\theta(a_t\mid s_t)
```

Offline RL dùng cùng dataset cố định nhưng cố tối đa hóa return, có thể ưu tiên actions tốt hơn behavior policy và phải xử lý out-of-distribution action/value estimation.

#### PPO khác VLA như thế nào?

PPO là một RL optimization algorithm. VLA là một model/policy class. Ta có thể dùng PPO hoặc RL algorithm khác để fine-tune một VLA nếu thiết kế được action likelihood, reward và quy trình rollout phù hợp.

#### π0.5 nằm ở đâu trong bản đồ lĩnh vực?

```text
Embodied AI
└── Robot Learning
    └── Foundation Robot Policies
        └── Vision-Language-Action models
            └── π family
                ├── π0
                ├── π0.5  ← supervised/co-trained generalist VLA
                ├── π*0.6 ← VLA + offline/online RL
                └── π0.7  ← steerable generalist VLA
```

---

### 13. Bản đồ ghi nhớ cuối cùng

```text
Markov Chain
    ↓ add reward
MRP
    ↓ add actions/control
MDP
    ↓ state hidden
POMDP
    ↓ learn from interaction or data
RL
├── model-free / model-based
├── value / policy / actor-critic
├── online / offline
├── on-policy / off-policy
├── single-agent / multi-agent
└── flat / hierarchical / safe / goal-conditioned / meta-RL ...

Một dòng phát triển khác:

demonstrations → imitation learning ───────────┐
Transformers → VLMs → semantic knowledge ─────┼→ VLA
diffusion/flow matching → action generation ──┘
                                                 ↓
                                      RL fine-tuning from experience
                                                 ↓
                                          π*0.6 / RLT
```

Ba câu cần nhớ:

1. **MDP là bài toán; RL là cách học lời giải.**
2. **VLA là policy/model class; không mặc định là RL.**
3. **π0.5 là VLA chủ yếu học bằng supervised co-training; π*0.6 mới là ví dụ rõ của VLA kết hợp offline và online RL.**

### Nguồn chính để đọc tiếp

- [Reinforcement Learning: An Introduction — Sutton & Barto](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- [OpenAI Spinning Up — taxonomy of RL algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [DQN — Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [π*0.6: A VLA That Learns From Experience](https://arxiv.org/abs/2511.14759)
- [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://www.pi.website/download/rlt.pdf)
- [π0.7: A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)

---
