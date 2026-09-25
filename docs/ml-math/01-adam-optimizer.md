# 3.1 Write or explain Adam optimizer pseudocode

[← Mục lục Machine Learning and Mathematics](./README.md)

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
