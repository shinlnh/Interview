# 3.6 Derive MSE from Gaussian Maximum Likelihood

[← Mục lục Machine Learning and Mathematics](./README.md)

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
