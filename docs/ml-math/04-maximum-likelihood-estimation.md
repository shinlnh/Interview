# 3.4 Maximum Likelihood Estimation — MLE

[← Mục lục Machine Learning and Mathematics](./README.md)

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
