# 3.3 Matrix decomposition

[← Mục lục Machine Learning and Mathematics](./README.md)

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
