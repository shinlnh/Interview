# 6.2 Fourier domain / tracking

[← Mục lục Robotics Systems](./README.md)

#### Why use the Fourier domain in correlation-filter tracking?

Trong correlation-filter tracker, mục tiêu cơ bản là lấy một **filter/template** $h$ rồi quét nó trên search region $f$ để tìm vị trí giống target nhất.

Nếu thực hiện trực tiếp trong spatial domain, ta phải dịch filter qua nhiều vị trí và tính correlation tại từng vị trí:

```math
g = f \star h
```

Trong Fourier domain, phép correlation trở thành phép nhân từng phần tử:

```math
\boxed{G = F \odot H^*}
```

với:

- $F = \mathcal{F}(f)$
- $H = \mathcal{F}(h)$
- $H^*$: complex conjugate
- $\odot$: element-wise multiplication

Sau đó:

```math
g = \mathcal{F}^{-1}(G)
```

MOSSE sử dụng chính cách này: biến đổi ảnh và filter bằng FFT, nhân element-wise trong frequency domain, rồi dùng inverse FFT để thu được response map.

Ý tưởng là:

```text
Spatial domain

search image ⋆ filter
        │
        │ khá tốn nếu quét trực tiếp
        ▼

Fourier domain

FFT(search) × conj(FFT(filter))
        │
        ▼
      IFFT
        │
        ▼
 response map
```

#### What is convolution in the spatial domain equivalent to in the Fourier domain?

Theo **Convolution Theorem**:

```math
\boxed{f * h \quad \Longleftrightarrow \quad F \odot H}
```

Tức là:

> **Convolution trong spatial domain = element-wise multiplication trong Fourier domain.**

Còn **correlation** hơi khác một chút:

```math
\boxed{f \star h \quad \Longleftrightarrow \quad F \odot H^*}
```

Khác biệt chính là complex conjugate $H^*$. MOSSE biểu diễn correlation dưới dạng:

```math
G = F \odot H^*
```

Vì vậy, có thể ghi nhớ:

```math
\boxed{\text{convolution} \rightarrow FH}
```

```math
\boxed{\text{correlation} \rightarrow FH^*}
```

#### Why can FFT reduce computation?

Nếu tính correlation trực tiếp cho mọi vị trí, mỗi vị trí cần một phép dot product với filter. Với tín hiệu có $N$ phần tử và filter có kích thước tương đương, chi phí có thể lên tới:

```math
O(N^2)
```

Khi dùng FFT, ta thực hiện ba bước:

1. Biến đổi tín hiệu sang Fourier domain bằng FFT.
2. Nhân element-wise.
3. Dùng inverse FFT để thu response map.

Tổng chi phí xấp xỉ:

```math
O(N\log N) + O(N) + O(N\log N) = O(N\log N)
```

Trong tracking, Fourier transform của filter thường được tái sử dụng, nên chi phí cho mỗi frame còn thấp hơn so với việc quét filter trực tiếp qua toàn bộ search region.

#### What is the complexity of FFT?

Với tín hiệu một chiều có $N$ phần tử, độ phức tạp của FFT là:

```math
O(N\log N)
```

Với ảnh hai chiều kích thước $M \times N$, FFT được thực hiện theo cả hàng và cột, nên độ phức tạp là:

```math
O\bigl(MN(\log M + \log N)\bigr)
```

thường được viết gọn là:

```math
O\bigl(MN\log(MN)\bigr)
```

#### What is a correlation filter?

Correlation filter là một template được học để xác định vị trí target trong search region. Filter được huấn luyện sao cho khi correlation với một image patch, nó tạo ra response map có đỉnh cao nhất tại vị trí của target.

Trong quá trình tracking:

1. Trích xuất search region quanh vị trí target trước đó.
2. Tính correlation giữa filter đã học và search region.
3. Chọn đỉnh của response map làm vị trí target mới.
4. Cập nhật filter bằng appearance mới của target.

Nhiều correlation-filter tracker sử dụng desired response có dạng Gaussian khi huấn luyện. Cách này khuyến khích response mạnh tại tâm target và response thấp ở background.

#### What is the relation between DCF/KCF/CSRT?

**DCF** là họ tracker **Discriminative Correlation Filter**. Các tracker này học một filter để phân biệt target với background xung quanh, sau đó dùng correlation để tạo response map.

**KCF (Kernelized Correlation Filter)** là một tracker thuộc họ DCF. Nó khai thác các cyclic shift của training sample và dùng kernel trick để học độ tương đồng phi tuyến trong khi vẫn tính toán nhanh ở Fourier domain. KCF hiệu quả nhưng có thể gặp khó khăn khi target thay đổi lớn về scale, bị biến dạng hoặc bị che khuất.

**CSRT (Channel and Spatial Reliability Tracking)** là một tracker DCF nâng cao hơn. Nó học độ tin cậy của từng feature channel và spatial region để tập trung vào các phần chứa nhiều thông tin của target. CSRT thường chính xác và ổn định hơn KCF, đặc biệt khi scale hoặc hình dạng thay đổi, nhưng chạy chậm hơn.

Tóm lại:

```math
\text{DCF family} \supset \{\text{KCF},\ \text{CSRT},\ldots\}
```

> KCF ưu tiên tốc độ, còn CSRT ưu tiên độ ổn định và chính xác.

#### Does moving computation to Fourier space always reduce total runtime?

Không. Tính toán trong Fourier domain có lợi khi signal hoặc filter lớn, khi cần tính correlation tại nhiều vị trí, hoặc khi có thể tái sử dụng filter đã được biến đổi.

Với input hoặc kernel nhỏ, tính trực tiếp trong spatial domain có thể nhanh hơn vì FFT tạo thêm các chi phí:

- Forward và inverse transform
- Padding và cấp phát bộ nhớ
- Phép toán trên số phức
- Di chuyển dữ liệu giữa các bước xử lý hoặc thiết bị

Vì vậy, phương pháp tối ưu phụ thuộc vào kích thước input, kích thước filter, phần cứng, cách triển khai và số lần tái sử dụng cùng một filter.

#### What does element-wise multiplication in frequency domain correspond to?

Element-wise multiplication trong frequency domain tương ứng với convolution trong spatial domain:

```math
F \odot H \quad \Longleftrightarrow \quad f * h
```

Nếu một spectrum được lấy complex conjugate, phép toán tương ứng với correlation:

```math
F \odot H^* \quad \Longleftrightarrow \quad f \star h
```

Vì Discrete Fourier Transform giả định signal có tính tuần hoàn, kết quả trực tiếp là **circular convolution hoặc correlation**. Muốn thu được dạng linear mà không bị wrap-around, cần zero-pad các input tới kích thước đủ lớn.

---
