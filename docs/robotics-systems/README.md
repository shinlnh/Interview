# Robotics Systems

### 6.1 Kalman Filter

- What problem does a Kalman Filter solve?

  A Kalman Filter estimates the hidden state of a dynamic system when the measurements are noisy. It combines

  \[
  \text{motion prediction} + \text{sensor measurement}
  \]

  to obtain a better state estimate.

  For example, in object tracking:

  \[
  x_k = [x, y, v_x, v_y]^T
  \]

  The camera measurement may be noisy, so the Kalman Filter combines the measurement with the previous motion information.

- What are the prediction and correction steps?

  The Kalman Filter has two main stages.

  **Prediction:**

  \[
  \hat{x}_k^- = F\hat{x}_{k-1}
  \]

  \[
  P_k^- = FP_{k-1}F^T + Q
  \]

  It predicts the current state using the previous state and the motion model.

  **Correction:**

  \[
  K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
  \]

  \[
  \hat{x}_k = \hat{x}_k^- + K_k(z_k - H\hat{x}_k^-)
  \]

  The correction step uses the new sensor measurement to correct the prediction.

- What is the state vector?

  The state vector contains the variables that describe the system and that we want to estimate.

  For example:

  \[
  x = [x, y, v_x, v_y]^T
  \]

  contains position and velocity.

  For bounding-box tracking, it could be:

  \[
  x = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T
  \]

  The exact state vector depends on the application.

- What is the covariance matrix P?

  The covariance matrix \(P\) represents the uncertainty of the current state estimate.

  A larger \(P\) means:

  > The filter is less confident about the estimated state.

  A smaller \(P\) means:

  > The filter is more confident about the estimated state.

  So:

  \[
  P \uparrow \;\Rightarrow\; \text{uncertainty} \uparrow
  \]

  \[
  P \downarrow \;\Rightarrow\; \text{uncertainty} \downarrow
  \]

- What are Q and R?

  \(Q\) is the **process noise covariance**.

  It represents uncertainty in the motion or system model.

  For example, we may assume constant velocity, but the object can actually accelerate.

  \(R\) is the **measurement noise covariance**.

  It represents the uncertainty or noise of the sensor measurement.

  A simple way to remember them is:

  > \(Q\) describes how much we distrust the motion model.

  > \(R\) describes how much we distrust the measurement.

- What is the Kalman Gain?

  The Kalman Gain is:

  \[
  K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
  \]

  It determines how much the filter should trust the measurement compared with the prediction.

  A large Kalman Gain means the measurement has more influence.

  A small Kalman Gain means the prediction has more influence.

  So the Kalman Gain can be viewed as a balance between the model prediction and the sensor measurement.

- How does measurement noise affect the Kalman Gain?

  Measurement noise is represented by \(R\). From the Kalman Gain equation:

  \[
  K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
  \]

  if \(R\) increases, the Kalman Gain generally decreases. The filter trusts the noisy measurement less and relies more on the prediction.

  If \(R\) decreases, the Kalman Gain generally increases. The filter considers the measurement more reliable and gives it more influence.

- What happens if Q is too large?

  If \(Q\) is too large, the filter assumes that the motion model is very uncertain. The predicted covariance \(P_k^-\) grows, which generally increases the Kalman Gain and makes the filter rely more on measurements.

  As a result, the estimate responds quickly to changes but may become noisy or unstable because it follows measurement noise too closely.

- What happens if R is too large?

  If \(R\) is too large, the filter assumes that the sensor measurements are very unreliable. The Kalman Gain becomes smaller, so the filter relies more on the motion prediction.

  The estimate becomes smoother but may react too slowly to real changes. If the motion model is inaccurate, the estimate can also drift away from the true state.

- Can a Kalman Filter fuse camera and radar measurements?

  Yes. A Kalman Filter can combine camera and radar measurements if both can be related to the same state.

  Each sensor has its own measurement model \(H\) and measurement noise covariance \(R\). Their measurements can be processed sequentially or combined into one measurement update.

  For example, a camera may provide accurate image position, while radar provides range and relative velocity. The system also needs correct spatial calibration, time synchronization, and data association. If the measurement relationship is nonlinear, an EKF or UKF may be more appropriate than a standard KF.

- What assumptions does a standard Kalman Filter make?

  A standard Kalman Filter mainly assumes that:

  - The state transition and measurement models are linear.
  - Process noise and measurement noise are zero-mean Gaussian noise.
  - The noise covariance matrices \(Q\) and \(R\) are known or estimated correctly.
  - Process noise, measurement noise, and the initial state error are mutually independent.
  - The motion and measurement models describe the real system reasonably well.

  Under linear-Gaussian assumptions, the Kalman Filter gives the optimal minimum-variance state estimate.

- KF vs EKF vs UKF?

  - **KF (Kalman Filter):** Used for linear state-transition and measurement models. It is simple, fast, and exact under linear-Gaussian assumptions.
  - **EKF (Extended Kalman Filter):** Used for nonlinear models. It linearizes the nonlinear functions using Jacobian matrices. It is computationally efficient but can be inaccurate or diverge when the system is strongly nonlinear.
  - **UKF (Unscented Kalman Filter):** Also used for nonlinear models. It propagates a set of sigma points through the nonlinear functions instead of calculating Jacobians. It usually represents nonlinear transformations more accurately than an EKF, but requires more computation.

- Why would a tracker use Kalman prediction during temporary occlusion?

  During occlusion, the tracker may not receive a reliable measurement. The Kalman Filter can still predict the object's next position from its previous state and motion model.

  This allows the tracker to maintain the track through a short gap and define a likely search region for matching the object when it becomes visible again. The covariance grows during prediction-only steps, representing increasing uncertainty. If the occlusion lasts too long, the prediction becomes unreliable and the track may need to be deleted.

---

### 6.2 Fourier domain / tracking

- Why use the Fourier domain in correlation-filter tracking?

  Trong correlation-filter tracker, mục tiêu cơ bản là lấy một **filter/template** \(h\) rồi quét nó trên search region \(f\) để tìm vị trí giống target nhất.

  Nếu thực hiện trực tiếp trong spatial domain, ta phải dịch filter qua nhiều vị trí và tính correlation tại từng vị trí:

  \[
  g = f \star h
  \]

  Trong Fourier domain, phép correlation trở thành phép nhân từng phần tử:

  \[
  \boxed{G = F \odot H^*}
  \]

  với:

  - \(F = \mathcal{F}(f)\)
  - \(H = \mathcal{F}(h)\)
  - \(H^*\): complex conjugate
  - \(\odot\): element-wise multiplication

  Sau đó:

  \[
  g = \mathcal{F}^{-1}(G)
  \]

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

- What is convolution in the spatial domain equivalent to in the Fourier domain?

  Theo **Convolution Theorem**:

  \[
  \boxed{f * h \quad \Longleftrightarrow \quad F \odot H}
  \]

  Tức là:

  > **Convolution trong spatial domain = element-wise multiplication trong Fourier domain.**

  Còn **correlation** hơi khác một chút:

  \[
  \boxed{f \star h \quad \Longleftrightarrow \quad F \odot H^*}
  \]

  Khác biệt chính là complex conjugate \(H^*\). MOSSE biểu diễn correlation dưới dạng:

  \[
  G = F \odot H^*
  \]

  Vì vậy, có thể ghi nhớ:

  \[
  \boxed{\text{convolution} \rightarrow FH}
  \]

  \[
  \boxed{\text{correlation} \rightarrow FH^*}
  \]

- Why can FFT reduce computation?

  Nếu tính correlation trực tiếp cho mọi vị trí, mỗi vị trí cần một phép dot product với filter. Với tín hiệu có \(N\) phần tử và filter có kích thước tương đương, chi phí có thể lên tới:

  \[
  O(N^2)
  \]

  Khi dùng FFT, ta thực hiện ba bước:

  1. Biến đổi tín hiệu sang Fourier domain bằng FFT.
  2. Nhân element-wise.
  3. Dùng inverse FFT để thu response map.

  Tổng chi phí xấp xỉ:

  \[
  O(N\log N) + O(N) + O(N\log N) = O(N\log N)
  \]

  Trong tracking, Fourier transform của filter thường được tái sử dụng, nên chi phí cho mỗi frame còn thấp hơn so với việc quét filter trực tiếp qua toàn bộ search region.

- What is the complexity of FFT?

  Với tín hiệu một chiều có \(N\) phần tử, độ phức tạp của FFT là:

  \[
  O(N\log N)
  \]

  Với ảnh hai chiều kích thước \(M \times N\), FFT được thực hiện theo cả hàng và cột, nên độ phức tạp là:

  \[
  O\bigl(MN(\log M + \log N)\bigr)
  \]

  thường được viết gọn là:

  \[
  O\bigl(MN\log(MN)\bigr)
  \]

- What is a correlation filter?

  Correlation filter là một template được học để xác định vị trí target trong search region. Filter được huấn luyện sao cho khi correlation với một image patch, nó tạo ra response map có đỉnh cao nhất tại vị trí của target.

  Trong quá trình tracking:

  1. Trích xuất search region quanh vị trí target trước đó.
  2. Tính correlation giữa filter đã học và search region.
  3. Chọn đỉnh của response map làm vị trí target mới.
  4. Cập nhật filter bằng appearance mới của target.

  Nhiều correlation-filter tracker sử dụng desired response có dạng Gaussian khi huấn luyện. Cách này khuyến khích response mạnh tại tâm target và response thấp ở background.

- What is the relation between DCF/KCF/CSRT?

  **DCF** là họ tracker **Discriminative Correlation Filter**. Các tracker này học một filter để phân biệt target với background xung quanh, sau đó dùng correlation để tạo response map.

  **KCF (Kernelized Correlation Filter)** là một tracker thuộc họ DCF. Nó khai thác các cyclic shift của training sample và dùng kernel trick để học độ tương đồng phi tuyến trong khi vẫn tính toán nhanh ở Fourier domain. KCF hiệu quả nhưng có thể gặp khó khăn khi target thay đổi lớn về scale, bị biến dạng hoặc bị che khuất.

  **CSRT (Channel and Spatial Reliability Tracking)** là một tracker DCF nâng cao hơn. Nó học độ tin cậy của từng feature channel và spatial region để tập trung vào các phần chứa nhiều thông tin của target. CSRT thường chính xác và ổn định hơn KCF, đặc biệt khi scale hoặc hình dạng thay đổi, nhưng chạy chậm hơn.

  Tóm lại:

  \[
  \text{DCF family} \supset \{\text{KCF},\ \text{CSRT},\ldots\}
  \]

  > KCF ưu tiên tốc độ, còn CSRT ưu tiên độ ổn định và chính xác.

- Does moving computation to Fourier space always reduce total runtime?

  Không. Tính toán trong Fourier domain có lợi khi signal hoặc filter lớn, khi cần tính correlation tại nhiều vị trí, hoặc khi có thể tái sử dụng filter đã được biến đổi.

  Với input hoặc kernel nhỏ, tính trực tiếp trong spatial domain có thể nhanh hơn vì FFT tạo thêm các chi phí:

  - Forward và inverse transform
  - Padding và cấp phát bộ nhớ
  - Phép toán trên số phức
  - Di chuyển dữ liệu giữa các bước xử lý hoặc thiết bị

  Vì vậy, phương pháp tối ưu phụ thuộc vào kích thước input, kích thước filter, phần cứng, cách triển khai và số lần tái sử dụng cùng một filter.

- What does element-wise multiplication in frequency domain correspond to?

  Element-wise multiplication trong frequency domain tương ứng với convolution trong spatial domain:

  \[
  F \odot H \quad \Longleftrightarrow \quad f * h
  \]

  Nếu một spectrum được lấy complex conjugate, phép toán tương ứng với correlation:

  \[
  F \odot H^* \quad \Longleftrightarrow \quad f \star h
  \]

  Vì Discrete Fourier Transform giả định signal có tính tuần hoàn, kết quả trực tiếp là **circular convolution hoặc correlation**. Muốn thu được dạng linear mà không bị wrap-around, cần zero-pad các input tới kích thước đủ lớn.

---

### 6.3 Camera + Radar

- What does a camera provide?
- What does radar provide?
- Why fuse camera and radar?
- What is sensor calibration?
- What is extrinsic calibration?
- What is intrinsic calibration?
- How do you transform radar points into the camera coordinate frame?
- How do you associate a radar point with a detected object?
- What happens if camera and radar disagree?
- What is data association?
- Can Hungarian matching be used for data association?
- Can a Kalman Filter perform sensor fusion?
- What is gating?
- What is Mahalanobis distance?
- Early fusion vs late fusion?
- What happens when radar has good depth but poor angular resolution?
- How do timestamps affect multi-sensor fusion?

---

### 6.4 Isaac Sim / Isaac Lab

- What is Isaac Sim?
- What is Isaac Lab?
- What is the difference between Isaac Sim and Isaac Lab?
- What simulator did you use for humanoid robots?
- What is an articulation?
- What is a joint / DOF?
- Position control vs velocity control vs torque control?
- What is sim-to-real?
- What is domain randomization?
- Why train robot policies in simulation?
- What is PhysX?
- What information belongs to an observation space?
- What information belongs to an action space?
- How do you define a reward in RL?
- How do you reset an environment?
- Why run many environments in parallel?
- How can Isaac Lab accelerate RL training?

---
