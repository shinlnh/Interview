# Robotics Systems

### 6.1 Kalman Filter

- What problem does a Kalman Filter solve?

  A Kalman Filter estimates the hidden state of a dynamic system when the measurements are noisy. It combines

  ```math
  \text{motion prediction} + \text{sensor measurement}
  ```

  to obtain a better state estimate.

  For example, in object tracking:

  ```math
  x_k = [x, y, v_x, v_y]^T
  ```

  The camera measurement may be noisy, so the Kalman Filter combines the measurement with the previous motion information.

- What are the prediction and correction steps?

  The Kalman Filter has two main stages.

  **Prediction:**

  ```math
  \hat{x}_k^- = F\hat{x}_{k-1}
  ```

  ```math
  P_k^- = FP_{k-1}F^T + Q
  ```

  It predicts the current state using the previous state and the motion model.

  **Correction:**

  ```math
  K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
  ```

  ```math
  \hat{x}_k = \hat{x}_k^- + K_k(z_k - H\hat{x}_k^-)
  ```

  The correction step uses the new sensor measurement to correct the prediction.

- What is the state vector?

  The state vector contains the variables that describe the system and that we want to estimate.

  For example:

  ```math
  x = [x, y, v_x, v_y]^T
  ```

  contains position and velocity.

  For bounding-box tracking, it could be:

  ```math
  x = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T
  ```

  The exact state vector depends on the application.

- What is the covariance matrix P?

  The covariance matrix $P$ represents the uncertainty of the current state estimate.

  A larger $P$ means:

  > The filter is less confident about the estimated state.

  A smaller $P$ means:

  > The filter is more confident about the estimated state.

  So:

  ```math
  P \uparrow \;\Rightarrow\; \text{uncertainty} \uparrow
  ```

  ```math
  P \downarrow \;\Rightarrow\; \text{uncertainty} \downarrow
  ```

- What are Q and R?

  $Q$ is the **process noise covariance**.

  It represents uncertainty in the motion or system model.

  For example, we may assume constant velocity, but the object can actually accelerate.

  $R$ is the **measurement noise covariance**.

  It represents the uncertainty or noise of the sensor measurement.

  A simple way to remember them is:

  > $Q$ describes how much we distrust the motion model.

  > $R$ describes how much we distrust the measurement.

- What is the Kalman Gain?

  The Kalman Gain is:

  ```math
  K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
  ```

  It determines how much the filter should trust the measurement compared with the prediction.

  A large Kalman Gain means the measurement has more influence.

  A small Kalman Gain means the prediction has more influence.

  So the Kalman Gain can be viewed as a balance between the model prediction and the sensor measurement.

- How does measurement noise affect the Kalman Gain?

  Measurement noise is represented by $R$. From the Kalman Gain equation:

  ```math
  K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
  ```

  if $R$ increases, the Kalman Gain generally decreases. The filter trusts the noisy measurement less and relies more on the prediction.

  If $R$ decreases, the Kalman Gain generally increases. The filter considers the measurement more reliable and gives it more influence.

- What happens if Q is too large?

  If $Q$ is too large, the filter assumes that the motion model is very uncertain. The predicted covariance $P_k^-$ grows, which generally increases the Kalman Gain and makes the filter rely more on measurements.

  As a result, the estimate responds quickly to changes but may become noisy or unstable because it follows measurement noise too closely.

- What happens if R is too large?

  If $R$ is too large, the filter assumes that the sensor measurements are very unreliable. The Kalman Gain becomes smaller, so the filter relies more on the motion prediction.

  The estimate becomes smoother but may react too slowly to real changes. If the motion model is inaccurate, the estimate can also drift away from the true state.

- Can a Kalman Filter fuse camera and radar measurements?

  Yes. A Kalman Filter can combine camera and radar measurements if both can be related to the same state.

  Each sensor has its own measurement model $H$ and measurement noise covariance $R$. Their measurements can be processed sequentially or combined into one measurement update.

  For example, a camera may provide accurate image position, while radar provides range and relative velocity. The system also needs correct spatial calibration, time synchronization, and data association. If the measurement relationship is nonlinear, an EKF or UKF may be more appropriate than a standard KF.

- What assumptions does a standard Kalman Filter make?

  A standard Kalman Filter mainly assumes that:

  - The state transition and measurement models are linear.
  - Process noise and measurement noise are zero-mean Gaussian noise.
  - The noise covariance matrices $Q$ and $R$ are known or estimated correctly.
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

- What is convolution in the spatial domain equivalent to in the Fourier domain?

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

- Why can FFT reduce computation?

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

- What is the complexity of FFT?

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

  ```math
  \text{DCF family} \supset \{\text{KCF},\ \text{CSRT},\ldots\}
  ```

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

  ```math
  F \odot H \quad \Longleftrightarrow \quad f * h
  ```

  Nếu một spectrum được lấy complex conjugate, phép toán tương ứng với correlation:

  ```math
  F \odot H^* \quad \Longleftrightarrow \quad f \star h
  ```

  Vì Discrete Fourier Transform giả định signal có tính tuần hoàn, kết quả trực tiếp là **circular convolution hoặc correlation**. Muốn thu được dạng linear mà không bị wrap-around, cần zero-pad các input tới kích thước đủ lớn.

---

### 6.3 Camera + Radar

- What does a camera provide?

  A camera provides rich visual information such as color, texture, object class, lane markings, segmentation masks, and accurate angular position in the image. However, a monocular camera does not directly measure depth or velocity; these values must be estimated from geometry, motion, or a neural network.

  In this project, the camera provides a 2D bounding box, class ID, segmentation mask, and an estimated ground position obtained from the bottom-center pixel of the bounding box.

- What does radar provide?

  Radar provides metric range, relative velocity through the Doppler effect, and an approximate direction to an object. It works well at long range and is relatively robust to darkness, rain, and fog, but usually has poorer angular resolution and provides little semantic information.

  In this project, radar supplies object position, velocity, closest point, and cluster dimensions, but its class ID is initially unknown.

- Why fuse camera and radar?

  They have complementary strengths:

  - Camera provides object classification and accurate visual/angular information.
  - Radar provides reliable distance and relative velocity.
  - Fusion improves robustness, range estimation, tracking stability, and redundancy when one sensor temporarily fails.

  In this implementation, the fused object takes its class, 2D bounding box, and mask from the camera, while position, velocity, acceleration, and closest point come from radar.

- What is sensor calibration?

  Sensor calibration determines the parameters needed to interpret sensor measurements and place data from different sensors into a common coordinate system.

  It normally includes:

  - **Intrinsic calibration:** The internal measurement model of a sensor.
  - **Extrinsic calibration:** The sensor's position and orientation relative to another frame.
  - **Temporal calibration:** The time offset between sensors.

- What is extrinsic calibration?

  Extrinsic calibration estimates the rigid transformation between two coordinate frames. It consists of rotation $R$ and translation $t$:

  ```math
  p_B = R_{B \leftarrow A}p_A + t_{B \leftarrow A}
  ```

  For camera-radar fusion, extrinsics describe where the camera and radar are mounted relative to the ego vehicle, allowing radar measurements to be transformed into the camera or ego frame.

- What is intrinsic calibration?

  Intrinsic calibration describes how a 3D point in the camera frame maps to an image pixel. The camera matrix is:

  ```math
  K =
  \begin{bmatrix}
  f_x & 0 & c_x \\
  0 & f_y & c_y \\
  0 & 0 & 1
  \end{bmatrix}
  ```

  It contains the focal lengths and principal point; a real camera model may also contain lens-distortion parameters.

  This project uses an ideal CARLA pinhole camera and calculates $f_x$, $f_y$, $c_x$, and $c_y$ from the horizontal field of view and image resolution.

- How do you transform radar points into the camera coordinate frame?

  First, transform the radar point into the ego frame, then transform it from the ego frame into the camera frame:

  ```math
  p_{\text{ego}} = T_{\text{ego} \leftarrow \text{radar}}p_{\text{radar}}
  ```

  ```math
  p_{\text{camera}} = T_{\text{camera} \leftarrow \text{ego}}p_{\text{ego}}
  ```

  To project it into the image:

  ```math
  \tilde{p} = Kp_{\text{camera}}, \qquad
  u = \frac{\tilde{p}_x}{\tilde{p}_z}, \quad
  v = \frac{\tilde{p}_y}{\tilde{p}_z}
  ```

  Points behind the camera or outside its field of view must be rejected. If the measurements are asynchronous, ego motion and object motion should also be used to transform them to the same timestamp.

- How do you associate a radar point with a detected object?

  Normally, I would:

  1. Transform the radar measurement into the camera or ego frame.
  2. Project it into the image.
  3. Check whether it falls inside or overlaps a camera bounding box.
  4. Compare its position and velocity with the predicted object state.
  5. Build an association cost and apply gating.
  6. Use a one-to-one assignment algorithm such as Hungarian matching.

  This project associates radar clusters or tracks rather than individual raw radar points. Its cost combines:

  - Mahalanobis distance over $[x, v_x, y, v_y]$.
  - Projected radar-box versus camera-box IoU.
  - A 20 m ground-distance guard unless image IoU is at least 0.35.
  - An absolute 60 m mismatch guard.
  - A final fusion-cost threshold of 13.280.

- What happens if camera and radar disagree?

  The measurements should not automatically be averaged. First, the system evaluates their consistency using spatial, temporal, velocity, and uncertainty gates.

  In this project:

  - A new pair that fails the gates is not fused and is emitted as camera-only and/or radar-only.
  - An existing fused identity can temporarily retain its previous association.
  - If one sensor disappears, the fused identity can coast for up to five cycles.
  - After that, the fused identity expires and falls back to a sensor-specific track.
  - For an accepted pair, camera semantics are trusted while radar kinematics are trusted.

- What is data association?

  Data association is the process of deciding which measurement belongs to which existing object or track.

  It is needed both:

  - **Temporally:** Matching current detections to previous tracks.
  - **Across sensors:** Matching a camera detection to a radar detection.

  A complete association process usually includes state prediction, cost calculation, gating, one-to-one assignment, and handling unmatched measurements.

- Can Hungarian matching be used for data association?

  Yes. Hungarian matching finds a globally optimal one-to-one assignment from a cost matrix.

  Possible costs include:

  - Mahalanobis distance.
  - Euclidean distance.
  - $1 - \operatorname{IoU}$.
  - Class incompatibility.
  - Velocity difference.
  - A weighted combination of these cues.

  Hungarian matching does not decide whether a match is physically valid, so gating and maximum-cost thresholds are still required. This repository uses Hungarian matching for both detection-to-track association and camera-to-radar association.

- Can a Kalman Filter perform sensor fusion?

  Yes. A Kalman Filter can predict an object state and update it using measurements from multiple sensors according to their measurement models and covariance matrices. More reliable measurements receive greater weight.

  However, a Kalman Filter does not solve calibration or data association by itself.

  In this project, separate camera and radar Kalman Filters provide tracking, timestamp prediction, state estimates, and covariance information for association. The final fusion is not a single joint Kalman update: it mainly selects camera semantics and radar kinematics.

- What is gating?

  Gating rejects physically implausible associations before accepting them as matches. It reduces false matches and also reduces the size of the assignment problem.

  Examples include:

  - Maximum Euclidean distance.
  - Maximum Mahalanobis distance.
  - Minimum bounding-box IoU.
  - Compatible object classes.
  - Maximum timestamp difference.
  - Camera field-of-view checks.

  The project implements distance guards inside the fusion cost and then validates the Hungarian result using a fusion threshold.

- What is Mahalanobis distance?

  Mahalanobis distance measures the difference between a measurement and a predicted state while accounting for uncertainty:

  ```math
  d_M = \sqrt{(z - \hat{z})^T S^{-1}(z - \hat{z})}
  ```

  Here:

  - $z - \hat{z}$ is the innovation.
  - $S$ is the innovation covariance.

  Unlike Euclidean distance, it treats an error in an uncertain direction as less significant than the same error in a highly accurate direction.

  The project calculates it using camera and radar states containing $x$, $v_x$, $y$, and $v_y$, together with their combined covariance.

- Early fusion vs late fusion?

  **Early fusion** combines raw sensor data or learned features before object detection.

  - It can preserve more information and learn complex cross-sensor relationships.
  - It requires accurate calibration and synchronization.
  - It is computationally expensive and tightly couples the sensors.

  **Late fusion** combines independently detected objects or tracks.

  - It is modular, easier to debug, and uses less bandwidth.
  - It can support sensors with different rates.
  - It loses some raw information and depends heavily on data association.

  `adas_service` performs late, object/track-level fusion.

- What happens when radar has good depth but poor angular resolution?

  The radar measurement may identify the correct range but cannot clearly determine which of several nearby objects produced the return. Its uncertainty is therefore narrow in range but wide laterally or angularly.

  A good fusion design should:

  - Trust radar more for range and range rate.
  - Trust camera more for bearing, lateral position, and class.
  - Represent radar uncertainty with an anisotropic covariance.
  - Avoid forcing a match when several camera objects are plausible.

  This project uses projected-box IoU and Mahalanobis distance to disambiguate such cases. One limitation is that the final fused position currently comes entirely from radar, including its lateral component; covariance-weighted lateral fusion could improve this.

- How do timestamps affect multi-sensor fusion?

  Measurements must represent approximately the same physical time. Otherwise, ego motion and object motion cause spatial errors:

  ```math
  \text{position error} \approx v\Delta t
  ```

  Timestamp errors can cause incorrect associations, duplicated objects, unstable velocity estimates, and ghost tracks.

  This project:

  - Buffers camera and radar results by timestamp.
  - Uses approximate-time synchronization.
  - Uses exact timestamps in CARLA ($\varepsilon = 0$).
  - Waits at most 200 ms for a missing sensor.
  - Predicts the older sensor track to the newer timestamp.
  - Does not reuse sensor data from a previous synchronized bundle.
  - Resets or clamps prediction when timestamps jump backward or differ by more than one second.

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
