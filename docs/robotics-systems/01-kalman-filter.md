# 6.1 Kalman Filter

[← Mục lục Robotics Systems](./README.md)

#### What problem does a Kalman Filter solve?

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

#### What are the prediction and correction steps?

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

#### What is the state vector?

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

#### What is the covariance matrix P?

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

#### What are Q and R?

$Q$ is the **process noise covariance**.

It represents uncertainty in the motion or system model.

For example, we may assume constant velocity, but the object can actually accelerate.

$R$ is the **measurement noise covariance**.

It represents the uncertainty or noise of the sensor measurement.

A simple way to remember them is:

> $Q$ describes how much we distrust the motion model.

> $R$ describes how much we distrust the measurement.

#### What is the Kalman Gain?

The Kalman Gain is:

```math
K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
```

It determines how much the filter should trust the measurement compared with the prediction.

A large Kalman Gain means the measurement has more influence.

A small Kalman Gain means the prediction has more influence.

So the Kalman Gain can be viewed as a balance between the model prediction and the sensor measurement.

#### How does measurement noise affect the Kalman Gain?

Measurement noise is represented by $R$. From the Kalman Gain equation:

```math
K_k = P_k^-H^T(HP_k^-H^T + R)^{-1}
```

if $R$ increases, the Kalman Gain generally decreases. The filter trusts the noisy measurement less and relies more on the prediction.

If $R$ decreases, the Kalman Gain generally increases. The filter considers the measurement more reliable and gives it more influence.

#### What happens if Q is too large?

If $Q$ is too large, the filter assumes that the motion model is very uncertain. The predicted covariance $P_k^-$ grows, which generally increases the Kalman Gain and makes the filter rely more on measurements.

As a result, the estimate responds quickly to changes but may become noisy or unstable because it follows measurement noise too closely.

#### What happens if R is too large?

If $R$ is too large, the filter assumes that the sensor measurements are very unreliable. The Kalman Gain becomes smaller, so the filter relies more on the motion prediction.

The estimate becomes smoother but may react too slowly to real changes. If the motion model is inaccurate, the estimate can also drift away from the true state.

#### Can a Kalman Filter fuse camera and radar measurements?

Yes. A Kalman Filter can combine camera and radar measurements if both can be related to the same state.

Each sensor has its own measurement model $H$ and measurement noise covariance $R$. Their measurements can be processed sequentially or combined into one measurement update.

For example, a camera may provide accurate image position, while radar provides range and relative velocity. The system also needs correct spatial calibration, time synchronization, and data association. If the measurement relationship is nonlinear, an EKF or UKF may be more appropriate than a standard KF.

#### What assumptions does a standard Kalman Filter make?

A standard Kalman Filter mainly assumes that:

- The state transition and measurement models are linear.
- Process noise and measurement noise are zero-mean Gaussian noise.
- The noise covariance matrices $Q$ and $R$ are known or estimated correctly.
- Process noise, measurement noise, and the initial state error are mutually independent.
- The motion and measurement models describe the real system reasonably well.

Under linear-Gaussian assumptions, the Kalman Filter gives the optimal minimum-variance state estimate.

#### KF vs EKF vs UKF?

- **KF (Kalman Filter):** Used for linear state-transition and measurement models. It is simple, fast, and exact under linear-Gaussian assumptions.
- **EKF (Extended Kalman Filter):** Used for nonlinear models. It linearizes the nonlinear functions using Jacobian matrices. It is computationally efficient but can be inaccurate or diverge when the system is strongly nonlinear.
- **UKF (Unscented Kalman Filter):** Also used for nonlinear models. It propagates a set of sigma points through the nonlinear functions instead of calculating Jacobians. It usually represents nonlinear transformations more accurately than an EKF, but requires more computation.

#### Why would a tracker use Kalman prediction during temporary occlusion?

During occlusion, the tracker may not receive a reliable measurement. The Kalman Filter can still predict the object's next position from its previous state and motion model.

This allows the tracker to maintain the track through a short gap and define a likely search region for matching the object when it becomes visible again. The covariance grows during prediction-only steps, representing increasing uncertainty. If the occlusion lasts too long, the prediction becomes unreliable and the track may need to be deleted.

---
