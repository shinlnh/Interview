# 6.3 Camera + Radar

[← Mục lục Robotics Systems](./README.md)

#### What does a camera provide?

A camera provides rich visual information such as color, texture, object class, lane markings, segmentation masks, and accurate angular position in the image. However, a monocular camera does not directly measure depth or velocity; these values must be estimated from geometry, motion, or a neural network.

In this project, the camera provides a 2D bounding box, class ID, segmentation mask, and an estimated ground position obtained from the bottom-center pixel of the bounding box.

#### What does radar provide?

Radar provides metric range, relative velocity through the Doppler effect, and an approximate direction to an object. It works well at long range and is relatively robust to darkness, rain, and fog, but usually has poorer angular resolution and provides little semantic information.

In this project, radar supplies object position, velocity, closest point, and cluster dimensions, but its class ID is initially unknown.

#### Why fuse camera and radar?

They have complementary strengths:

- Camera provides object classification and accurate visual/angular information.
- Radar provides reliable distance and relative velocity.
- Fusion improves robustness, range estimation, tracking stability, and redundancy when one sensor temporarily fails.

In this implementation, the fused object takes its class, 2D bounding box, and mask from the camera, while position, velocity, acceleration, and closest point come from radar.

#### What is sensor calibration?

Sensor calibration determines the parameters needed to interpret sensor measurements and place data from different sensors into a common coordinate system.

It normally includes:

- **Intrinsic calibration:** The internal measurement model of a sensor.
- **Extrinsic calibration:** The sensor's position and orientation relative to another frame.
- **Temporal calibration:** The time offset between sensors.

#### What is extrinsic calibration?

Extrinsic calibration estimates the rigid transformation between two coordinate frames. It consists of rotation $R$ and translation $t$:

```math
p_B = R_{B \leftarrow A}p_A + t_{B \leftarrow A}
```

For camera-radar fusion, extrinsics describe where the camera and radar are mounted relative to the ego vehicle, allowing radar measurements to be transformed into the camera or ego frame.

#### What is intrinsic calibration?

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

#### How do you transform radar points into the camera coordinate frame?

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

#### How do you associate a radar point with a detected object?

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

#### What happens if camera and radar disagree?

The measurements should not automatically be averaged. First, the system evaluates their consistency using spatial, temporal, velocity, and uncertainty gates.

In this project:

- A new pair that fails the gates is not fused and is emitted as camera-only and/or radar-only.
- An existing fused identity can temporarily retain its previous association.
- If one sensor disappears, the fused identity can coast for up to five cycles.
- After that, the fused identity expires and falls back to a sensor-specific track.
- For an accepted pair, camera semantics are trusted while radar kinematics are trusted.

#### What is data association?

Data association is the process of deciding which measurement belongs to which existing object or track.

It is needed both:

- **Temporally:** Matching current detections to previous tracks.
- **Across sensors:** Matching a camera detection to a radar detection.

A complete association process usually includes state prediction, cost calculation, gating, one-to-one assignment, and handling unmatched measurements.

#### Can Hungarian matching be used for data association?

Yes. Hungarian matching finds a globally optimal one-to-one assignment from a cost matrix.

Possible costs include:

- Mahalanobis distance.
- Euclidean distance.
- $1 - \mathrm{IoU}$.
- Class incompatibility.
- Velocity difference.
- A weighted combination of these cues.

Hungarian matching does not decide whether a match is physically valid, so gating and maximum-cost thresholds are still required. This repository uses Hungarian matching for both detection-to-track association and camera-to-radar association.

#### Can a Kalman Filter perform sensor fusion?

Yes. A Kalman Filter can predict an object state and update it using measurements from multiple sensors according to their measurement models and covariance matrices. More reliable measurements receive greater weight.

However, a Kalman Filter does not solve calibration or data association by itself.

In this project, separate camera and radar Kalman Filters provide tracking, timestamp prediction, state estimates, and covariance information for association. The final fusion is not a single joint Kalman update: it mainly selects camera semantics and radar kinematics.

#### What is gating?

Gating rejects physically implausible associations before accepting them as matches. It reduces false matches and also reduces the size of the assignment problem.

Examples include:

- Maximum Euclidean distance.
- Maximum Mahalanobis distance.
- Minimum bounding-box IoU.
- Compatible object classes.
- Maximum timestamp difference.
- Camera field-of-view checks.

The project implements distance guards inside the fusion cost and then validates the Hungarian result using a fusion threshold.

#### What is Mahalanobis distance?

Mahalanobis distance measures the difference between a measurement and a predicted state while accounting for uncertainty:

```math
d_M = \sqrt{(z - \hat{z})^T S^{-1}(z - \hat{z})}
```

Here:

- $z - \hat{z}$ is the innovation.
- $S$ is the innovation covariance.

Unlike Euclidean distance, it treats an error in an uncertain direction as less significant than the same error in a highly accurate direction.

The project calculates it using camera and radar states containing $x$, $v_x$, $y$, and $v_y$, together with their combined covariance.

#### Early fusion vs late fusion?

**Early fusion** combines raw sensor data or learned features before object detection.

- It can preserve more information and learn complex cross-sensor relationships.
- It requires accurate calibration and synchronization.
- It is computationally expensive and tightly couples the sensors.

**Late fusion** combines independently detected objects or tracks.

- It is modular, easier to debug, and uses less bandwidth.
- It can support sensors with different rates.
- It loses some raw information and depends heavily on data association.

`adas_service` performs late, object/track-level fusion.

#### What happens when radar has good depth but poor angular resolution?

The radar measurement may identify the correct range but cannot clearly determine which of several nearby objects produced the return. Its uncertainty is therefore narrow in range but wide laterally or angularly.

A good fusion design should:

- Trust radar more for range and range rate.
- Trust camera more for bearing, lateral position, and class.
- Represent radar uncertainty with an anisotropic covariance.
- Avoid forcing a match when several camera objects are plausible.

This project uses projected-box IoU and Mahalanobis distance to disambiguate such cases. One limitation is that the final fused position currently comes entirely from radar, including its lateral component; covariance-weighted lateral fusion could improve this.

#### How do timestamps affect multi-sensor fusion?

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
