# Robotics Systems

### 6.1 Kalman Filter

- What problem does a Kalman Filter solve?
- What are the prediction and correction steps?
- What is the state vector?
- What is the covariance matrix P?
- What are Q and R?
- What is the Kalman Gain?
- How does measurement noise affect the Kalman Gain?
- What happens if Q is too large?
- What happens if R is too large?
- Can a Kalman Filter fuse camera and radar measurements?
- What assumptions does a standard Kalman Filter make?
- KF vs EKF vs UKF?
- Why would a tracker use Kalman prediction during temporary occlusion?

---

### 6.2 Fourier domain / tracking

- Why use the Fourier domain in correlation-filter tracking?
- What is convolution in the spatial domain equivalent to in the Fourier domain?
- Why can FFT reduce computation?
- What is the complexity of FFT?
- What is a correlation filter?
- What is the relation between DCF/KCF/CSRT?
- Does moving computation to Fourier space always reduce total runtime?
- What does element-wise multiplication in frequency domain correspond to?

Key relationship:

\[
\text{convolution in spatial domain}
\leftrightarrow
\text{multiplication in frequency domain}
\]

FFT complexity is approximately:

\[
O(n\log n)
\]

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
