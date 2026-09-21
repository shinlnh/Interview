# Paper Reproduction and SOTA Model Redesign

### 1.1 Replicate an architecture from a research paper

**Mức đã trao đổi:** **4/5**

Ý nghĩa của mức 4: em có thể đọc paper, hiểu architecture và triển khai lại với một ít guidance/reference.

### Interviewer có thể hỏi thêm

- Can you describe a model that you reproduced from a research paper?
  - I have reproduced RCLane from paper in the past. First of all, I encountered defect with curved lane, when teams use LaneATT to detect lane, LaneATT use linear model to detect lane, so when LaneATT meet curved lane, it doesn't detect. So, my task is find model to replace LaneATT to detect curved lane. I find out RCLane, a new model in 2025 on arxiv. However, this model is belonging to Huawei. So, it doesn't have open-source for C++. I will try find open-source on github and find source for RCLane. But, this source was developed by an individual using Mindscope, a proprietary programming language from Huawei. So, I port code from Mindscope to C++, that programming native on Jetson Xavier NX.

- How do you go from a paper diagram to actual code?
  - Let me take RCLane as an example. First of all, I read the diagram and turn each block into a module in the code. RCLane has the following general flow :
  - Second of all, I investigate each detail block. For example RCLane, I find that it has input image block, that require one camera, raw, pitch, yaw when it use training and FOV 30 when I want to detect far lane. I will reproduce follow all information that paper mention in block.
  - Continously, I invest next block that is SegFormer backbone, it has three header including Segment head, Distance head, Transfer head. In the block, I devide block to two part. With part 1, I code segment head to predict binary lane mask, distance head to predict distance map, and transfer head to predict transfer map. Distance and transfer have two direction forward and backward. After that, with part 2, I decide tensor input and output for each module. With RCLane, paper define R(HxWx3), I will code similar define for input tensor.
  - After that, I will read method to learn about transfer head and distance head. Transfer head give direction/vector to go from lane point currently to next lane point. About distance map, it has information about distance follow lane. Model predict forward and backward to recover both side of lane.
  - After segmentation, paper get foreground points using Point-NMS to keep keypoint that segmentation created. Each keypoint can become starting point to recover a lane. Forward branch follow :
    - pi+1 = pi + Tf . This is head model, it includes Segmentation head, Transfer head, Distance head
  - And final, we have decoder to handle from S,T,D to lane final.

- What do you do when the paper does not provide enough implementation details?
  - When I encountered that, I will adress assumptions. And I am very important about metric evaluate when paper leak of information. Because when I try to all of assumptions that I adress it, I will compare with metric of paper. If the paper has been accepted on conference or journey, has index on scopus, I will try to exceed the metric paper.

- How do you verify that your implementation matches the original paper?
  - I rely on the evaluation of the paper's metrics. So, I will try all my assumptions. It will create all different benchmark, and I will compare benchmark that I just create with benchmark paper, so if it is same, I will accept it.

- What is the difference between reproducing a paper and simply using its official repository?
  - It means when I reproducing a paper, I need to understand it in great detail. And if I use open-source offical, I will try to understand it and try to port code to native with pipeline

- Have you ever modified an architecture after reproducing it?
  - Yes, I have been using CSRT and modify header, I cut head layer and I removed the output layer and kept only the backbone; for CSRT, I only need the center point. I look at the heatmap—where the peak occurs is the center—while the CNN component handles the adjustment of width and height.

- How do you debug a reproduced model if the results are much worse than the paper?
  - If the result are much worse than the paper, I will check each block. First, the aspect most likely to differ significantly from model is the data itself or the data preprocessing
  - Second of all, I will check checkpoint, training processing to detect anomalies and the subsequent block.

### Em nên chuẩn bị

Có thể lấy ví dụ từ các model em đã làm việc như **RCLane / LaneATT / StreamPETR / tracking models**. Cần nói rõ:

`paper → architecture → data format → loss → training → evaluation → deployment`

---

### 1.2 Redesign or modify a state-of-the-art model

**Mức đã trao đổi:** **3–4/5** — **chưa xác nhận chính xác em đã bấm 3 hay 4**

Sometimes, we must update knowledge everyday, because if we don't update, we encountered issue about when we focus intently on a topic, someone else is already working on it while we are conducting our research.

### Interviewer có thể hỏi thêm

- What does it mean to redesign a SOTA model?
  - Redesigning a SOTA model means taking a strong existing model and changing part of its architecture to better fit a new problem or improve some limitation.

- Tell me about a model architecture that you modified.
  - I have defined SafeFail, as you can see. So, with year 2026, FailSafe is the SOTA paper. And I convert from revert current task to revert choose task. I add VLM to detect which sub task model should choose in pipeline. It can task 1, 2,... some thing like that.

### Em nên chuẩn bị

Một câu chuyện rõ ràng theo format:

`Problem → limitation of baseline → modification → reason → experiment → result`

---
