# Distributed Training: DDP and FSDP

**Mức đã trao đổi:** **2–3/5** — **chưa xác nhận chính xác**

**DDP — Distributed Data Parallel:** mỗi GPU giữ một bản đầy đủ của model, mỗi GPU xử lý một phần batch và gradients được đồng bộ.

**FSDP — Fully Sharded Data Parallel:** parameters, gradients và optimizer states có thể được shard giữa nhiều GPU để giảm memory trên từng GPU.

### Interviewer có thể hỏi thêm

- What is distributed training?
  + It includes DDP and FSDP in library torch.nn.parallel
  + Code :

import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler

-----------------------------------------------------------

import torch.distributed as dist


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


- What is the difference between DataParallel and DistributedDataParallel?
  + DataParallel use a main process to control GPU. Model has been replicated to GPU, batch has been break down. Others GPU run forward/backward, and the result gather to main GPU.
  + And DDP (DistributedDataParallel) use each process for each GPU. And each process keep full model, each GPU processes its own batch.
- How does DDP synchronize gradients?
  + It uses Allreduce to add gradient from all GPU : sum avegare
- What problem does FSDP solve?
  + GPU memory, with large model, VRAM GPU is not contain full model, so FSDP will handle it as devide model for many GPU.
- What does FSDP shard?
  + FSDP shards the model—including parameters, gradients, and weights—across the GPUs. This means that during the optimization step, it doesn't use AllReduce; instead, it performs optimization directly on each specific shard of the model.

---
