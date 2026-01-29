### Deep Thinking by Markov Chain of Continuous Thoughts

This is an anonymous repo for paper "Deep Thinking by Markov Chain of Continuous Thoughts".

### Environment
* GPU: 8xH200 141GB

### Training

#### Phase 1

```shell
torchrun --standalone --nproc_per_node=4 train.py
```

#### Phase 2

Set phase='2' in train.py and replace "phase1_saved.pt" with the checkpoint path saved from Phase 1, then run Phase 2 training:

```shell
torchrun --standalone --nproc_per_node=4 train.py
```

### Testing

In sample.py, replace "phase2_saved.pt" with the checkpoint path saved from Phase 2, then run testing:

```shell
torchrun --standalone --nproc_per_node=1 sample.py
```
