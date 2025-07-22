# MAFE
**Manifold-Aware Point Cloud Completion via Geodesic-Attentive Hierarchical Feature Learning**

MAFE is a novel framework for point cloud completion that leverages geodesic-aware attention and hierarchical feature learning. This repository contains the official implementation and training/testing scripts.

---

## Environment

- **PyTorch 1.12.0** with NVIDIA GPU support recommended

### Dependencies

- [pointnet2_ops_lib](./extension/pointnet2_ops_lib)
- [Chamfer Distance](./extension/ChamferDistance)

Install dependencies (under `extension` folder):

```bash
python3 setup.py install
```

---

## Dataset & Python Requirements

- Please follow [PoinTr's dataset and environment preparation](https://github.com/yuxumin/PoinTr/tree/master).
- Mainstream datasets supported include ShapeNet, PCN, MVP, KITTI, etc.

---

## Pretrained Models

> Pretrained models will be released soon. Stay tuned.

---

## Training & Testing

Example distributed training command:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -m torch.distributed.launch \
  --master_port=12223 --nproc_per_node=2 main.py \
  --launcher pytorch --sync_bn \
  --config ./MAFE/cfgs/PCN_models/Mafe_attention.yaml  \
  --exp_name test --val_freq 1 --val_interval 50
```

Please adjust parameters according to your GPU availability and config file paths.

---

## Citation

If you find this project helpful for your research, please cite our paper (to be released).

---

## Acknowledgment

- Our code is based on [PoinTr](https://github.com/yuxumin/PoinTr).
- We thank all related open-source projects.
