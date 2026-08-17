# CLAST Base 项目解读

这是一个基于 **VAE + GMM(高斯混合模型)先验** 的无监督聚类与"图谱(atlas)构建"项目,以 MNIST 作为基准数据集。项目名称里的 atlas/Scheme-A 暗示其最终目的可能是迁移到脑影像或医学图像的群体图谱构建,这里先用 MNIST 做基线验证。

## 项目结构

```
clast_base/
├── config.py          # 超参数配置 (CFG 类)
├── train.py           # 训练主入口
├── models/vae.py      # 残差卷积 VAE + 可训练 GMM 先验
├── data/data_utils.py # 平衡采样的 MNIST subset
├── training/trainer.py# 单 epoch 训练 + 评估保存
└── utils/
    ├── clustering.py    # sklearn GMM 重拟合、Hungarian 准确率
    ├── visualization.py # 图谱构建、聚类样本网格保存
    └── general.py       # 种子、CSV 等工具
```

## 核心方法

### 1. 模型 (`models/vae.py`)

- `Encoder/Decoder`:Conv + ResidualBlock 的卷积 VAE,潜变量维度 `latent_dim=8`
- `StaticGMMVAE`:在潜空间维护一个**可训练的对角协方差 GMM 先验**(`gmm_means`、`gmm_logvars`、`logit_pi`),用 `K=10` 个分量

### 2. 损失 (`training/trainer.py:42-56`)

```
loss = w_recon · BCE(x̂,x) + w_kl · KL(q(z|x)‖N(0,I)) + w_gmm · -log Σ π_k N(z|μ_k,σ_k²)
```

权重默认 `1:1:1`,可选 `pi_balance` 鼓励分量均衡。

### 3. 双 GMM 策略(关键设计)

- **训练中**:用网络内嵌的可训练 GMM 计算 prior loss
- **每个 epoch 结束**:用 sklearn 在 `μ(x)` 上**重拟合(refit)**一个 full-cov GMM,做评估
- **第 20 epoch**:用 refit GMM 的参数**硬覆盖**网络内的 GMM 先验(`overwrite_vae_prior_from_refit_gmm`),相当于一次"prior 复位"

### 4. Scheme-A Atlas (`build_cluster_atlas_A`)

对每个簇,在**像素空间**取属于该簇样本的中位数(或均值)作为该簇的"图谱"。配合 `conf_thresh=0.9` 和 `gamma_power=2.0` 锐化责任度,只用高置信样本平均。这就是项目核心的输出之一。

## 评估与产物

每个 epoch 写到 `results_schemeA_atlas_refit/`:

- `metrics.csv`:Hungarian 准确率 / NMI / ARI
- `cluster_samples/`:每簇 top-64 高置信样本网格
- `cluster_purity_epoch_XXX.csv`:每簇真实数字组成
- `atlas_A/`:Scheme-A 像素空间图谱(K 张 + 合并 grid)
- `atlas_refit_means/`:`decode(GMM_means)` 得到的潜空间图谱
- `vae_prior_means_decoded_epoch_XXX.png`:可训练先验 means 解码图
- `best_vae.pth`:按 test refit acc 取最优

## 一句话总结

> 用 VAE 把图像编码到 8 维潜空间,在该空间用 GMM 同时做生成先验和聚类;对每个簇分别用"潜空间均值解码"和"像素空间稳健均值"两种方式构建簇图谱,以 MNIST 作为 atlas-building 流程的可验证基线("CLAST base")。
