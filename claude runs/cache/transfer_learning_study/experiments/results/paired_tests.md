| comparison | metric | A - B | 95% CI | verdict |
|---|---|---|---|---|
| **06_resnet18_pretrained** vs **06_resnet18_scratch**: Transfer learning: ImageNet-pretrained ResNet-18 vs the SAME net from random weights | | | | |
| | AUC any (pooled) | +0.047 | [+0.008, +0.080] | A better |
| | AUC any (within-variety) | +0.021 | [-0.027, +0.064] | no clear difference |
| | AUC red blotch (within) | -0.008 | [-0.053, +0.035] | no clear difference |
| | AUC leafroll (within) | +0.021 | [-0.031, +0.077] | no clear difference |
| **08_ssl_pretrained** vs **08_ssl_scratch**: Self-supervised pretraining vs the SAME net from random weights | | | | |
| | AUC any (pooled) | +0.000 | [-0.011, +0.011] | no clear difference |
| | AUC any (within-variety) | -0.014 | [-0.034, +0.007] | no clear difference |
| | AUC red blotch (within) | -0.001 | [-0.017, +0.015] | no clear difference |
| | AUC leafroll (within) | -0.021 | [-0.047, +0.008] | no clear difference |
| **06_resnet18_pretrained** vs **04_pixel_cnn1d**: Pretrained whole-leaf ResNet vs pixel-level 1D CNN from scratch | | | | |
| | AUC any (pooled) | +0.053 | [+0.008, +0.097] | A better |
| | AUC any (within-variety) | +0.036 | [-0.019, +0.091] | no clear difference |
| | AUC red blotch (within) | -0.046 | [-0.096, +0.005] | no clear difference |
| | AUC leafroll (within) | -0.007 | [-0.074, +0.071] | no clear difference |
| **05_attention_mil** vs **04_pixel_mlp**: Attention-MIL pooling vs plain averaging of pixel predictions | | | | |
| | AUC any (pooled) | +0.018 | [-0.008, +0.049] | no clear difference |
| | AUC any (within-variety) | +0.005 | [-0.042, +0.058] | no clear difference |
| | AUC red blotch (within) | +0.006 | [-0.030, +0.043] | no clear difference |
| | AUC leafroll (within) | +0.001 | [-0.050, +0.052] | no clear difference |
| **07_dofa_probe** vs **06_resnet18_pretrained**: Hyperspectral foundation model (frozen) vs ImageNet ResNet | | | | |
| | AUC any (pooled) | -0.107 | [-0.155, -0.063] | B better |
| | AUC any (within-variety) | -0.024 | [-0.076, +0.038] | no clear difference |
| | AUC red blotch (within) | -0.005 | [-0.055, +0.056] | no clear difference |
| | AUC leafroll (within) | -0.005 | [-0.081, +0.070] | no clear difference |
| **07_dinov2_probe** vs **02_plsda__mean_snv__byvariety**: [post-hoc] best run overall vs best classical run | | | | |
| | AUC any (pooled) | +0.003 | [-0.034, +0.043] | no clear difference |
| | AUC any (within-variety) | +0.006 | [-0.046, +0.073] | no clear difference |
| | AUC red blotch (within) | -0.029 | [-0.090, +0.023] | no clear difference |
| | AUC leafroll (within) | +0.033 | [-0.042, +0.103] | no clear difference |
