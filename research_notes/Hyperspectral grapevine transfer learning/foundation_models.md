# Hyperspectral foundation models and cross-dataset transfer for a ~1000-sample leaf-level, Specim IQ (204 bands, 400-1000 nm) disease task

Method note: all facts below come from fetched pages or search snippets (2026-10). "Verified" = read on the fetched page; "claimed" = stated by authors/abstract only or from a search summary. Full-text tables of the original papers were not read for SpectralGPT, DOFA, HyperFree supplement, or Copernicus-FM.

## Q1. What pretrained hyperspectral/spectral foundation models exist, what were they pretrained on, are weights public?

### Takeaway
Nearly all are remote-sensing models (satellite/airborne, 30 m to ~1 m pixels). Public weights are verified for HyperSIGMA, DOFA, Copernicus-FM and SpectralEarth. SpectralGPT's official weights could not be confirmed, and HyperFree's release was not confirmed. Only SpectralEarth, HyperSIGMA and HyperFree are truly hyperspectral; SpectralGPT, SatMAE, Copernicus-FM and DOFA's training data are largely multispectral.

### Cited Findings
- SpectralGPT: ViT-B 3D masked autoencoder; keeps 12 of Sentinel-2's 13 bands; pretrained 200 epochs on fMoW-S2 then 100 on BigEarthNet-S2 (128x128x12); about 1M images — [arXiv 2311.07113 (search summary)](https://arxiv.org/pdf/2311.07113). Reported SpectralGPT+ numbers (claimed, via secondary overview): EuroSAT 99.21%, BigEarthNet 88.22 macro-mAP, SegMunich 51.0 mIoU — [alphaXiv](https://www.alphaxiv.org/overview/2311.07113v3).
- SpectralGPT weights: a third-party reproduction on Hugging Face (OneScience-Group) says weights "will be uploaded soon"; it requires 12-band Sentinel-2 style input, GPL-3.0 — [HF OneScience-Group/SpectralGPT](https://huggingface.co/OneScience-Group/SpectralGPT). Official release not confirmed by me (gap).
- HyperSIGMA: pretrained on HyperGlobal-450K, about 447k 64x64 patches, 247,072 from EO-1 (Hyperion) and 200,000 from GF-5 — [arXiv 2406.11519v2](https://arxiv.org/html/2406.11519v2). Weights: six checkpoints (Spatial_MAE and Spectral_MAE, ViT-B/L/H) on Hugging Face `WHU-Sigma/HyperSIGMA` and Baidu; Apache-2.0 for the repo (weight license unstated) — [GitHub WHU-Sigma/HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA). The README is internally inconsistent about band configuration ("hundreds of bands" vs "three-band").
- HyperSIGMA classification OA reported by authors: IP 85.54, PU 93.52, HC 94.44, HH 94.87, HU 86.80, ZY 94.92 — [arXiv 2406.11519v2](https://arxiv.org/html/2406.11519v2).
- HyperFree (CVPR 2025): SAM-style ViT-base; pretraining set Hyper-Seg with about 41.9k to 50k AVIRIS images, 224 channels, 400-2500 nm at 10 nm, 0.6-5 m GSD, plus processed multispectral fMoW/SpaceNet (about 150k images total); wavelength weight dictionary spans 400-2500 nm — [arXiv 2503.21841](https://arxiv.org/html/2503.21841v1). Paper text says code/dataset "would be accessible" at [project page](https://rsidea.whu.edu.cn/hyperfree.htm); weight release not verified by me.
- DOFA: wavelength-conditioned dynamic hypernetwork; pretrained on five EO modalities; weights public — [arXiv 2403.15356](https://arxiv.org/abs/2403.15356). Weights on HF `XShadow/DOFA`, torch.hub and TorchGeo, MIT license; accepts any number of channels if wavelengths (micrometres) are given — [GitHub zhu-xlab/DOFA](https://github.com/zhu-xlab/DOFA). Examples in the README span 0.48 to 2.19 um (NAIP to Sentinel-2 SWIR).
- Copernicus-FM: Fourier-encoded wavelength/bandwidth to hypernetwork patch embedding; pretrained on Copernicus-Pretrain (18.7M images, Sentinel-1 to 5P) — [arXiv 2503.11849](https://arxiv.org/pdf/2503.11849). Weights on HF `wangyi111/Copernicus-FM`, CC-BY-4.0 — [GitHub](https://github.com/zhu-xlab/Copernicus-FM). Page shows no hyperspectral/EnMAP results.
- SpectralEarth: EnMAP, 538,974 patches (128x128, 30 m), 202 bands after removing water-absorption bands; ResNet-50, ViT-S/B/L/H/g with a spectral adapter; MoCo-v2, DINO, MAE — [arXiv 2408.08447](https://arxiv.org/abs/2408.08447), [search summary of paper](https://arxiv.org/html/2408.08447v2). Weights via a DESY SyncAndShare link; license not stated; dataset was replaced 25 Sep 2025 — [GitHub AABNassim/spectral_earth](https://github.com/AABNassim/spectral_earth). Downstream tasks are land cover, crop type, tree species.
- HyperSL: used in a 2026 cross-domain study; pixel-wise spectral tokenizer using band central wavelengths; pretraining data and band count not stated in the page I read — [arXiv 2604.26478](https://arxiv.org/html/2604.26478).
- SatMAE is multispectral, not HSI — [arXiv 2604.26478](https://arxiv.org/html/2604.26478). SpectralFormer is a supervised architecture, not a released pretrained foundation model (my knowledge; not verified here).
- The only proximal-sensing HSI foundation model cited by the 2026 study (Laprade et al.) "is not publicly available" — [arXiv 2604.26478](https://arxiv.org/html/2604.26478).
- An unnamed 2025 MAE trained on 16.5k images (7 proximal datasets + 3 self-collected with a Specim IQ + 8,504 EnMAP patches, about 388-2445 nm) exists, but code/data/weights were "URL to come" — [arXiv 2503.01628](https://arxiv.org/html/2503.01628v1). This is the closest match to the user's sensor and domain, and it is not usable.

### Inferences
- For a Specim IQ 400-1000 nm leaf task, the practical public candidates are DOFA, Copernicus-FM, HyperSIGMA, SpectralEarth, and maybe HyperFree. None was pretrained on close-range leaf data.
- SpectralGPT-style models need 12 Sentinel-2 bands, which is a poor fit for 204 contiguous bands.

### Gaps
- Official SpectralGPT repo/weights status not confirmed; HyperFree weights status not confirmed; DOFA's five pretraining datasets and Copernicus-FM spectral ranges not read in full text; SpectralEarth/HyperSIGMA per-sensor wavelength ranges not obtained (Hyperion is about 400-2500 nm and EnMAP 420-2450 nm from my knowledge, unverified here).

## Q2. Arbitrary band counts/wavelengths, close-range 400-1000 nm leaf domain, evidence of domain gap

### Takeaway
Wavelength-conditioned models (DOFA, Copernicus-FM, HyperFree, HyperSL) can technically ingest 204 bands at known wavelengths; HyperSIGMA and SpectralGPT cannot natively and need re-initialised embeddings or PCA. Direct evidence on leaf-level 400-1000 nm is thin: one 2026 paper on proximal segmentation, and no foundation-model paper on leaf disease found.

### Cited Findings
- HyperSIGMA: random continuous channel subsets at pretraining, spectral branch averages adjacent channels into a fixed token count; if the downstream band count differs, the patch embedding is randomly initialised — [arXiv 2406.11519v2](https://arxiv.org/html/2406.11519v2). This partly discards pretraining on the input layer.
- SpectralGPT: input fixed at 12 bands; a 2025 study reduced each hyperspectral dataset (115 to 270 bands) to 12 bands with PCA and 32x32 patches — [arXiv 2505.15334](https://arxiv.org/html/2505.15334v1). PCA destroys the wavelength correspondence the model learned.
- DOFA: "can handle images with any number of input channels" if wavelengths are provided — [GitHub](https://github.com/zhu-xlab/DOFA). Copernicus-FM claims any spectral or non-spectral sensor — [GitHub](https://github.com/zhu-xlab/Copernicus-FM).
- HyperFree: dynamic embedding, wavelength dictionary 400-2500 nm at 10 nm; authors note the training data is mostly airborne and key-channel selection relies on expert knowledge of sensors — [arXiv 2503.21841](https://arxiv.org/html/2503.21841v1) (limitations are my reading as summarized by the fetch tool, not an authors' limitations section).
- Proximal transfer evidence (HS3-Bench, semantic segmentation, frozen HyperSL backbone; datasets HCV2 128 bands 450-950 nm, HyKo2 15 bands 470-630 nm, HSI-Drive 25 bands 600-975 nm): mIoU average 60.60 for remote-sensing HSI FM vs 57.68 in-domain RU-Net and 53.36 U-Net, but 76.58 for RGB-pretrained DINOv2 with an HSI adapter — [arXiv 2604.26478](https://arxiv.org/html/2604.26478). So a remote-sensing HSI FM helped a bit and an RGB foundation model with an adapter helped far more.
- Authors of the 2026 study frame transfer as "cross-domain" from remote sensing to proximal sensing and say it needs no RGB-HSI modality bridging — [abstract](https://arxiv.org/abs/2604.26478) (claimed).
- A 2025 review says deep learning at close range is less explored and flags limited annotated data — [Waikato Research Commons](https://researchcommons.waikato.ac.nz/items/4eac55d3-4f71-47c6-8725-720f0c276ab6).

### Inferences
- At 400-1000 nm, a model whose pretraining (EnMAP, Hyperion, AVIRIS, Sentinel-2) has substantial SWIR and 30 m mixed pixels sees only the VNIR portion in the leaf task; leaf-level pure-pixel spectra with red-edge/pigment signatures differ in statistics from mixed landscape pixels. I found no paper that measures this gap directly for leaf disease.
- Small objects: Specim IQ cubes (512x512) of single leaves differ in spatial scale from 30 m patches, so pixel-wise spectral tokenisers (HyperSL-style) or spectral branches are more plausible fits than spatial patch ViTs (my inference).

### Gaps
- No paper found applying SpectralGPT, HyperSIGMA, DOFA, HyperFree or SpectralEarth to leaf-level plant disease classification. A targeted search returned only RGB foundation-model adaptation and non-foundation hyperspectral work, e.g. [Chen et al. ICCVW 2023](https://openaccess.thecvf.com/content/ICCV2023W/CVPPA/papers/Chen_Adapting_Vision_Foundation_Models_for_Plant_Phenotyping_ICCVW_2023_paper.pdf) (RGB, frozen backbone plus LoRA).

## Q3. Does pretraining on public HSI datasets (Indian Pines, Pavia, Salinas) help a different sensor/task? Positive or negative transfer?

### Takeaway
Evidence is mostly positive for same-domain or similar-range remote-sensing transfer, mixed for cross-domain, and I found no clean negative-transfer measurement for Indian Pines/Pavia pretraining. Gains shrink or reverse in some settings (below).

### Cited Findings
- Negative/mixed result: on HSI-Drive (600-975 nm, close to VNIR), in-domain RU-Net beat the frozen-FM variant at 10% and 25% of the data, and only the 100% setting favored the FM; largest positive gain was about +12 mIoU on HyKo2 at 10% data; HCV2 gain about +1.6 — [arXiv 2604.26478](https://arxiv.org/html/2604.26478). Exact subset numbers appear only in figures.
- Survey-level and snippet-level claims that transfer gains grow as labeled samples shrink — [search results incl. arXiv 2112.01800 survey](https://arxiv.org/pdf/2112.01800); I did not read specific numbers (claimed).
- Cross-sensor transfer pairs used in the literature (Pavia to Houston, Indian Pines to Botswana) were chosen to have similar resolution and spectral range — [MDPI Remote Sens. 13(5):930](https://www.doi.org/10.3390/RS13050930) (snippet). This implies success depends on similarity, not on arbitrary sensors.
- A cross-domain few-shot paper notes alignment can "exacerbate negative transfer" under domain shift; its Indian Pines OA gain was about 69.21 to 71.24 — [search snippet, paper not identified](https://pure.nwpu.edu.cn/zh/publications/few-shot-hyperspectral-image-classification-using-meta-learning-a) (weak, secondary).
- Classic Indian Pines/Pavia/Salinas are small, few-scene, same-scene train/test benchmarks. Foundation models cited here were not pretrained on them as the main source; HyperSIGMA only fine-tunes on them — [arXiv 2406.11519v2](https://arxiv.org/html/2406.11519v2).

### Inferences
- Pretraining on IP/Pavia/Salinas alone would be a very small, scene-specific source (each is one image); large multi-sensor corpora (EnMAP, EO-1, AVIRIS) are the realistic source.
- Since the user's target (VNIR, leaf, 204 bands) overlaps only the visible/NIR portion of any source, negative or null transfer is plausible, particularly with frozen backbones and small data (as on HSI-Drive).

### Gaps
- No published controlled study of Indian Pines/Pavia/Salinas pretraining followed by transfer to a different sensor with from-scratch baselines was found.

## Q4. Reported gains of transfer vs from-scratch for small labeled hyperspectral sets

### Takeaway
Gains are large when "scratch" means a data-hungry ViT/ResNet on tiny data, but small or absent against well-tuned lightweight baselines, and ImageNet-pretrained RGB models are often competitive.

### Cited Findings
- Proximal spectral MAE (16.5k images, includes Specim IQ data), F1: DriedFoods 93.18, Wheat 66.49, BigEarthNet 69.34 vs ResNet-50 scratch 18.42 / 16.08 / 68.36, ResNet-50 ImageNet 91.21 / 46.65 / 67.89, ViT-L scratch 0 / 33.34 / 0, ViT-L ImageNet 97.98 / 56.17 / 70.96 — [arXiv 2503.01628 Table 4](https://arxiv.org/html/2503.01628v1). ImageNet ViT-L beat the HSI model on two of three tasks; weights unreleased.
- HyperFree few-shot (5 shots baselines vs 1 point prompt per class, frozen): LongKou OA 93.39 vs best baseline MambaHSI 92.65; HanChuan 75.47 vs 73.33 — [arXiv 2503.21841](https://arxiv.org/html/2503.21841v1). Gains are about 1-2 OA points; the fetch tool noted the paper's "about 12-point F1" claim could not be verified from the tables.
- SpectralGPT full fine-tuning OA: IP 85.62, Pavia 93.74, Houston 90.68, Botswana 98.97, LongKou 97.55; vs HyperSIGMA 87.26 / 84.80 / 82.81 / 98.65 / 99.45 and MambaHSI 95.77 / 90.61 / 86.70 / 95.51 / 99.62; the paper has no from-scratch baseline — [arXiv 2505.15334](https://arxiv.org/html/2505.15334v1). Indian Pines used 695 training pixels, Botswana 420. Note SpectralGPT (12-band PCA input) beat HyperSIGMA on three of five datasets, and a specialised Mamba model beat both on Indian Pines.
- HyperSL on proximal segmentation: +3 mIoU average over RU-Net in-domain (60.60 vs 57.68); up to +12 at 10% data on HyKo2 — [arXiv 2604.26478](https://arxiv.org/html/2604.26478).
- Plant-trait regression: GreenHyperSpectra (cross-sensor, cross-ecosystem pretraining set) reports label-efficient pretrained models outperform the supervised baseline; no numbers on the abstract page — [arXiv 2507.06806](https://arxiv.org/abs/2507.06806) (claimed).

### Inferences
- For about 1000 leaf-level samples, a lightweight spectral-feature model (1D CNN/MLP/PLS/SVM on mean or pixel spectra) is the baseline any foundation model must beat; the literature above provides no leaf-disease evidence that it would. A fair test: frozen/PEFT DOFA or HyperSIGMA vs scratch 1D models vs an ImageNet RGB-adapter baseline, with leaf/plant-level splits.

### Gaps
- No from-scratch vs pretrained comparison on a leaf-level disease dataset with 204 bands at 400-1000 nm; HyperSIGMA/DOFA/SpectralEarth numbers on non-remote-sensing tasks were not found.
