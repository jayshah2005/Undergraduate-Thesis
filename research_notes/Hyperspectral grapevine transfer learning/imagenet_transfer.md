# ImageNet-pretrained 2D CNN/ViT transfer to hyperspectral leaf cubes (204 bands, ~150x150 px, ~1000 leaves)

Scope note: I did about 14 tool calls. Direct evidence for "ImageNet backbone on hyperspectral leaf-disease cubes" is thin. Most numbers below come from adjacent domains (synthetic multichannel, remote sensing, other HSI classification). Items marked [snippet-only] come from search-result summaries that I did not open in full.

## 1. Adaptation strategies (false-color/band-subset, PCA, learned 1x1 reduction, weight inflation to N channels)

### Takeaway
The one controlled comparison I found (Perez & Maji 2021, ImageNet VGG/ResNet, multichannel non-leaf data) shows a learned linear (1x1-style) adaptor to 3 channels or a band subset is about as good as or better than "inflate first-layer filters to N channels". Gains over RGB-only come from using all bands. Simple adaptors are hard to beat, and deeper adaptors add little. Nothing I found directly benchmarks these on leaf-disease cubes.

### Cited Findings
- Perez & Maji tested adaptors in front of ImageNet-pretrained VGG-D, ResNet18 and ResNet50. The adaptors were linear projection to 3 channels, multi-layer, inflated (average and replicate first-layer filters) and subset selection. They conclude that "simple methods, such as linear projection or subset selection, are often the most effective", and that a multi-view adaptor gives further gains. — [arXiv abstract 2108.01555](https://arxiv.org/abs/2108.01555); tables via [ar5iv](https://ar5iv.arxiv.org/html/2108.01555)
- VGG-D accuracy, linear / inflated / multi-layer / subset, on the real-HSI sets. These are fixed train/test splits with no separate validation split described, so treat them as single-split numbers. — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
  - EuroSAT (13 bands, 10k train): 97.1 / 96.7 / 97.7 / 96.6. From scratch: 95.7.
  - EuroSAT-small (1000 train samples): 88.4 / 83.6 / 92.8 / 86.6. From scratch: 75.2.
  - LEGUS-small (1000 train samples): 54.9 / 51.8 / 53.1 / 51.3. From scratch: 27.2.
  - So2Sat-small (1000 train samples): 45.1 / 40.8 / 40.8 / 43.8. From scratch: 34.5.
- Inflation was worst or near-worst in most synthetic cases. For example, CUB-15 gave 26.0 (inflated) vs 41.2 (linear) vs 45.9 (subset). — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- Multi-view adaptor (several parallel linear or subset 3-channel views fused by order-invariant pooling in an intermediate layer) added up to 11.4% on CUB. Performance plateaued at about 5 views. EuroSAT-small rose to 94.4. — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- Initialization matters little. PCA-initialized linear adaptors were equal to or slightly worse than random initialization (EuroSAT 94.7 vs 97.1; Cars-5 67.0 vs 70.5; So2Sat 53.9 vs 52.9). Raising the adaptor learning rate 10x gave average gains of 3.7% (synthetic) and 5.8% (real). — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- Using all bands beat RGB conversion. EuroSAT RGB was 70.3 vs 97.7 with 13 bands. LEGUS RGB conversion was 62.8 vs 65.3 with 5 bands. — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- Pretrained networks are sensitive to channel order and semantics for natural-image-like inputs. Birds accuracy fell from 72.0 to 63.6-66.8 under channel permutations. — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- PCA to a low-dimensional input helped a classical model on leaves. In wheat powdery mildew, an SVM on PCA-reduced bands reached 93.33% cross-validated accuracy vs 88.00% on full-band images. — [PMC7464903](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7464903/) (from search summary, not opened in full)
- [snippet-only] An SJSU thesis converted HSI samples to 3 channels (PCA or band selection) and fine-tuned ResNet/DenseNet/VGG. It reports 73.6% on Pavia University and 82.8% on Salinas with 25 training samples per class. — [SJSU thesis](https://scholarworks.sjsu.edu/etd_theses/5167)
- [snippet-only] On maize seeds, hyperspectral bands were split into 25 groups of 10 bands and an ImageNet-based network was applied per group. — [arXiv 1805.11784](https://arxiv.org/pdf/1805.11784)
- [snippet-only] For potato late blight, selecting high-reflectance bands improved several models, including a 2D-CNN, by about 5-7%. — [arXiv 2111.12155](https://arxiv.org/pdf/2111.12155)

### Inferences
- For 204 bands, the most defensible baselines are (a) a learned 1x1 conv, 204 to 3 and optionally 204 to 3k for a multi-view ensemble, trained at a higher learning rate than the backbone, and (b) a hand-picked or learned band subset. Plain inflation is the weakest option in the one comparison I found, so it should not be the default.
- Perez & Maji's data are not leaf cubes. The ranking may shift for narrow-band, highly correlated spectra, where PCA to 3 channels may capture most of the variance.
- Their "small" setting is 1000 training samples (not leaves), which is a similar order of magnitude to the user's ~1000 leaves. The ~15-point gains of pretrained adaptors over from-scratch in that regime (EuroSAT-small 75.2 to 88.4) are the most relevant effect-size evidence.

### Gaps
- No peer-reviewed head-to-head of pseudo-RGB vs PCA-3 vs 1x1-conv vs inflation on hyperspectral leaf disease. A direct search found none.
- The Perez & Maji ResNet18/50 results are only in their supplement, which I did not read.

## 2. Frozen features vs fine-tuning; best backbones; linear probing

### Takeaway
I found no study that compares frozen vs fine-tuned ConvNeXt, DINOv2 or ViT on hyperspectral leaf data. Related evidence is mixed. Pretraining helped most in the few-sample regime (Perez & Maji), but Lee et al. 2019 found no pretraining advantage for shallow HSI networks.

### Cited Findings
- On the few-sample sets above, ImageNet-pretrained VGG-D with an adaptor outperformed from-scratch training by a large margin (EuroSAT-small 88.4 vs 75.2; LEGUS-small 54.9 vs 27.2). The gap shrank on the full datasets (EuroSAT 97.1 vs 95.7). — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- Backbone ordering on RGB fine-tuning for VGG-D, ResNet18 and ResNet50: CUB 72.0 / 70.5 / 77.0; Cars 78.2 / 80.6 / 86.7; Aircraft 81.1 / 76.9 / 84.0. This is the fine-tuning upper bound on RGB, not hyperspectral. — [ar5iv 2108.01555](https://ar5iv.arxiv.org/html/2108.01555)
- Lee et al. 2019 pretrained a CNN on several hyperspectral source datasets and fine-tuned on a target. Their ablation found "a network trained from scratch performs as good as a network fine-tuned from a pre-trained network", with pretraining helping only for deeper networks. This is HSI-to-HSI pretraining, not ImageNet. The abstract gives no numbers or validation scheme. — [arXiv 1901.08658](https://arxiv.org/abs/1901.08658)
- DINOv2 is evaluated by linear probing on frozen features (the standard protocol). — [DINOv2 paper](https://arxiv.org/pdf/2304.07193). Its use on hyperspectral inputs requires reducing channels to 3 or changing the patch embedding. I found no paper testing this.
- Wheat leaf HSI, EfficientNet with a 2D-conv input (best of 4 models), reached 81% overall accuracy on 1,447 leaf images. Inception and 3D-conv variants were worse. The model was not stated to be ImageNet-pretrained, and the validation scheme was not found. — [NIAB/Cambridge summary](https://www.niab.com/node/3587)

### Inferences
- With ~1000 leaves, the evidence-backed default is fine-tuning a ResNet-class model with a learned adaptor, with differential learning rates. Frozen linear probing is a cheap baseline, but I have no HSI evidence for or against it.
- ConvNeXt, DINOv2 and ViT remain untested here. Treat them as hypotheses.

### Gaps
- No quantitative frozen-vs-fine-tuned or backbone comparison on hyperspectral plant data. DINOv2 and ConvNeXt on HSI were not found.
- The Lee 2019 numbers and datasets are not in what I read.

## 3. Plant-disease HSI studies: transfer vs scratch vs classical

### Takeaway
Published leaf-level HSI studies mostly train CNNs from scratch (3D or 2D) or use classical models. They show modest CNN gains over random forest / PLS-DA, and none that I found reports ImageNet-pretrained vs scratch.

### Cited Findings
- Grapevine leafroll/red blotch (Sawyer et al. 2023): 3D-CNN on 40 bands (510-710 nm), about 496 leaf images from 248 plants. Validation was stratified 5-fold CV, with no grouping by vine stated, so leakage between two leaves of one vine is possible. The CNN was not described as pretrained. — [PMC10036814](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
  - Binary, symptomatic: CNN 87.0 vs RF 82.4.
  - Binary, pre-symptomatic: CNN 85.6 vs RF 82.8.
  - Binary, entire dataset: CNN 80.9 vs RF 79.5.
  - Four-class, entire dataset: CNN 67.0 vs RF 62.2.
  - Four-class, pre-symptomatic: CNN 76.9 vs RF 77.7 (RF slightly better).
  - RF used the mean spectrum per leaf. The CNN used the full cube.
- Grapevine Esca: a 1D CNN on 900-1700 nm pixel spectra for asymptomatic/symptomatic/healthy leaves outperformed PLS-DA for segmentation. — [UniNavarra record](https://academica-e.unavarra.es/handle/2454/52132) (search summary; I did not get the numbers)
- Tea anthracnose-type disease: CARS-LSTM on HSI reached 95% vs 70% for the best RGB CNN (WT-ResNet18). This shows the spectral information advantage over RGB, but not pretrained vs scratch. — [Plant Methods s13007-023-01074-2](https://link.springer.com/article/10.1186/s13007-023-01074-2) (search summary)
- Wheat multi-infection: 81% overall, 72% on yellow rust + mildew. The authors say deep learning still works on a relatively small dataset, but rare classes need more data. — [NIAB](https://www.niab.com/node/3587)
- A rice-disease Frontiers paper examines deep transfer learning on spectral data. I did not open it and have no numbers. — [Frontiers 2021](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.693521/epub) [snippet-only]

### Inferences
- Realistic expectations for ~1000 leaves: CNN gains over classical baselines of roughly 1-5 points in the one grapevine study (Sawyer). The classical baseline should always be reported. A leaf-mean spectrum + RF/SVM/PLS-DA is cheap and sometimes better.
- Splits must be grouped by plant (vine), or the numbers will be optimistic.

### Gaps
- No leaf-disease HSI study with a pretrained-vs-scratch ablation was found.
- Several secondary papers were only seen as search summaries.

## 4. Self-supervised pretraining on target data (MAE, contrastive, autoencoders)

### Takeaway
SSL on hyperspectral data shows gains in the remote sensing literature, but at much larger scale (tens of thousands of tiles) than 1000 leaves. I found no evidence for small leaf datasets.

### Cited Findings
- TerraMAE (channel-grouped MAE) was pretrained on about 22,078 EnMAP 64x64 tiles with 218 bands. With a frozen encoder and a light conv head it gave mean accuracy of 54.79% (crop type) vs 49.91% for supervised ResNet-50 and 50.87% for a baseline MAE. For land cover it reached 83.82 vs 76.09 and 76.40. Soil texture MAE was 0.0806 vs 0.0863 and 0.0956 (5 runs). Classification used an 80/20 tile split. — [arXiv 2508.07020](https://arxiv.org/html/2508.07020v1)
- HyperspectralMAE (50% spatial and 50% spectral masking) reports strong Indian Pines results with few labeled pixels. The model is about 0.18B parameters. — [arXiv 2505.05710](https://arxiv.org/pdf/2505.05710) [snippet-only]
- An SPIE paper compares MAE masking dimension, fraction and coherence for HSI. It is early-stage. — [SPIE](https://biomedicaloptics.spiedigitallibrary.org/conference-proceedings-of-spie/13031/130310J/Adapting-self-supervised-learning-to-the-hyperspectral-domain--methods/10.1117/12.3013487.full) [snippet-only]
- An MAE for hyperspectral plant images exists, but it is a systems paper about distributed training. — [arXiv 2506.21411](https://arxiv.org/pdf/2506.21411) [snippet-only]

### Inferences
- A leaf dataset has ~1000 x ~20k = ~20M labeled-leaf pixels, but they are highly correlated. Pixel or small-patch spectral autoencoders or masked-band prediction are plausible as an auxiliary pretext step, but the evidence is by analogy only.
- A reasonable test is to compare against (supervised from scratch) and (ImageNet + adaptor) on the same grouped CV.

### Gaps
- No study of SSL (MAE, contrastive, spectral AE) on a small HSI plant or leaf dataset was found. No comparison against ImageNet transfer was found.

## 5. Augmentation and pixel-level / MIL training for per-leaf labels

### Takeaway
Geometric augmentation is standard in leaf HSI CNNs, and spectral augmentation helps in pixel-level work. I found no MIL-specific hyperspectral leaf paper.

### Cited Findings
- Sawyer et al. applied random flips, shifts, scaling and rotation (Albumentations, p=0.5 per copy) to CNN training images only. — [PMC10036814](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
- Relighting augmentation of pixel spectra (11x data) improved a CNN at every training-set size tested. — [arXiv 1611.09007](https://arxiv.org/pdf/1611.09007) [snippet-only]
- Another approach adds neighbor-pixel augmentation plus a loss penalizing weight differences between adjacent wavelengths, evaluated with no train/test pixel overlap. — [arXiv 1711.05512](https://arxiv.com/abs/1711.05512v1) [snippet-only]
- A 2025 preprint reports guided-diffusion augmentation on PRISMA data. — [arXiv 2510.08363](https://arxiv.org/abs/2510.08363v1) [snippet-only]
- HyperLeaf2024 (2,410 wheat HSIs) notes that most HSI datasets are per-pixel, and that whole-image/leaf-level datasets are scarce. — [CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/FGVC11/papers/Laprade_HyperLeaf2024_-_A_Hyperspectral_Imaging_Dataset_for_Classification_and_Regression_CVPRW_2024_paper.pdf) (search summary; the PDF returned 403)

### Inferences
- Pixel or patch training with per-leaf label inheritance, followed by mean-probability aggregation per leaf, multiplies the number of training samples, but diseased leaves have many healthy-looking pixels, so label noise is a risk. An attention-based MIL pooling is a natural fix but is untested in the sources I found.
- Spectral augmentations (intensity scaling or relighting, small band-wise noise) are low-risk. Flips and rotations are valid for leaf images.

### Gaps
- No MIL or pixel-to-leaf aggregation study on HSI plant disease was found.
- No ablation of augmentation types for leaf-level HSI was found.
