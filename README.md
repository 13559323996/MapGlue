# MapGlue: Multimodal Remote Sensing Image Matching
## Abstract
Multimodal remote sensing image (MRSI) matching is pivotal for cross-modal fusion, localization, and object detection, but it faces severe challenges due to geometric, radiometric, and viewpoint discrepancies across imaging modalities. Existing unimodal datasets lack scale and diversity, limiting deep learning solutions. This paper proposes MapGlue, a universal MRSI matching framework, and MapData, a large-scale multimodal dataset addressing these gaps. Our contributions are twofold. MapData, a globally diverse dataset spanning 233 sampling points, offers original images (7,000×5,000 to 20,000×15,000 pixels). After rigorous cleaning, it provides 121,781 aligned electronic map–visible image pairs (512×512 pixels) with hybrid manual-automated ground truth, addressing the scarcity of scalable multimodal benchmarks. MapGlue integrates semantic context with a dual graph-guided mechanism to extract cross-modal invariant features. This structure enables global-to-local interaction, enhancing descriptor robustness against modality-specific distortions. Extensive evaluations on MapData and five public datasets demonstrate MapGlue’s superiority in matching accuracy under complex conditions, outperforming state-of-the-art methods. Notably, MapGlue generalizes effectively to unseen modalities without retraining, highlighting its adaptability. This work addresses longstanding challenges in MRSI matching by combining scalable dataset construction with a robust, semantics-driven framework. Furthermore, MapGlue shows strong generalization capabilities on other modality matching tasks for which it was not specifically trained.

![multi_matching_5_0319_github](https://github.com/user-attachments/assets/0a63abdd-04d4-48fd-8209-b18c95a7763d)
Fig. 1. Qualitative Results of Multimodal Image Matching. "``Easy``," "``Normal``," and "``Hard``" denote different levels of transformations applied to the images.

## MapData Dataset
![数据集分布_2_github](https://github.com/user-attachments/assets/5c3476ae-c466-47ba-899b-f06780ce0a90)
Fig. 2. Geographic Distribution of Sampled Images in the MapData Dataset.

MapData is a globally diverse dataset spanning ``233`` geographic sampling points. It offers original high-resolution images ranging from 7,000×5,000 to 20,000×15,000 pixels. After rigorous cleaning, the dataset provides ``121,781`` aligned electronic map–visible image pairs (each standardized to 512×512 pixels) with hybrid manual-automated ground truth—addressing the scarcity of scalable multimodal benchmarks.

Within the MapData structure, each of the 233 folders represents a unique geographic sampling point. Inside each folder, there are three subfolders (named ``1``, ``2``, and ``3``) corresponding to three different image pairs sampled at that location. Each subfolder further contains two directories: ``L`` for source images and ``R`` for target images.  
### Data format  
```text
MapData/
├── 001/             # sampling points
│   ├── 1/
│   │   ├── L/       # Source images
│   │   └── R/       # Target images
│   ├── 2/
│   │   ├── L/
│   │   └── R/
│   └── 3/
│       ├── L/
│       └── R/
└── ...
```

The MapData-test dataset can be obtained at this link xxxxxxx, and the full dataset will be available after our paper is accepted!
# Runing MapGlue
## Requirements
```python
conda create -n mapglue python=3.12
conda activate mapglue
git clone xxx
cd MapGlue
pip install -r requirements.txt
```
## Usage
### Running Inference
The TorchScript model accepts inputs as ``torch.Tensor`` with shape ``(H, W, C)`` in RGB format. Values may be in the range [0, 255] (uint8) or [0, 1]. The model internally converts the image to the required format.  
Below is a demo:
```python

import cv2
import torch
# Load the TorchScript model
model = torch.jit.load('./weights/mapglue_model.pt')
model.eval()

# Read input images using OpenCV
image0 = cv2.imread('./assets/image0.jpg')
image1 = cv2.imread('./assets/image1.jpg')

# Convert BGR to RGB
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Convert numpy arrays to torch.Tensor
image0 = torch.from_numpy(image0)
image1 = torch.from_numpy(image1)
num_keypoints = torch.tensor(2048) # Defaults to 2048

# Run inference
points_tensor = model(image0, image1, num_keypoints)
points0 = points_tensor[:, :2]
points1 = points_tensor[:, 2:]
```
# Citation
If you find our work useful in your research, please consider giving a star ⭐ and a citation
```python
```



