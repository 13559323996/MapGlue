# MapGlue: Multimodal Remote Sensing Image Matching
## Abstract
Multimodal remote sensing image (MRSI) matching is pivotal for cross-modal fusion, localization, and object detection, but it faces severe challenges due to geometric, radiometric, and viewpoint discrepancies across imaging modalities. Existing unimodal datasets lack scale and diversity, limiting deep learning solutions. This paper proposes MapGlue, a universal MRSI matching framework, and MapData, a large-scale multimodal dataset addressing these gaps. Our contributions are twofold. MapData, a globally diverse dataset spanning 233 sampling points, offers original images (7,000×5,000 to 20,000×15,000 pixels). After rigorous cleaning, it provides 121,781 aligned electronic map–visible image pairs (512×512 pixels) with hybrid manual-automated ground truth, addressing the scarcity of scalable multimodal benchmarks. MapGlue integrates semantic context with a dual graph-guided mechanism to extract cross-modal invariant features. This structure enables global-to-local interaction, enhancing descriptor robustness against modality-specific distortions. Extensive evaluations on MapData and five public datasets demonstrate MapGlue’s superiority in matching accuracy under complex conditions, outperforming state-of-the-art methods. Notably, MapGlue generalizes effectively to unseen modalities without retraining, highlighting its adaptability. This work addresses longstanding challenges in MRSI matching by combining scalable dataset construction with a robust, semantics-driven framework. Furthermore, MapGlue shows strong generalization capabilities on other modality matching tasks for which it was not specifically trained.

![multi_matching_5_0319_github](https://github.com/user-attachments/assets/0a63abdd-04d4-48fd-8209-b18c95a7763d)
Fig. 1. Qualitative Results of Multimodal Image Matching. "Easy," "Normal," and "Hard" denote different levels of transformations applied to the images.

## MapData Dataset
![数据集分布_2_github](https://github.com/user-attachments/assets/5c3476ae-c466-47ba-899b-f06780ce0a90)
Fig. 2. Geographic Distribution of Sampled Images in the MapData Dataset.
MapData, a globally diverse dataset spanning 233 sampling points, offers original images (7,000×5,000 to 20,000×15,000 pixels). After rigorous cleaning, it provides 121,781 aligned electronic map–visible image pairs (512×512 pixels) with hybrid manual-automated ground truth, addressing the scarcity of scalable multimodal benchmarks.




