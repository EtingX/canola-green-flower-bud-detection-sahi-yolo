# canola-green-flower-bud-detection-sahi-yolo

Accurate detection of individual green buds in canola fields is critical for high-throughput phenotyping in breeding, hybrid seed production, and regulatory field trials. However, these buds are extremely small and camouflaged, posing significant challenges for conventional object detectors. This study presented an advanced detection pipeline that integrates SPD-Conv, C3K2-PPA, and ASFF modules into a YOLOv11 backbone, coupled with Slicing Aided Hyper Inference (SAHI) and multi-scale training strategies. Extensive evaluations demonstrated that the proposed model outperformed baseline and ablation variants across image-level and object-level tasks, achieving an R² of 0.879 and mAP@50 of 0.910 under 960-pixel SAHI inference. Grad-CAM visualisations confirmed the model’s ability to produce spatially precise and centre-focused activations. Beyond canola, this modular architecture is adaptable to other crops featuring small and low-contrast phenotypic traits. The results offer a generalizable framework for improving small object detection in complex field environments, supporting scalable and accurate phenotyping across diverse agricultural systems.

Model:https://huggingface.co/Eting0308/canola-green-flower-bud-detection-sahi-yolo_model/tree/main
Dataset:https://huggingface.co/datasets/Eting0308/canola-green-flower-bud-detection-sahi-yolo_dataset/blob/main/canola_dataset.zip

![image](https://github.com/user-attachments/assets/ed5f491d-d00c-46c6-9262-ff7831f0d8b0)

	R²	mAP@50	mAP@50:95	R² diff	mAP@50 diff	mAP (50-95) diff
Original 	0.825 	0.832 	0.425 	-	-	-
SPD-Conv	0.844 	0.884 	0.468 	0.019	0.053	0.043
C3K2-PPA	0.873 	0.862 	0.447 	0.048	0.031	0.022
ASFFHead	0.857 	0.855 	0.420 	0.032	0.023	-0.006
SPD-Conv+C3K2-PPA	0.844 	0.920 	0.510 	0.019	0.089	0.085
SPD-Conv+ASFFHead	0.736 	0.853 	0.469 	-0.089	0.021	0.044
C3K2-PPA+ASFFHead	0.894 	0.894 	0.474 	0.069	0.062	0.049
Advanced	0.879 	0.910 	0.498 	0.054	0.078	0.072

