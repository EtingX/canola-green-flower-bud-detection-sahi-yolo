# canola-green-flower-bud-detection-sahi-yolo

Accurate detection of individual green buds in canola fields is critical for high-throughput phenotyping in breeding, hybrid seed production, and regulatory field trials. However, these buds are extremely small and camouflaged, posing significant challenges for conventional object detectors. This study presented an advanced detection pipeline that integrates SPD-Conv, C3K2-PPA, and ASFF modules into a YOLOv11 backbone, coupled with Slicing Aided Hyper Inference (SAHI) and multi-scale training strategies. Extensive evaluations demonstrated that the proposed model outperformed baseline and ablation variants across image-level and object-level tasks, achieving an R² of 0.879 and mAP@50 of 0.910 under 960-pixel SAHI inference. Grad-CAM visualisations confirmed the model’s ability to produce spatially precise and centre-focused activations. Beyond canola, this modular architecture is adaptable to other crops featuring small and low-contrast phenotypic traits. The results offer a generalizable framework for improving small object detection in complex field environments, supporting scalable and accurate phenotyping across diverse agricultural systems.

Model:https://huggingface.co/Eting0308/canola-green-flower-bud-detection-sahi-yolo_model/tree/main
Dataset:https://huggingface.co/datasets/Eting0308/canola-green-flower-bud-detection-sahi-yolo_dataset/blob/main/canola_dataset.zip

![image](https://github.com/user-attachments/assets/ed5f491d-d00c-46c6-9262-ff7831f0d8b0)

![image](https://github.com/user-attachments/assets/78fee248-2840-4fcc-aef7-09a4bcc75826)


