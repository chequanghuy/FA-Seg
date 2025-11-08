# FA-Seg: A Fast and Accurate Diffusion-Based Method for Open-Vocabulary Segmentation

By [Huy Che](https://scholar.google.com/citations?user=k7lUdFAAAAAJ&hl), [Vinh-Tiep Nguyen](https://scholar.google.com/citations?user=DulHk_YAAAAJ&hl).

## Paper
We are pleased to announce that our paper has been published in the journal Neurocomputing (Elsevier). You can check it out on [Elsevier](https://doi.org/10.1016/j.neucom.2025.131844) or on [arxiv](https://arxiv.org/abs/2506.23323).

## Citing FA-Seg
If you find DiffSegmenter useful in your research, please consider citing:
```bibtex
@article{CHE2026131844,
        title = {FA-Seg: A fast and accurate diffusion-based method for open-vocabulary segmentation},
        journal = {Neurocomputing},
        volume = {660},
        pages = {131844},
        year = {2026},
        issn = {0925-2312},
        doi = {https://doi.org/10.1016/j.neucom.2025.131844},
        url = {https://www.sciencedirect.com/science/article/pii/S0925231225025160},
        author = {Huy Che and Vinh-Tiep Nguyen}
}
```



## Installation


  
* Requirements
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Dataset preparation

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download datasets. Alternatively, you can directly [download the pre-processed data](https://drive.google.com/file/d/1TRo_4cvGp0l0IRb88LBCPg_mN5xUJ7MR/view?usp=sharing) provided by us.

```bash
gdown 1TRo_4cvGp0l0IRb88LBCPg_mN5xUJ7MR
```

```

└── dataset
    ├── coco_stuff164k
    │   ├── images
    │   │   └── val2017
    │   └── annotation_object
    │       └── val2017
    └── VOCdevkit
        ├── VOC2010
        │   ├── JPEGImages
        │   ├── SegmentationClassContextColor
        └── VOC2012
            ├── JPEGImages
            ├── SegmentationClass
```

### Open Vocabulary Semantic Segmentation

#### Evaluation


Step 1: Generate segmentation masks for the dataset.

```
python main_{voc/coco/context}.py
```

Step 2: Run the evaluation script to compute the mIoU score based on the generated masks

```
python evaluation/eval_{voc/coco/context}.py
```

## License
This project is licensed under <a rel="license" href="https://github.com/mc-lan/SmooSeg/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## Acknowledgement
This work would not have been possible without the valuable contributions of the following authors.

* [Dataset Diffusion](https://github.com/VinAIResearch/Dataset-Diffusion)
* [Null-Text Inversion](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)

## Contact
If you have any questions, please feel free to reach out at `huycq@uit.edu.vn`.
