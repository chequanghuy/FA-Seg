# FA-Seg: A Fast and Accurate Diffusion-Based Method for Open-Vocabulary Segmentation

By [Huy Che](https://scholar.google.com/citations?user=k7lUdFAAAAAJ&hl), [Vinh-Tiep Nguyen](https://scholar.google.com/citations?user=DulHk_YAAAAJ&hl).

## New Paper🎉
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

Step 2: Run ptp_stable_voc10.py to generate segmentation results.

```
python evaluation/eval_{voc/coco/context}.py
```
