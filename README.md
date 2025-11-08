# A Comparative Analysis of Deep Learning Architectures for Real-World Parking Occupancy Detection

This is the official repository for our CPE110-1 research project. This study conducts a rigorous comparative analysis of different deep learning methodologies (patch-based classification vs. full-scene object detection) to determine the most robust and generalizable approach for a real-world parking detection system in Davao City.

The project's key finding demonstrates a significant "domain gap" when using academic-trained object detectors on local data (0.12% mAP), while patch-based classifiers generalized exceptionally well (97.9% accuracy). A final experiment proves that this domain gap can be successfully bridged by fine-tuning a pre-trained detector on a small, local dataset (achieving 80.85% mAP).

**Group Members:**
* Mohammad Jameel Jibreel N. Mamogkat
* Duff S. Bastasa
* Jake Lloyd A. Ganoy

---

## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Systems Engineering Overview](#systems-engineering-overview)
3.  [Folder Structure Explained](#folder-structure-explained)
4.  [Project Workflow](#project-workflow)
5.  [Setup and Dependencies](#setup-and-dependencies)

---

## Project Goal

The primary objective of this research is to move beyond a simple comparison and investigate the real-world viability of common smart parking architectures. We aim to answer the question: "What is the most robust and effective methodology for deploying a parking occupancy system in a new environment like Davao City?"

To do this, we compare two main approaches:
1.  **Patch-Based Classification:** (MobileNetV2, EfficientNetB0) - Models that classify small, pre-cropped images of single parking spaces.
2.  **Full-Scene Object Detection:** (YOLOv8, Mask R-CNN) - Models that scan an entire image to find and classify all occupied spaces.

Our methodology is designed to intentionally test the "domain gap" by training models on public academic datasets and evaluating them on a custom "Davao Real-World Set."

---

## Systems Engineering Overview

The project is architected as a multi-stage experimental pipeline. Each numbered script corresponds to a logical step in our research process.

#### Subsystem 1: Data Preprocessing & Harmonization
* **Function:** Ingests raw academic datasets (PKLot, CNRPark-EXT) and converts them into two formats: (1) A patch-based dataset for the classifiers and (2) A full-frame dataset (COCO format) for the detectors.
* **Responsible Script:** `scripts/1_harmonize_data.py`

#### Subsystem 2: Model Training (on Academic Data)
* **Function:** Trains the different model architectures on the processed academic datasets.
* **Responsible Scripts:**
    * `scripts/2a_train_mobilenet.py`
    * `scripts/2b_train_efficientnet.py`
    * `scripts/2c_train_yolo_academic.py`
    * `scripts/2d_train_maskrcnn.py` (Placeholder for Mask R-CNN)

#### Subsystem 3: Evaluation of Academic-Trained Models (on Davao Set)
* **Function:** Tests the models from Subsystem 2 against our custom Davao Real-World Set to measure the "domain gap".
* **Responsible Scripts:**
    * `scripts/3a_evaluate_classifiers.py`: Evaluates the trained classifiers on the Davao *patches*.
    * `scripts/3b_prep_yolo_test_data.py`: Converts our custom Davao COCO JSON annotations into the YOLO `.txt` format required for evaluation.
    * `scripts/3c_evaluate_yolo_academic.py`: Evaluates the academic-trained YOLO model on the full-frame Davao images.

#### Subsystem 4: Fine-Tuning Experiment (on Davao Data)
* **Function:** Proves our hypothesis that the "domain gap" can be fixed by fine-tuning the detector on a small, local dataset.
* **Responsible Scripts:**
    * `scripts/4a_split_davao_set.py`: Splits our 50 annotated Davao images into `train` (40) and `valid` (10) sets for this new experiment.
    * `scripts/4b_finetune_yolo_on_davao.py`: Fine-tunes the *pre-trained* `yolov8n.pt` (not the one from 2c!) on our 40 Davao training images.
    * `scripts/4c_evaluate_finetuned_yolo.py`: Evaluates the new fine-tuned model on the 10 unseen Davao validation images.

#### Subsystem 5: Final Analysis
* **Function:** (Future Work) Ingests all final accuracy and mAP scores for statistical comparison (e.g., ANOVA test).
* **Responsible Script:** `scripts/5_run_anova.py`

#### Utility Scripts
* **Function:** Simple scripts for running quick, single-image predictions for debugging and visualization.
* **Location:** `scripts/utils/`

---

## Folder Structure Explained

* `📁 data/`: Contains all project data.
    * `📁 raw/`: **(Git Ignored)** Holds the original downloaded academic datasets (`archive`, `CNR-EXT-Patches-150x150`).
    * `📁 processed/`: **(Git Ignored)** Contains all datasets created by our scripts.
        * `classifier/`: The patch-based dataset (train/test) from academic data.
        * `detector/`: The full-frame dataset (train/valid/test) from academic data.
        * `davao_finetune_set/`: The new, small dataset for the fine-tuning experiment (train/valid).
    * `📁 real_world_test_set/`: Contains our custom 50+ Davao images and their annotations.
        * `images/`: The 50+ raw `.jpg` images.
        * `annotations/`: The original COCO `.json` files from the annotation tool.
        * `labels/`: The YOLO-compatible `.txt` files generated by `3b_prep_yolo_test_data.py`.
        * `davao_classifier_test_set/`: The manually cropped patches for testing the classifiers (`occupied/`, `vacant/`).

* `📁 models/`: **(Git Ignored)** Final trained model files (`.pth`) for the classifiers are saved here.

* `📁 results/`: Contains all visual outputs from our evaluations.
    * `📁 charts/`: Confusion matrix `.png` files saved from the classifier evaluation.

* `📁 runs/`: **(Git Ignored)** Auto-generated by YOLO. This is where all YOLO training and evaluation results are saved (e.g., in `runs/detect/train/`, `runs/detect/val3/`, `runs/detect/finetune_davao/`).

* `📁 scripts/`: Contains all executable Python code, organized by subsystem.
    * `📁 utils/`: Contains helper scripts for quick predictions.

* `📜 parking_data.yaml`: Config file for training YOLO on the *academic* dataset.
* `📜 yolo_davao_test_config.yaml`: Config file for *evaluating* the academic YOLO on the Davao set.
* `📜 yolo_davao_finetune_config.yaml`: Config file for *fine-Tuning* YOLO on the Davao set.

---

## Project Workflow

This project is run as a series of experiments.

1.  **Setup the Environment:** Run `pip install -r requirements.txt` (ideally in a virtual environment).
2.  **Download Academic Data:** Place raw datasets into `data/raw/`.
3.  **Prepare Academic Data:** Run `python scripts/1_harmonize_data.py` to create the `data/processed/` folders.
4.  **Train Academic Models:**
    * Run `python scripts/2a_train_mobilenet.py` (best on Colab).
    * Run `python scripts/2b_train_efficientnet.py` (best on Colab).
    * Run `python scripts/2c_train_yolo_academic.py` (locally or on Colab).
5.  **Manually Annotate & Prep Davao Set:**
    * Create the 50 images and annotate them (outputting JSON files into `data/real_world_test_set/annotations/`).
    * Manually create the classifier patches in `data/real_world_test_set/davao_classifier_test_set/`.
    * Run `python scripts/3b_prep_yolo_test_data.py` to create the YOLO `.txt` labels.
6.  **Run Evaluation (The First Finding):**
    * Run `python scripts/3a_evaluate_classifiers.py` to get the high-accuracy (95-98%) results.
    * Run `python scripts/3c_evaluate_yolo_academic.py` to get the failed (0.12%) results.
7.  **Run Fine-Tuning Experiment (The Second Finding):**
    * Run `python scripts/4a_split_davao_set.py` to create the new 40/10 split.
    * Run `python scripts/4b_finetune_yolo_on_davao.py` to train the new model.
    * Run `python scripts/4c_evaluate_finetuned_yolo.py` to get the new, successful (80.85%) results.
8.  **Analyze Results:** Compare the results from `3a`, `3c`, and `4c` to form your final conclusion.

---

## Setup and Dependencies

To set up the project environment on a new machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/JibbsMamogkat/cv-models-comparison.git](https://github.com/JibbsMamogkat/cv-models-comparison.git)
    cd cv-models-comparison
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows (Git Bash):
    source venv/Scripts/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Academic Data:** Manually download the PKLot and CNRPark+EXT datasets and place the `.zip` files in the `data/raw/` directory.

The project is now ready for development and execution.