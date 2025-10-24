# --- Standard Python Libraries ---
import os
import random
import cv2
import torch
import numpy as np

# --- Detectron2 Libraries ---
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def main():
    """
    Main function to handle the entire training and evaluation pipeline.
    """

    # --- 1. SETUP LOGGING AND PATHS ---
    setup_logger()
    print("--- Starting Local Training and Evaluation ---")

    base_data_path = os.path.join("data", "raw", "archive")
    output_dir = "./output_local"

    # --- 2. REGISTER DATASETS ---
    print(f"Looking for dataset in: {base_data_path}")
    try:
        for d in ["train", "valid", "test"]:
            register_coco_instances(
                f"parking_{d}", {},
                os.path.join(base_data_path, d, "_annotations.coco.json"),
                os.path.join(base_data_path, d)
            )
        print("✅ Datasets registered successfully!")
    except AssertionError as e:
        print(f"Warning: Datasets might already be registered. {e}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not register datasets. Please check your `base_data_path`.")
        print(f"   Error details: {e}")
        return  # Exit if data isn't found

    # --- 3. CONFIGURE THE MODEL (High-Accuracy Config) ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("parking_train",)
    cfg.DATASETS.TEST = ("parking_valid",)
    cfg.DATALOADER.NUM_WORKERS = 4  # You can try increasing this if you have a good CPU
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    cfg.SOLVER.IMS_PER_BATCH = 2  # Keep this at 2 for MX250 GPU
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 40000

    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = (30000, 36000)
    cfg.SOLVER.GAMMA = 0.1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save a checkpoint every 1000 iterations
    cfg.TEST.EVAL_PERIOD = 0  # Turn off evaluation during training

    # Set the output directory for checkpoints and logs
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n✅ High-accuracy configuration loaded.")
    print(f"   - MAX_ITER set to {cfg.SOLVER.MAX_ITER}.")
    print(f"   - Model checkpoints will be saved to '{cfg.OUTPUT_DIR}'.")

    # --- 4. TRAIN THE MODEL (With Resume Logic) ---
    trainer = DefaultTrainer(cfg)

    # Smart resume: check if a checkpoint exists and resume if it does.
    # This is crucial for long local training sessions.
    trainer.resume_or_load(resume=True)
    print("\n--- Starting Training ---")
    trainer.train()
    print("\n--- Training Complete! ---")

    print(f"Explicitly saving configuration to {cfg.OUTPUT_DIR}/config.yaml")
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump()) # Dump the config to the file
    print("Configuration saved.")

    # --- 5. EVALUATE THE FINAL MODEL ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluator = COCOEvaluator("parking_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "parking_test")
    results = inference_on_dataset(trainer.model, test_loader, evaluator)
    print("\n--- Evaluation Results ---")
    print(results)

    # --- 6. VISUALIZE A RANDOM PREDICTION ---
    print("\n--- Visualizing a Random Prediction ---")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    test_dataset_dicts = DatasetCatalog.get("parking_test")
    random_image_dict = random.choice(test_dataset_dicts)
    im = cv2.imread(random_image_dict["file_name"])

    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        MetadataCatalog.get("parking_test"),
        scale=1.2
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Show the image in a new window on desktop
    print("Displaying prediction. Press any key in the image window to exit.")
    cv2.imshow("Faster R-CNN Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
