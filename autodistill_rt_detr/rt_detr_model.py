import os
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import supervision as sv
import torch
import torchvision
from autodistill.detection import DetectionTargetModel
from torch.utils.data import DataLoader
from transformers import (AutoModelForObjectDetection, RTDetrForObjectDetection,
                          RTDetrImageProcessor)

HOME = os.path.expanduser("~")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Update to use RT-DETR Image Processor
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor):
        annotation_file_path = os.path.join(
            image_directory_path, "_annotations.coco.json"
        )
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id": image_id, "annotations": annotations}
        encoding = self.image_processor(
            images=images, annotations=annotations, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

class RTDetr(pl.LightningModule):
    def __init__(
        self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader, id2label
    ):
        super().__init__()
        # Update to use RT-DETR model
        self.model = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="PekingU/rtdetr_r50vd",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.model_train_dataloader = train_dataloader
        self.model_val_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

    def train_dataloader(self):
        return self.model_train_dataloader

    def val_dataloader(self):
        return self.model_val_dataloader

@dataclass
class RT_DETR(DetectionTargetModel):
    model: RTDetrForObjectDetection = None

    def __init__(self, lr=7.5e-5, lr_backbone=1e-5, weight_decay=1e-4):
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        with torch.no_grad():
            # load image and predict
            image = cv2.imread(input)
            inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)

            results = image_processor.post_process_object_detection(
                outputs=outputs, threshold=confidence, target_sizes=target_sizes
            )[0]

            return sv.Detections.from_transformers(transformers_results=results)

    def create_dataloader(self, dataset, batch_size=12, shuffle=False, num_workers=0, prefetch_factor=None):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

    def train(self, dataset, epochs: int = 100):
        train_directory = os.path.join(dataset, "train")
        val_directory = os.path.join(dataset, "valid")

        train_dataset = CocoDetection(
            image_directory_path=train_directory,
            image_processor=image_processor,
        )

        val_dataset = CocoDetection(
            image_directory_path=val_directory,
            image_processor=image_processor,
        )

        labels = train_dataset.coco.cats

        id2label = {k: v["name"] for k, v in labels.items()}

        train_dataloader = self.create_dataloader(
            train_dataset, batch_size=12, shuffle=True
        )
        val_dataloader = self.create_dataloader(
            val_dataset, batch_size=12, shuffle=False,
        )

        model = RTDetr(
            lr=self.lr,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            id2label=id2label,
        )

        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            max_epochs=epochs,
            gradient_clip_val=0.1,
            accumulate_grad_batches=8,
            log_every_n_steps=5,
        )

        trainer.fit(model)

        model.model.save_pretrained("rtdetr_model")

        self.model = RTDetrForObjectDetection.from_pretrained("rtdetr_model").to(DEVICE)

