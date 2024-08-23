from autodistill_rt_detr import RT_DETR

# Now you can use RT_DETR as intended
target_model = RT_DETR()


# train for 10 epochs
target_model.train("./forest-fire-detection.v1i.coco", epochs=3)

# run inference on an image
results = target_model.predict("./forest-fire-detection.v1i.coco/valid/f14_jpg.rf.6b71c87175a5d38415da4711d372d200.jpg")

print ('results',results)

