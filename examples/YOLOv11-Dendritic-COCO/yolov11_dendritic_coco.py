from ultralytics import YOLO

from perforatedai import globals_perforatedai as GPA

GPA.pc.append_module_names_to_convert(["Conv", "DWConv"])
GPA.pc.append_module_ids_to_track([".model.23.dfl"])
GPA.pc.set_unwrapped_modules_confirmed(True)
GPA.pc.set_testing_dendrite_capacity(False)
GPA.pc.set_weight_decay_accepted(True)
GPA.pc.set_n_epochs_to_switch(25)
# If using perforated backpropagation set the additional dendritic parameters
if GPA.pc.get_perforated_backpropagation():
    GPA.pc.set_initial_correlation_batches(7)
    GPA.pc.set_debugging_backwards_nan(True)
    GPA.pc.set_candidate_grad_clipping(1.0)

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model - using coco128 for more images (128)
# Set nbs=batch to disable gradient accumulation (run backward every step)
# Epochs ignored for dendritic versions since training controlled by PAI
results = model.train(data="coco128.yaml", epochs=100, imgsz=640, batch=16, nbs=16)
