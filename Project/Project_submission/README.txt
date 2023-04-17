In order to replicate our results, follow the following instructions.

1) Train the model for 100 epochs with a batch size of 128
2) The model will checkpoint itself every epoch in a new folder called "ckpt". It will also output a .txt and a .csv with some training metrics.
3) Run the resultparser.ipynb notebook to identify the epoch that had the best test accuracy. For us, this was epoch 98.
4) Modify "torch_to_onnx.py" to select the chosen epoch. 
5) Run "torch_to_onnx.py" to generate an initial onnx. It will put it in a newly created "onnxs" folder
6) Run "quantize_onnx.py" on the previously generated onnx. This will generate two new onnxs. The one we care about is the one ending int "_int8.onnx"
7) Deploy this onnx on the raspberry pi using the included deploy_onnx.py file