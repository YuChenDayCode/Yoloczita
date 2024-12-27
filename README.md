# Yoloczita
 > .NET Framework 4.6.1下使用Yolov5、v11的onnx模型进行目标检测。
 > Using the onnx model of YOLOV5 or V11 for target detection under .NET Framework 4.6.1.

 ```csharp
 var yolopredictor = new YoloPredictor("./model/xx.onnx")
 var results = yolopredictor.Predict(imgpath, 0.75f);
 ```