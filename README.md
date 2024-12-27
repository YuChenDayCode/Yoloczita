# Yoloczita
 > .NET Framework 4.6.1��ʹ��Yolov5��v11��onnxģ�ͽ���Ŀ���⡣
 > Using the onnx model of YOLOV5 or V11 for target detection under .NET Framework 4.6.1.

 ```csharp
 var yolopredictor = new YoloPredictor("./model/xx.onnx")
 var results = yolopredictor.Predict(imgpath, 0.75f);
 ```