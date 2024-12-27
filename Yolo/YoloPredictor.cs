using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yoloczita
{
    /// <summary>
    /// 检测对象
    /// </summary>
    public class YoloPredictor : IDisposable
    {
        public YoloMetadata Metadata { get; }
        public YoloSession YoloSession { get; }
        private readonly InferenceSession _session;
        private bool _disposed;


        public YoloPredictor(string path) : this(File.ReadAllBytes(path))
        {
        }

        public YoloPredictor(byte[] model)
        {
            _session = new InferenceSession(model);
            Metadata = YoloMetadata.Parse(_session);

            var shpeInfo = new SessionIoShapeInfo(_session, Metadata);
            YoloSession = new YoloSession(_session, Metadata, shpeInfo);
        }

        /// <summary>
        /// 图片检测
        /// </summary>
        /// <param name="imgPath">图片路径</param>
        /// <param name="confidence">置信度</param>
        /// <returns>检测信息</returns>
        public List<DetectionResult> Predict(string imgPath, float confidence = 0.35f)
        {
            try
            {
                Size originImageSize; //图片原始大小
                var inputs = PrePprocess(imgPath, out originImageSize);
                using (var outputresults = _session.Run(inputs))
                {
                    var output = outputresults.First().AsTensor<float>();
                    switch (Metadata.Architecture)
                    {
                        case YoloArchitecture.YoloV5:
                            return PredictV5(output, originImageSize, confidence);
                        case YoloArchitecture.YoloV8Or11:
                            return PredictV11(output, originImageSize, confidence);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("检测发生异常：" + ex.Message, ex.InnerException);
            }

            return new List<DetectionResult>();
        }

        /// <summary>
        /// v5
        /// </summary>
        /// <param name="output"></param>
        /// <param name="originImageSize">图像原始大小</param>
        /// <returns></returns>
        private List<DetectionResult> PredictV5(Tensor<float> output, Size originImageSize, float confidence)
        {
            List<DetectionResult> results = new List<DetectionResult>();

            var batch_size = output.Dimensions[0]; // 1
            var ModelOutputCount = output.Dimensions[1];
            var ModelOutputDimensions = output.Dimensions[2]; // 

            // 首先处理所有输出的置信度乘积
            Parallel.For(0, ModelOutputCount, i =>
            {
                if (output[0, i, 4] <= confidence) return;

                // 计算置信度乘积
                for (int j = 5; j < ModelOutputDimensions; j++)
                {
                    output[0, i, j] *= output[0, i, 4];
                }
            });

            Parallel.For(0, ModelOutputCount * (ModelOutputDimensions - 5), index =>
            {
                int i = index / (ModelOutputDimensions - 5); // 获取预测组索引
                int k = index % (ModelOutputDimensions - 5) + 5; // 获取类别索引

                if (output[0, i, 4] < confidence) return; // 检查基础置信度

                // 计算边界框坐标
                float xMin = output[0, i, 0] - output[0, i, 2] / 2;
                float yMin = output[0, i, 1] - output[0, i, 3] / 2;
                float xMax = output[0, i, 0] + output[0, i, 2] / 2;
                float yMax = output[0, i, 1] + output[0, i, 3] / 2;

                results.Add(new DetectionResult
                {
                    Index = index,
                    BoundBox = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin),
                    ClassIndex = k - 5,
                    Confidence = output[0, i, k],
                });
            });
            results = Apply(results, 0.3f);
            for (int i = 0; i < results.Count; i++)
            {
                var result = results[i];
                result.BoundBox = ImageAdjust(result.BoundBox, originImageSize);
            }

            return results;
        }

        private List<DetectionResult> PredictV11(Tensor<float> output, Size originImageSize, float confidence)
        {
            List<DetectionResult> results = new List<DetectionResult>();

            var boxStride = output.Strides[1]; //步长
            var boxSize = output.Dimensions[2]; //边框数量

            var namesCount = Metadata.Names.Length;
            var outputTensor = output.AsEnumerable<float>().ToArray();
            for (var boxIndex = 0; boxIndex < boxSize; boxIndex++)
            {
                for (var nameIndex = 0; nameIndex < namesCount; nameIndex++)
                {
                    var _confidence = outputTensor[(nameIndex + 4) * boxStride + boxIndex];
                    if (_confidence < confidence) continue;

                    var x = outputTensor[0 + boxIndex];
                    var y = outputTensor[1 * boxStride + boxIndex];
                    var w = outputTensor[2 * boxStride + boxIndex];
                    var h = outputTensor[3 * boxStride + boxIndex];

                    var bounds = new RectangleF(x - w / 2, y - h / 2, w, h);
                    results.Add(new DetectionResult
                    {
                        Index = boxIndex,
                        BoundBox = bounds,
                        ClassIndex = nameIndex,
                        Confidence = _confidence,
                    });
                }
            }

            results = Apply(results, 0.3f);

            for (int i = 0; i < results.Count; i++)
            {
                var result = results[i];
                result.BoundBox = ImageAdjust(result.BoundBox, originImageSize);
            }

            return results;
        }


        #region 图片前处理

        private List<NamedOnnxValue> PrePprocess(string imgPath, out Size imageSize)
        {
            using (var image = new Bitmap(imgPath))
            {
                imageSize = image.Size;
                using (var resized =
                       image.Width != Metadata.ImageSize.Width || image.Height != Metadata.ImageSize.Height
                           ? Utils.ResizeImage(image, Metadata.ImageSize.Width, Metadata.ImageSize.Height)
                           : new Bitmap(image))
                {
                    return new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(),
                            Utils.ExtractPixels(resized))
                    };
                }
            }
        }

        #endregion

        #region feizuidayizhichuli

        private List<DetectionResult> Apply(List<DetectionResult> boxes, float iouThreshold)
        {
            if (boxes.Count == 0)
                return new List<DetectionResult>();

            boxes.Sort((x, y) => y.Confidence.CompareTo(x.Confidence));

            var result = new List<DetectionResult>(8)
            {
                boxes[0]
            };

            for (var i = 1; i < boxes.Count; i++)
            {
                var box1 = boxes[i];
                var addToResult = true;

                for (var j = 0; j < result.Count; j++)
                {
                    var box2 = result[j];

                    if (box1.ClassIndex != box2.ClassIndex)
                        continue;

                    if (CalculateIoU(box1, box2) > iouThreshold)
                    {
                        addToResult = false;
                        break;
                    }
                }

                if (addToResult)
                    result.Add(box1);
            }

            return result.ToList();
        }

        protected virtual float CalculateIoU(DetectionResult box1, DetectionResult box2)
        {
            var rect1 = box1.BoundBox;
            var rect2 = box2.BoundBox;

            var area1 = rect1.Width * rect1.Height;

            if (area1 <= 0f)
                return 0f;

            var area2 = rect2.Width * rect2.Height;

            if (area2 <= 0f)
                return 0f;

            var intersection = RectangleF.Intersect(rect1, rect2);
            var intersectionArea = intersection.Width * intersection.Height;

            return (float)intersectionArea / (area1 + area2 - intersectionArea);
        }

        #endregion


        #region 图片后置处理

        public Rectangle
            ImageAdjust(RectangleF rectangle, Size size) //((int X, int Y) padding, (float X, float Y) ratio) adjustment
        {
            var padding = CalculatePadding(size);
            var ratio = CalculateRatio(size);

            var x = (rectangle.X - padding.X) * ratio.X;
            var y = (rectangle.Y - padding.Y) * ratio.Y;
            var w = (rectangle.Width) * ratio.X;
            var h = (rectangle.Height) * ratio.Y;

            return new Rectangle((int)x, (int)y, (int)w, (int)h);
        }

        private (int X, int Y) CalculatePadding(Size size)
        {
            var model = Metadata.ImageSize;

            var xPadding = 0;
            var yPadding = 0;

            (float X, float Y) reductionRatio = (model.Width / (float)size.Width, model.Height / (float)size.Height);

            xPadding = (int)((model.Width - size.Width * reductionRatio.X) / 2);
            yPadding = (int)((model.Height - size.Height * reductionRatio.Y) / 2);

            return (xPadding, yPadding);
        }

        private (float X, float Y) CalculateRatio(Size size)
        {
            var model = Metadata.ImageSize;

            var xRatio = (float)size.Width / model.Width;
            var yRatio = (float)size.Height / model.Height;

            // var ratio = Math.Max(xRatio, yRatio);
            //
            // xRatio = ratio;
            // yRatio = ratio;

            return (xRatio, yRatio);
        }

        #endregion

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _session.Dispose();
            _disposed = true;

            GC.SuppressFinalize(this);
        }
    }


    public class DetectionResult
    {
        public int Index { get; set; }
        public RectangleF BoundBox { get; set; }
        public float Confidence { get; set; }
        public int ClassIndex { get; set; }
    }
}