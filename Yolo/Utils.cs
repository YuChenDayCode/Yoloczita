using System;
using System.Drawing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yoloczita
{
    public class Utils
    {
        public static float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public static Bitmap ResizeImage(Image image, int targetWidth, int targetHeight)
        {
            // var equalPropSize = CalculateEqualProp(image.Size, target_width, target_height);
            var equalPropSize = new Size(targetWidth, targetHeight);

            var resizedImage = new Bitmap(equalPropSize.Width, equalPropSize.Height);
            using (var graphics = Graphics.FromImage(resizedImage))
            {
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(image, 0, 0, equalPropSize.Width, equalPropSize.Height);
            }

            return resizedImage;
        }

        private static Size CalculateEqualProp(Size source, int width, int height)
        {
            int targetWidth = width;
            int targetHeight = height;

            float percentHeight = Math.Abs(height / (float)source.Height);
            float percentWidth = Math.Abs(width / (float)source.Width);

            float ratio = height / (float)width;
            float sourceRatio = source.Height / (float)source.Width;

            if (sourceRatio < ratio)
                targetHeight = (int)Math.Round(source.Height * percentWidth);
            else
                targetWidth = (int)Math.Round(source.Width * percentHeight);

            return new Size(Math.Max(1, targetWidth), Math.Max(1, targetHeight));
        }

        public static Tensor<float> ExtractPixels(Bitmap image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    tensor[0, 0, y, x] = pixel.R / 255.0F; // r
                    tensor[0, 1, y, x] = pixel.G / 255.0F; // g
                    tensor[0, 2, y, x] = pixel.B / 255.0F; // b
                }
            }

            return tensor;
        }

        public static float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        public static float Sigmoid(float value)
        {
            return 1 / (1 + (float)Math.Exp(-value));
        }
    }
}