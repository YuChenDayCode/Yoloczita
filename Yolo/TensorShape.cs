using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;

namespace Yoloczita
{
    public class SessionIoShapeInfo
    {
        public TensorShape Input0 { get; }

        public TensorShape Output0 { get; }

        public TensorShape? Output1 { get; }

        public SessionIoShapeInfo(InferenceSession session, YoloMetadata metadata)
        {
            var inputMetadata = session.InputMetadata.Values;
            var outputMetadata = session.OutputMetadata.Values;
            
            Input0 = new TensorShape(inputMetadata.First().Dimensions);
            Output0 = new TensorShape(outputMetadata.First().Dimensions);
            
            if (session.OutputMetadata.Count == 2)
            {
                Output1 = new TensorShape(outputMetadata.Last().Dimensions);
            }
        }
        
    }

    public readonly struct TensorShape
    {
        public int Length { get; }
        //public bool IsDynamic { get; }
        public int[] Dimensions { get; }
        //public long[] Dimensions64 { get; }

        public TensorShape(int[] shape)
        {
            if (shape.Any(x => x < 0))
            {
                //IsDynamic = true;
                Length = -1;
            }
            else
            {
                Length = GetSizeForShape(shape);
            }

            Dimensions = shape;
            //Dimensions64 = [.. shape.Select(x => (long)x)];
        }

        private static int GetSizeForShape(ReadOnlySpan<int> shape)
        {
            var product = 1;
            for (var i = 0; i < shape.Length; i++)
            {
                var dimension = shape[i];

                if (dimension < 0)
                {
                    throw new ArgumentOutOfRangeException($"Shape must not have negative elements: {dimension}");
                }

                product = checked(product * dimension);
            }

            return product;
        }
    }
}
