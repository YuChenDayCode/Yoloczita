using Microsoft.ML.OnnxRuntime;

namespace Yoloczita
{
    public class YoloSession
    {
        public YoloSession(InferenceSession session, YoloMetadata metadata, SessionIoShapeInfo shapeInfo)
        {
            Metadata = metadata;
            Session = session;
            ShapeInfo = shapeInfo;
        }

        public YoloMetadata Metadata { get; }

        public InferenceSession Session { get; }

        public SessionIoShapeInfo ShapeInfo { get; }
    }
}