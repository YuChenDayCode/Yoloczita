using Microsoft.ML.OnnxRuntime;
using System;
using System.Linq;
using System.Drawing;

namespace Yoloczita
{
    public class YoloMetadata
    {
        public string Author { get; }

        public string Description { get; }

        public string Version { get; }

        public int BatchSize { get; }

        public Size ImageSize { get; }

        public YoloName[] Names { get; }

        /// <summary>
        /// 模型类型
        /// </summary>
        public YoloArchitecture Architecture { get; }

        internal YoloMetadata(InferenceSession session)
            :
            this(session, ParseYoloArchitecture(session))
        {
        }

        internal YoloMetadata(InferenceSession session, YoloArchitecture architecture)
        {
            var metadata = session.ModelMetadata.CustomMetadataMap;
            Author = metadata.ContainsKey("author") ? metadata["author"] : "";
            Description = metadata.ContainsKey("description") ? metadata["description"] : "";
            Version = metadata.ContainsKey("version") ? metadata["version"] : "";


            Architecture = architecture;
            BatchSize = metadata.ContainsKey("batch") ? int.Parse(metadata["batch"]) : 1;
            ImageSize = metadata.ContainsKey("imgsz")
                ? ParseSize(metadata["imgsz"])
                : new Size(session.InputMetadata.First().Value.Dimensions[2],
                    session.InputMetadata.First().Value.Dimensions[3]);
            Names = metadata.ContainsKey("names") ? ParseNames(metadata["names"]) : new YoloName[0];
        }

        public static YoloMetadata Parse(InferenceSession session)
        {
            return new YoloMetadata(session);
        }

        private static YoloArchitecture ParseYoloArchitecture(InferenceSession session)
        {
            var metadata = session.ModelMetadata.CustomMetadataMap;

            var output0 = session.OutputMetadata["output0"];

            if (output0.Dimensions[2] == 6) // YOLOv5 output0: [1, 123123, 6]
            {
                return YoloArchitecture.YoloV5;
            }

            return YoloArchitecture.YoloV8Or11;
        }

        #region Parsers

        private static Size ParseSize(string text)
        {
            //text = text.Trim('[', ']'); // '[640, 640]' => '640, 640'

            //var split = text.Split(new string[] { ": " }, StringSplitOptions.None);

            //var y = int.Parse(split[0]);
            //var x = int.Parse(split[1]);

            //return new Size(x, y);
            int[] result = text.Trim('[', ']').Split(',')
                .Select(x => int.Parse(x.Trim())).ToArray();
            var imageSize = new Size(result[0], result[1]);
            return imageSize;
        }

        private static YoloName[] ParseNames(string text)
        {
            text = text.Trim('{', '}');

            var split = text.Split(',');
            var count = split.Length;

            var names = new YoloName[count];

            for (int i = 0; i < count; i++)
            {
                var value = split[i];
                var splitvalue = value.Split(new string[] { ": " }, StringSplitOptions.None);
                var id = int.Parse(splitvalue[0]);
                var name = splitvalue[1].TrimStart('\'').TrimEnd('\'');
                names[i] = new YoloName(id, name);
            }

            return names;
        }

        #endregion
    }

    public class YoloName
    {
        public YoloName(int id, string name)
        {
            Id = id;
            Name = name;
        }

        public int Id { get; }

        public string Name { get; }

        public override string ToString()
        {
            return $"{Id}: '{Name}'";
        }
    }

    public enum YoloArchitecture
    {
        YoloV8Or11,
        YoloV5
    }
}