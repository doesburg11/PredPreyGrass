using System;
using System.IO;

namespace MyExercise_6
{
    public class FileCreator
    {
        public const string k_Directory = "OutputFiles";

        public void CreateEmptyFile(string fileName)
        {
            CreateFile(fileName, String.Empty);
        }
        
        public void CreateFile(string fileName, string content)
        {
            using (var stream = File.CreateText(Path.Combine(k_Directory, fileName)))
            {
                stream.Write(content);
            }
        }
    }
}