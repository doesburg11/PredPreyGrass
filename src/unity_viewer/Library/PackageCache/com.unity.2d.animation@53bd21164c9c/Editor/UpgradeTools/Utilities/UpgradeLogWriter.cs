using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace UnityEditor.U2D.Animation.Upgrading
{
    internal static class UpgradeLogWriter
    {
        const string k_LogSavePath = "/Logs/";

        public static string Generate(string content)
        {
            if (string.IsNullOrEmpty(content))
                return string.Empty;

            var path = Path.GetDirectoryName(Application.dataPath);
            path = Path.Join(path, k_LogSavePath);

            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            var dateStamp = DateTime.Now.Ticks.ToString();
            var filePath = Path.Join(path, $"AssetUpgradingLog_{dateStamp}.txt");

            content = AddHeaderToContent(content);
            using (var file = File.CreateText(filePath))
            {
                file.Write(content);
            }

            return filePath;
        }

        static string AddHeaderToContent(string content)
        {
            var sb = new StringBuilder();
            sb.AppendLine(DateTime.Now.ToString());
            sb.AppendLine("Asset Upgrading");
            sb.AppendLine("---------------");
            sb.AppendLine(content);
            return sb.ToString();
        }
    }
}
