using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace UnityEditor.U2D.Animation.Upgrading
{
    internal enum UpgradeMode
    {
        SpriteLibrary,
        AnimationClip
    }

    internal enum UpgradeResult
    {
        None = 0,
        Successful = 1,
        Warning = 2,
        Error = 3
    }

    internal struct ObjectIndexPair
    {
        public Object Target;
        public int Index;
    }

    internal struct UpgradeEntry
    {
        public Object Target;
        public int Index;
        public UpgradeResult Result;
        public string Message;
    }

    internal struct UpgradeReport
    {
        public List<UpgradeEntry> UpgradeEntries;
        public string Log;
    }

    internal class Logger
    {
        StringBuilder m_Log = new StringBuilder();

        public void Add(string entry) => m_Log.AppendLine(entry);
        public void AddLineBreak() => m_Log.AppendLine("");
        public void Clear() => m_Log.Clear();
        public string GetLog() => m_Log.ToString();
    }

    internal static class UpgradeUtilities
    {
        const string k_PsbImporterSignature = "UnityEditor.U2D.PSD.PSDImporter";

        public static bool IsPsbImportedFile(string path)
        {
            return AssetImporter.GetAtPath(path).GetType().ToString() == k_PsbImporterSignature;
        }
    }
}
