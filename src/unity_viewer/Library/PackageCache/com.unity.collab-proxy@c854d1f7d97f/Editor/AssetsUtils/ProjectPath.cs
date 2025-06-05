using System.IO;

namespace Unity.PlasticSCM.Editor.AssetUtils
{
    internal static class ProjectPath
    {
        internal static string Get()
        {
            return Path.GetDirectoryName(Path.GetFullPath(
                ApplicationDataPath.Get()));
        }
    }
}
