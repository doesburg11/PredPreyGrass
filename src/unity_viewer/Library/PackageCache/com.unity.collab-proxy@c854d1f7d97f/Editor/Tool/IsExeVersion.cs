using System;
using System.Diagnostics;

namespace Unity.PlasticSCM.Editor.Tool
{
    internal static class IsExeVersion
    {
        internal static bool GreaterOrEqual(string exePath, string minVersionString)
        {
            try
            {
                FileVersionInfo fileVersionInfo = FileVersionInfo.GetVersionInfo(exePath);

                Version version = new Version(fileVersionInfo.FileVersion);
                Version minVersion = new Version(minVersionString);

                return (version >= minVersion);
            }
            catch (Exception)
            {
                return false;
            }
        }
    }
}
