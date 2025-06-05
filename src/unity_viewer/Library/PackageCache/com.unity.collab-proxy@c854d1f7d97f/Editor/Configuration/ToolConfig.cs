using System.IO;

using Codice.Utils;

namespace Unity.PlasticSCM.Editor.Configuration
{
    internal static class ToolConfig
    {
        internal static string GetUnityPlasticLogConfigFile()
        {
            if (!string.IsNullOrEmpty(mLogConfigFolder))
                return Path.Combine(mLogConfigFolder, LOG_CONFIG_FILE);

            return GetConfigFilePath(LOG_CONFIG_FILE);
        }

        internal static void InitializeLogConfigFolderForTesting(string logConfigFolder)
        {
            mLogConfigFolder = logConfigFolder;
        }

        internal static void Reset()
        {
            mLogConfigFolder = null;
        }

        static string GetConfigFilePath(string configfile)
        {
            string file = Path.Combine(
                ApplicationLocation.GetAppPath(), configfile);

            if (File.Exists(file))
                return file;

            return UserConfigFolder.GetConfigFile(configfile);
        }

        static string mLogConfigFolder;

        const string LOG_CONFIG_FILE = "unityplastic.log.conf";
    }
}
