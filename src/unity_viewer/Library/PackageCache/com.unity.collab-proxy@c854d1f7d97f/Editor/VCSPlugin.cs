using UnityEditor;

namespace Unity.PlasticSCM.Editor
{
    internal static class VCSPlugin
    {
        internal static bool IsEnabled()
        {
            return GetVersionControl() == "PlasticSCM";
        }

        internal static void Disable()
        {
            SetVersionControl("Visible Meta Files");

            AssetDatabase.SaveAssets();
        }

        internal static bool IsAnyProviderEnabled()
        {
            return !IsVisibleMetaFilesMode() && !IsHiddenMetaFilesMode();
        }

        static string GetVersionControl()
        {
            return VersionControlSettings.mode;
        }

        static void SetVersionControl(string versionControl)
        {
            VersionControlSettings.mode = versionControl;
        }

        static bool IsVisibleMetaFilesMode()
        {
            return GetVersionControl() == "Visible Meta Files";
        }

        static bool IsHiddenMetaFilesMode()
        {
            return GetVersionControl() == "Hidden Meta Files";
        }
    }
}
