using System;
using System.Reflection;

using UnityEditor;

using Codice.Client.Common.Threading;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Settings
{
    internal static class OpenPlasticProjectSettings
    {
        internal static void ByDefault()
        {
            PlasticProjectSettingsProvider provider = OpenInPlasticProjectSettings();

            if (provider == null)
                return;

            provider.OpenAllFoldouts();
        }

        internal static void InDiffAndMergeFoldout()
        {
            PlasticProjectSettingsProvider provider = OpenInPlasticProjectSettings();

            if (provider == null)
                return;

            provider.OpenDiffAndMergeFoldout();
        }

        internal static void InShelveAndSwitchFoldout()
        {
            PlasticProjectSettingsProvider provider = OpenInPlasticProjectSettings();

            if (provider == null)
                return;

            provider.OpenShelveAndSwitchFoldout();
        }

        internal static void InOtherFoldout()
        {
            PlasticProjectSettingsProvider provider = OpenInPlasticProjectSettings();

            if (provider == null)
                return;

            provider.OpenOtherFoldout();
        }

        internal static PlasticProjectSettingsProvider OpenInPlasticProjectSettings()
        {
            EditorWindow settingsWindow = OpenProjectSettingsWithPlasticSelected();
            return GetPlasticProvider(settingsWindow);
        }

        internal static EditorWindow OpenProjectSettingsWithPlasticSelected()
        {
            return SettingsService.OpenProjectSettings(
                UnityConstants.PROJECT_SETTINGS_TAB_PATH);
        }

        internal static PlasticProjectSettingsProvider GetPlasticProvider(
            EditorWindow settingsWindow)
        {
            try
            {
                /* The following code must be compiled only for editor versions that allow our code
                 to access internal code from the editor, otherwise the ProjectSettingsWindow is not
                 accessible and the compilation fails.
                 We don't know yet the version number that allows us to access this code, so for the
                 moment the code is commented

#if UNITY_6000_0_OR_NEWER
                ProjectSettingsWindow projectSettingsWindow = settingsWindow as ProjectSettingsWindow;
                return projectSettingsWindow.GetCurrentProvider() as PlasticProjectSettingsProvider;
#else            */

                MethodInfo getCurrentProviderMethod = settingsWindow.GetType().GetMethod(
                    "GetCurrentProvider",
                    BindingFlags.Instance | BindingFlags.NonPublic);

                return getCurrentProviderMethod.Invoke(
                    settingsWindow, null) as PlasticProjectSettingsProvider;
//#endif
            }
            catch (Exception ex)
            {
                ExceptionsHandler.LogException("OpenPlasticProjectSettingsProvider", ex);
                return null;
            }
        }
    }
}
