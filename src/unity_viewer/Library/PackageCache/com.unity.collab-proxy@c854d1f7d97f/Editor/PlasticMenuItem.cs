using UnityEditor;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor
{
    internal static class PlasticMenuItem
    {
#if UNITY_6000_1_OR_NEWER
        [MenuItem(MENU_ITEM_NAME, false, 0)]
#else
        [MenuItem(MENU_ITEM_NAME, false)]
#endif
        static void ShowPanel()
        {
            PlasticPlugin.OpenPlasticWindowDisablingOfflineModeIfNeeded();
        }

        [MenuItem(MENU_ITEM_NAME, true)]
        static bool ValidateMenu()
        {
            return !VCSPlugin.IsAnyProviderEnabled();
        }

        // The Window menu was refactored in Unity 6000.1.0a4 to host both UVCS & External providers (Perforce)
        const string MENU_ITEM_NAME =
#if UNITY_6000_1_OR_NEWER
            "Window/Version Control/" + UnityConstants.PLASTIC_WINDOW_TITLE;
#else
            "Window/" + UnityConstants.PLASTIC_WINDOW_TITLE;
#endif
    }
}
