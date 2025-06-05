using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Aseprite
{
    internal class ImportSettings
    {
        const bool k_DefaultBackgroundImport = false;
        const string k_BackgroundImportKey = UserSettings.settingsUniqueKey + "ImportSettings.backgroundImport";
        const string k_SettingsTitle = "Aseprite Importer";
        static readonly GUIContent k_BackgroundImportLabel = EditorGUIUtility.TrTextContent("Background import", "Enable asset import when the Unity Editor is in the background.");

        public static bool backgroundImport
        {
            get => EditorPrefs.GetBool(k_BackgroundImportKey, k_DefaultBackgroundImport);
            private set => EditorPrefs.SetBool(k_BackgroundImportKey, value);
        }

        public static void SetupUI(VisualElement rootElement)
        {
            var container = new VisualElement();
            container.style.paddingLeft = 5;
            rootElement.Add(container);
            
            var header = new Label(k_SettingsTitle);
            header.style.paddingLeft = 5;
            header.style.paddingBottom = 10;
            header.style.fontSize = 20;
            header.style.unityFontStyleAndWeight = new StyleEnum<FontStyle>(FontStyle.Bold);
            container.Add(header);
            
            var toggle = new Toggle(k_BackgroundImportLabel.text)
            {
                tooltip = k_BackgroundImportLabel.tooltip,
                value = backgroundImport
            };
            toggle.RegisterValueChangedCallback(x =>
            {
                backgroundImport = x.newValue;
            });
            container.Add(toggle);
        }
    }

    internal class UserSettings : SettingsProvider
    {
        public const string settingsUniqueKey = "UnityEditor.U2D.Aseprite/";

        UserSettings() : base("Preferences/2D/Aseprite Importer", SettingsScope.User)
        {
            guiHandler = OnGUI;
        }

        [SettingsProvider]
        static SettingsProvider CreateSettingsProvider() { return new UserSettings(); }

        public override void OnActivate(string searchContext, VisualElement rootElement)
        {
            ImportSettings.SetupUI(rootElement);
        }
    }
}
