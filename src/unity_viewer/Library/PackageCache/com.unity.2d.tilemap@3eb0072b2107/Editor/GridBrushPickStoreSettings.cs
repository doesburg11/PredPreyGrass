using System;
using UnityEditorInternal;
using UnityEngine;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    internal class GridBrushPickStoreSettings : ScriptableObject
    {
        [SerializeField]
        public GridBrushPickStore m_UserPickStore;
    }

    [Serializable]
    internal class GridBrushPickStoreSettingsProvider : ScriptableSingleton<GridBrushPickStoreSettingsProvider>
    {
        private const string kSettingsAssetPath = "ProjectSettings/TilePaletteBrushPicks.asset";

        internal class SettingsProperties
        {
            public static GUIStyle header = null;

            public static readonly GUIContent brushPicksLabel = EditorGUIUtility.TrTextContent("Brush Picks");

            public static readonly GUIContent brushPicksAssetLabel = EditorGUIUtility.TrTextContent("Brush Picks Asset", "Asset for storing Brush Picks for the Project.");
            public static readonly GUIContent createNewLabel = EditorGUIUtility.TrTextContent("New", "Creates a new Brush Picks Asset in the Project.");
            public static readonly GUIContent createCloneLabel = EditorGUIUtility.TrTextContent("Clone", "Creates a new Brush Picks Asset in the Project from the current Brush Picks Asset (From default if there is none set).");
        }

        [SerializeField]
        private GridBrushPickStore m_PickStorePreference;
        [SerializeField]
        private GridBrushPickStoreSettings m_PickStoreSettings;

        private void OnEnable()
        {
            LoadGridBrushPickStoreSettings();
        }

        private void LoadGridBrushPickStoreSettings()
        {
            var objs = InternalEditorUtility.LoadSerializedFileAndForget(kSettingsAssetPath);
            if (objs != null && objs.Length > 0)
            {
                m_PickStoreSettings = objs[0] as GridBrushPickStoreSettings;
                m_PickStorePreference = m_PickStoreSettings != null ? m_PickStoreSettings.m_UserPickStore : null;
            }
        }

        internal static GridBrushPickStore GetUserBrushPickStore()
        {
            var settings = GetGridBrushPickStoreSettings();
            if (settings != null)
                return settings.m_UserPickStore;
            return null;
        }

        internal static GridBrushPickStoreSettings GetGridBrushPickStoreSettings()
        {
            if (instance.m_PickStoreSettings == null && instance.m_PickStoreSettings is not null)
            {
                instance.LoadGridBrushPickStoreSettings();
            }
            return instance.m_PickStoreSettings;
        }

        private static GridBrushPickStoreSettings LoadOrCreateGridBrushPickStoreSettings()
        {
            var settings = GetGridBrushPickStoreSettings();
            if (settings == null)
            {
                instance.m_PickStoreSettings = ScriptableObject.CreateInstance<GridBrushPickStoreSettings>();
                InternalEditorUtility.SaveToSerializedFileAndForget(new [] { instance.m_PickStoreSettings }, kSettingsAssetPath, EditorSettings.serializationMode != SerializationMode.ForceBinary);
                settings = instance.m_PickStoreSettings;
            }
            return settings;
        }

        internal static void PreferencesGUI()
        {
            using (new SettingsWindow.GUIScope())
            {
                if (SettingsProperties.header == null)
                    SettingsProperties.header = "SettingsHeader";

                GUILayout.Label(SettingsProperties.brushPicksLabel, SettingsProperties.header, GUILayout.MinWidth(160));
                EditorGUILayout.BeginHorizontal();
                var pickStorePreference = (GridBrushPickStore) EditorGUILayout.ObjectField(SettingsProperties.brushPicksAssetLabel, instance.m_PickStorePreference, typeof(GridBrushPickStore), false);
                if (pickStorePreference != instance.m_PickStorePreference)
                {
                    instance.m_PickStorePreference = pickStorePreference;
                    var settings = LoadOrCreateGridBrushPickStoreSettings();
                    settings.m_UserPickStore = instance.m_PickStorePreference;
                    Apply();
                }
                if (GUILayout.Button(SettingsProperties.createNewLabel, GUILayout.Width(55)))
                {
                    CreateNew();
                }
                if (GUILayout.Button(SettingsProperties.createCloneLabel, GUILayout.Width(55)))
                {
                    CreateClone();
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.Space();
            }
        }

        private static void Apply()
        {
            if (instance.m_PickStoreSettings == null)
                return;

            InternalEditorUtility.SaveToSerializedFileAndForget(new [] { instance.m_PickStoreSettings }, kSettingsAssetPath, EditorSettings.serializationMode != SerializationMode.ForceBinary);
            GridPaletteBrushes.FlushCache();
            GridPaintingState.brushPickStore = GetUserBrushPickStore();
        }

        private static void CreateNew()
        {
            var brushStore = ScriptableObject.CreateInstance<GridBrushPickStore>();
            var defaultPath = ProjectBrowser.s_LastInteractedProjectBrowser ? ProjectBrowser.s_LastInteractedProjectBrowser.GetActiveFolderPath() : "Assets";
            var filePath = EditorUtility.SaveFilePanel("Create Brush Picks Asset into folder ", defaultPath, "BrushPick", "asset");
            if (string.IsNullOrEmpty(filePath))
                return;

            filePath = FileUtil.GetProjectRelativePath(filePath);
            var fileName = FileUtil.UnityGetFileNameWithoutExtension(filePath);
            brushStore.name = fileName;
            AssetDatabase.CreateAsset(brushStore, filePath);

            var settings = LoadOrCreateGridBrushPickStoreSettings();
            settings.m_UserPickStore = brushStore;
            Apply();
            instance.m_PickStorePreference = brushStore;
            Selection.activeObject = brushStore;
        }

        private static void CreateClone()
        {
            var defaultPath = ProjectBrowser.s_LastInteractedProjectBrowser
                ? ProjectBrowser.s_LastInteractedProjectBrowser.GetActiveFolderPath()
                : "Assets";
            var filePath = EditorUtility.SaveFilePanel("Create Brush Picks Asset into folder ", defaultPath,
                "BrushPick", "asset");
            if (string.IsNullOrEmpty(filePath))
                return;

            GridBrushPickStore newBrushStore;
            var currentBrushStore = GetUserBrushPickStore();
            if (currentBrushStore == null)
            {
                newBrushStore = ScriptableObject.CreateInstance<GridBrushPickStore>();
            }
            else
            {
                newBrushStore = Object.Instantiate(currentBrushStore);
            }

            filePath = FileUtil.GetProjectRelativePath(filePath);
            var fileName = FileUtil.UnityGetFileNameWithoutExtension(filePath);
            newBrushStore.name = fileName;
            AssetDatabase.CreateAsset(newBrushStore, filePath);

            // Clone Library Brushes if cloning from default instance
            if (currentBrushStore == null)
            {
                currentBrushStore = GridBrushPickStore.LoadOrCreateLibraryGridBrushPickAsset();
                foreach (var userBrush in currentBrushStore.userSavedBrushes)
                {
                    newBrushStore.AddNewUserSavedBrush(userBrush);
                }
            }

            var settings = LoadOrCreateGridBrushPickStoreSettings();
            settings.m_UserPickStore = newBrushStore;
            Apply();
            instance.m_PickStorePreference = newBrushStore;
            Selection.activeObject = newBrushStore;
        }
    }
}
