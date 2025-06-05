using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEditorInternal;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    internal class GridPaletteBrushes : ScriptableSingleton<GridPaletteBrushes>
    {
        internal static readonly string s_SessionStateLastUsedBrush = "GridPaletteBrushes.LastUsedBrush";

        internal static readonly string s_LibraryPath = "Library/GridBrush";

        private static readonly string s_GridBrushExtension = ".asset";

        private static bool s_RefreshCache;

        [SerializeField] private List<GridBrushBase> m_Brushes;
        public static List<GridBrushBase> brushes
        {
            get
            {
                if (instance.m_Brushes == null || instance.m_Brushes.Count == 0 || s_RefreshCache)
                {
                    instance.RefreshBrushesCache();
                    s_RefreshCache = false;
                }

                return instance.m_Brushes;
            }
        }

        private string[] m_BrushNames;
        public static string[] brushNames
        {
            get
            {
                return instance.m_BrushNames;
            }
        }

        private string[] m_BrushTooltips;

        public static string[] brushTooltips
        {
            get
            {
                return instance.m_BrushTooltips;
            }
        }

        private Dictionary<Type, Texture2D> m_BrushIcons;

        public static Texture2D GetBrushIcon(Type brushType)
        {
            Texture2D iconTex;
            if (instance.m_BrushIcons != null && instance.m_BrushIcons.TryGetValue(brushType, out iconTex))
            {
                return iconTex;
            }
            return Texture2D.redTexture;
        }

        private void OnDisable()
        {
            FlushCache();
        }

        public GridBrushBase GetLastUsedBrush()
        {
            var sessionIndex = SessionState.GetInt(s_SessionStateLastUsedBrush, -1);
            if (sessionIndex >= 0 && brushes.Count > sessionIndex)
                return brushes[sessionIndex];
            if (sessionIndex == -1)
                StoreLastUsedBrush(brushes[0]);
            return brushes[0];
        }

        public void StoreLastUsedBrush(GridBrushBase brush)
        {
            int index = brushes.IndexOf(brush);
            SessionState.SetInt(s_SessionStateLastUsedBrush, index);
        }

        public static Type GetDefaultBrushType()
        {
            Type defaultType = typeof(GridBrush);
            int count = 0;
            foreach (var type in TypeCache.GetTypesWithAttribute<CustomGridBrushAttribute>())
            {
                var attrs = type.GetCustomAttributes(typeof(CustomGridBrushAttribute), false) as CustomGridBrushAttribute[];
                if (attrs != null && attrs.Length > 0)
                {
                    if (attrs[0].defaultBrush)
                    {
                        defaultType = type;
                        count++;
                    }
                }
            }
            if (count > 1)
            {
                Debug.LogWarning("Multiple occurrences of defaultBrush == true found. It should only be declared once.");
            }
            return defaultType;
        }

        public static void GridBrushAssetChanged(GridBrushBase brush)
        {
            if (brush == null)
                return;

            if (IsLibraryBrush(brush))
            {
                instance.SaveLibraryGridBrushAsset(brush);
            }
        }

        private void RefreshBrushesCache()
        {
            if (m_Brushes == null)
                m_Brushes = new List<GridBrushBase>();

            if (m_Brushes.Count == 0 || !(m_Brushes[0] is GridBrush))
            {
                Type defaultType = GetDefaultBrushType();
                GridBrushBase defaultBrush = LoadOrCreateLibraryGridBrushAsset(defaultType);
                m_Brushes.Insert(0, defaultBrush);
                m_Brushes[0].name = GetBrushDropdownName(m_Brushes[0]);
            }

            var brushTypes = TypeCache.GetTypesDerivedFrom<GridBrushBase>().Where(t => t != typeof(GridBrush));
            foreach (var brushType in brushTypes)
            {
                if (IsDefaultInstanceVisibleGridBrushType(brushType))
                {
                    var brush = LoadOrCreateLibraryGridBrushAsset(brushType);
                    if (brush != null)
                        m_Brushes.Add(brush);
                }
            }

            string[] guids = AssetDatabase.FindAssets("t:GridBrushBase");
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var brush = AssetDatabase.LoadAssetAtPath(path, typeof(GridBrushBase)) as GridBrushBase;
                if (brush != null && IsAssetVisibleGridBrushType(brush.GetType()))
                    m_Brushes.Add(brush);
            }

            m_BrushNames = new string[m_Brushes.Count];
            m_BrushTooltips = new string[m_Brushes.Count];
            m_BrushIcons = new Dictionary<Type, Texture2D>();
            for (int i = 0; i < m_Brushes.Count; i++)
            {
                m_BrushNames[i] = m_Brushes[i].name;
                var editor = Editor.CreateEditor(m_Brushes[i]) as GridBrushEditorBase;
                m_BrushTooltips[i] = editor != null ? editor.tooltip : null;
                m_BrushIcons[m_Brushes[i].GetType()] = editor != null ? editor.icon : Texture2D.redTexture;
                DestroyImmediate(editor);
            }
        }

        internal static bool IsDefaultInstanceVisibleGridBrushType(Type brushType)
        {
            CustomGridBrushAttribute[] customBrushes = brushType.GetCustomAttributes(typeof(CustomGridBrushAttribute), false) as CustomGridBrushAttribute[];
            if (customBrushes != null && customBrushes.Length > 0)
            {
                return !customBrushes[0].hideDefaultInstance;
            }
            return false;
        }

        private bool IsAssetVisibleGridBrushType(Type brushType)
        {
            CustomGridBrushAttribute[] customBrushes = brushType.GetCustomAttributes(typeof(CustomGridBrushAttribute), false) as CustomGridBrushAttribute[];
            if (customBrushes != null && customBrushes.Length > 0 && GridBrushPickStoreSettingsProvider.GetUserBrushPickStore() == null)
            {
                return !customBrushes[0].hideAssetInstances;
            }
            return false;
        }

        private void SaveLibraryGridBrushAsset(GridBrushBase brush)
        {
            var gridBrushPath = GenerateGridBrushInstanceLibraryPath(brush.GetType());
            string folderPath = Path.GetDirectoryName(gridBrushPath);
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }
            InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] { brush }, gridBrushPath, EditorSettings.serializationMode != SerializationMode.ForceBinary);
        }

        private GridBrushBase LoadOrCreateLibraryGridBrushAsset(Type brushType)
        {
            var serializedObjects = InternalEditorUtility.LoadSerializedFileAndForget(GenerateGridBrushInstanceLibraryPath(brushType));
            if (serializedObjects != null && serializedObjects.Length > 0)
            {
                GridBrushBase brush = serializedObjects[0] as GridBrushBase;
                if (brush != null && brush.GetType() == brushType)
                    return brush;
            }
            return CreateLibraryGridBrushAsset(brushType);
        }

        private GridBrushBase CreateLibraryGridBrushAsset(Type brushType)
        {
            GridBrushBase brush = ScriptableObject.CreateInstance(brushType) as GridBrushBase;
            brush.hideFlags = HideFlags.DontSave;
            brush.name = GetBrushDropdownName(brush);
            SaveLibraryGridBrushAsset(brush);
            return brush;
        }

        private string GenerateGridBrushInstanceLibraryPath(Type brushType)
        {
            var path = FileUtil.CombinePaths(s_LibraryPath, brushType + s_GridBrushExtension);
            path = FileUtil.NiceWinPath(path);
            return path;
        }

        private string GetBrushDropdownName(GridBrushBase brush)
        {
            // Asset Brushes use the asset name
            if (!IsLibraryBrush(brush))
                return brush.name;

            // Library Brushes
            CustomGridBrushAttribute[] customBrushes = brush.GetType().GetCustomAttributes(typeof(CustomGridBrushAttribute), false) as CustomGridBrushAttribute[];
            if (customBrushes != null && customBrushes.Length > 0 && customBrushes[0].defaultName.Length > 0)
                return customBrushes[0].defaultName;

            if (brush.GetType() == typeof(GridBrush))
                return "Default Brush";

            return brush.GetType().Name;
        }

        private static bool IsLibraryBrush(GridBrushBase brush)
        {
            return !AssetDatabase.Contains(brush);
        }

        // TODO: Better way of clearing caches than AssetPostprocessor
        public class AssetProcessor : AssetPostprocessor
        {
            public override int GetPostprocessOrder()
            {
                return 1;
            }

            private static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromPath)
            {
                if (!GridPaintingState.savingPalette)
                    FlushCache();
            }
        }

        internal static void FlushCache()
        {
            s_RefreshCache = true;
            if (instance.m_Brushes != null)
            {
                foreach (var brush in instance.m_Brushes)
                {
                    if (!EditorUtility.IsPersistent(brush))
                        DestroyImmediate(brush);
                }
                instance.m_Brushes.Clear();
                GridPaintingState.FlushCache();
            }
        }

        internal static void RefreshCache()
        {
            if (s_RefreshCache)
            {
                s_RefreshCache = false;
                instance.RefreshBrushesCache();
            }
        }
    }
}
