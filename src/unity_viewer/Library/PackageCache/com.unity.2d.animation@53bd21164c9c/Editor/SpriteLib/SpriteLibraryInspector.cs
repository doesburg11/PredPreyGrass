using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Scripting.APIUpdating;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation
{
    [CustomEditor(typeof(SpriteLibrary))]
    [CanEditMultipleObjects]
    [MovedFrom("UnityEditor.Experimental.U2D.Animation")]
    internal class SpriteLibraryInspector : Editor
    {
        static class Style
        {
            public static readonly string createNew = L10n.Tr("New");
            public static readonly string exportToAssetInfo = L10n.Tr($"There are overrides in this Sprite Library. To save the override data, select 'Export to Sprite Library Asset'. Learn mode about creating and modifying overrides <a href={k_OverrideEntriesDocsLink}>here</a>.");
            public static readonly string exportToAsset = L10n.Tr("Export to Sprite Library Asset");
            public static readonly string selectSaveLocation = L10n.Tr("Select save location");
            public static readonly string selectSaveLocationMessage = L10n.Tr("Select save location for the Sprite Library Asset in your project.");
            public static readonly string openInSpriteLibraryEditor = L10n.Tr("Open Sprite Library Editor");
            public static readonly string exportIncorrectSavePath = L10n.Tr("Asset save path must be inside the Assets folder.");
        }

        SerializedProperty m_MasterLibraryProperty;
        SpriteLibraryAsset m_MasterLibraryObject;

        Dictionary<Object, SerializedObject> m_CachedSerializedObjects;
        List<SpriteResolver> m_CachedResolvers;

        const string k_RootFolderName = "Assets";
        const string k_OverrideEntriesDocsLink = "\"https://docs.unity3d.com/Packages/com.unity.2d.animation@latest/index.html?subfolder=/manual/SL-component.html%23overriding-entries\"";

        public void OnEnable()
        {
            m_MasterLibraryProperty = serializedObject.FindProperty(SpriteLibraryComponentPropertyString.spriteLibraryAsset);

            UpdateMasterLibraryReference();
            CacheSerializedObjects();
            CacheResolvers();
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            UpdateMasterLibraryReference();

            EditorGUILayout.BeginHorizontal();

            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_MasterLibraryProperty);
            if (EditorGUI.EndChangeCheck())
            {
                serializedObject.ApplyModifiedProperties();

                UpdateMasterLibraryReference();
                UpdateSpriteResolvers();
            }

            if (m_MasterLibraryObject == null && !m_MasterLibraryProperty.hasMultipleDifferentValues)
            {
                if (GUILayout.Button(Style.createNew) && HandleCreateNewAsset())
                {
                    UpdateMasterLibraryReference();
                    UpdateSpriteResolvers();
                }
            }

            EditorGUILayout.EndHorizontal();

            EditorGUI.BeginDisabledGroup(m_MasterLibraryObject == null || m_MasterLibraryProperty.hasMultipleDifferentValues);
            if (GUILayout.Button(Style.openInSpriteLibraryEditor))
            {
                Selection.objects = new Object[] { m_MasterLibraryObject };
                SpriteLibraryEditor.SpriteLibraryEditorWindow.OpenWindow();
            }

            EditorGUI.EndDisabledGroup();

            if (targets.Any(t => HasLocalOverride(m_CachedSerializedObjects[t])))
            {
                EditorStyles.helpBox.richText = true;
                EditorGUILayout.HelpBox(Style.exportToAssetInfo, MessageType.Info);
                if (GUILayout.Button(Style.exportToAsset) && HandleExportOverrides())
                {
                    UpdateMasterLibraryReference();
                    UpdateSpriteResolvers();
                }
            }
        }

        void CacheSerializedObjects()
        {
            m_CachedSerializedObjects = new Dictionary<Object, SerializedObject>();
            foreach (var t in targets)
                m_CachedSerializedObjects[t] = new SerializedObject(t);
        }

        void CacheResolvers()
        {
            m_CachedResolvers = new List<SpriteResolver>();
            foreach (var t in targets)
            {
                var sl = (SpriteLibrary)t;
                var resolvers = sl.GetComponentsInChildren<SpriteResolver>();
                m_CachedResolvers.AddRange(resolvers);
            }
        }

        void UpdateMasterLibraryReference()
        {
            serializedObject.Update();
            m_MasterLibraryObject = (SpriteLibraryAsset)m_MasterLibraryProperty.objectReferenceValue;
        }

        void UpdateSpriteResolvers()
        {
            foreach (var resolver in m_CachedResolvers)
            {
                resolver.ResolveSpriteToSpriteRenderer();
                resolver.spriteLibChanged = true;
            }
        }

        bool HandleCreateNewAsset()
        {
            var createPath = GetFileSavePath(target.name);
            if (!string.IsNullOrEmpty(createPath))
            {
                var emptyLibrary = CreateInstance<SpriteLibrarySourceAsset>();
                SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(emptyLibrary, createPath);
                DestroyImmediate(emptyLibrary);

                AssetDatabase.ImportAsset(createPath);
                var newLibraryAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(createPath);
                if (newLibraryAsset != null)
                {
                    m_MasterLibraryProperty.objectReferenceValue = newLibraryAsset;
                    serializedObject.ApplyModifiedProperties();

                    return true;
                }
            }

            return false;
        }

        bool HandleExportOverrides()
        {
            if (targets.Length == 1)
            {
                var exportPath = GetFileSavePath(target.name);
                if (!string.IsNullOrEmpty(exportPath))
                {
                    SpriteLibraryUtilitiesEditor.ExportSpriteLibraryToAssetFile(target as SpriteLibrary, exportPath);
                    return true;
                }
            }
            else
            {
                var exportDirectory = GetSaveDirectory();
                if (!string.IsNullOrEmpty(exportDirectory))
                {
                    foreach (var t in targets)
                    {
                        if (HasLocalOverride(m_CachedSerializedObjects[t]))
                        {
                            var exportPath = $"{exportDirectory}/{t.name}{SpriteLibrarySourceAsset.extension}";
                            exportPath = AssetDatabase.GenerateUniqueAssetPath(exportPath);
                            SpriteLibraryUtilitiesEditor.ExportSpriteLibraryToAssetFile(t as SpriteLibrary, exportPath);
                        }
                    }

                    return true;
                }
            }

            return false;
        }

        static string GetFileSavePath(string suggestedFileName)
        {
            var title = $"{Style.selectSaveLocation} ({suggestedFileName})";
            var defaultName = suggestedFileName + SpriteLibrarySourceAsset.extension;
            var extension = SpriteLibrarySourceAsset.extension.Substring(1);
            var path = EditorUtility.SaveFilePanelInProject(title, defaultName, extension, Style.selectSaveLocationMessage);
            return path;
        }

        static string GetSaveDirectory()
        {
            var saveDirectory = EditorUtility.SaveFolderPanel(Style.selectSaveLocation, k_RootFolderName, "");
            if (string.IsNullOrEmpty(saveDirectory))
                return string.Empty;

            saveDirectory = GetPathRelativeToAssetsRoot(saveDirectory);
            if (string.IsNullOrEmpty(saveDirectory))
            {
                Debug.Log(Style.exportIncorrectSavePath);
                return string.Empty;
            }

            return saveDirectory;
        }

        static string GetPathRelativeToAssetsRoot(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !path.StartsWith(Application.dataPath))
                return string.Empty;

            var pathStartIndex = path.IndexOf(k_RootFolderName);
            return pathStartIndex == -1 ? string.Empty : path.Substring(pathStartIndex);
        }

        static bool HasLocalOverride(SerializedObject serializedObject)
        {
            serializedObject.Update();
            var library = serializedObject.FindProperty(SpriteLibraryComponentPropertyString.library);
            return library != null && library.arraySize > 0;
        }
    }
}