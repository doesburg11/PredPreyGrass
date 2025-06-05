using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    internal class SceneViewOpenTilePaletteHelper : ScriptableSingleton<SceneViewOpenTilePaletteHelper>
    {
        private bool m_RegisteredEventHandlers;
        private bool m_IsSelectionValid;
        private bool m_HighlightHelper;

        internal static bool highlight
        {
            get => instance.m_HighlightHelper;
            set => instance.m_HighlightHelper = value;
        }

        [InitializeOnLoadMethod]
        private static void Initialize()
        {
            instance.RegisterEventHandlers();
        }

        private void OnEnable()
        {
            RegisterEventHandlers();
        }

        private void RegisterEventHandlers()
        {
            if (m_RegisteredEventHandlers)
                return;

            Selection.selectionChanged += SelectionChanged;
            EditorApplication.hierarchyChanged += SelectionChanged;

            m_IsSelectionValid = IsSelectionValid();

            m_RegisteredEventHandlers = true;
        }

        private void OnDisable()
        {
            Selection.selectionChanged -= SelectionChanged;
            EditorApplication.hierarchyChanged -= SelectionChanged;
            m_RegisteredEventHandlers = false;
        }

        internal static void OpenTilePalette()
        {
            GridPaintPaletteWindow.OpenTilemapPalette();
            instance.m_HighlightHelper = false;

            var target = Selection.activeGameObject;
            if (target != null)
            {
                if (PrefabUtility.IsPartOfPrefabAsset(target))
                {
                    var path = AssetDatabase.GetAssetPath(target);
                    if (AssetDatabase.LoadAssetAtPath<GridPalette>(path))
                    {
                        GridPaintingState.palette = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                    }
                }
                else if (GridPaintingState.validTargets != null)
                {
                    var grid = target.GetComponent<GridLayout>();
                    if (grid != null)
                    {
                        foreach (var validTarget in GridPaintingState.validTargets)
                        {
                            if (validTarget == target)
                            {
                                GridPaintingState.scenePaintTarget = target;
                                break;
                            }
                        }
                    }
                }
            }
        }

        internal static bool IsActive()
        {
            if (GridPaintingState.isEditing)
                return false;
            return instance.m_IsSelectionValid;
        }

        internal static bool IsSelectionValid()
        {
            if (Selection.activeObject == null)
                return false;
            if (Selection.activeObject is TileBase)
                return true;
            if (Selection.activeGameObject != null && Selection.activeGameObject.GetComponent<GridLayout>() != null)
                return true;
            return false;
        }

        private void SelectionChanged()
        {
            var old = m_IsSelectionValid;
            m_IsSelectionValid = IsSelectionValid();
            if (m_IsSelectionValid != old)
                m_HighlightHelper = m_IsSelectionValid;
        }

        internal class SceneViewOpenTilePaletteProperties
        {
            public static readonly string showInSceneViewEditorPref = "OpenTilePalette.ShowInSceneView";
            public static readonly string showInSceneViewLookup = "Show Open Tile Palette in Scene View";

            public static readonly GUIContent showInSceneViewLabel = EditorGUIUtility.TrTextContent(showInSceneViewLookup, "Shows an overlay in the SceneView for opening the Tile Palette when selecting an object that interacts with the Tile Palette.");
        }

        internal static bool showInSceneViewActive
        {
            get { return EditorPrefs.GetBool(SceneViewOpenTilePaletteProperties.showInSceneViewEditorPref, true); }
            set { EditorPrefs.SetBool(SceneViewOpenTilePaletteProperties.showInSceneViewEditorPref, value); }
        }

        internal static void PreferencesGUI()
        {
            using (new SettingsWindow.GUIScope())
            {
                EditorGUI.BeginChangeCheck();
                var val = EditorGUILayout.Toggle(SceneViewOpenTilePaletteProperties.showInSceneViewLabel, showInSceneViewActive);
                if (EditorGUI.EndChangeCheck())
                {
                    showInSceneViewActive = val;
                    SceneView.RepaintAll();
                }
            }
        }
    }
}
