using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    internal class TilemapFocusModeUtility
    {
        internal enum TilemapFocusMode
        {
            None = 0,
            Tilemap = 1,
            Grid = 2
        }
        private static readonly string k_TilemapFocusModeEditorPref = "TilemapFocusMode";

        internal static TilemapFocusMode focusMode
        {
            get
            {
                return (TilemapFocusMode)EditorPrefs.GetInt(k_TilemapFocusModeEditorPref, (int)TilemapFocusMode.None);
            }
            set
            {
                EditorPrefs.SetInt(k_TilemapFocusModeEditorPref, (int)value);
            }
        }

        internal static void OnSceneViewGUI(SceneView sceneView)
        {
            if ((GridPaintingState.defaultBrush == null || GridPaintingState.scenePaintTarget == null) && focusMode != TilemapFocusMode.None)
            {
                // case 946284: Disable Focus if focus mode is set but there is nothing to focus on
                DisableFocus();
                focusMode = TilemapFocusMode.None;
            }
        }

        internal static void OnScenePaintTargetChanged(GameObject scenePaintTarget)
        {
            DisableFocus();
            EnableFocus();
            SceneView.RepaintAll();
        }

        internal static void OnBrushChanged(GridBrushBase brush)
        {
            DisableFocus();
            if (brush is GridBrush)
                EnableFocus();
            SceneView.RepaintAll();
        }

        internal static void SetFocusMode(TilemapFocusMode tilemapFocusMode)
        {
            if (tilemapFocusMode != focusMode)
            {
                DisableFocus();
                focusMode = tilemapFocusMode;
                EnableFocus();
            }
        }

        private static void EnableFocus()
        {
            if (GridPaintingState.scenePaintTarget == null)
                return;

            switch (focusMode)
            {
                case TilemapFocusMode.Tilemap:
                {
                    FilterSingleSceneObjectInScene(GridPaintingState.scenePaintTarget.GetInstanceID());
                    break;
                }
                case TilemapFocusMode.Grid:
                {
                    Tilemap tilemap = GridPaintingState.scenePaintTarget.GetComponent<Tilemap>();
                    if (tilemap != null && tilemap.layoutGrid != null)
                    {
                        FilterSingleSceneObjectInScene(tilemap.layoutGrid.gameObject.GetInstanceID());
                    }
                    break;
                }
            }
        }

        private static void DisableFocus()
        {
            if (focusMode == TilemapFocusMode.None)
                return;

            StageHandle currentStageHandle = StageUtility.GetCurrentStageHandle();
            if (currentStageHandle.IsValid() && !currentStageHandle.isMainStage)
            {
                HierarchyProperty.ClearSceneObjectsFilterInScene(new[] { currentStageHandle.customScene });
            }
            else
            {
                HierarchyProperty.ClearSceneObjectsFilter();
            }

            if (SceneView.lastActiveSceneView != null)
            {
                SceneView.lastActiveSceneView.SetSceneViewFiltering(false);
                SceneView.lastActiveSceneView.Repaint();
            }
        }

        private static void FilterSingleSceneObjectInScene(int instanceID)
        {
            if (SceneView.lastActiveSceneView != null)
                SceneView.lastActiveSceneView.SetSceneViewFiltering(true);

            StageHandle currentStageHandle = StageUtility.GetCurrentStageHandle();
            if (currentStageHandle.IsValid() && !currentStageHandle.isMainStage)
            {
                HierarchyProperty.FilterSingleSceneObjectInScene(instanceID
                    , false
                    , new[] { currentStageHandle.customScene });
            }
            else
            {
                HierarchyProperty.FilterSingleSceneObject(instanceID, false);
            }

            if (SceneView.lastActiveSceneView != null)
                SceneView.lastActiveSceneView.Repaint();
        }
    }
}
