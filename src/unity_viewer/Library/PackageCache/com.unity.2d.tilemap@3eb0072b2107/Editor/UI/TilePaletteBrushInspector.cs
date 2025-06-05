using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteBrushInspector
    {
        private Vector2 m_Scroll;

        private static class Styles
        {
            public static readonly GUIContent lockZPosition = EditorGUIUtility.TrTextContent("Lock Z Position", "Toggle editing of Z position");
            public static readonly GUIContent sceneViewZPosition = EditorGUIUtility.TrTextContent("SceneView Z Position", "Set a Z position for the active Brush for painting in the SceneView");
            public static readonly GUIContent paletteZPosition = EditorGUIUtility.TrTextContent("Palette Z Position", "Set a Z position for the active Brush for painting in the Palette");
            public static readonly GUIContent resetZPosition = EditorGUIUtility.TrTextContent("Reset", "Reset Z position for the active Brush");
        }

        public void OnGUI()
        {
            if (GridPaintingState.gridBrush == null)
                return;

            m_Scroll = GUILayout.BeginScrollView(m_Scroll);

            // Brush Inspector GUI
            EditorGUI.BeginChangeCheck();
            if (GridPaintingState.activeBrushEditor != null)
                GridPaintingState.activeBrushEditor.OnPaintInspectorGUI();
            else if (GridPaintingState.fallbackEditor != null)
                GridPaintingState.fallbackEditor.OnInspectorGUI();
            if (EditorGUI.EndChangeCheck())
            {
                GridPaintingState.ActiveGridBrushAssetChanged();
            }

            // Z Position Inspector
            var hasSceneViewGrid = GridPaintingState.paintableSceneViewGrid != null;
            var hasClipboard = GridPaintPaletteClipboard.instances.Count > 0 && GridPaintPaletteClipboard.instances[0] != null;

            using (new EditorGUI.DisabledScope(!hasSceneViewGrid))
            {
                var lockZPosition = false;
                if (GridPaintingState.activeBrushEditor != null)
                {
                    EditorGUI.BeginChangeCheck();
                    lockZPosition = EditorGUILayout.Toggle(Styles.lockZPosition, !GridPaintingState.activeBrushEditor.canChangeZPosition);
                    if (EditorGUI.EndChangeCheck())
                        GridPaintingState.activeBrushEditor.canChangeZPosition = !lockZPosition;
                }
                using (new EditorGUI.DisabledScope(lockZPosition))
                {
                    HandleGridZPosition(hasSceneViewGrid ? GridPaintingState.paintableSceneViewGrid : null, Styles.sceneViewZPosition, !hasSceneViewGrid);
                    HandleGridZPosition(hasClipboard ? GridPaintPaletteClipboard.instances[0] : null, Styles.paletteZPosition, !hasClipboard);
                }
            }
            GUILayout.EndScrollView();
        }

        private void HandleGridZPosition(PaintableGrid paintableGrid, GUIContent label, bool disabled)
        {
            using (new EditorGUI.DisabledScope(disabled))
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUI.BeginChangeCheck();
                var paletteZPosition = EditorGUILayout.DelayedIntField(label, paintableGrid != null ? paintableGrid.zPosition : 0);
                if (EditorGUI.EndChangeCheck())
                {
                    if (GridPaintingState.lastActiveGrid == paintableGrid)
                        GridPaintingState.gridBrush.ChangeZPosition(paletteZPosition - paintableGrid.zPosition);
                    paintableGrid.zPosition = paletteZPosition;
                }
                if (GUILayout.Button(Styles.resetZPosition))
                {
                    if (GridPaintingState.lastActiveGrid == paintableGrid)
                        GridPaintingState.gridBrush.ResetZPosition();
                    paintableGrid.ResetZPosition();
                }
                EditorGUILayout.EndHorizontal();
            }
        }
    }
}
