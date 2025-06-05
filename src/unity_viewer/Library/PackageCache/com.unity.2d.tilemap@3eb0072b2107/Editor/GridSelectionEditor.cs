using System;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    [CustomEditor(typeof(GridSelection))]
    internal class GridSelectionEditor : Editor
    {
        private const float iconSize = 32f;

        static class Styles
        {
            public static readonly GUIContent gridSelectionLabel = EditorGUIUtility.TrTextContent("Grid Selection");

            public static readonly string iconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/GridSelection.png";
        }

        private void OnValidate()
        {
            var position = GridSelection.position;
            GridSelection.position = new BoundsInt(position.min, position.max - position.min);
        }

        public override void OnInspectorGUI()
        {
            EditorGUI.BeginChangeCheck();
            if (GridPaintingState.activeBrushEditor && GridSelection.active)
            {
                var canEdit =
                    !GridPaintingState.IsPartOfActivePalette(GridSelection.target)
                    || GridPaintingState.isPaletteEditable;
                using (new EditorGUI.DisabledScope(!canEdit))
                {
                    GridPaintingState.activeBrushEditor.OnSelectionInspectorGUI();
                }
            }
            if (EditorGUI.EndChangeCheck())
            {
                if (GridPaintingState.IsPartOfActivePalette(GridSelection.target))
                {
                    GridPaintingState.UnlockGridPaintPaletteClipboardForEditing();
                    GridPaintingState.RepaintGridPaintPaletteWindow();
                }
                else
                {
                    GridSelection.SaveStandalone();
                }
            }
        }

        protected override void OnHeaderGUI()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.inspectorBig);
            Texture2D icon = EditorGUIUtility.LoadIcon(Styles.iconPath);
            GUILayout.Label(icon, GUILayout.Width(iconSize), GUILayout.Height(iconSize));
            EditorGUILayout.BeginVertical();
            GUILayout.Label(Styles.gridSelectionLabel);
            EditorGUI.BeginChangeCheck();
            GridSelection.position = EditorGUILayout.BoundsIntField(GUIContent.none, GridSelection.position);
            if (EditorGUI.EndChangeCheck())
            {
                OnValidate();
            }
            EditorGUILayout.EndVertical();
            EditorGUILayout.EndHorizontal();
            DrawHeaderHelpAndSettingsGUI(GUILayoutUtility.GetLastRect());
        }

        public bool HasFrameBounds()
        {
            return GridSelection.active;
        }

        public Bounds OnGetFrameBounds()
        {
            Bounds bounds = new Bounds();
            if (GridSelection.active)
            {
                Vector3Int gridMin = GridSelection.position.min;
                Vector3Int gridMax = GridSelection.position.max;

                Vector3 min = GridSelection.grid.CellToWorld(gridMin);
                Vector3 max = GridSelection.grid.CellToWorld(gridMax);

                bounds = new Bounds((max + min) * .5f, max - min);
            }

            return bounds;
        }
    }
}
