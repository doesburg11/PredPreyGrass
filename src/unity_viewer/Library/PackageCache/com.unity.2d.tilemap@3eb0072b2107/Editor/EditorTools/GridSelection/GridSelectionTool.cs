using System;
using System.Linq;
using UnityEditor.EditorTools;
using UnityEngine;
using UnityEngine.Tilemaps;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Abstract class for Editor Tool used to handle a GridSelection.
    /// </summary>
    public abstract class GridSelectionTool : EditorTool
    {
        private TileBase[] m_SelectionTiles;
        private Color[] m_SelectionColors;
        private Matrix4x4[] m_SelectionMatrices;
        private TileFlags[] m_SelectionFlagsArray;
        private Sprite[] m_SelectionSprites;
        private Tile.ColliderType[] m_SelectionColliderTypes;
        private int m_FirstCellWithTile;

        private int selectionCellCount => Math.Abs(GridSelection.position.size.x * GridSelection.position.size.y * GridSelection.position.size.z);

        /// <summary>
        /// Does the GUI for the GridSelectionTool for an EditorWindow.
        /// </summary>
        /// <param name="window">EditorWindow which GUI is being done.</param>
        public override void OnToolGUI(EditorWindow window)
        {
            var selection = Selection.activeObject as GridSelection;
            if (selection == null)
                return;

            if (window is SceneView && GridSelection.target != null && GridPaintingState.IsPartOfActivePalette(GridSelection.target))
                return;

            OnToolGUI();
        }

        internal void OnToolGUI()
        {
            if (GridSelection.target == null)
                return;

            var brushTarget = GridSelection.target;
            var tilemap = brushTarget.GetComponent<Tilemap>();
            if (tilemap == null)
                return;

            UpdateSelection(tilemap);
            if (m_SelectionFlagsArray == null || m_SelectionFlagsArray.Length <= 0)
                return;

            bool transformFlagsAllEqual = m_SelectionFlagsArray.All(flags => (flags & TileFlags.LockTransform) == (m_SelectionFlagsArray.First() & TileFlags.LockTransform));
            if (!transformFlagsAllEqual || (m_SelectionFlagsArray[0] & TileFlags.LockTransform) != 0)
                return;

            var index = m_FirstCellWithTile != -1 ? m_FirstCellWithTile : 0;
            var transformMatrix = m_SelectionMatrices[index];
            var p = (Vector3)transformMatrix.GetColumn(3);
            var r = Quaternion.identity;
            var s = transformMatrix.lossyScale;
            Vector3 selectionPosition = GridSelection.position.position;
            selectionPosition += tilemap.tileAnchor;
            if (selectionCellCount > 1)
            {
                selectionPosition.x = GridSelection.position.center.x;
                selectionPosition.y = GridSelection.position.center.y;
            }
            var gizmoPosition = tilemap.LocalToWorld(tilemap.CellToLocalInterpolated(selectionPosition)) + p;
            var originalP = gizmoPosition;
            var originalR = r;
            var originalS = s;
            EditorGUI.BeginChangeCheck();
            HandleTool(ref gizmoPosition, ref r, ref s);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(new Object[] { tilemap, tilemap.gameObject }, "Move");
                var deltaPos = gizmoPosition - originalP;
                var deltaRotation = Quaternion.Inverse(originalR) * r;
                var deltaScale = new Vector3(s.x / originalS.x, s.y / originalS.y, s.z / originalS.z);
                foreach (var cellPosition in GridSelection.position.allPositionsWithin)
                {
                    if (tilemap.HasTile(cellPosition))
                    {
                        var trs = tilemap.GetTransformMatrix(cellPosition);
                        var tilePosition = trs.GetPosition() + deltaPos;
                        var tileRotation = trs.rotation * deltaRotation;
                        var tileScale = trs.lossyScale;
                        tileScale.x *= deltaScale.x;
                        tileScale.y *= deltaScale.y;
                        tileScale.z *= deltaScale.z;
                        trs = Matrix4x4.TRS(tilePosition, tileRotation, tileScale);
                        tilemap.SetTransformMatrix(cellPosition, trs);
                    }
                }
                InspectorWindow.RepaintAllInspectors();
            }
        }

        /// <summary>
        /// Handles the gizmo for the GridSelectionTool.
        /// Implement this the handle the gizmo for the GridSelectionTool.
        /// </summary>
        /// <param name="position">Position of the GridSelection gizmo.</param>
        /// <param name="rotation">Rotation of the GridSelection gizmo.</param>
        /// <param name="scale">Scale of the GridSelection gizmo.</param>
        public abstract void HandleTool(ref Vector3 position, ref Quaternion rotation, ref Vector3 scale);

        internal static bool IsActive()
        {
            return ToolManager.activeToolType != null && ToolManager.activeToolType.IsSubclassOf(typeof(GridSelectionTool));
        }

        private void UpdateSelection(Tilemap tilemap)
        {
            m_FirstCellWithTile = -1;
            var selection = GridSelection.position;
            var cellCount = selectionCellCount;
            if (m_SelectionTiles == null || m_SelectionTiles.Length != selectionCellCount)
            {
                m_SelectionTiles = new TileBase[cellCount];
                m_SelectionColors = new Color[cellCount];
                m_SelectionMatrices = new Matrix4x4[cellCount];
                m_SelectionFlagsArray = new TileFlags[cellCount];
                m_SelectionSprites = new Sprite[cellCount];
                m_SelectionColliderTypes = new Tile.ColliderType[cellCount];
            }

            int index = 0;
            foreach (var p in selection.allPositionsWithin)
            {
                m_SelectionTiles[index] = tilemap.GetTile(p);
                m_SelectionColors[index] = tilemap.GetColor(p);
                m_SelectionMatrices[index] = tilemap.GetTransformMatrix(p);
                m_SelectionFlagsArray[index] = tilemap.GetTileFlags(p);
                m_SelectionSprites[index] = tilemap.GetSprite(p);
                m_SelectionColliderTypes[index] = tilemap.GetColliderType(p);

                if (m_FirstCellWithTile == -1 && m_SelectionTiles[index] != null)
                    m_FirstCellWithTile = index;
                index++;
            }
        }
    }
}
