using UnityEditor.EditorTools;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// An `EditorTool` for handling Moves for a `GridSelection`.
    /// </summary>
    public class GridSelectionMoveTool : GridSelectionTool
    {
        private static class Styles
        {
            public static GUIContent toolbarIcon = EditorGUIUtility.TrTextContentWithIcon("Move", "Shows a Gizmo in the Scene view for changing the offset for the Grid Selection", "MoveTool");
        }

        /// <summary>
        /// Toolbar icon for the `GridSelectionMoveTool`.
        /// </summary>
        public override GUIContent toolbarIcon => Styles.toolbarIcon;

        /// <summary>
        /// Handles the gizmo for managing Moves for the `GridSelectionMoveTool`.
        /// </summary>
        /// <param name="position">Position of the `GridSelection` gizmo.</param>
        /// <param name="rotation">Rotation of the `GridSelection` gizmo.</param>
        /// <param name="scale">Scale of the `GridSelection` gizmo.</param>
        public override void HandleTool(ref Vector3 position, ref Quaternion rotation, ref Vector3 scale)
        {
            position = Handles.PositionHandle(position, rotation);
        }
    }
}
