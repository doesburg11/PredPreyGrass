using UnityEditor.EditorTools;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// An `EditorTool` for handling Scale for a `GridSelection`.
    /// </summary>
    public class GridSelectionScaleTool : GridSelectionTool
    {
        private static class Styles
        {
            public static readonly GUIContent toolbarIcon = EditorGUIUtility.TrTextContentWithIcon("Scale", "Shows a Gizmo in the Scene view for changing the scale for the Grid Selection", "ScaleTool");
        }

        /// <summary>
        /// Toolbar icon for the `GridSelectionScaleTool`.
        /// </summary>
        public override GUIContent toolbarIcon => Styles.toolbarIcon;

        /// <summary>
        /// Handles the gizmo for managing Rotation for the `GridSelectionScaleTool`.
        /// </summary>
        /// <param name="position">Position of the `GridSelection` gizmo.</param>
        /// <param name="rotation">Rotation of the `GridSelection` gizmo.</param>
        /// <param name="scale">Scale of the `GridSelection` gizmo.</param>
        public override void HandleTool(ref Vector3 position, ref Quaternion rotation, ref Vector3 scale)
        {
            scale = Handles.ScaleHandle(scale, position, rotation, HandleUtility.GetHandleSize(position));
        }
    }
}
