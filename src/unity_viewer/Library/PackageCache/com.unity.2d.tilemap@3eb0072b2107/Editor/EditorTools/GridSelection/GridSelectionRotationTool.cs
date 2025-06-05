using UnityEditor.EditorTools;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// An `EditorTool` for handling Rotation for a `GridSelection`.
    /// </summary>
    public class GridSelectionRotateTool : GridSelectionTool
    {
        private static class Styles
        {
            public static readonly GUIContent toolbarIcon = EditorGUIUtility.TrTextContentWithIcon("Rotate", "Shows a Gizmo in the Scene view for changing the rotation for the Grid Selection", "RotateTool");
        }

        /// <summary>
        /// Toolbar icon for the `GridSelectionRotateTool`.
        /// </summary>
        public override GUIContent toolbarIcon => Styles.toolbarIcon;

        private Quaternion before = Quaternion.identity;

        private void Reset()
        {
            before = Quaternion.identity;
        }

        /// <summary>
        /// Handles the gizmo for managing Rotation for the `GridSelectionRotateTool`.
        /// </summary>
        /// <param name="position">Position of the `GridSelection` gizmo.</param>
        /// <param name="rotation">Rotation of the `GridSelection` gizmo.</param>
        /// <param name="scale">Scale of the `GridSelection` gizmo.</param>
        public override void HandleTool(ref Vector3 position, ref Quaternion rotation, ref Vector3 scale)
        {
            if (Event.current.GetTypeForControl(GUIUtility.hotControl) == EventType.MouseUp)
                Reset();

            EditorGUI.BeginChangeCheck();
            var after = Handles.RotationHandle(before, position);
            if (EditorGUI.EndChangeCheck())
            {
                rotation *= Quaternion.Inverse(before) * after;
            }
            before = after;
        }
    }
}
