using UnityEditor.Overlays;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [CustomEditor(typeof(GridSelectionTool), true)]
    internal class GridSelectionToolEditor : Editor, ICreateHorizontalToolbar, ICreateVerticalToolbar
    {
        public override VisualElement CreateInspectorGUI()
        {
            return null;
        }

        public OverlayToolbar CreateHorizontalToolbarContent()
        {
            return null;
        }

        public OverlayToolbar CreateVerticalToolbarContent()
        {
            return null;
        }
    }
}
