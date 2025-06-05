using UnityEditor.Toolbars;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteBrushModalElement : VisualElement
    {
        private static readonly string ussClassName = "unity-tilepalette-splitview-brushes";
        private static readonly string brushesToolbarUssClassName = "unity-tilepalette-splitview-brushes-toolbar";
        private static readonly string rightToolbarUssClassName = brushesToolbarUssClassName + "-right";

        private EditorToolbarToggle m_SettingsButton;
        private EditorToolbarToggle m_PickButton;
        private TilePaletteBrushInspectorElement m_BrushInspectorElement;

        public TilePaletteBrushModalElement()
        {
            name = "tilePaletteBrushModalElement";
            AddToClassList(ussClassName);

            TilePaletteOverlayUtility.SetStyleSheet(this);

            var he = new VisualElement();
            he.AddToClassList(brushesToolbarUssClassName);

            var rightToolbarElement = new VisualElement();
            rightToolbarElement.AddToClassList(rightToolbarUssClassName);

            string[] rightToolbarElements = new[] {
                TilePaletteBrushElementToggle.k_ToolbarId
            };
            var rightToolbar = EditorToolbar.CreateOverlay(rightToolbarElements);
            rightToolbarElement.Add(rightToolbar);

            he.Add(new TilePaletteBrushesPopup());
            he.Add(rightToolbarElement);
            Add(he);

            m_BrushInspectorElement = new TilePaletteBrushInspectorElement();
            Add(m_BrushInspectorElement);
        }
    }
}
