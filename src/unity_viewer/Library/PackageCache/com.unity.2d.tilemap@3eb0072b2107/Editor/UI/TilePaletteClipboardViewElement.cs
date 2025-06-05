using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [UxmlElement]
    public partial class TilePaletteClipboardViewElement : VisualElement
    {
        private static readonly string ussClassName = "unity-tilepalette-clipboard-view-element";
        private static readonly string k_Name = L10n.Tr("Tile Palette Clipboard View Element");


        public TilePaletteClipboardViewElement()
        {
            AddToClassList(ussClassName);

            name = k_Name;
            TilePaletteOverlayUtility.SetStyleSheet(this);

        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            GridPaintingState.beforePaletteChanged += BeforePaletteChanged;
            GridPaintingState.paletteChanged += PaletteChanged;
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            GridPaintingState.beforePaletteChanged -= BeforePaletteChanged;
            GridPaintingState.paletteChanged -= PaletteChanged;
        }

        private void BeforePaletteChanged()
        {
        }

        private void PaletteChanged(GameObject palette)
        {
        }
    }
}
