using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteClipboardPanelPopup : BoolFieldOverlayPopupWindow
    {
        private BaseField<bool> trigger;

        public void CreateGUI()
        {
            var clipboardElement = new TilePaletteElement();
            rootVisualElement.Add(clipboardElement);
        }
    }
}
