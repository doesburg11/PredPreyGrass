using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteBrushPickPanelPopup : BoolFieldOverlayPopupWindow
    {
        private BaseField<bool> trigger;

        public void CreateGUI()
        {
            var brushPickElement = new TilePaletteBrushPickElement();
            rootVisualElement.Add(brushPickElement);
        }
    }
}
