using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [EditorToolbarElement(k_ToolbarId)]
    internal class TilePaletteBrushPickButton : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/BrushPick Button";
        private const string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.BrushPicks.png";

        public TilePaletteBrushPickButton()
        {
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);

            var input = this.Q<VisualElement>(className: Toggle.inputUssClassName);
            var arrow = new VisualElement();
            arrow.AddToClassList("unity-icon-arrow");
            arrow.pickingMode = PickingMode.Ignore;
            input.Add(arrow);
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            if (value)
                BoolFieldOverlayPopupWindow.ShowOverlayPopup<TilePaletteBrushPickPanelPopup>(this, new Vector2(370, 320), false);
            else
                BoolFieldOverlayPopupWindow.CloseAllWindows<TilePaletteBrushPickPanelPopup>();
        }
    }
}
