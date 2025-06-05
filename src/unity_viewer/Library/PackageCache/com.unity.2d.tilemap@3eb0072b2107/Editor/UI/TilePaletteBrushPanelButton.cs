using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [EditorToolbarElement(k_ToolbarId)]
    internal class TilePaletteBrushPanelButton : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Brush Panel Button";
        private const string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.BrushSettings.png";

        public TilePaletteBrushPanelButton()
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
                BoolFieldOverlayPopupWindow.ShowOverlayPopup<TilePaletteBrushInspectorPopup>(this, new Vector2(300, 180), false);
            else
                BoolFieldOverlayPopupWindow.CloseAllWindows<TilePaletteBrushInspectorPopup>();
        }
    }
}
