using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [EditorToolbarElement(k_ToolbarId)]
    internal class TilePaletteBrushesDropdownToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Brushes Dropdown Toggle";

        private new static readonly string ussClassName = "unity-tilepalette-brushesdropdown-toggle";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.BrushSettings.png";
        private static readonly string k_Tooltip = L10n.Tr("Toggles the visibility of the Brush Settings Overlay");

        public TilePaletteBrushesDropdownToggle()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_Tooltip;

            var input = this.Q<VisualElement>(className: Toggle.inputUssClassName);
            var arrow = new VisualElement();
            arrow.AddToClassList("unity-icon-arrow");
            arrow.pickingMode = PickingMode.Ignore;
            input.Add(arrow);
        }

        public override bool value
        {
            get => base.value;
            set
            {
                base.value = value;
                if (value)
                {
                    ClickEvent();
                }
                else
                {
                    CloseEvent();
                }
            }
        }

        private void ClickEvent()
        {
            BoolFieldOverlayPopupWindow.ShowOverlayPopup<TilePaletteBrushInspectorPopup>(this, new Vector2(300, 180), false);
        }

        private void CloseEvent()
        {
            BoolFieldOverlayPopupWindow.CloseAllWindows<TilePaletteBrushInspectorPopup>();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            CloseEvent();
        }
    }
}
