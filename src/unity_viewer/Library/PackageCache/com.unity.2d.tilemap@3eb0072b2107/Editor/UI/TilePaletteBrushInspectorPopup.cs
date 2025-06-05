using System;
using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteBrushInspectorPopup : BoolFieldOverlayPopupWindow
    {
        private static readonly string ussClassName = "unity-tilepalette-brushinspectorpopup";
        private static readonly string horizontalClassName = ussClassName + "__horizontal";

        private BaseField<bool> trigger;
        private bool isLocked;
        private Rect screenRect;
        private Vector2 size;

        public void CreateGUI()
        {
            rootVisualElement.AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(rootVisualElement);

            var imguiContainer = new TilePaletteBrushInspectorElement();

            var horizontalElement = new VisualElement();
            horizontalElement.AddToClassList(horizontalClassName);

            var label = new Label("Brush Settings");
            horizontalElement.Add(label);

            this.rootVisualElement.Add(horizontalElement);
            this.rootVisualElement.Add(imguiContainer);
        }

        protected override void OnEnable()
        {
            SceneView.duringSceneGui += DuringSceneGui;
        }

        private void DuringSceneGui(SceneView obj)
        {
            if (Event.current.isMouse)
                this.Repaint();
        }

        protected override void OnDisable()
        {
            SceneView.duringSceneGui -= DuringSceneGui;

            base.OnDisable();
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteBrushInspectorLockToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Brush Inspector Lock";

        private const string k_ToolSettingsClass = "unity-tool-settings";
        private static string k_TooltipText = L10n.Tr("Locks the Brush Inspector");

        public Action<bool> ToggleChanged;

        public TilePaletteBrushInspectorLockToggle()
        {
            name = "Tile Palette Brush Inspector Lock";
            AddToClassList(k_ToolSettingsClass);

            icon = EditorGUIUtility.LoadIconRequired("LockIcon");
            tooltip = k_TooltipText;
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            ToggleChanged?.Invoke(value);
        }
    }
}
