using System;
using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Dropdown Button for selecting the Active Brush for Grid Painting.
    /// </summary>
    [EditorToolbarElement(k_ToolbarId)]
    [UxmlElement]
    public sealed partial class TilePaletteBrushesButton : EditorToolbarDropdown
    {
        /// <summary>
        /// Factory for TilePaletteBrushesButton.
        /// </summary>
        [Obsolete("TilePaletteBrushesButtonFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushesButtonFactory : UxmlFactory<TilePaletteBrushesButton, TilePaletteBrushesButtonUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for TilePaletteBrushesButton.
        /// </summary>
        [Obsolete("TilePaletteBrushesButtonUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushesButtonUxmlTraits : UxmlTraits {}

        private new static readonly string ussClassName = "unity-tilepalette-brushes-button";

        internal const string k_ToolbarId = "Tile Palette/Brushes Button";
        private const string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.DefaultBrush.png";

        private Texture2D m_DefaultIcon;

        /// <summary>
        /// Initializes and returns an instance of TilePaletteBrushesButton.
        /// </summary>
        public TilePaletteBrushesButton()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);
            m_DefaultIcon = EditorGUIUtility.LoadIcon(k_IconPath);
            icon = m_DefaultIcon;

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            clicked += OnClicked;
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            GridPaintingState.brushChanged += OnBrushChanged;
            UpdateBrush();
        }

        private void OnBrushChanged(GridBrushBase obj)
        {
            UpdateBrush();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            GridPaintingState.brushChanged -= OnBrushChanged;
        }

        private void UpdateBrush()
        {
            var defaultTooltip = TilePaletteBrushesLabel.kNullBrushName;
            var defaultIcon = m_DefaultIcon;
            tooltip = GridPaintingState.gridBrush != null ? GridPaintingState.gridBrush.name : defaultTooltip;
            icon = GridPaintingState.activeBrushEditor != null && GridPaintingState.activeBrushEditor.icon != null ? GridPaintingState.activeBrushEditor.icon : defaultIcon;
        }

        private void OnClicked()
        {
            IGenericMenu menu = new TilePaletteBrushesDropdownMenu();
            menu.DropDown(worldBound, this, true);
        }
    }
}
