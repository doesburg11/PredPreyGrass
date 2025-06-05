using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [UxmlElement]
    internal partial class TilePaletteBrushesLabel : TextElement
    {
        public static string kNullBrushName = L10n.Tr("No Valid Brush");
        public static string kLabelTooltip =
            L10n.Tr("Specifies the currently active Brush used for painting in the Scene View.");

        private bool m_AppendSettings;
        public bool appendSettings
        {
            get { return m_AppendSettings; }
            set
            {
                m_AppendSettings = value;
                style.unityFontStyleAndWeight = m_AppendSettings ? FontStyle.Bold : FontStyle.Normal;
            }
        }

        [Obsolete("TilePaletteBrushesLabelFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class
            TilePaletteBrushesLabelFactory : UxmlFactory<TilePaletteBrushesLabel, TilePaletteBrushesLabelUxmlTraits>
        {
        }

        [Obsolete("TilePaletteBrushesLabelUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushesLabelUxmlTraits : UxmlTraits
        {
        }

        /// <summary>
        /// USS class name of elements of this type.
        /// </summary>
        public new static readonly string ussClassName = "unity-tilepalette-brushes-label";

        /// <summary>
        /// Initializes and returns an instance of TilePaletteBrushesLabel.
        /// </summary>
        public TilePaletteBrushesLabel()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);
            tooltip = kLabelTooltip;

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
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

        private string FormatBrushName(GridBrushBase brush)
        {
            if (brush != null)
            {
                if (appendSettings)
                    return String.Format("{0} Settings", brush.name);
                return brush.name;
            }
            return kNullBrushName;
        }

        private void UpdateBrush()
        {
            text = FormatBrushName(GridPaintingState.gridBrush);
        }
    }
}
