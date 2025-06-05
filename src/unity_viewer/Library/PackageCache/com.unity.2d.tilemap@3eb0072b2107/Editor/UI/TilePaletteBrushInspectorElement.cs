using System;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Visual Element showing the Inspector for the Active Brush for Grid Painting.
    /// </summary>
    [UxmlElement]
    public partial class TilePaletteBrushInspectorElement : IMGUIContainer
    {
        /// <summary>
        /// Factory for TilePaletteBrushInspectorElement.
        /// </summary>
        [Obsolete("TilePaletteBrushInspectorElementFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushInspectorElementFactory : UxmlFactory<TilePaletteBrushInspectorElement, TilePaletteBrushInspectorElementUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for TilePaletteBrushInspectorElement.
        /// </summary>
        [Obsolete("TilePaletteBrushInspectorElementUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushInspectorElementUxmlTraits : UxmlTraits {}

        /// <summary>
        /// USS class name of elements of this type.
        /// </summary>
        private new static readonly string ussClassName = "unity-tilepalette-brushinspector";

        private TilePaletteBrushInspector m_BrushInspector = new TilePaletteBrushInspector();

        /// <summary>
        /// Initializes and returns an instance of TilePaletteBrushInspectorElement.
        /// </summary>
        public TilePaletteBrushInspectorElement()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            onGUIHandler = m_BrushInspector.OnGUI;
        }
    }
}
