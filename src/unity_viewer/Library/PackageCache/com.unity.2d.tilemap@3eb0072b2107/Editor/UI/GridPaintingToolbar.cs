using UnityEngine.UIElements;
using UnityEditor.Toolbars;
using System;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// A VisualElement displaying a Toolbar showing EditorTools for GridPainting.
    /// </summary>
    /// <description>
    /// This shows the EditorTools available for the active Brush.
    /// </description>
    [UxmlElement]
    public partial class GridPaintingToolbar : VisualElement
    {
        /// <summary>
        /// Factory for GridPaintingToolbar.
        /// </summary>
        [Obsolete("GridPaintingToolbarFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class GridPaintingToolbarFactory : UxmlFactory<GridPaintingToolbar, GridPaintingToolbarUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for GridPaintingToolbar.
        /// </summary>
        [Obsolete("GridPaintingToolbarUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class GridPaintingToolbarUxmlTraits : UxmlTraits {}


        private static readonly string ussClassName = "unity-tilepalette-toolbar";
        private static readonly string k_Name = L10n.Tr("Tile Palette Toolbar");
        private static readonly string[] k_ToolbarElements = { TilemapEditorToolbarStrip.k_ToolbarId };

        /// <summary>
        /// Initializes and returns an instance of GridPaintingToolbar.
        /// </summary>
        public GridPaintingToolbar() : this(null) { }

        /// <summary>
        /// Initializes and returns an instance of GridPaintingToolbar.
        /// </summary>
        /// <param name="editorWindow">Editor Window containing this VisualElement.</param>
        public GridPaintingToolbar(EditorWindow editorWindow)
        {
            AddToClassList(ussClassName);

            name = k_Name;
            var ot = EditorToolbar.CreateOverlay(k_ToolbarElements, editorWindow);
            Add(ot);
        }
    }
}
