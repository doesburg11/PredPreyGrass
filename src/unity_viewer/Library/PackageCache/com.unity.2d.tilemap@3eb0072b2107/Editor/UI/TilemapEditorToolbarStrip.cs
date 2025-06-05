using UnityEditor.Toolbars;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// A VisualElement displaying a Toolbar showing EditorTools for GridPainting.
    /// </summary>
    /// <description>
    /// This shows the EditorTools available for the active Brush.
    /// </description>
    [EditorToolbarElement(k_ToolbarId)]
    internal class TilemapEditorToolbarStrip : VisualElement
    {
        private static readonly string ussClassName = "unity-tilepalette-toolbar-strip";
        internal const string k_ToolbarId = "Tools/Tile Palette Tools";
        private static readonly string k_Name = L10n.Tr("Tile Palette Tools");

        private bool isUpdating;

        /// <summary>
        /// Constructor for TilemapEditorToolbarStrip.
        /// </summary>
        public TilemapEditorToolbarStrip()
        {
            name = k_Name;
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
        }

        void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            TilemapEditorToolPreferences.tilemapEditorToolsChanged += OnTilemapEditorToolsChanged;
            GridPaintingState.brushToolsChanged += OnTilemapEditorToolsChanged;

            OnTilemapEditorToolsChanged();
        }

        void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            RemoveTilemapEditorTools();

            GridPaintingState.brushToolsChanged -= OnTilemapEditorToolsChanged;
            TilemapEditorToolPreferences.tilemapEditorToolsChanged -= OnTilemapEditorToolsChanged;
        }

        private void OnTilemapEditorToolsChanged()
        {
            UpdateTilemapEditorTools();
        }

        private void UpdateTilemapEditorTools()
        {
            if (isUpdating)
                return;

            isUpdating = true;
            RemoveTilemapEditorTools();
            var tools = TilemapEditorTool.tilemapEditorTools;
            foreach (var tool in tools)
            {
                var button = new TilemapEditorToolButton(tool as TilemapEditorTool);
                Add(button);
                button.SetEnabled(true);
            }
            EditorToolbarUtility.SetupChildrenAsButtonStrip(this);
            isUpdating = false;
        }

        private void RemoveTilemapEditorTools()
        {
            Clear();
        }
    }
}
