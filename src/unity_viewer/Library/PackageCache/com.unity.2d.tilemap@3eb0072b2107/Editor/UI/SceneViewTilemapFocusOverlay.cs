using System;
using UnityEditor.Overlays;
using UnityEditor.Toolbars;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [Overlay(typeof(SceneView), k_OverlayId, k_DisplayName
        , defaultDockZone = DockZone.RightColumn
        , defaultDockPosition = DockPosition.Bottom
        , defaultDockIndex = 0
        , defaultLayout = Layout.Panel)]
    internal class SceneViewTilemapFocusOverlay : ToolbarOverlay, ITransientOverlay
    {
        internal const string k_OverlayId = "Scene View/Tilemap Focus";
        private const string k_DisplayName = "Tilemap Focus";

        public SceneViewTilemapFocusOverlay() : base(new[] {"Tile Palette/Focus Label", "Tile Palette/Focus Dropdown"})
        {}

        public bool visible =>
            GridPaintingState.isEditing
                && GridPaintingState.scenePaintTarget != null
                && GridPaintingState.defaultBrush != null;
    }

    [EditorToolbarElement("Tile Palette/Focus Label")]
    sealed class TilePaletteFocusLabel : VisualElement
    {
        const string k_ToolSettingsClass = "unity-tool-settings";

        readonly TextElement m_Label;

        private static string k_LabelText = L10n.Tr("Focus On");

        public TilePaletteFocusLabel()
        {
            name = "Focus Label";
            AddToClassList(k_ToolSettingsClass);

            m_Label = new TextElement();
            m_Label.AddToClassList(EditorToolbar.elementLabelClassName);
            m_Label.text = k_LabelText;
            Add(m_Label);
        }
    }

    /// <summary>
    /// A `VisualElement` displaying a `Dropdown` for selecting the Focus Mode while painting on Tilemaps.
    /// </summary>
    [EditorToolbarElement("Tile Palette/Focus Dropdown")]
    [UxmlElement]
    public sealed partial class TilePaletteFocusDropdown : EditorToolbarDropdown
    {
        /// <summary>
        /// A factory for `TilePaletteFocusDropdown`.
        /// </summary>
        [Obsolete("TilePaletteFocusDropdownFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteFocusDropdownFactory : UxmlFactory<TilePaletteFocusDropdown, TilePaletteFocusDropdownUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for `TilePaletteFocusDropdown`.
        /// </summary>
        [Obsolete("TilePaletteFocusDropdownUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteFocusDropdownUxmlTraits : UxmlTraits {}

        const string k_DropdownIconClass = "unity-toolbar-dropdown-label-icon";
        const string k_ToolSettingsClass = "unity-tool-settings";

        private static readonly string k_Name = L10n.Tr("Focus Dropdown");
        private static readonly string k_FocusNoneIconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.FocusNone.png";
        private static readonly string k_FocusTilemapIconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.FocusTilemap.png";
        private static readonly string k_FocusGridIconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.FocusGrid.png";

        readonly TextElement m_Label;
        readonly VisualElement m_Icon;

        readonly GUIContent m_None;
        readonly GUIContent m_Tilemap;
        readonly GUIContent m_Grid;

        /// <summary>
        /// Constructor for `TilePaletteFocusDropdown`.
        /// </summary>
        public TilePaletteFocusDropdown()
        {
            name = k_Name;
            AddToClassList(k_ToolSettingsClass);

            m_None = EditorGUIUtility.TrTextContentWithIcon("None",
                "Focus Mode is not active.",
                EditorGUIUtility.LoadIcon(k_FocusNoneIconPath));
            m_Tilemap = EditorGUIUtility.TrTextContentWithIcon("Tilemap",
                "Focuses on the active Tilemap. Filters out all other Renderers.",
                EditorGUIUtility.LoadIcon(k_FocusTilemapIconPath));
            m_Grid = EditorGUIUtility.TrTextContentWithIcon("Grid",
                "Focuses on all Renderers with the active Grid. Filters out all other Renderers.",
                EditorGUIUtility.LoadIcon(k_FocusGridIconPath));

            clicked += OpenContextMenu;

            FocusModeChanged();
        }

        void OpenContextMenu()
        {
            var menu = new GenericMenu();
            var focusMode = TilemapFocusModeUtility.focusMode;
            menu.AddItem(m_None, focusMode == TilemapFocusModeUtility.TilemapFocusMode.None, () => SetFocusMode(TilemapFocusModeUtility.TilemapFocusMode.None));
            menu.AddItem(m_Tilemap, focusMode == TilemapFocusModeUtility.TilemapFocusMode.Tilemap, () => SetFocusMode(TilemapFocusModeUtility.TilemapFocusMode.Tilemap));
            menu.AddItem(m_Grid, focusMode == TilemapFocusModeUtility.TilemapFocusMode.Grid, () => SetFocusMode(TilemapFocusModeUtility.TilemapFocusMode.Grid));
            menu.DropDown(worldBound);
        }

        void SetFocusMode(TilemapFocusModeUtility.TilemapFocusMode mode)
        {
            TilemapFocusModeUtility.SetFocusMode(mode);
            FocusModeChanged();
        }

        void FocusModeChanged()
        {
            var content = m_None;
            switch (TilemapFocusModeUtility.focusMode)
            {
                case TilemapFocusModeUtility.TilemapFocusMode.Tilemap:
                    content = m_Tilemap;
                    break;
                case TilemapFocusModeUtility.TilemapFocusMode.Grid:
                    content = m_Grid;
                    break;
            }

            text = content.text;
            tooltip = content.tooltip;
            icon = content.image as Texture2D;
        }
    }
}
