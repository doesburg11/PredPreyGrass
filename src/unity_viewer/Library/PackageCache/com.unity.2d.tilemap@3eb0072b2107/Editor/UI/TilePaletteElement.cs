using System;
using UnityEditor.ShortcutManagement;
using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// A Visual Element which handles and displays a Tile Palette Clipboard
    /// and its associated tools.
    /// </summary>
    /// <description>
    /// This Visual Element includes other Visual Elements that help with managing
    /// the Tile Palette Clipboard, which includes a popup dropdown for selecting
    /// the active Palette and a toolbar for enabling edits and showing visual aids
    /// in the Tile Palette Clipboard.
    /// </description>
    [UxmlElement]
    public partial class TilePaletteElement : VisualElement
    {
        /// <summary>
        /// Factory for TilePaletteElement.
        /// </summary>
        [Obsolete("TilePaletteElementFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteElementFactory : UxmlFactory<TilePaletteElement, TilePaletteElementUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for TilePaletteElement.
        /// </summary>
        [Obsolete("TilePaletteElementUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteElementUxmlTraits : UxmlTraits {}

        private static readonly string ussClassName = "unity-tilepalette-element";
        private static readonly string toolbarUssClassName = ussClassName + "-toolbar";
        private static readonly string rightToolbarUssClassName = toolbarUssClassName + "-right";

        private static readonly string kTilePaletteElementHideOnPickEditorPref = "TilePaletteElementHideOnPick";

        private static readonly string k_Name = L10n.Tr("Tile Palette Element");

        private static string[] k_LeftToolbarElements = new[]
        {
            TilePaletteActivePalettePopupIcon.k_ToolbarId
            , TilePaletteActivePalettePopup.k_ToolbarId
        };

        private static string[] k_RightToolbarElements = new[] {
            TilePaletteEditToggle.k_ToolbarId
            , TilePaletteGridToggle.k_ToolbarId
            , TilePaletteGizmoToggle.k_ToolbarId
            , TilePaletteHideClipboardToggle.k_ToolbarId
        };
        private static bool[] k_TilePaletteWindowActiveRightToolbar = new[] { true, true, true, false };
        private static bool[] k_TilePaletteOverlayActiveRightToolbar = new[] { true, true, true, true };

        private VisualElement m_ToolbarElement;
        private TilePaletteClipboardElement m_ClipboardElement;
        private TilePaletteEditToggle m_EditToggle;
        private TilePaletteGridToggle m_GridToggle;
        private TilePaletteGizmoToggle m_GizmoToggle;
        private TilePaletteHideClipboardToggle m_HideToggle;
        private VisualElement m_RightToolbar;

        private event Action onBrushPickedInternal;

        /// <summary>
        /// Whether the clipboard is unlocked for editing.
        /// </summary>
        public bool clipboardUnlocked
        {
            get => m_ClipboardElement.clipboardUnlocked;
            set
            {
                m_ClipboardElement.clipboardUnlocked = value;
                m_EditToggle.SetValueWithoutNotify(value);
            }
        }

        /// <summary>
        /// Whether the clipboard is hidden when the Pick EditorTool is activated on it.
        /// </summary>
        public bool hideOnPick
        {
            get => EditorPrefs.GetBool(kTilePaletteElementHideOnPickEditorPref, true);
            set
            {
                EditorPrefs.SetBool(kTilePaletteElementHideOnPickEditorPref, value);
                m_HideToggle.SetValueWithoutNotify(value);
            }
        }

        /// <summary>
        /// Callback when the active Brush does a Pick on the Clipboard.
        /// </summary>
        public event Action onBrushPicked
        {
            add
            {
                onBrushPickedInternal += value;
                SetupRightToolbar();
            }
            remove
            {
                onBrushPickedInternal -= value;
                SetupRightToolbar();
            }
        }

        internal GridPaintPaletteClipboard clipboardView => m_ClipboardElement.clipboardView;

        /// <summary>
        /// Initializes and returns an instance of TilePaletteElement.
        /// </summary>
        public TilePaletteElement()
        {
            AddToClassList(ussClassName);

            name = k_Name;
            TilePaletteOverlayUtility.SetStyleSheet(this);

            m_ToolbarElement = new VisualElement();
            m_ToolbarElement.AddToClassList(toolbarUssClassName);
            Add(m_ToolbarElement);

            m_ClipboardElement = new TilePaletteClipboardElement();
            m_ClipboardElement.onBrushPicked += OnBrushPicked;
            Add(m_ClipboardElement);

            var leftToolbar = EditorToolbar.CreateOverlay(k_LeftToolbarElements);
            m_ToolbarElement.Add(leftToolbar);

            var rightToolbarElement = new VisualElement();
            rightToolbarElement.AddToClassList(rightToolbarUssClassName);
            m_ToolbarElement.Add(rightToolbarElement);

            m_RightToolbar = EditorToolbar.CreateOverlay(k_RightToolbarElements);
            SetupRightToolbar();
            rightToolbarElement.Add(m_RightToolbar);

            m_EditToggle = this.Q<TilePaletteEditToggle>();
            m_EditToggle.SetValueWithoutNotify(false);
            m_EditToggle.ToggleChanged += OnEditToggleChanged;

            m_GridToggle = this.Q<TilePaletteGridToggle>();
            m_GridToggle.SetValueWithoutNotify(GridPaintingState.drawGridGizmo);
            m_GridToggle.ToggleChanged += OnGridToggleChanged;

            m_GizmoToggle = this.Q<TilePaletteGizmoToggle>();
            m_GizmoToggle.SetValueWithoutNotify(GridPaintingState.drawGizmos);
            m_GizmoToggle.ToggleChanged += OnGizmoToggleChanged;

            m_HideToggle = this.Q<TilePaletteHideClipboardToggle>();
            m_HideToggle.SetValueWithoutNotify(hideOnPick);
            m_HideToggle.ToggleChanged += OnHideClipboardToggleChanged;

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            UpdateToggleState();
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            m_ClipboardElement.clipboardUnlockedChanged += OnUnlockedChanged;
            GridPaintingState.palettesChanged += UpdateToggleState;
            GridPaintingState.paletteChanged += UpdateEditToggleState;
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            GridPaintingState.paletteChanged -= UpdateEditToggleState;
            GridPaintingState.palettesChanged -= UpdateToggleState;
            m_ClipboardElement.clipboardUnlockedChanged -= OnUnlockedChanged;
        }

        private void OnBrushPicked()
        {
            if (hideOnPick)
                onBrushPickedInternal?.Invoke();
        }

        private void OnEditToggleChanged(bool unlock)
        {
            clipboardUnlocked = unlock;
            clipboardView.Repaint();
        }

        private void OnUnlockedChanged(bool unlock)
        {
            m_EditToggle.SetValueWithoutNotify(unlock);
        }

        private void OnGridToggleChanged(bool drawGridGizmo)
        {
            GridPaintingState.drawGridGizmo = drawGridGizmo;
            clipboardView.Repaint();
        }

        private void OnGizmoToggleChanged(bool drawGizmo)
        {
            GridPaintingState.drawGizmos = drawGizmo;
            clipboardView.Repaint();
        }

        private void UpdateToggleState()
        {
            var hasPalette = GridPaintingState.palettes != null && GridPaintingState.palettes.Count > 0;
            m_EditToggle.SetEnabled(hasPalette && GridPaintingState.isPaletteEditable);
            m_GridToggle.SetEnabled(hasPalette);
            m_GizmoToggle.SetEnabled(hasPalette);
        }

        private void UpdateEditToggleState(GameObject palette)
        {
            m_EditToggle.SetEnabled(GridPaintingState.isPaletteEditable);
        }

        private void OnHideClipboardToggleChanged(bool hideClipboard)
        {
            hideOnPick = hideClipboard;
        }

        private void SetupRightToolbar()
        {
            TilePaletteOverlayUtility.SetupChildrenAsButtonStripForVisible(m_RightToolbar,
                onBrushPickedInternal != null ? k_TilePaletteOverlayActiveRightToolbar : k_TilePaletteWindowActiveRightToolbar);
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteEditToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Edit";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_ElementClass = "unity-tilepalette-element-tilepaletteedit";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.EditPalette.png";
        private static readonly string k_TooltipText = L10n.Tr("Toggles Tile Palette Edit");

        public Action<bool> ToggleChanged;

        public TilePaletteEditToggle()
        {
            name = "Tile Palette Edit";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_TooltipText;
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            ToggleChanged?.Invoke(value);
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteGridToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Grid";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_ElementClass = "unity-tilepalette-element-gridtoggle";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.ShowGrid.png";
        private static readonly string k_TooltipText = L10n.Tr("Toggle visibility of the Grid in the Tile Palette");

        public Action<bool> ToggleChanged;

        public TilePaletteGridToggle()
        {
            name = "Tile Palette Grid";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_TooltipText;
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            ToggleChanged?.Invoke(value);
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteBrushElementToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Brush Element";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_ElementClass = "unity-tilepalette-element-brushelement";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.BrushSettings.png";
        private static readonly string k_TooltipText = L10n.Tr("Toggle visibility of the Brush Inspector in the Tile Palette");

        public Action<bool> ToggleChanged;

        public TilePaletteBrushElementToggle()
        {
            name = "Tile Palette Grid";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_TooltipText;
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            ToggleChanged?.Invoke(value);
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteGizmoToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Gizmo";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_ElementClass = "unity-tilepalette-element-gizmotoggle";

        private static readonly string k_IconPath = "GizmosToggle";
        private static readonly string k_TooltipText = L10n.Tr("Toggle visibility of Gizmos in the Tile Palette");

        public Action<bool> ToggleChanged;

        public TilePaletteGizmoToggle()
        {
            name = "Tile Palette Gizmo";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_TooltipText;
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            ToggleChanged?.Invoke(value);
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteHideClipboardToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Hide Clipboard";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_ElementClass = "unity-tilepalette-element-hideclipboard";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.ShowTilePalette.png";
        private static readonly string k_TooltipFormatText = L10n.Tr("Hides Tile Palette on Pick. ( {0} ) to show/hide Tile Palette.");
        private static readonly string k_ShortcutId = "Grid Painting/Toggle SceneView Palette";

        public Action<bool> ToggleChanged;

        public TilePaletteHideClipboardToggle()
        {
            name = "Tile Palette Hide Clipboard";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            ShortcutIntegration.instance.profileManager.shortcutBindingChanged += OnShortcutBindingChanged;
            UpdateTooltip();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            ShortcutIntegration.instance.profileManager.shortcutBindingChanged -= OnShortcutBindingChanged;
        }

        private void OnShortcutBindingChanged(IShortcutProfileManager arg1, Identifier arg2, ShortcutBinding arg3, ShortcutBinding arg4)
        {
            UpdateTooltip();
        }

        private void UpdateTooltip()
        {
            tooltip = String.Format(k_TooltipFormatText, ShortcutManager.instance.GetShortcutBinding(k_ShortcutId));
        }

        protected override void ToggleValue()
        {
            base.ToggleValue();
            ToggleChanged?.Invoke(value);
        }
    }
}
