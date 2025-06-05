using System;
using UnityEngine;
using UnityEditor.Overlays;
using UnityEditor.ShortcutManagement;
using UnityEditor.Toolbars;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [Overlay(typeof(SceneView), k_OverlayId, k_DisplayName
        , defaultDockZone = DockZone.RightColumn
        , defaultDockPosition = DockPosition.Bottom
        , defaultDockIndex = 0
        , defaultLayout = Layout.Panel
        , defaultWidth = k_DefaultWidth
        , defaultHeight = k_DefaultHeight)]
    internal class TilePaletteClipboardOverlay : Overlay, ICreateHorizontalToolbar, ICreateVerticalToolbar, ITransientOverlay
    {
        public const string k_OverlayId = "Scene View/Tile Palette Clipboard";
        internal const string k_DisplayName = "Tile Palette Clipboard";

        internal readonly static string[] k_ElementIds = new[] { TilePaletteClipboardButton.k_ToolbarId };

        private SceneView sceneView => containerWindow as SceneView;
        private bool m_DisplayAsPanel = false;
        private TilePaletteElement m_PaletteElement;

        public bool visible => GridPaintPaletteWindow.isActive && m_DisplayAsPanel;

        private const float k_DefaultWidth = 390;
        private const float k_DefaultHeight = 300;

        public TilePaletteClipboardOverlay()
        {
            minSize = new Vector2(k_DefaultWidth, k_DefaultHeight);
            maxSize = new Vector2(k_DefaultWidth * 3, k_DefaultHeight * 3);
            collapsedIcon = EditorGUIUtility.LoadIcon(TilePaletteClipboardToggle.k_IconPath);
        }

        public override void OnWillBeDestroyed()
        {
            if (m_PaletteElement != null)
            {
                m_PaletteElement.onBrushPicked -= TogglePopup;
                m_PaletteElement = null;
            }
            base.OnWillBeDestroyed();
        }

        public void Hide()
        {
            m_DisplayAsPanel = false;
        }

        public void Show()
        {
            m_DisplayAsPanel = true;
        }

        public void TogglePopup()
        {
            m_DisplayAsPanel = !m_DisplayAsPanel;
        }

        public void TogglePopup(Vector2 mousePosition)
        {
            m_DisplayAsPanel = !m_DisplayAsPanel;
            if (m_DisplayAsPanel)
                MoveToMousePosition(mousePosition);
        }

        private void MoveToMousePosition(Vector2 mousePosition)
        {
            if (m_PaletteElement == null)
                return;

            var width = m_PaletteElement.rect.width;
            var height = m_PaletteElement.rect.height;

            // Try to position middle on the panel on mouse position
            mousePosition.x -= width / 2;
            mousePosition.y -= height / 2;

            floatingPosition = mousePosition;
        }

        public void Repaint()
        {
            if (sceneView != null)
                sceneView.Repaint();
        }

        public override VisualElement CreatePanelContent()
        {
            m_PaletteElement = new TilePaletteElement();
            m_PaletteElement.onBrushPicked += TogglePopup;
            var clipboardElement = m_PaletteElement.Q<TilePaletteClipboardElement>();
            clipboardElement.AddToClassList(TilePaletteClipboardElement.overlayClassName);
            return m_PaletteElement;
        }

        public OverlayToolbar CreateHorizontalToolbarContent()
        {
            var content = EditorToolbar.CreateOverlay(k_ElementIds, containerWindow);
            return content;
        }

        public OverlayToolbar CreateVerticalToolbarContent()
        {
            var content = EditorToolbar.CreateOverlay(k_ElementIds, containerWindow);
            return content;
        }
    }

    [TilePaletteOverlayToolbarElement]
    [EditorToolbarElement(k_ToolbarId)]
    internal class TilePaletteClipboardToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Clipboard Toggle";

        private new static readonly string ussClassName = "unity-tilepalette-clipboard-toggle";

        internal static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.TilePalette.png";

        private static readonly string k_Tooltip = L10n.Tr("Toggles the visibility of the Tile Palette Overlay ({0})");

        private static readonly string k_ShortcutId = GridPaintPaletteWindow.ShortcutIds.k_ToggleSceneViewPalette;

        private SceneView m_SceneView;
        private TilePaletteClipboardOverlay m_ClipboardOverlay;

        public TilePaletteClipboardToggle()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = GetTooltipText();
        }

        private TilePaletteClipboardOverlay GetOverlayFromSceneView(SceneView sceneView)
        {
            if (sceneView == null)
                return null;

            if (sceneView.TryGetOverlay(TilePaletteClipboardOverlay.k_OverlayId, out var overlay)
                && overlay is TilePaletteClipboardOverlay clipboardOverlay)
            {
                return clipboardOverlay;
            }
            return null;
        }

        private static string GetTooltipText()
        {
            return String.Format(k_Tooltip, ShortcutIntegration.instance.GetKeyCombinationFor(k_ShortcutId));
        }

        private void OnAttachToPanel(AttachToPanelEvent evt)
        {
            SceneView.lastActiveSceneViewChanged += LastActiveSceneViewChanged;
            if (SceneView.lastActiveSceneView != null)
            {
                m_SceneView = SceneView.lastActiveSceneView;
            }
            else
            {
                if (SceneView.sceneViews.Count > 0)
                {
                    m_SceneView = SceneView.sceneViews[0] as SceneView;
                }
            }
            ShortcutIntegration.instance.profileManager.shortcutBindingChanged += UpdateTooltips;
            UpdateClipboardOverlay();
        }

        private void UpdateClipboardOverlay()
        {
            m_ClipboardOverlay = GetOverlayFromSceneView(m_SceneView);
            if (m_ClipboardOverlay != null)
            {
                SetValueWithoutNotify(m_ClipboardOverlay.visible);
                m_ClipboardOverlay.displayedChanged += ClipboardOnDisplayChanged;
            }
        }

        private void LastActiveSceneViewChanged(SceneView oldSceneView, SceneView newSceneView)
        {
            if (m_ClipboardOverlay != null)
            {
                m_ClipboardOverlay.displayedChanged -= ClipboardOnDisplayChanged;
            }
            m_SceneView = newSceneView;
            UpdateClipboardOverlay();
        }

        private void ClipboardOnDisplayChanged(bool _)
        {
            SetValueWithoutNotify(m_ClipboardOverlay.visible);
        }

        public override bool value
        {
            get => base.value;
            set
            {
                if (m_SceneView == null || m_ClipboardOverlay == null)
                    return;

                if (value != base.value)
                {
                    if (value)
                    {
                        m_ClipboardOverlay.Show();
                    }
                    else
                    {
                        m_ClipboardOverlay.Hide();
                    }
                    base.value = value;
                    m_SceneView.RepaintImmediately();
                }
            }
        }

        public override void SetValueWithoutNotify(bool newValue)
        {
            // Force value to be tied to m_BrushSettingsOverlay.visible
            // As there is a bug where the arrow toggle affects the value
            base.SetValueWithoutNotify(m_ClipboardOverlay.visible);
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            if (m_ClipboardOverlay != null)
            {
                m_ClipboardOverlay.displayedChanged -= ClipboardOnDisplayChanged;
            }
            ShortcutIntegration.instance.profileManager.shortcutBindingChanged -= UpdateTooltips;
            SceneView.lastActiveSceneViewChanged -= LastActiveSceneViewChanged;
        }

        private void UpdateTooltips(IShortcutProfileManager obj, Identifier identifier, ShortcutBinding oldBinding, ShortcutBinding newBinding)
        {
            tooltip = GetTooltipText();
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal class TilePaletteClipboardDropdownToggle : EditorToolbarDropdownToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Clipboard Dropdown Toggle";

        private new static readonly string ussClassName = "unity-tilepalette-clipboarddropdown-toggle";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.TilePalette.png";

        private static readonly string k_Tooltip = L10n.Tr("Toggles the visiblity of the Tile Palette Overlay");

        private SceneView m_SceneView;
        private TilePaletteClipboardOverlay m_ClipboardOverlay;

        public TilePaletteClipboardDropdownToggle()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_Tooltip;
            dropdownClicked += ClickEvent;
        }

        private void OnAttachToPanel(AttachToPanelEvent evt)
        {
            foreach (var sceneViewObj in SceneView.sceneViews)
            {
                var sceneView = sceneViewObj as SceneView;
                if (sceneView == null)
                    continue;

                var common = this.FindCommonAncestor(sceneView.rootVisualElement);
                if (common == null)
                    continue;

                m_SceneView = sceneView;
                if (m_SceneView.TryGetOverlay(TilePaletteClipboardOverlay.k_OverlayId, out var overlay)
                    && overlay is TilePaletteClipboardOverlay clipboardOverlay)
                {
                    m_ClipboardOverlay = clipboardOverlay;
                    SetValueWithoutNotify(m_ClipboardOverlay.visible);
                    m_ClipboardOverlay.displayedChanged += ClipboardOnDisplayChanged;
                }
                break;
            }
        }

        private void ClipboardOnDisplayChanged(bool _)
        {
            SetValueWithoutNotify(m_ClipboardOverlay.visible);
        }

        public override bool value
        {
            get => base.value;
            set
            {
                if (value != base.value)
                {
                    if (value)
                    {
                        m_ClipboardOverlay.Show();
                    }
                    else
                    {
                        m_ClipboardOverlay.Hide();
                    }
                    base.value = value;
                }
            }
        }

        public override void SetValueWithoutNotify(bool newValue)
        {
            // Force value to be tied to m_BrushSettingsOverlay.visible
            // As there is a bug where the arrow toggle affects the value
            base.SetValueWithoutNotify(m_ClipboardOverlay.visible);
        }

        private void ClickEvent()
        {
            BoolFieldOverlayPopupWindow.ShowOverlayPopup<TilePaletteClipboardPanelPopup>(this, new Vector2(370, 260), false);
        }

        private void CloseEvent()
        {
            BoolFieldOverlayPopupWindow.CloseAllWindows<TilePaletteClipboardPanelPopup>();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            if (m_ClipboardOverlay != null)
                m_ClipboardOverlay.displayedChanged -= ClipboardOnDisplayChanged;
            CloseEvent();
        }
    }
}
