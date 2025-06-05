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
        , defaultDockIndex = 1
        , defaultLayout = Layout.Panel
        , defaultWidth = k_DefaultWidth
        , defaultHeight = k_DefaultHeight)]
    internal class TilePaletteBrushPickOverlay : Overlay, ICreateHorizontalToolbar, ICreateVerticalToolbar, ITransientOverlay
    {
        public const string k_OverlayId = "Scene View/Tile Palette Brush Pick";
        internal const string k_DisplayName = "Tile Palette Brush Pick";

        internal readonly static string[] k_ElementIds = new[] { TilePaletteBrushPickButton.k_ToolbarId };

        private SceneView sceneView => containerWindow as SceneView;
        private bool m_DisplayAsPanel = false;
        private TilePaletteBrushPickElement m_BrushPickElement;

        public bool visible => GridPaintPaletteWindow.isActive && m_DisplayAsPanel;

        private const float k_DefaultWidth = 390;
        private const float k_DefaultHeight = 300;

        public TilePaletteBrushPickOverlay()
        {
            minSize = new Vector2(k_DefaultWidth, k_DefaultHeight);
            maxSize = new Vector2(k_DefaultWidth * 3, k_DefaultHeight * 3);
            collapsedIcon = EditorGUIUtility.LoadIcon(TilePaletteBrushPickToggle.k_IconPath);
        }

        public override void OnWillBeDestroyed()
        {
            if (m_BrushPickElement != null)
            {
                m_BrushPickElement.onBrushPicked -= TogglePopup;
                m_BrushPickElement = null;
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
            if (m_BrushPickElement == null)
                return;

            var width = m_BrushPickElement.rect.width;
            var height = m_BrushPickElement.rect.height;

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
            m_BrushPickElement = new TilePaletteBrushPickElement();
            m_BrushPickElement.onBrushPicked += TogglePopup;
            return m_BrushPickElement;
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
    internal class TilePaletteBrushPickToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Brush Pick Toggle";

        private new static readonly string ussClassName = "unity-tilepalette-brushpick-toggle";

        internal static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.BrushPicks.png";

        private static readonly string k_Tooltip = L10n.Tr("Toggles the visibility of the Tile Palette Brush Picks Overlay ({0})");

        private static readonly string k_ShortcutId = GridPaintPaletteWindow.ShortcutIds.k_ToggleSceneViewBrushPick;

        private SceneView m_SceneView;
        private TilePaletteBrushPickOverlay m_BrushPickOverlay;

        public TilePaletteBrushPickToggle()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = GetTooltipText();
        }

        private TilePaletteBrushPickOverlay GetOverlayFromSceneView(SceneView sceneView)
        {
            if (sceneView == null)
                return null;

            if (sceneView.TryGetOverlay(TilePaletteBrushPickOverlay.k_OverlayId, out var overlay)
                && overlay is TilePaletteBrushPickOverlay brushPickOverlay)
            {
                return brushPickOverlay;
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
            UpdateBrushPickOverlay();
        }

        private void UpdateBrushPickOverlay()
        {
            m_BrushPickOverlay = GetOverlayFromSceneView(m_SceneView);
            if (m_BrushPickOverlay != null)
            {
                SetValueWithoutNotify(m_BrushPickOverlay.visible);
                m_BrushPickOverlay.displayedChanged += BrushPickOnDisplayChanged;
            }
        }

        private void LastActiveSceneViewChanged(SceneView oldSceneView, SceneView newSceneView)
        {
            if (m_BrushPickOverlay != null)
            {
                m_BrushPickOverlay.displayedChanged -= BrushPickOnDisplayChanged;
            }
            m_SceneView = newSceneView;
            UpdateBrushPickOverlay();
        }

        private void BrushPickOnDisplayChanged(bool _)
        {
            SetValueWithoutNotify(m_BrushPickOverlay.visible);
        }

        public override bool value
        {
            get => base.value;
            set
            {
                if (m_SceneView == null || m_BrushPickOverlay == null)
                    return;

                if (value != base.value)
                {
                    if (value)
                    {
                        m_BrushPickOverlay.Show();
                    }
                    else
                    {
                        m_BrushPickOverlay.Hide();
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
            base.SetValueWithoutNotify(m_BrushPickOverlay.visible);
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            if (m_BrushPickOverlay != null)
            {
                m_BrushPickOverlay.displayedChanged -= BrushPickOnDisplayChanged;
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
    internal class TilePaletteBrushPickDropdownToggle : EditorToolbarDropdownToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Brush Selection Dropdown Toggle";

        private new static readonly string ussClassName = "unity-tilepalettebrushPickdropdown-toggle";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.BrushPicks.png";

        private static readonly string k_Tooltip = L10n.Tr("Toggles the visiblity of the Tile Palette Brush Selection Overlay");

        private SceneView m_SceneView;
        private TilePaletteBrushPickOverlay m_BrushPickOverlay;

        public TilePaletteBrushPickDropdownToggle()
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
                if (m_SceneView.TryGetOverlay(TilePaletteBrushPickOverlay.k_OverlayId, out var overlay)
                    && overlay is TilePaletteBrushPickOverlay brushPickOverlay)
                {
                    m_BrushPickOverlay = brushPickOverlay;
                    SetValueWithoutNotify(m_BrushPickOverlay.visible);
                    m_BrushPickOverlay.displayedChanged += BrushPickOnDisplayChanged;
                }
                break;
            }
        }

        private void BrushPickOnDisplayChanged(bool _)
        {
            SetValueWithoutNotify(m_BrushPickOverlay.visible);
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
                        m_BrushPickOverlay.Show();
                    }
                    else
                    {
                        m_BrushPickOverlay.Hide();
                    }
                    base.value = value;
                }
            }
        }

        public override void SetValueWithoutNotify(bool newValue)
        {
            // Force value to be tied to m_BrushSettingsOverlay.visible
            // As there is a bug where the arrow toggle affects the value
            base.SetValueWithoutNotify(m_BrushPickOverlay.visible);
        }

        private void ClickEvent()
        {
            BoolFieldOverlayPopupWindow.ShowOverlayPopup<TilePaletteBrushPickPanelPopup>(this, new Vector2(370, 260), false);
        }

        private void CloseEvent()
        {
            BoolFieldOverlayPopupWindow.CloseAllWindows<TilePaletteBrushPickPanelPopup>();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            if (m_BrushPickOverlay != null)
                m_BrushPickOverlay.displayedChanged -= BrushPickOnDisplayChanged;
            CloseEvent();
        }
    }
}
