using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor.Overlays;
using UnityEngine;
using UnityEngine.U2D.Animation;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SceneOverlays
{
    internal interface INavigableElement
    {
        public event Action<int> onSelectionChange;

        int itemCount { get; }
        int selectedIndex { get; }

        public void Select(int index);
        void SetItems(IList items);
        object GetItem(int index);

        VisualElement visualElement { get; }
    }

    internal enum PropertyAnimationState
    {
        NotAnimated,
        Animated,
        Candidate,
        Recording
    }

    [Overlay(typeof(SceneView), overlayId, k_DefaultName,
        defaultDisplay = k_DefaultVisibility,
        defaultDockZone = DockZone.RightColumn,
        defaultDockPosition = DockPosition.Bottom,
        defaultLayout = Overlays.Layout.Panel,
        defaultWidth = k_DefaultWidth + k_WidthPadding,
        defaultHeight = k_DefaultHeight + k_HeightPadding)]
    [Icon("Packages/com.unity.2d.animation/Editor/Assets/ComponentIcons/Animation.SpriteResolver.png")]
    internal class SpriteSwapOverlay : Overlay
    {
        public static class Settings
        {
            public const float minThumbnailSize = 20.0f + k_ThumbnailPadding;
            public const float maxThumbnailSize = 110.0f + k_ThumbnailPadding;
            public const float defaultThumbnailSize = 50.0f + k_ThumbnailPadding;
            const float k_ThumbnailPadding = 8.0f;

            const string k_FilterKey = UserSettings.kSettingsUniqueKey + "SpriteSwapOverlay.filter";
            const string k_LockKey = UserSettings.kSettingsUniqueKey + "SpriteSwapOverlay.lock";
            const string k_ThumbnailSizeKey = UserSettings.kSettingsUniqueKey + "SpriteSwapOverlay.thumbnailSize";
            const string k_PreferredWidthKey = UserSettings.kSettingsUniqueKey + "SpriteSwapOverlay.preferredWidth";
            const string k_PreferredHeightKey = UserSettings.kSettingsUniqueKey + "SpriteSwapOverlay.preferredHeight";

            public static bool filter
            {
                get => EditorPrefs.GetBool(k_FilterKey, false);
                set => EditorPrefs.SetBool(k_FilterKey, value);
            }

            public static bool locked
            {
                get => EditorPrefs.GetBool(k_LockKey, false);
                set => EditorPrefs.SetBool(k_LockKey, value);
            }

            public static float thumbnailSize
            {
                get => EditorPrefs.GetFloat(k_ThumbnailSizeKey, defaultThumbnailSize);
                set => EditorPrefs.SetFloat(k_ThumbnailSizeKey, Mathf.Clamp(value, minThumbnailSize, maxThumbnailSize));
            }

            public static float preferredWidth
            {
                get => EditorPrefs.GetFloat(k_PreferredWidthKey, k_DefaultWidth);
                set => EditorPrefs.SetFloat(k_PreferredWidthKey, value);
            }

            public static float preferredHeight
            {
                get => EditorPrefs.GetFloat(k_PreferredHeightKey, k_DefaultHeight);
                set => EditorPrefs.SetFloat(k_PreferredHeightKey, value);
            }
        }

        public const string overlayId = "Scene View/Sprite Swap";
        public const string rootStyle = "sprite-swap-overlay";

        const float k_DefaultWidth = 230.0f;
        const float k_DefaultHeight = 133.0f;
        const float k_WidthPadding = 6.0f;
        const float k_HeightPadding = 19.0f;

        const bool k_DefaultVisibility = false;

        const string k_DefaultName = "Sprite Swap";

        public SpriteSwapVisualElement mainVisualElement => m_MainVisualElement;

        internal SpriteResolver[] selection => m_Selection;

        bool isViewInitialized => m_MainVisualElement != null;

        SpriteResolver[] m_Selection = Array.Empty<SpriteResolver>();

        SpriteSwapVisualElement m_MainVisualElement;

        public SpriteSwapOverlay()
        {
            minSize = new Vector2(k_DefaultWidth + k_WidthPadding, k_DefaultHeight + k_HeightPadding);
            maxSize = new Vector2(k_DefaultWidth * 10.0f + k_WidthPadding, k_DefaultHeight * 10.0f + k_HeightPadding);
        }

        public override VisualElement CreatePanelContent()
        {
            var overlayElement = new SpriteSwapVisualElement { style = { width = Settings.preferredWidth, height = Settings.preferredHeight } };
            var toolbar = overlayElement.Q<OverlayToolbar>();
            toolbar.onFilterToggled += OnFilterToggled;
            toolbar.onLockToggled += OnLockToggled;
            toolbar.onResetSliderValue += OnResetThumbnailSize;
            toolbar.onSliderValueChanged += OnChangeThumbnailSize;
            overlayElement.RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            overlayElement.RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
            overlayElement.styleSheets.Add(ResourceLoader.Load<StyleSheet>("SpriteSwap/SpriteSwapOverlay.uss"));
            return overlayElement;
        }

        public override void OnCreated()
        {
            base.OnCreated();
            EditorApplication.hierarchyChanged += OnHierarchyChanged;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            Selection.selectionChanged += OnSelectionChanged;
            SceneView.duringSceneGui += OnSceneGUI;

            UpdateSelection(true);
        }

        public override void OnWillBeDestroyed()
        {
            EditorApplication.hierarchyChanged -= OnHierarchyChanged;
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            Selection.selectionChanged -= OnSelectionChanged;
            SceneView.duringSceneGui -= OnSceneGUI;
            base.OnWillBeDestroyed();
        }

        void OnSceneGUI(SceneView sceneView)
        {
            if (containerWindow != sceneView || m_MainVisualElement == null)
                return;

            m_MainVisualElement.OnSceneGUI();
        }

        internal void OnSelectionChanged()
        {
            UpdateSelection();
        }

        void OnHierarchyChanged()
        {
            var needUpdate = false;
            foreach (var spriteResolver in m_Selection)
            {
                if (spriteResolver == null)
                {
                    needUpdate = true;
                    break;
                }
            }

            UpdateSelection(needUpdate);
        }

        void OnPlayModeStateChanged(PlayModeStateChange newState)
        {
            if (newState is PlayModeStateChange.EnteredEditMode)
                UpdateSelection(true);
        }

        void OnAttachToPanel(AttachToPanelEvent evt)
        {
            var element = (SpriteSwapVisualElement)evt.target;
            if (element != null)
            {
                m_MainVisualElement = element;
                m_MainVisualElement.parent.RegisterCallback<GeometryChangedEvent>(OnParentGeometryChanged);
                if (collapsed || isInToolbar)
                {
                    m_MainVisualElement.style.width = m_MainVisualElement.style.maxWidth = Settings.preferredWidth;
                    m_MainVisualElement.style.height = m_MainVisualElement.style.maxHeight = Settings.preferredHeight;
                }

                UpdateResolverList();
            }
        }

        void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            var element = (SpriteSwapVisualElement)evt.currentTarget;
            if (element == m_MainVisualElement)
                m_MainVisualElement = null;
        }

        void OnParentGeometryChanged(GeometryChangedEvent evt)
        {
            if (m_MainVisualElement == null)
                return;

            if (!isInToolbar && !collapsed)
            {
                Settings.preferredWidth = evt.newRect.width;
                Settings.preferredHeight = evt.newRect.height;
            }

            m_MainVisualElement.style.width = evt.newRect.width;
        }

        void OnFilterToggled(bool filter)
        {
            if (Settings.filter == filter)
                return;

            Settings.filter = filter;

            if (!isViewInitialized)
                return;

            UpdateResolverList();
        }

        void OnLockToggled(bool locked)
        {
            if (Settings.locked == locked)
                return;

            Settings.locked = locked;
            if (!locked)
                UpdateSelection();
        }

        void OnChangeThumbnailSize(float newSize)
        {
            if (Math.Abs(Settings.thumbnailSize - newSize) < 0.01f)
                return;

            Settings.thumbnailSize = newSize;

            RefreshResolverList();
        }

        void OnResetThumbnailSize()
        {
            if (Math.Abs(Settings.thumbnailSize - Settings.defaultThumbnailSize) < 0.01f)
                return;

            Settings.thumbnailSize = Settings.defaultThumbnailSize;

            RefreshResolverList();
        }

        void UpdateSelection(bool force = false)
        {
            if (Settings.locked && !force)
                return;

            var gameObjects = Selection.gameObjects.Where(go => go.activeInHierarchy).ToArray();
            m_Selection = GetSelectedSpriteResolvers(gameObjects);

            UpdateResolverList();
        }

        void UpdateResolverList()
        {
            if (!isViewInitialized)
                return;

            var filtered = false;
            var filteredSelection = Settings.filter ? FilterSelection(out filtered) : selection;
            m_MainVisualElement.SetSpriteResolvers(filteredSelection);
            m_MainVisualElement.SetFiltered(filtered);
        }

        void RefreshResolverList()
        {
            if (!isViewInitialized)
                return;

            m_MainVisualElement.RefreshSpriteResolvers();
        }

        SpriteResolver[] FilterSelection(out bool filtered)
        {
            filtered = false;
            var filteredSelection = new List<SpriteResolver>();
            if (selection != null)
            {
                for (var i = 0; i < selection.Length; i++)
                {
                    var spriteResolver = selection[i];
                    var spriteLibrary = spriteResolver.spriteLibrary;
                    if (spriteLibrary == null)
                    {
                        filtered = true;
                        continue;
                    }

                    var selectedCategory = spriteResolver.GetCategory();
                    if (string.IsNullOrEmpty(selectedCategory))
                    {
                        filtered = true;
                        continue;
                    }

                    var labelNames = spriteLibrary.GetEntryNames(selectedCategory);
                    if (labelNames != null && labelNames.Count() > 1)
                        filteredSelection.Add(spriteResolver);
                    else
                        filtered = true;
                }
            }

            return filteredSelection.ToArray();
        }

        static SpriteResolver[] GetSelectedSpriteResolvers(GameObject[] selectedGameObjects)
        {
            var spriteResolvers = new HashSet<SpriteResolver>();
            if (selectedGameObjects != null)
            {
                for (var o = 0; o < selectedGameObjects.Length; o++)
                {
                    var gameObject = selectedGameObjects[o];
                    if (gameObject == null || !gameObject.activeSelf)
                        continue;

                    var children = gameObject.GetComponentsInChildren<SpriteResolver>();
                    for (var c = 0; c < children.Length; c++)
                    {
                        var spriteResolver = children[c];
                        spriteResolvers.Add(spriteResolver);
                    }
                }
            }

            return spriteResolvers.ToArray();
        }
    }
}
