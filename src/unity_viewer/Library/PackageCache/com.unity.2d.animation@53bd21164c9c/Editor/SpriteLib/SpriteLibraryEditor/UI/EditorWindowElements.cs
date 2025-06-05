using System;
using System.Collections.Generic;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.U2D.Animation;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class EditorMainWindow : VisualElement
    {
#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<EditorMainWindow, UxmlTraits> { }
#endif

        ViewEvents m_ViewEvents;

        CategoriesTab m_CategoriesTab;
        LabelsTab m_LabelsTab;
        TwoPaneSplitView m_SplitView;

        const float k_MinPaneWidth = 150f;

        internal void BindElements(ControllerEvents controllerEvents, ViewEvents viewEvents)
        {
            m_ViewEvents = viewEvents;

            m_CategoriesTab = this.Q<CategoriesTab>();
            m_CategoriesTab.BindElements(controllerEvents, viewEvents);
            m_LabelsTab = this.Q<LabelsTab>();
            m_LabelsTab.BindElements(controllerEvents, viewEvents);

            RegisterCallback<GeometryChangedEvent>(GeometryInitialized);
        }

        void GeometryInitialized(GeometryChangedEvent evt)
        {
            m_SplitView = this.Q<TwoPaneSplitView>();
            m_SplitView.fixedPaneInitialDimension = m_SplitView.resolvedStyle.width * SpriteLibraryEditorWindow.Settings.splitPaneSize;
            m_SplitView.fixedPane.style.minWidth = m_SplitView.flexedPane.style.minWidth = k_MinPaneWidth;
            m_SplitView.flexedPane.RegisterCallback<GeometryChangedEvent>(OnSplitViewResize);

            UnregisterCallback<GeometryChangedEvent>(GeometryInitialized);
        }

        void OnSplitViewResize(GeometryChangedEvent evt)
        {
            m_ViewEvents.onMainUISplitPaneSizeChanged?.Invoke(1f - evt.newRect.width / m_SplitView.resolvedStyle.width);
        }
    }

#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class EditorBottomToolbar : VisualElement
    {
#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<EditorBottomToolbar, UxmlTraits> { }
#endif

        public void BindElements(ControllerEvents controllerEvents, ViewEvents viewEvents)
        {
            var listButton = this.Q<Button>("ListButton");
            listButton.clicked += () => viewEvents.onViewTypeUpdate?.Invoke(ViewType.List);

            var gridButton = this.Q<Button>("GridButton");
            gridButton.clicked += () => viewEvents.onViewTypeUpdate?.Invoke(ViewType.Grid);

            var slider = this.Q<Slider>("SizeSlider");
            slider.RegisterValueChangedCallback(evt => viewEvents.onViewSizeUpdate?.Invoke(evt.newValue));
            controllerEvents.onViewChanged.AddListener(viewData => slider.SetValueWithoutNotify(viewData.absoluteSize));

            viewEvents.onViewSizeUpdate?.Invoke(SpriteLibraryEditorWindow.Settings.viewSize);
        }
    }

#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class EditorTopToolbar : VisualElement
    {
#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<EditorTopToolbar, UxmlTraits> { }
#endif

        ControllerEvents m_ControllerEvents;
        ViewEvents m_ViewEvents;

        ObjectField m_ObjectField;
        ToolbarBreadcrumbs m_Breadcrumbs;
        ToolbarPopupSearchField m_SearchField;
        Button m_SaveButton;
        Button m_RevertButton;
        Toggle m_AutoSaveToggle;

        SearchType m_CurrentSearchColumn;

        SpriteLibraryAsset m_SelectedAsset;

        public void BindElements(ControllerEvents controllerEvents, ViewEvents viewEvents)
        {
            m_ControllerEvents = controllerEvents;
            m_ViewEvents = viewEvents;

            m_ControllerEvents.onMainLibraryChanged.AddListener(OnMainAssetChanged);
            m_ControllerEvents.onSelectedLibrary.AddListener(OnSelectedAssetChanged);
            m_ControllerEvents.onLibraryDataChanged.AddListener(SetSaveRevertEnabled);

            this.Q<Image>("SpriteLibraryIcon").image = EditorIconUtility.LoadIconResource("Animation.SpriteLibrary", "ComponentIcons", "ComponentIcons");

            var searchController = this.Q("SearchController");
            searchController.style.flexGrow = 1;
            searchController.style.flexDirection = FlexDirection.RowReverse;

            m_SearchField = new ToolbarPopupSearchField();
            m_SearchField.style.flexBasis = 150f;
            m_SearchField.style.flexGrow = 0;
            m_SearchField.Q("unity-text-input").style.paddingRight = 0;

            m_SearchField.menu.AppendAction(
                EditorGUIUtility.TrTextContent("Category and Label").text,
                _ => SetSearchType(SearchType.CategoryAndLabel),
                _ => m_CurrentSearchColumn == SearchType.CategoryAndLabel ? DropdownMenuAction.Status.Checked : DropdownMenuAction.Status.Normal);

            m_SearchField.menu.AppendAction(
                EditorGUIUtility.TrTextContent("Category").text,
                _ => SetSearchType(SearchType.Category),
                _ => m_CurrentSearchColumn == SearchType.Category ? DropdownMenuAction.Status.Checked : DropdownMenuAction.Status.Normal);

            m_SearchField.menu.AppendAction(
                EditorGUIUtility.TrTextContent("Label").text,
                _ => SetSearchType(SearchType.Label),
                _ => m_CurrentSearchColumn == SearchType.Label ? DropdownMenuAction.Status.Checked : DropdownMenuAction.Status.Normal);

            m_SearchField.RegisterValueChangedCallback(evt => viewEvents.onSelectedFilter?.Invoke(evt.newValue));

            searchController.Add(m_SearchField);

            m_ObjectField = this.Q<ObjectField>();
            m_ObjectField.allowSceneObjects = false;
            m_ObjectField.focusable = false;
            m_ObjectField.objectType = typeof(SpriteLibraryAsset);
            m_ObjectField.tooltip = TextContent.spriteLibraryMainLibraryTooltip;

            m_ObjectField.labelElement.style.minWidth = 50;
            m_ObjectField.labelElement.style.maxWidth = m_ObjectField.labelElement.style.width = 103;
            m_ObjectField.RegisterValueChangedCallback(OnSelectedMainAsset);

            m_Breadcrumbs = this.Q<ToolbarBreadcrumbs>();

            m_AutoSaveToggle = this.Q<Toggle>();
            m_AutoSaveToggle.RegisterValueChangedCallback(OnSelectAutoSave);
            m_SaveButton = this.Q<Button>("SaveButton");
            m_SaveButton.clicked += () => m_ViewEvents.onSave?.Invoke();
            m_RevertButton = this.Q<Button>("RevertButton");
            m_RevertButton.clicked += () => m_ViewEvents.onRevert?.Invoke();
            AutoSaveChanged(SpriteLibraryEditorWindow.Settings.autoSave);
        }

        void OnSelectAutoSave(ChangeEvent<bool> evt)
        {
            var autoSave = evt.newValue;
            AutoSaveChanged(autoSave);

            m_ViewEvents.onToggleAutoSave?.Invoke(autoSave);
        }

        void SetSearchType(SearchType searchType)
        {
            m_CurrentSearchColumn = searchType;
            m_ViewEvents.onSelectedFilterType?.Invoke(searchType);
        }

        void AutoSaveChanged(bool autoSave)
        {
            m_AutoSaveToggle.SetValueWithoutNotify(autoSave);
            m_SaveButton.style.display = autoSave ? DisplayStyle.None : DisplayStyle.Flex;
            m_RevertButton.style.display = autoSave ? DisplayStyle.None : DisplayStyle.Flex;
        }

        void OnSelectedMainAsset(ChangeEvent<UnityEngine.Object> evt)
        {
            var libraryAsset = evt.newValue as SpriteLibraryAsset;
            m_ViewEvents.onSetMainAsset?.Invoke(libraryAsset);
        }

        void OnSelectedAssetChanged(SpriteLibraryAsset selectedAsset)
        {
            m_SelectedAsset = selectedAsset;

            var breadcrumbList = SpriteLibrarySourceAssetImporter.GetAssetParentChain(selectedAsset);
            UpdateBreadCrumbs(breadcrumbList);

            m_ObjectField.SetValueWithoutNotify(SpriteLibrarySourceAssetImporter.GetAssetParent(selectedAsset));

            SetSaveRevertEnabled(false);
        }

        void OnMainAssetChanged(SpriteLibraryAsset mainAsset)
        {
            var breadcrumbList = SpriteLibrarySourceAssetImporter.GetAssetParentChain(mainAsset);
            if (mainAsset != null)
                breadcrumbList.Insert(0, mainAsset);
            UpdateBreadCrumbs(breadcrumbList);

            m_ObjectField.SetValueWithoutNotify(mainAsset);
        }

        void UpdateBreadCrumbs(List<SpriteLibraryAsset> breadcrumbList)
        {
            m_Breadcrumbs.Clear();

            if (breadcrumbList.Count > 0)
            {
                for (var i = breadcrumbList.Count; i-- > 0;)
                {
                    var asset = breadcrumbList[i];
                    m_Breadcrumbs.PushItem(asset.name, () => OnBreadcrumbClicked(asset));
                }
            }

            if (m_SelectedAsset != null)
                m_Breadcrumbs.PushItem(m_SelectedAsset.name, () => OnBreadcrumbClicked(m_SelectedAsset));
        }

        static void OnBreadcrumbClicked(SpriteLibraryAsset asset)
        {
            Selection.objects = new UnityEngine.Object[] { asset };
            EditorGUIUtility.PingObject(asset);
        }

        void SetSaveRevertEnabled(bool toggleEnabled)
        {
            m_SaveButton.SetEnabled(toggleEnabled);
            m_RevertButton.SetEnabled(toggleEnabled);
        }
    }

#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class EditorTabHeader : VisualElement
    {
#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<EditorTabHeader, UxmlTraits> { }
#endif
    }
}
