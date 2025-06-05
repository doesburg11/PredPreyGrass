using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.U2D.Animation;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class CategoriesTab : VisualElement
    {
#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<CategoriesTab, UxmlTraits> { }
#endif

        ViewEvents m_ViewEvents;

        ScrollView m_CategoryListsScrollContainer;
        VisualElement m_LocalListContainer;
        VisualElement m_InheritedListContainer;
        VisualElement m_CategoryListsContainer;
        Foldout m_LocalFoldout;
        Foldout m_InheritedFoldout;
        Button m_AddButton;
        Label m_InfoLabel;

        RenamableListView m_LocalListView;
        ListView m_InheritedListView;
        List<string> m_SelectedCategories = new();

        List<CategoryData> m_Categories = new();
        List<CategoryData> m_LocalCategoriesData = new();
        List<CategoryData> m_InheritedCategoriesData = new();

        const string k_LabelContainerName = SpriteLibraryEditorWindow.editorWindowClassName + "__category-label-container";
        const string k_ListContainerClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__category-list-container";
        const string k_ListTextClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__category-list-text";
        const string k_ListItemClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__category-list-item";

        const int k_FoldoutHeight = 25;
        const int k_ListItemHeight = 20;

        bool m_LibrarySelected;
        bool m_IsFiltered;

        bool CanDragStart() => m_LibrarySelected;

        bool CanModifyCategories() => m_LibrarySelected && !m_IsFiltered;

        public void BindElements(ControllerEvents controllerEvents, ViewEvents viewEvents)
        {
            m_ViewEvents = viewEvents;

            var tabHeaderLabel = this.Q(SpriteLibraryEditorWindow.tabHeaderName).Q<Label>();
            tabHeaderLabel.text = "Categories";
            tabHeaderLabel.tooltip = TextContent.spriteLibraryCategoriesTooltip;

            var containerOverlay = new VisualElement { pickingMode = PickingMode.Ignore };
            Add(containerOverlay);
            containerOverlay.StretchToParentSize();

            this.AddManipulator(new ContextualMenuManipulator(ContextualManipulatorAddActions));

            this.AddManipulator(new DragAndDropManipulator(containerOverlay, CanDragStart, (spritesData, alt) =>
            {
                m_ViewEvents.onAddDataToCategories?.Invoke(spritesData, alt, null);
            }));

            controllerEvents.onModifiedCategories.AddListener(OnModifiedCategories);
            controllerEvents.onSelectedCategories.AddListener(SetSelection);
            controllerEvents.onSelectedLibrary.AddListener(OnLibraryAssetSelected);

            m_CategoryListsScrollContainer = this.Q<ScrollView>("CategoryListsScrollView");
            m_CategoryListsScrollContainer.pickingMode = PickingMode.Ignore;
            m_CategoryListsScrollContainer.Q<VisualElement>("unity-content-and-vertical-scroll-container").pickingMode = PickingMode.Ignore;
            m_CategoryListsScrollContainer.contentContainer.pickingMode = PickingMode.Ignore;

            m_CategoryListsContainer = new VisualElement { name = "CategoryListsContainer", pickingMode = PickingMode.Ignore };
            m_CategoryListsScrollContainer.Add(m_CategoryListsContainer);

            SetupInheritedList();
            SetupLocalList();
            SetupUI();
        }

        void SetupLocalList()
        {
            m_LocalListContainer = new VisualElement { name = "Local" };
            m_LocalListContainer.AddToClassList(k_ListContainerClassName);

            m_LocalListView = new RenamableListView
            {
                CanRenameAtIndex = CanRenameAtId,
                selectionType = SelectionType.Multiple,
                fixedItemHeight = k_ListItemHeight,
                reorderable = true,
                name = "LocalListView"
            };
            m_LocalListView.RegisterCallback<DragPerformEvent>(ReorderCategories);
            m_LocalListView.SetSourceItems(m_LocalCategoriesData);
            m_LocalListView.makeItem += OnMakeLocalItem;
            m_LocalListView.bindItem += OnBindLocalItem;
            m_LocalListView.unbindItem += OnUnbindItem;
            m_LocalListView.onRename += (i, newName) => OnCategoryRenamed(m_LocalListView, i, newName);
            m_LocalListView.selectionChanged += OnSelectionChanged;
            m_LocalListView.Rebuild();
            m_LocalListContainer.Add(m_LocalListView);
        }

        void ReorderCategories(DragPerformEvent dragPerformEvent)
        {
            if (!CanModifyCategories())
                return;

            var inherited = m_InheritedCategoriesData.Select(cat => cat.name);
            var local = m_LocalListView.itemsSource.Cast<CategoryData>().Select(cat => cat.name);
            m_ViewEvents.onReorderCategories.Invoke(inherited.Concat(local).ToList());
        }

        void OnBindLocalItem(VisualElement e, int i)
        {
            var category = m_LocalCategoriesData[i];
            var categoryName = category.name;

            var label = e.Q<Label>();

            label.text = categoryName;

            var useText = i == m_LocalListView.renamingIndex;
            label.style.display = useText ? DisplayStyle.None : DisplayStyle.Flex;

            var text = e.Q<TextField>();
            if (text != null)
                e.Remove(text);
            if (useText)
            {
                text = new TextField { name = IRenamableCollection.textFieldElementName, value = categoryName, selectAllOnMouseUp = false };
                text.AddToClassList(k_ListTextClassName);
                e.Insert(2, text);
                text.Focus();
                text.RegisterCallback<FocusOutEvent>(OnTextFocusOut);
            }

            var overlay = e.Q(className: DragAndDropManipulator.overlayClassName);
            overlay.userData = categoryName;
        }

        void OnTextFocusOut(FocusOutEvent evt)
        {
            evt.StopPropagation();
            m_LocalListView.EndRename();
        }

        VisualElement OnMakeLocalItem()
        {
            var e = new VisualElement { name = "LocalItemParent" };

            var label = new Label { name = IRenamableCollection.labelElementName, pickingMode = PickingMode.Ignore, style = { display = DisplayStyle.None } };

            var overlay = new VisualElement { pickingMode = PickingMode.Ignore };
            e.Add(overlay);
            overlay.StretchToParentSize();

            label.AddToClassList(k_ListTextClassName);
            e.AddToClassList(k_ListItemClassName);

            e.Add(label);

            e.AddManipulator(new DragAndDropManipulator(overlay, CanDragStart, (spritesData, alt) =>
            {
                m_ViewEvents.onAddDataToCategories?.Invoke(spritesData, alt, overlay.userData.ToString());
            }));

            return e;
        }

        void SetupInheritedList()
        {
            m_InheritedListContainer = new VisualElement { name = "LocalListContainer" };
            m_InheritedListContainer.AddToClassList(k_ListContainerClassName);

            m_InheritedListView = new ListView
            {
                selectionType = SelectionType.Multiple,
                itemsSource = m_InheritedCategoriesData,
                fixedItemHeight = k_ListItemHeight,
                name = "InheritedListView"
            };
            m_InheritedListView.bindItem += OnBindInheritedItem;
            m_InheritedListView.makeItem += OnMakeInheritedItem;
            m_InheritedListView.unbindItem += OnUnbindItem;
            m_InheritedListView.selectionChanged += OnSelectionChanged;
            m_InheritedListView.Rebuild();
            m_InheritedListContainer.Add(m_InheritedListView);
        }

        VisualElement OnMakeInheritedItem()
        {
            var e = new VisualElement { name = "InheritedItemParent" };

            var label = new Label() { pickingMode = PickingMode.Ignore };
            label.AddToClassList(k_ListTextClassName);
            e.AddToClassList(k_ListItemClassName);

            var overlay = new VisualElement { name = DragAndDropManipulator.overlayClassName, pickingMode = PickingMode.Ignore };
            e.Add(overlay);
            overlay.StretchToParentSize();

            e.Add(label);

            e.AddManipulator(new DragAndDropManipulator(overlay, CanDragStart, (spritesData, alt) =>
            {
                m_ViewEvents.onAddDataToCategories?.Invoke(spritesData, alt, overlay.userData.ToString());
            }));

            return e;
        }

        void OnBindInheritedItem(VisualElement e, int i)
        {
            var label = e.Q<Label>();
            var category = m_InheritedCategoriesData[i];
            label.text = category.name;

            e.EnableInClassList(SpriteLibraryEditorWindow.overrideClassName, category.isOverride);

            var overlay = e.Q(className: DragAndDropManipulator.overlayClassName);
            overlay.userData = category.name;
        }

        static void OnUnbindItem(VisualElement e, int i)
        {
            var overlay = e.Q(className: DragAndDropManipulator.overlayClassName);
            overlay.userData = null;

            e.RemoveFromClassList(SpriteLibraryEditorWindow.overrideClassName);
            e.Blur();
        }

        void SetupUI()
        {
            m_AddButton = this.Q<Button>();
            m_AddButton.style.backgroundImage = Background.FromTexture2D(EditorGUIUtility.IconContent("CreateAddNew").image as Texture2D);
            m_AddButton.clicked += CreateNewCategory;
            m_AddButton.tooltip = TextContent.spriteLibraryAddCategoryTooltip;
            m_AddButton.focusable = false;
            m_AddButton.SetEnabled(false);

            m_LocalFoldout = new Foldout { name = "LocalFoldout", text = "Local" };
            m_LocalFoldout.Q<Label>().AddToClassList(k_LabelContainerName);
            m_LocalFoldout.contentContainer.Add(m_LocalListContainer);
            m_LocalFoldout.RegisterValueChangedCallback(_ => UpdateFoldouts());
            m_LocalFoldout.Q<Label>().tooltip = TextContent.spriteLibraryLocalCategoryTooltip;

            m_InheritedFoldout = new Foldout { name = "InheritedFoldout", text = "Inherited" };
            m_InheritedFoldout.Q<Label>().AddToClassList(k_LabelContainerName);
            m_InheritedFoldout.contentContainer.Add(m_InheritedListContainer);
            m_InheritedFoldout.RegisterValueChangedCallback(_ => UpdateFoldouts());
            m_InheritedFoldout.Q<Label>().tooltip = TextContent.spriteLibraryInheritedCategoryTooltip;

            m_CategoryListsContainer.Add(m_InheritedFoldout);
            m_CategoryListsContainer.Add(m_LocalFoldout);

            m_InfoLabel = new Label
            {
                name = "InfoLabel",
                text = TextContent.spriteCategoryColumnEmpty,
                style = { display = DisplayStyle.None },
                pickingMode = PickingMode.Ignore
            };
            m_InfoLabel.AddToClassList(SpriteLibraryEditorWindow.infoLabelClassName);
            Add(m_InfoLabel);
            m_InfoLabel.StretchToParentSize();

            RegisterCallback<ValidateCommandEvent>(evt =>
            {
                if (evt.commandName is SpriteLibraryEditorWindow.deleteCommandName or SpriteLibraryEditorWindow.softDeleteCommandName or SpriteLibraryEditorWindow.renameCommandName)
                    evt.StopPropagation();
            });
            RegisterCallback<ExecuteCommandEvent>(evt =>
            {
                if (evt.commandName is SpriteLibraryEditorWindow.deleteCommandName or SpriteLibraryEditorWindow.softDeleteCommandName)
                {
                    evt.StopPropagation();
                    DeleteSelected();
                }
                else if (evt.commandName == SpriteLibraryEditorWindow.renameCommandName)
                {
                    if (m_LocalListView.selectedIndices != null && m_LocalListView.selectedIndices.Count() == 1)
                    {
                        evt.StopPropagation();
                        RenameSelected();
                    }
                }
            });
        }

        void OnModifiedCategories(List<CategoryData> categories, bool filtered)
        {
            m_LocalListView.EndRename();

            m_Categories = categories;
            m_IsFiltered = filtered;

            m_LocalListView.reorderable = !m_IsFiltered;

            m_LocalCategoriesData = new List<CategoryData>();
            m_InheritedCategoriesData = new List<CategoryData>();
            foreach (var categoryData in m_Categories)
            {
                if (categoryData.fromMain)
                    m_InheritedCategoriesData.Add(categoryData);
                else
                    m_LocalCategoriesData.Add(categoryData);
            }

            m_LocalListView.SetSourceItems(m_LocalCategoriesData);
            m_InheritedListView.itemsSource = m_InheritedCategoriesData;
            m_InheritedListView.RefreshItems();

            UpdateFoldouts();
            SetSelection(m_SelectedCategories);

            PostRefreshUI();
        }

        void CreateNewCategory()
        {
            if (!CanModifyCategories())
                return;

            m_LocalFoldout.SetValueWithoutNotify(true);
            m_ViewEvents.onCreateNewCategory?.Invoke(null, null);

            m_LocalListView.SetSelectionWithoutNotify(new List<int> { m_LocalListView.itemsSource.Count - 1 });
            m_CategoryListsScrollContainer.scrollOffset = new Vector2(0f, m_CategoryListsScrollContainer.contentContainer.resolvedStyle.height);

            RenameSelected();
        }

        void RenameSelected()
        {
            m_LocalListView.StartRename();
        }

        void DeleteSelected()
        {
            m_ViewEvents.onDeleteCategories?.Invoke();
        }

        void OnCategoryRenamed(RenamableListView listView, int i, string newName)
        {
            if (i < 0 || string.IsNullOrEmpty(newName))
                return;

            var category = (CategoryData)listView.itemsSource[i];

            if (category == null || string.IsNullOrEmpty(newName) || newName == category.name)
                return;

            m_ViewEvents.onRenameCategory?.Invoke(newName);
        }

        void OnSelectionChanged(IEnumerable<object> selection)
        {
            var newSelection = selection.Cast<CategoryData>().Select(d => d.name).ToList();
            m_ViewEvents.onSelectCategories?.Invoke(newSelection);
        }

        void UpdateFoldouts()
        {
            var localCount = m_LocalCategoriesData.Count;
            var inheritedCount = m_InheritedCategoriesData.Count;

            var displayInherited = m_Categories.Any(cat => cat.fromMain);
            m_InheritedFoldout.style.display = displayInherited ? DisplayStyle.Flex : DisplayStyle.None;

            var inheritedHeight = m_InheritedFoldout.value && displayInherited ? inheritedCount * k_ListItemHeight + k_FoldoutHeight : k_FoldoutHeight;
            m_InheritedFoldout.style.minHeight = m_InheritedFoldout.style.height = inheritedHeight;

            var localHeight = m_LocalFoldout.value ? localCount * k_ListItemHeight + k_FoldoutHeight : k_FoldoutHeight;
            m_LocalFoldout.style.minHeight = m_LocalFoldout.style.height = localHeight;

            var offset = displayInherited ? k_FoldoutHeight : 0;
            m_CategoryListsContainer.style.minHeight = m_CategoryListsContainer.style.height = localHeight + inheritedHeight + offset;
        }

        void SetSelection(List<string> categories)
        {
            m_SelectedCategories = categories;

            var localSelection = new List<int>();
            var inheritedSelection = new List<int>();
            foreach (var category in categories.Select(categoryName => m_Categories.FirstOrDefault(c => c.name == categoryName)))
            {
                var localIndex = m_LocalCategoriesData.IndexOf(category);
                var inheritedIndex = m_InheritedCategoriesData.IndexOf(category);
                if (localIndex >= 0)
                    localSelection.Add(localIndex);

                if (inheritedIndex >= 0)
                    inheritedSelection.Add(inheritedIndex);
            }

            if (!m_LocalListView.selectedIndices.SequenceEqual(localSelection))
                m_LocalListView.SetSelectionWithoutNotify(localSelection);
            if (!m_InheritedListView.selectedIndices.SequenceEqual(inheritedSelection))
                m_InheritedListView.SetSelectionWithoutNotify(inheritedSelection);
        }

        void OnLibraryAssetSelected(SpriteLibraryAsset libraryAsset)
        {
            m_LibrarySelected = libraryAsset != null;
        }

        void PostRefreshUI()
        {
            var show = m_Categories.Count == 0 && !m_IsFiltered;
            m_InfoLabel.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
            m_CategoryListsScrollContainer.style.display = show ? DisplayStyle.None : DisplayStyle.Flex;

            m_AddButton.SetEnabled(CanModifyCategories());
            m_AddButton.tooltip = m_IsFiltered ? TextContent.spriteLibraryAddCategoryTooltipNotAvailable : TextContent.spriteLibraryAddCategoryTooltip;
        }

        bool CanRenameAtId(int i)
        {
            var selectionCount = m_LocalListView.selectedIndices?.Count() ?? 0;
            return selectionCount == 1;
        }

        void ContextualManipulatorAddActions(ContextualMenuPopulateEvent evt)
        {
            if (!CanModifyCategories())
                return;

            evt.menu.AppendAction(TextContent.spriteLibraryCreateCategory, _ => CreateNewCategory());

            var selectionCount = m_LocalListView.selectedIndices?.Count() ?? 0;
            var canRenameStatus = selectionCount == 1 ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Disabled;
            evt.menu.AppendAction(TextContent.spriteLibraryRenameCategory, _ => RenameSelected(), canRenameStatus);
            var canDeleteCategoryStatus = selectionCount > 0 ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Disabled;
            evt.menu.AppendAction(TextContent.spriteLibraryDeleteCategories, _ => DeleteSelected(), canDeleteCategoryStatus);
        }
    }
}