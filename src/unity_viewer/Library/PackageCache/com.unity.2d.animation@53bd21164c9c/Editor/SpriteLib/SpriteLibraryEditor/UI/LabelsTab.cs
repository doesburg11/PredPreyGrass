using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class LabelsTab : VisualElement
    {
#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<LabelsTab, UxmlTraits> { }
#endif

        ViewEvents m_ViewEvents;

        VisualElement m_LabelsContainer;
        Button m_AddButton;
        Label m_InfoLabel;

        IRenamableCollection m_ItemsCollection;
        List<string> m_SelectedCategories = new();
        List<string> m_SelectedLabels = new();

        List<LabelData> m_LabelData = new();

        ViewType m_CurrentViewType;
        float m_CurrentViewSize;
        float m_AdjustedViewSize;

        const string k_ListLabelClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__label-list-label";
        const string k_GridLabelClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__label-grid-label";
        const string k_ListTextClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__label-list-text";
        const string k_GridTextClassName = SpriteLibraryEditorWindow.editorWindowClassName + "__label-grid-text";

        const float k_MinListSize = 20f;
        const float k_MaxListSize = 30f;
        const float k_MinGridSize = 80f;
        const float k_MaxGridSize = 200f;

        bool m_IsFiltered;

        bool IsTabActive() => m_SelectedCategories != null && m_SelectedCategories.Count == 1;

        bool CanModifyLabels() => IsTabActive() && !m_IsFiltered;

        public void BindElements(ControllerEvents controllerEvents, ViewEvents viewEvents)
        {
            m_ViewEvents = viewEvents;

            m_CurrentViewType = SpriteLibraryEditorWindow.Settings.viewType;
            m_CurrentViewSize = SpriteLibraryEditorWindow.Settings.viewSize;

            m_LabelsContainer = this.Q<VisualElement>("LabelsContainer");
            m_LabelsContainer.pickingMode = PickingMode.Ignore;

            var tabHeaderLabel = this.Q(SpriteLibraryEditorWindow.tabHeaderName).Q<Label>();
            tabHeaderLabel.text = "Labels";
            tabHeaderLabel.tooltip = TextContent.spriteLibraryLabelsTooltip;

            var container = new VisualElement { pickingMode = PickingMode.Ignore };
            Add(container);
            container.StretchToParentSize();

            this.AddManipulator(new ContextualMenuManipulator(ContextualManipulatorAddActions));

            this.AddManipulator(new DragAndDropManipulator(container, IsTabActive, (spritesData, alt) =>
            {
                m_ViewEvents.onAddDataToLabels?.Invoke(spritesData, alt, null);
            }));

            controllerEvents.onModifiedLabels.AddListener(OnModifiedLabels);
            controllerEvents.onSelectedCategories.AddListener(OnSelectedCategories);
            controllerEvents.onViewChanged.AddListener(OnViewChanged);
            controllerEvents.onSelectedLabels.AddListener(SetSelection);

            SetupUI();

            SetViewType(m_CurrentViewType);

            Debug.Assert(m_ItemsCollection != null);

            m_AdjustedViewSize = GetAdjustedViewSize(m_CurrentViewType, m_CurrentViewSize);
            m_ItemsCollection.SetElementSize(m_AdjustedViewSize);
        }

        void OnViewChanged(ViewData viewData)
        {
            if (viewData.viewType != m_CurrentViewType)
            {
                SetViewType(viewData.viewType);
                m_ItemsCollection.SetSourceItems(m_LabelData);
            }

            m_CurrentViewSize = viewData.relativeSize;
            m_AdjustedViewSize = GetAdjustedViewSize(m_CurrentViewType, m_CurrentViewSize);
            m_ItemsCollection.SetElementSize(m_AdjustedViewSize);

            SetSelection(m_SelectedLabels);

            PostRefreshUI();
        }

        void OnModifiedLabels(List<LabelData> labels, bool filter)
        {
            m_ItemsCollection.EndRename();

            m_LabelData = labels;
            m_IsFiltered = filter;

            m_ItemsCollection.SetSourceItems(m_LabelData);

            PostRefreshUI();
        }

        void OnSelectedCategories(List<string> categories)
        {
            m_SelectedCategories = categories;
            PostRefreshUI();
        }

        void SetupUI()
        {
            m_AddButton = this.Q<Button>();
            m_AddButton.style.backgroundImage = Background.FromTexture2D(EditorGUIUtility.IconContent("CreateAddNew").image as Texture2D);
            m_AddButton.clicked += CreateNewLabel;
            m_AddButton.tooltip = TextContent.spriteLibraryAddLabelTooltip;
            m_AddButton.focusable = false;
            m_AddButton.SetEnabled(false);

            m_InfoLabel = new Label
            {
                name = "InfoLabel",
                text = TextContent.spriteCategoryNoSelection,
                style = { display = DisplayStyle.None },
                pickingMode = PickingMode.Ignore
            };
            m_InfoLabel.AddToClassList(SpriteLibraryEditorWindow.infoLabelClassName);
            m_LabelsContainer.Add(m_InfoLabel);
            m_InfoLabel.StretchToParentSize();
        }

        void PostRefreshUI()
        {
            m_AddButton.SetEnabled(CanModifyLabels());
            m_AddButton.tooltip = m_IsFiltered ? TextContent.spriteLibraryAddLabelTooltipNotAvailable : TextContent.spriteLibraryAddLabelTooltip;

            if (m_CurrentViewType == ViewType.List)
                ((ListView)m_ItemsCollection).reorderable = !m_IsFiltered;

            UpdateShowInfoLabel();
        }

        void UpdateShowInfoLabel()
        {
            var show = false;
            if (m_SelectedCategories.Count == 1 && m_LabelData.Count == 0)
            {
                m_InfoLabel.text = TextContent.spriteLabelColumnEmpty;
                show = true;
            }
            else if (m_SelectedCategories.Count > 1)
            {
                m_InfoLabel.text = TextContent.spriteCategoryMultiSelect;
                show = true;
            }
            else if (m_SelectedCategories.Count == 0 && !m_IsFiltered)
            {
                m_InfoLabel.text = TextContent.spriteCategoryNoSelection;
                show = true;
            }

            m_InfoLabel.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
            ((VisualElement)m_ItemsCollection).style.display = show ? DisplayStyle.None : DisplayStyle.Flex;
        }

        void SetViewType(ViewType newView)
        {
            if (m_ItemsCollection != null)
                m_LabelsContainer.Remove(m_ItemsCollection as VisualElement);

            m_CurrentViewType = newView;

            if (newView == ViewType.List)
            {
                m_ItemsCollection = new RenamableListView
                    { reorderable = true, CanRenameAtIndex = CanRenameAtId, selectionType = SelectionType.Multiple };

                var renamableList = m_ItemsCollection as RenamableListView;
                Debug.Assert(renamableList != null);
                renamableList.makeItem += MakeListItem;
                renamableList.bindItem += BindListItem;
                renamableList.unbindItem += UnbindCollectionItem;
                renamableList.selectionChanged += OnSelectionChanged;
            }
            else
            {
                m_ItemsCollection = new RenamableGridView { CanRenameAtIndex = CanRenameAtId, selectionType = SelectionType.Multiple };
                var renamableGrid = m_ItemsCollection as RenamableGridView;
                Debug.Assert(renamableGrid != null);
                renamableGrid.makeItem += MakeGridItem;
                renamableGrid.bindItem += BindGridItem;
                renamableGrid.unbindItem += UnbindCollectionItem;
                renamableGrid.selectionChanged += OnSelectionChanged;
            }

            InitializeItemsCollection();
        }

        static void UnbindCollectionItem(VisualElement e, int i)
        {
            var overlay = e.Q(className: DragAndDropManipulator.overlayClassName);
            overlay.userData = null;
        }

        void InitializeItemsCollection()
        {
            var collectionVisualItem = (VisualElement)m_ItemsCollection;
            m_LabelsContainer.Add(collectionVisualItem);
            collectionVisualItem.StretchToParentSize();

            m_ItemsCollection.onRename += OnRenameLabel;
            m_ItemsCollection.onItemsReordered += OnItemsReordered;

            m_AdjustedViewSize = GetAdjustedViewSize(m_CurrentViewType, m_CurrentViewSize);
            m_ItemsCollection.SetElementSize(m_AdjustedViewSize);
            m_ItemsCollection.SetWidth(resolvedStyle.width);

            RegisterCallback<GeometryChangedEvent>(OnGeometryChanged);

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
                    var selectedId = m_SelectedLabels.Select(label => m_LabelData.FindIndex(l => l.name == label)).FirstOrDefault();
                    if (CanRenameAtId(selectedId))
                    {
                        evt.StopPropagation();
                        RenameSelected();
                    }
                }
            });

            SetSelection(m_SelectedLabels);
        }

        void OnGeometryChanged(GeometryChangedEvent evt)
        {
            var width = resolvedStyle.width;
            m_ItemsCollection.SetWidth(width);
        }

        void OnItemsReordered()
        {
            if (!CanModifyLabels())
                return;

            var inherited = m_LabelData.Where(l => l.fromMain);
            var toReorder = m_ItemsCollection.GetItemSource().Cast<LabelData>().Except(inherited);
            m_ViewEvents.onReorderLabels?.Invoke(inherited.Select(l => l.name).Concat(toReorder.Select(l => l.name)).ToList());
        }

        void OnSelectionChanged(IEnumerable<object> selection)
        {
            var newSelection = selection.Cast<LabelData>().Select(d => d.name).ToList();
            m_ViewEvents.onSelectLabels?.Invoke(newSelection);
        }

        void OnRenameLabel(int i, string newName)
        {
            if (!CanModifyLabels())
                return;

            if (i < 0 || i >= m_LabelData.Count || string.IsNullOrEmpty(newName))
                return;

            var label = m_LabelData[i];
            if (label == null || newName == label.name)
                return;

            m_ViewEvents.onRenameLabel?.Invoke(newName);
        }

        void SetSelection(List<string> labels)
        {
            m_SelectedLabels = labels;
            var selectedIndices = m_SelectedLabels.Select(label => m_LabelData.FindIndex(l => l.name == label));
            m_ItemsCollection.SetSelectionWithoutNotify(selectedIndices);
        }

        void CreateNewLabel()
        {
            m_ViewEvents.onCreateNewLabel.Invoke(null);

            var lastId = m_ItemsCollection.GetItemSource().Count - 1;
            m_ItemsCollection.SetSelectionWithoutNotify(new List<int> { lastId });

            m_ItemsCollection.ScrollToItem(lastId);

            RenameSelected();
        }

        void RenameSelected()
        {
            m_ItemsCollection.StartRename();
        }

        void DeleteSelected()
        {
            m_ViewEvents.onDeleteLabels?.Invoke();
        }

        void RevertLabelOverride(bool revertAll)
        {
            var labels = revertAll ? m_LabelData.Select(l => l.name).ToList() : m_SelectedLabels;
            m_ViewEvents.onRevertOverridenLabels?.Invoke(labels);
        }

        void SpriteReferenceChanged(ObjectField objectField)
        {
            var i = (int)objectField.userData;
            var label = m_LabelData[i].name;

            var sprite = objectField.value as Sprite;
            m_ViewEvents.onSetLabelSprite?.Invoke(label, sprite);
        }

        VisualElement MakeListItem()
        {
            var e = new VisualElement { name = "ListElementParent" };

            var overlay = new VisualElement { pickingMode = PickingMode.Ignore };
            e.Add(overlay);
            overlay.StretchToParentSize();

            const int spriteSizeMargin = 5;
            var spriteSizeAdjustedForMargin = m_AdjustedViewSize - spriteSizeMargin;
            var spriteSlot = new Image { name = "ListSpriteSlot", pickingMode = PickingMode.Ignore, style = { width = spriteSizeAdjustedForMargin, height = spriteSizeAdjustedForMargin } };
            e.Add(spriteSlot);

            var label = new Label { name = IRenamableCollection.labelElementName, pickingMode = PickingMode.Ignore };
            label.AddToClassList(k_ListLabelClassName);
            e.Add(label);

            var objField = new ObjectField { objectType = typeof(Sprite), name = "LabelSpriteObjectField", allowSceneObjects = false, focusable = false };
            objField.RegisterValueChangedCallback(_ => SpriteReferenceChanged(objField));
            e.Add(objField);

            e.AddManipulator(new DragAndDropManipulator(overlay, IsTabActive, (spritesData, alt) =>
            {
                m_ViewEvents.onAddDataToLabels?.Invoke(spritesData, alt, overlay.userData.ToString());
            }));

            return e;
        }

        void BindListItem(VisualElement e, int i)
        {
            var labelData = m_LabelData[i];

            var label = e.Q<Label>();
            label.text = label.tooltip = labelData.name;

            var image = e.Q<Image>();
            image.sprite = labelData.sprite;

            var useText = i == m_ItemsCollection.renamingIndex;
            label.style.display = useText ? DisplayStyle.None : DisplayStyle.Flex;
            var text = e.Q<TextField>();
            if (text != null)
            {
                e.Blur();
                e.Remove(text);
            }

            if (useText)
            {
                text = new TextField { name = IRenamableCollection.textFieldElementName, value = labelData.name, label = null, selectAllOnMouseUp = false };
                text.AddToClassList(k_ListTextClassName);
                e.Insert(2, text);
                text.Focus();

                text.RegisterCallback<FocusOutEvent>(OnTextFocusOut);
            }

            var objRef = e.Q<ObjectField>();
            objRef.SetValueWithoutNotify(labelData.sprite);
            objRef.userData = i;

            objRef.EnableInClassList(SpriteLibraryEditorWindow.overrideClassName, labelData.spriteOverride);

            e.RemoveFromClassList(SpriteLibraryEditorWindow.overrideClassName);
            if (labelData.categoryFromMain && !labelData.fromMain)
                e.AddToClassList(SpriteLibraryEditorWindow.overrideClassName);

            var overlay = e.Q(className: DragAndDropManipulator.overlayClassName);
            overlay.userData = labelData.name;
        }

        VisualElement MakeGridItem()
        {
            var e = new VisualElement { name = "GridElementParent" };

            var spriteSlot = new Image { name = "GridElementImage", pickingMode = PickingMode.Ignore };
            e.Add(spriteSlot);

            var overlay = new VisualElement { pickingMode = PickingMode.Ignore };
            spriteSlot.Add(overlay);
            overlay.StretchToParentSize();

            var label = new Label { name = IRenamableCollection.labelElementName, pickingMode = PickingMode.Ignore };
            label.AddToClassList(k_GridLabelClassName);
            e.Add(label);

            e.AddManipulator(new DragAndDropManipulator(overlay, IsTabActive, (spritesData, alt) =>
            {
                m_ViewEvents.onAddDataToLabels?.Invoke(spritesData, alt, overlay.userData.ToString());
            }));

            return e;
        }

        void BindGridItem(VisualElement e, int i)
        {
            var labelData = m_LabelData[i];

            var image = e.Q<Image>();
            image.sprite = labelData.sprite;

            var label = e.Q<Label>();
            label.text = label.tooltip = labelData.name;

            var useText = i == m_ItemsCollection.renamingIndex;
            label.style.display = useText ? DisplayStyle.None : DisplayStyle.Flex;

            var text = e.Q<TextField>();
            if (text != null)
            {
                text.Blur();
                e.Remove(text);
            }

            if (useText)
            {
                text = new TextField { name = IRenamableCollection.textFieldElementName, label = null, value = labelData.name, selectAllOnMouseUp = false };
                text.AddToClassList(k_GridTextClassName);
                e.Add(text);
                text.Focus();

                text.RegisterCallback<FocusOutEvent>(OnTextFocusOut);
            }

            if (labelData.categoryFromMain && !labelData.fromMain || labelData.spriteOverride)
                image.AddToClassList(SpriteLibraryEditorWindow.overrideClassName);
            else
                image.RemoveFromClassList(SpriteLibraryEditorWindow.overrideClassName);

            if (labelData.categoryFromMain && !labelData.fromMain)
                label.AddToClassList(SpriteLibraryEditorWindow.overrideClassName);
            else
                label.RemoveFromClassList(SpriteLibraryEditorWindow.overrideClassName);

            var overlay = e.Q(className: DragAndDropManipulator.overlayClassName);
            overlay.userData = labelData.name;
        }

        void OnTextFocusOut(FocusOutEvent evt)
        {
            evt.StopPropagation();
            m_ItemsCollection.EndRename();
        }

        static bool CanRevertLabel(LabelData labelData) => labelData.categoryFromMain && !labelData.fromMain || labelData.spriteOverride;

        bool CanModifyAtId(int i)
        {
            if (!CanModifyLabels())
                return false;

            if (m_LabelData == null || i < 0 || i >= m_LabelData.Count)
                return false;

            var label = m_LabelData[i];
            return label != null && !label.fromMain;
        }

        bool CanRenameAtId(int i)
        {
            return m_SelectedLabels.Count == 1 && CanModifyAtId(i);
        }

        void ContextualManipulatorAddActions(ContextualMenuPopulateEvent evt)
        {
            if (CanModifyLabels())
            {
                evt.menu.AppendAction(TextContent.spriteLibraryCreateLabel, _ => CreateNewLabel());

                var selectedId = m_SelectedLabels.Select(label => m_LabelData.FindIndex(l => l.name == label)).FirstOrDefault();
                var canModifyAt = CanModifyAtId(selectedId);
                var canModifyStatus = canModifyAt ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Disabled;
                var canRename = CanRenameAtId(selectedId);
                var canRenameStatus = canRename ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Disabled;
                evt.menu.AppendAction(TextContent.spriteLibraryRenameLabel, _ => RenameSelected(), _ => canRenameStatus);
                evt.menu.AppendAction(TextContent.spriteLibraryDeleteLabels, _ => DeleteSelected(), _ => canModifyStatus);
                evt.menu.AppendSeparator();

                var canRevertSelectedStatus = m_SelectedLabels.Any(l => m_LabelData.Any(labelData => CanRevertLabel(labelData) && labelData.name == l)) ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Disabled;
                var canRevertAnyStatus = m_LabelData.Any(CanRevertLabel) ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Disabled;
                evt.menu.AppendAction(TextContent.spriteLibraryRevertLabels, _ => RevertLabelOverride(false), _ => canRevertSelectedStatus);
                evt.menu.AppendAction(TextContent.spriteLibraryRevertAllLabels, _ => RevertLabelOverride(true), _ => canRevertAnyStatus);
                evt.menu.AppendSeparator();
            }

            var sprite = m_SelectedLabels.Any() ? m_SelectedLabels.Select(label => m_LabelData.FirstOrDefault(l => l.name == label)).FirstOrDefault(l => l?.sprite != null)?.sprite : null;
            evt.menu.AppendAction(TextContent.spriteLibraryShowLabel, _ => Selection.objects = new UnityEngine.Object[] { sprite }, _ => sprite != null ? DropdownMenuAction.Status.Normal : DropdownMenuAction.Status.Hidden);
        }

        static float GetAdjustedViewSize(ViewType viewType, float size)
        {
            if (viewType == ViewType.List)
                return Mathf.Lerp(k_MinListSize, k_MaxListSize, size);

            if (viewType == ViewType.Grid)
                return Mathf.Lerp(k_MinGridSize, k_MaxGridSize, size);

            return size;
        }
    }
}