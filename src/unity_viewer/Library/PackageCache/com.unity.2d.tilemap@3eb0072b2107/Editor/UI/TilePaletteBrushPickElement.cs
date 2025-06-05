using System;
using System.Collections.Generic;
using UnityEditor.EditorTools;
using UnityEditor.ShortcutManagement;
using UnityEditor.Tilemaps.External;
using UnityEditor.Toolbars;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [UxmlElement]
    internal partial class TilePaletteBrushPickElement : VisualElement
    {
        /// <summary>
        /// Factory for TilePaletteBrushPickElement.
        /// </summary>
        [Obsolete("TilePaletteBrushPickElementFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushPickElementFactory : UxmlFactory<TilePaletteBrushPickElement, TilePaletteBrushPickElementUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for TilePaletteBrushPickElement.
        /// </summary>
        [Obsolete("TilePaletteBrushPickElementUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteBrushPickElementUxmlTraits : UxmlTraits {}

        /// <summary>
        /// USS class name of elements of this type.
        /// </summary>
        private static readonly string ussClassName = "unity-tilepalette-brushpick";
        private static readonly string labelToolbarUssClassName = "unity-tilepalette-label-toolbar";
        private static readonly string viewToolbarUssClassName = "unity-tilepalette-brushpick-view-toolbar";
        private static readonly string lastUsedUssClassName = "unity-tilepalette-brushpick-lastused";
        private static readonly string emptyViewUssClassName = "unity-tilepalette-brushpick-emptyview";

        private static readonly string kTilePaletteBrushPickHideOnPickEditorPref = "TilePaletteBrushPickHideOnPick";
        private static readonly string kTilePaletteBrushPickLastSelectionEditorPref = "TilePaletteBrushPickLastSelection";

        private static readonly string kLastUsed = L10n.Tr("Last Picked");
        private static readonly string kPersonal = L10n.Tr("Saved Picks");

        private static readonly string kSizeSliderTooltip = L10n.Tr("Adjusts the size of the Picks");
        private static readonly string kGridViewTooltip = L10n.Tr("View Picks in a Grid");
        private static readonly string kListViewTooltip = L10n.Tr("View Picks in a List");
        private static readonly string kListViewNameColumn = L10n.Tr("Name");
        private static readonly string kListViewTypeColumn = L10n.Tr("Type");

        private static readonly string kEmptyViewSaveBrushPickText = L10n.Tr("To save a Brush Pick: ");
        private static readonly string kEmptyViewSaveBrushPickText1 = L10n.Tr("<indent=1em>\u2022<indent=2em>Activate the Pick Tool ");
        private static readonly string kEmptyViewSaveBrushPickText2 = L10n.Tr("<indent=1em>\u2022<indent=2em>Pick with the Brush from the Tile Palette or SceneView");
        private static readonly string kEmptyViewSaveBrushPickText3 = L10n.Tr("<indent=1em>\u2022<indent=2em>Hit the Save Button ");
        private static readonly string kEmptyViewSaveBrushPickText3a = L10n.Tr(" above to save it");
        private static readonly string kEmptyViewSaveBrushPickText1Alt = L10n.Tr("<indent=1em>\u2022<indent=2em>Set up the Brush in Brush Settings");

        private static readonly string brushPickUserViewEditorPref = "TilePaletteBrushPickUserView";
        private static readonly string brushPickScaleEditorPref = "TilePaletteBrushPickScale";
        private static readonly string brushPickFilteredEditorPref = "TilePaletteBrushPickFiltered";
        private static readonly string brushPickFilterTextEditorPref = "TilePaletteBrushPickFilterText";

        private static int kDefaultTextSize = 220;
        private static int kDefaultTypeTextSize = 160;
        private static int kDefaultItemSize = 96;
        private static int kDefaultBorderSize = 8;

        private static int kMinItemSize = kDefaultItemSize + kDefaultBorderSize;
        private static float kSliderScaleMinValue = 0.4f;
        private static float kSliderScaleMaxValue = 2.0f;

        private static float kSliderListThreshold = 0.8f;
        private static float kSliderGridThreshold = 1.2f;

        private static string[] k_RightToolbarElements  = new[] {
            TilePaletteBrushPickActiveBrushToggle.k_ToolbarId,
            TilePaletteBrushPickSaveButton.k_ToolbarId,
            TilePaletteHidePicksToggle.k_ToolbarId
        };
        private static bool[] k_TilePaletteWindowActiveRightToolbar = new[] { true, true, false };
        private static bool[] k_TilePaletteOverlayActiveRightToolbar = new[] { true, true, true };

        private readonly VisualElement m_EmptyPicksView;
        private readonly GridView m_PersonalGridView;
        private readonly MultiColumnListView m_PersonalListView;

        private readonly TilePaletteHidePicksToggle m_HideToggle;
        private readonly VisualElement m_RightToolbar;
        private readonly VisualElement m_ViewToolbar;
        private readonly Slider m_Slider;
        private readonly EditorToolbarToggle m_ListToggle;
        private readonly EditorToolbarToggle m_GridToggle;

        private IVisualElementScheduledItem m_GeometryResizeItem;
        private IVisualElementScheduledItem m_FilterTextItem;

        private GridBrushPickStore pickStore => GridPaintingState.brushPickStore;

        private int itemSize
        {
            get => Mathf.FloorToInt(kDefaultItemSize * brushPickScale);
        }

        private bool isShowGrid
        {
            get => EditorPrefs.GetBool(brushPickUserViewEditorPref, true);
            set => EditorPrefs.SetBool(brushPickUserViewEditorPref, value);
        }

        private float brushPickScale
        {
            get => EditorPrefs.GetFloat(brushPickScaleEditorPref, 1.0f);
            set => EditorPrefs.SetFloat(brushPickScaleEditorPref, Mathf.Clamp(value, kSliderScaleMinValue, kSliderScaleMaxValue));
        }

        private bool isBrushPickFiltered => brushPickFiltered || !String.IsNullOrWhiteSpace(brushPickFilterText);

        private bool brushPickFiltered
        {
            get => EditorPrefs.GetBool(brushPickFilteredEditorPref, false);
            set => EditorPrefs.SetBool(brushPickFilteredEditorPref, value);
        }

        private string brushPickFilterText
        {
            get => EditorPrefs.GetString(brushPickFilterTextEditorPref, "");
            set => EditorPrefs.SetString(brushPickFilterTextEditorPref, value);
        }

        /// <summary>
        /// Whether the Brush Picks overlay is hidden when a Brush Pick is selected.
        /// </summary>
        public bool hideOnPick
        {
            get => EditorPrefs.GetBool(kTilePaletteBrushPickHideOnPickEditorPref, true);
            set
            {
                EditorPrefs.SetBool(kTilePaletteBrushPickHideOnPickEditorPref, value);
                m_HideToggle.SetValueWithoutNotify(value);
            }
        }

        internal int lastSelection
        {
            get => EditorPrefs.GetInt(kTilePaletteBrushPickLastSelectionEditorPref, -1);
            set => EditorPrefs.SetInt(kTilePaletteBrushPickLastSelectionEditorPref, value);
        }

        private bool m_InitialScrollToLastSelection;

        /// <summary>
        /// Callback when the active Brush does a Pick on the Clipboard.
        /// </summary>
        public event Action onBrushPicked
        {
            add
            {
                onBrushPickedInternal += value;
                if (onBrushPickedInternal != null)
                {
                    TilePaletteOverlayUtility.SetupChildrenAsButtonStripForVisible(m_RightToolbar, k_TilePaletteOverlayActiveRightToolbar);
                }
            }
            remove
            {
                onBrushPickedInternal -= value;
                if (onBrushPickedInternal == null)
                {
                    TilePaletteOverlayUtility.SetupChildrenAsButtonStripForVisible(m_RightToolbar, k_TilePaletteWindowActiveRightToolbar);
                }
            }
        }

        private event Action onBrushPickedInternal;

        /// <summary>
        /// Initializes and returns an instance of TilePaletteBrushPickElement.
        /// </summary>
        public TilePaletteBrushPickElement()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            m_EmptyPicksView = new VisualElement();
            m_EmptyPicksView.AddToClassList(emptyViewUssClassName);
            m_EmptyPicksView.style.display = DisplayStyle.None;
            m_EmptyPicksView.style.visibility = Visibility.Hidden;

            UpdateEmptyPicksView();

            m_PersonalListView = new MultiColumnListView
            {
                showAlternatingRowBackgrounds = AlternatingRowBackground.ContentOnly,
                reorderable = true,
                reorderMode = ListViewReorderMode.Animated,
                columns =
                {
                    new Column()
                    {
                        title = kListViewNameColumn,
                        minWidth = kDefaultItemSize * kSliderScaleMinValue + kDefaultTextSize,
                        width = itemSize + kDefaultTextSize,
                    },
                    new Column()
                    {
                        title = kListViewTypeColumn,
                        minWidth = kDefaultTypeTextSize,
                        width = kDefaultTypeTextSize,
                    },
                }
            };
            m_PersonalListView.itemIndexChanged += UserItemIndexChanged;
            m_PersonalListView.columns[0].makeCell = MakeUserItem;
            m_PersonalListView.columns[0].bindCell = BindUserItem;
            m_PersonalListView.columns[1].makeCell = MakeUserType;
            m_PersonalListView.columns[1].bindCell = BindUserType;

            m_PersonalListView.itemsChosen += OnUserItemChosen;
            m_PersonalListView.selectionChanged += OnUserSelectionChanged;
            m_PersonalListView.style.display = isShowGrid ? DisplayStyle.None : DisplayStyle.Flex;

            m_PersonalGridView = new GridView();
            m_PersonalGridView.selectionType = SelectionType.Single;
            m_PersonalGridView.showBorder = true;
            m_PersonalGridView.makeItem = MakeUserItem;
            m_PersonalGridView.bindItem = BindUserItem;
            m_PersonalGridView.onItemsChosen += OnUserItemChosen;
            m_PersonalGridView.onSelectionChange += OnUserSelectionChanged;
            m_PersonalGridView.style.display = isShowGrid ? DisplayStyle.Flex : DisplayStyle.None;

            var personalToolbar = new VisualElement();
            personalToolbar.AddToClassList(labelToolbarUssClassName);
            personalToolbar.Add(new Label(kPersonal));

            var emptySpace = new VisualElement();
            emptySpace.style.flexGrow = 1.0f;
            emptySpace.style.flexShrink = 0f;
            personalToolbar.Add(emptySpace);

            var filterField = new ToolbarSearchField();
            filterField.SetValueWithoutNotify(brushPickFilterText);
            filterField.RegisterValueChangedCallback(FilterTextChanged);
            personalToolbar.Add(filterField);

            m_RightToolbar = EditorToolbar.CreateOverlay(k_RightToolbarElements);
            TilePaletteOverlayUtility.SetupChildrenAsButtonStripForVisible(m_RightToolbar, k_TilePaletteWindowActiveRightToolbar);

            var filterButton = m_RightToolbar.Q<EditorToolbarToggle>(null, TilePaletteBrushPickActiveBrushToggle.k_ElementClass);
            filterButton.RegisterValueChangedCallback(FilterChanged);
            filterButton.SetValueWithoutNotify(brushPickFiltered);
            var saveButton = m_RightToolbar.Q<EditorToolbarButton>(null, TilePaletteBrushPickSaveButton.k_ElementClass);
            saveButton.clicked += AddCurrent;

            m_HideToggle = m_RightToolbar.Q<TilePaletteHidePicksToggle>();
            m_HideToggle.SetValueWithoutNotify(hideOnPick);
            m_HideToggle.ToggleChanged += OnHidePicksToggleChanged;

            personalToolbar.Add(m_RightToolbar);

            Add(personalToolbar);

            Add(m_EmptyPicksView);
            Add(m_PersonalListView);
            Add(m_PersonalGridView);

            m_ViewToolbar = new VisualElement();
            m_ViewToolbar.AddToClassList(viewToolbarUssClassName);
            emptySpace = new VisualElement();
            emptySpace.style.flexGrow = 1f;
            emptySpace.style.flexShrink = 0f;
            m_ViewToolbar.Add(emptySpace);
            m_Slider = new Slider(kSliderScaleMinValue, kSliderScaleMaxValue);
            m_Slider.SetValueWithoutNotify(brushPickScale);
            m_Slider.RegisterValueChangedCallback(SliderChanged);
            m_Slider.tooltip = kSizeSliderTooltip;
            m_ViewToolbar.Add(m_Slider);
            m_ListToggle = new EditorToolbarToggle();
            m_ListToggle.AddToClassList("list-button");
            m_ListToggle.RegisterValueChangedCallback(ListToggleChanged);
            m_ListToggle.SetValueWithoutNotify(!isShowGrid);
            m_ListToggle.tooltip = kListViewTooltip;
            m_ViewToolbar.Add(m_ListToggle);
            m_GridToggle = new EditorToolbarToggle();
            m_GridToggle.AddToClassList("grid-button");
            m_GridToggle.RegisterValueChangedCallback(GridToggleChanged);
            m_GridToggle.SetValueWithoutNotify(isShowGrid);
            m_GridToggle.tooltip = kGridViewTooltip;
            m_ViewToolbar.Add(m_GridToggle);
            Add(m_ViewToolbar);

            var lastUsedToolbar = new VisualElement();
            lastUsedToolbar.AddToClassList(labelToolbarUssClassName);
            lastUsedToolbar.AddToClassList(lastUsedUssClassName);
            lastUsedToolbar.Add(new Label(kLastUsed));

            Add(lastUsedToolbar);

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
            RegisterCallback<GeometryChangedEvent>(GeometryChanged);
        }

        private void UpdateEmptyPicksView()
        {
            m_EmptyPicksView.Clear();

            TilemapEditorTool pickingTool = null;
            foreach (var tool in TilemapEditorTool.tilemapEditorTools)
            {
                if (tool is PickingTool editorTool)
                {
                    pickingTool = editorTool;
                    break;
                }
            }

            if (pickingTool != null)
            {
                m_EmptyPicksView.Add(new Label(kEmptyViewSaveBrushPickText));

                var line1Text = new VisualElement();
                line1Text.style.flexDirection = FlexDirection.Row;
                line1Text.Add(new Label(kEmptyViewSaveBrushPickText1));
                var pickToolButton = new TilemapEditorToolButton(pickingTool);
                var pickToolStrip = new VisualElement();
                pickToolStrip.Add(pickToolButton);
                EditorToolbarUtility.SetupChildrenAsButtonStrip(pickToolStrip);
                line1Text.Add(pickToolStrip);
                m_EmptyPicksView.Add(line1Text);

                m_EmptyPicksView.Add(new Label(kEmptyViewSaveBrushPickText2));
            }
            else
            {
                m_EmptyPicksView.Add(new Label(kEmptyViewSaveBrushPickText1Alt));
            }

            // Save
            var line3Text = new VisualElement();
            line3Text.style.flexDirection = FlexDirection.Row;
            line3Text.Add(new Label(kEmptyViewSaveBrushPickText3));
            var saveButton = new TilePaletteBrushPickSaveButton();
            saveButton.clicked += AddCurrent;
            var saveStrip = new VisualElement();
            saveStrip.Add(saveButton);
            EditorToolbarUtility.SetupChildrenAsButtonStrip(saveStrip);
            line3Text.Add(saveStrip);
            line3Text.Add(new Label(kEmptyViewSaveBrushPickText3a));
            m_EmptyPicksView.Add(line3Text);
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            GridPaintingState.brushChanged += OnBrushChanged;
            GridPaintingState.brushPickChanged += OnBrushPickChanged;
            GridPaintingState.brushPickStoreChanged += UpdatePick;
            AssemblyReloadEvents.afterAssemblyReload += UpdatePick;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            RegisterCallback<KeyDownEvent>(OnKeyPress);
            UpdatePick();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            UnregisterCallback<KeyDownEvent>(OnKeyPress);
            AssemblyReloadEvents.afterAssemblyReload -= UpdatePick;
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            GridPaintingState.brushPickStoreChanged -= UpdatePick;
            GridPaintingState.brushPickChanged -= OnBrushPickChanged;
            GridPaintingState.brushChanged -= OnBrushChanged;
        }

        private void FilterTextChanged(ChangeEvent<string> evt)
        {
            brushPickFilterText = evt.newValue;
            ClearFilterTextEvent();
            var count = 500;
            if (String.IsNullOrWhiteSpace(brushPickFilterText))
                count = 200;
            m_FilterTextItem = schedule.Execute(UpdatePick).StartingIn(count);
        }

        private void FilterChanged(ChangeEvent<bool> evt)
        {
            brushPickFiltered = evt.newValue;
            UpdatePick();
        }

        private void SliderChanged(ChangeEvent<float> evt)
        {
            brushPickScale = evt.newValue;
            var changed = false;
            if (evt.newValue < evt.previousValue && evt.newValue < kSliderListThreshold)
            {
                m_GridToggle.SetValueWithoutNotify(false);
                m_ListToggle.SetValueWithoutNotify(true);
                changed = true;
            }
            else if (evt.newValue > evt.previousValue && evt.newValue > kSliderGridThreshold)
            {
                m_GridToggle.SetValueWithoutNotify(true);
                m_ListToggle.SetValueWithoutNotify(false);
                changed = true;
            }

            if (changed)
                UpdatePersonalDisplay();
            else
                UpdateSize();
        }

        private void ListToggleChanged(ChangeEvent<bool> evt)
        {
            m_GridToggle.SetValueWithoutNotify(!evt.newValue);
            UpdatePersonalDisplay();
        }

        private void GridToggleChanged(ChangeEvent<bool> evt)
        {
            m_ListToggle.SetValueWithoutNotify(!evt.newValue);
            UpdatePersonalDisplay();
        }

        private void ClearGeometryResizeEvent()
        {
            if (m_GeometryResizeItem == null)
                return;

            m_GeometryResizeItem.Pause();
            m_GeometryResizeItem = null;
        }

        private void GeometryChanged(GeometryChangedEvent evt)
        {
            ClearGeometryResizeEvent();
            var count = 100;
            m_GeometryResizeItem = schedule.Execute(UpdateSize).StartingIn(count);
        }

        private void UpdateSize()
        {
            m_PersonalListView.fixedItemHeight = itemSize;
            if (!isShowGrid)
                EditorApplication.delayCall += () => m_PersonalListView.Rebuild();

            var inspectorWidth = resolvedStyle.width;
            var gridColumnCount = Mathf.Max(1, Mathf.FloorToInt(inspectorWidth / itemSize));

            m_PersonalGridView.itemHeight = itemSize;
            m_PersonalGridView.columnCount = gridColumnCount;
            if (isShowGrid)
                EditorApplication.delayCall += () => m_PersonalGridView.Refresh();

            var scrollIndex = m_PersonalGridView.selectedIndex;
            if (!m_InitialScrollToLastSelection)
            {
                scrollIndex = lastSelection;
                m_InitialScrollToLastSelection = true;
            }
            ScrollToSelectedDelayed(scrollIndex);
        }

        private void UpdatePersonalDisplay()
        {
            var showGrid = m_GridToggle.value;
            isShowGrid = showGrid;
            if (showGrid)
            {
                m_PersonalGridView.style.display = DisplayStyle.Flex;
                m_PersonalListView.style.display = DisplayStyle.None;
                m_PersonalGridView.selectedIndex = m_PersonalListView.selectedIndex;
                if (m_Slider.value < kSliderListThreshold)
                {
                    m_Slider.SetValueWithoutNotify(kSliderListThreshold);
                    brushPickScale = kSliderListThreshold;
                }
            }
            else
            {
                m_PersonalGridView.style.display = DisplayStyle.None;
                m_PersonalListView.style.display = DisplayStyle.Flex;
                m_PersonalListView.selectedIndex = m_PersonalGridView.selectedIndex;
                if (m_Slider.value > kSliderGridThreshold)
                {
                    m_Slider.SetValueWithoutNotify(kSliderGridThreshold);
                    brushPickScale = kSliderGridThreshold;
                }
            }
            UpdateSize();
        }

        private void ClearFilterTextEvent()
        {
            if (m_FilterTextItem == null)
                return;

            m_FilterTextItem.Pause();
            m_FilterTextItem = null;
        }

        private void UpdatePick()
        {
            ClearFilterTextEvent();
            pickStore.SetUserBrushFilterType(brushPickFiltered ? GridPaintingState.gridBrush.GetType() : null, brushPickFilterText);
            m_PersonalListView.itemsSource = pickStore.filteredUserSavedBrushes;
            m_PersonalListView.RefreshItems();
            m_PersonalGridView.itemsSource = pickStore.filteredUserSavedBrushes;

            if (pickStore.userSavedBrushes.Count == 0)
            {
                m_EmptyPicksView.style.display = DisplayStyle.Flex;
                m_EmptyPicksView.style.visibility = Visibility.Visible;
                m_PersonalListView.style.display = DisplayStyle.None;
                m_PersonalGridView.style.display = DisplayStyle.None;

                m_ViewToolbar.SetEnabled(false);
            }
            else
            {
                m_EmptyPicksView.style.display = DisplayStyle.None;
                m_EmptyPicksView.style.visibility = Visibility.Hidden;
                m_ViewToolbar.SetEnabled(true);

                if (m_PersonalGridView.style.display == DisplayStyle.None
                    && m_PersonalListView.style.display == DisplayStyle.None)
                {
                    UpdatePersonalDisplay();
                    if (!isShowGrid)
                        EditorApplication.delayCall += () => m_PersonalListView.Rebuild();
                    if (isShowGrid)
                        EditorApplication.delayCall += () => m_PersonalGridView.Refresh();
                }
            }
        }

        private void UserItemIndexChanged(int oldIdx, int newIdx)
        {
            if (oldIdx == newIdx)
                return;

            if (isBrushPickFiltered)
            {
                oldIdx = pickStore.GetIndexOfUserBrush(pickStore.filteredUserSavedBrushes[oldIdx]);
                newIdx = pickStore.GetIndexOfUserBrush(pickStore.filteredUserSavedBrushes[newIdx]);
            }

            pickStore.SwapUserSavedBrushes(oldIdx, newIdx);
            UpdatePick();
        }

        private void OnUserSelectionChanged(IEnumerable<object> objs)
        {
            foreach (var obj in objs)
            {
                var idx = pickStore.filteredUserSavedBrushes.IndexOf(obj as GridBrushBase);
                if (idx >= 0)
                {
                    LoadBrush(true, idx);
                    lastSelection = idx;
                }
                break;
            }
        }

        private void BindUserItem(VisualElement item, int idx)
        {
            var brush = pickStore.filteredUserSavedBrushes[idx];
            var brushPickItemElement = item.Q<TilePaletteBrushPickItemElement>();
            brushPickItemElement.SetBrush(brush);
            if (!isShowGrid)
            {
                brushPickItemElement.SetSize(itemSize);
            }
            if (brush != null)
            {
                brushPickItemElement.pointerUpEvent = () => LoadBrush(true, idx);
                brushPickItemElement.renameEvent = newName => RenameUser(idx, newName);
            }
        }

        private VisualElement MakeUserItem()
        {
            var element = new TilePaletteBrushPickItemElement(true);
            return element;
        }

        private void BindUserType(VisualElement item, int idx)
        {
            var brush = pickStore.filteredUserSavedBrushes[idx];
            var brushPickTypeElement = item.Q<TilePaletteBrushPickTypeElement>();
            brushPickTypeElement.SetBrush(brush);
        }

        private VisualElement MakeUserType()
        {
            var element = new TilePaletteBrushPickTypeElement();
            return element;
        }

        private void OnPlayModeStateChanged(PlayModeStateChange playModeStateChange)
        {
            if (playModeStateChange != PlayModeStateChange.EnteredEditMode)
                return;

            UpdatePick();
        }

        private void OnBrushChanged(GridBrushBase brush)
        {
            UpdateEmptyPicksView();
            UpdatePick();
        }

        private void OnBrushPickChanged()
        {
            UpdatePick();
            m_PersonalGridView.ClearSelection();
            m_PersonalListView.ClearSelection();
        }

        private void OnHidePicksToggleChanged(bool hidePicks)
        {
            hideOnPick = hidePicks;
        }

        private void OnKeyPress(KeyDownEvent evt)
        {
            switch (evt.keyCode)
            {
                case KeyCode.F2:
                    ActivateRenameUser();
                    break;

                case KeyCode.Backspace:
                    if ((Application.platform == RuntimePlatform.OSXEditor)
                        && evt.commandKey)
                    {
                        DeleteUser();
                    }
                    break;
                case KeyCode.Delete:
                    {
                        DeleteUser();
                    }
                    break;
            }
        }

        private void OnUserItemChosen(IEnumerable<object> objs)
        {
            EditorApplication.delayCall += ActivateRenameUser;
        }

        private void LoadBrush(bool user, int idx)
        {
            if (user && isBrushPickFiltered)
            {
                idx = pickStore.GetIndexOfUserBrushFromFilteredIdx(idx);
            }
            GridPaintingState.SetPickOnActiveGridBrush(user, idx);

            if (!TilemapEditorTool.IsActive(typeof(PaintTool)))
            {
                foreach (var tilemapEditorTool in TilemapEditorTool.tilemapEditorTools)
                {
                    if (tilemapEditorTool is not PaintTool)
                        continue;

                    ToolManager.SetActiveTool(tilemapEditorTool);
                    break;
                }
            }
            if (hideOnPick)
                onBrushPickedInternal?.Invoke();
        }

        private void ActivateRenameUser()
        {
            TilePaletteBrushPickItemElement selectedItem = null;
            if (isShowGrid)
            {
                selectedItem = m_PersonalGridView.GetElementAt(m_PersonalGridView.selectedIndex) as TilePaletteBrushPickItemElement;
            }
            else
            {
                var ve = m_PersonalListView.GetRootElementForIndex(m_PersonalListView.selectedIndex);
                if (ve != null)
                    selectedItem = ve.Q<TilePaletteBrushPickItemElement>();
            }
            if (selectedItem != null)
                selectedItem.ToggleRename();
        }

        private void RenameUser(int idx, string newName)
        {
            if (idx < 0 || idx >= pickStore.filteredUserSavedBrushes.Count)
                return;

            var brush = pickStore.filteredUserSavedBrushes[idx];
            if (brush.name == newName)
                return;

            brush.name = newName;
            if (isBrushPickFiltered)
            {
                idx = pickStore.GetIndexOfUserBrush(brush);
            }
            pickStore.SaveUserSavedBrush(idx, brush);
        }

        private void DeleteUser()
        {
            var isGridView = m_GridToggle.value;
            var idx = m_PersonalListView.selectedIndex;
            if (isGridView)
                idx = m_PersonalGridView.selectedIndex;
            if (isBrushPickFiltered)
            {
                idx = pickStore.GetIndexOfUserBrushFromFilteredIdx(idx);
            }
            if (pickStore.RemoveUserSavedBrush(idx))
                UpdatePick();
        }

        private void AddCurrent()
        {
            pickStore.AddNewUserSavedBrush(GridPaintingState.gridBrush);
            UpdatePick();

            var idx = pickStore.filteredUserSavedBrushes.Count - 1;
            m_PersonalListView.selectedIndex = idx;
            m_PersonalGridView.selectedIndex = idx;
            ScrollToSelectedDelayed(idx);

            if (isShowGrid)
            {
                var element = m_PersonalGridView.GetElementAt(idx);
                if (element != null)
                    element.Focus();
            }
            else
            {
                var element = m_PersonalListView.GetRootElementForIndex(idx);
                if (element != null)
                    element.Focus();
            }
        }

        private void ScrollToSelectedDelayed(int index)
        {
            if (index == -1)
                return;

            EditorApplication.delayCall += () =>
            {
                m_PersonalListView.ScrollToItem(index);
                m_PersonalGridView.ScrollToItem(index);
            };
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteBrushPickActiveBrushToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Brush Pick Filter";
        internal static readonly string k_ElementClass = "unity-tilepalette-brushpick-filter-toggle";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";

        private static readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/EditorUI.Filter.png";
        private static readonly string k_TooltipText = L10n.Tr("Filters Picks by Active Brush Type");

        public TilePaletteBrushPickActiveBrushToggle()
        {
            name = "Tile Palette Filter Pick";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = k_TooltipText;
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteBrushPickSaveButton : EditorToolbarButton
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Brush Pick Save";
        internal static readonly string k_ElementClass = "unity-tilepalette-brushpick-save-button";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_TooltipText = L10n.Tr("Adds the current Brush as a new saved Pick");

        public TilePaletteBrushPickSaveButton()
        {
            name = "Tile Palette Save Pick";
            AddToClassList(k_ToolSettingsClass);
            AddToClassList(k_ElementClass);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            icon = EditorGUIUtility.LoadIcon("SaveAs");
            tooltip = k_TooltipText;
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal sealed class TilePaletteHidePicksToggle : EditorToolbarToggle
    {
        internal const string k_ToolbarId = "Tile Palette/Tile Palette Hide Picks";

        private static readonly string k_ToolSettingsClass = "unity-tool-settings";
        private static readonly string k_ElementClass = "unity-tilepalette-element-hidepicks";

        private static readonly string k_IconPath =
            "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.ShowTilePalette.png";

        private static readonly string k_TooltipFormatText =
            L10n.Tr("Hides Brush Picks on Pick. ( {0} ) to show/hide Brush Picks.");

        private static readonly string k_ShortcutId = "Grid Painting/Toggle SceneView BrushPick";

        public Action<bool> ToggleChanged;

        public TilePaletteHidePicksToggle()
        {
            name = "Tile Palette Hide Picks";
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

        private void OnShortcutBindingChanged(IShortcutProfileManager arg1, Identifier arg2, ShortcutBinding arg3,
            ShortcutBinding arg4)
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
