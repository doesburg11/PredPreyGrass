using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps.External
{
    /// <summary>
    /// A view containing recycled rows with items inside.
    /// </summary>
    [UxmlElement]
    internal partial class GridView : BindableElement, ISerializationCallbackReceiver
    {
        const int k_ExtraVisibleRows = 2;

        /// <summary>
        /// The USS class name for GridView elements.
        /// </summary>
        /// <remarks>
        /// Unity adds this USS class to every instance of the GridView element. Any styling applied to
        /// this class affects every GridView located beside, or below the stylesheet in the visual tree.
        /// </remarks>
        const string k_UssClassName = "unity-grid-view";

        /// <summary>
        /// The USS class name for GridView elements with a border.
        /// </summary>
        /// <remarks>
        /// Unity adds this USS class to an instance of the GridView element if the instance's
        /// <see cref="GridView.showBorder"/> property is set to true. Any styling applied to this class
        /// affects every such GridView located beside, or below the stylesheet in the visual tree.
        /// </remarks>
        const string k_BorderUssClassName = k_UssClassName + "--with-border";

        /// <summary>
        /// The USS class name of item elements in GridView elements.
        /// </summary>
        /// <remarks>
        /// Unity adds this USS class to every item element the GridView contains. Any styling applied to
        /// this class affects every item element located beside, or below the stylesheet in the visual tree.
        /// </remarks>
        const string k_ItemUssClassName = k_UssClassName + "__item";

        /// <summary>
        /// The USS class name of selected item elements in the GridView.
        /// </summary>
        /// <remarks>
        /// Unity adds this USS class to every selected element in the GridView. The <see cref="GridView.selectionType"/>
        /// property decides if zero, one, or more elements can be selected. Any styling applied to
        /// this class affects every GridView item located beside, or below the stylesheet in the visual tree.
        /// </remarks>
        internal const string itemSelectedVariantUssClassName = k_ItemUssClassName + "--selected";

        /// <summary>
        /// The USS class name of rows in the GridView.
        /// </summary>
        const string k_RowUssClassName = k_UssClassName + "__row";

        const int k_DefaultItemHeight = 30;

        static CustomStyleProperty<int> s_ItemHeightProperty = new CustomStyleProperty<int>("--unity-item-height");

        internal readonly ScrollView scrollView;

        readonly List<int> m_SelectedIds = new List<int>();

        readonly List<int> m_SelectedIndices = new List<int>();

        readonly List<object> m_SelectedItems = new List<object>();

        Action<VisualElement, int> m_BindItem;

        int m_ColumnCount = 1;

        int m_FirstVisibleIndex;

        Func<int, int> m_GetItemId;

        int m_ItemHeight = k_DefaultItemHeight;

        bool m_ItemHeightIsInline;

        IList m_ItemsSource;

        float m_LastHeight;

        Func<VisualElement> m_MakeItem;

        int m_RangeSelectionOrigin = -1;

        List<RecycledRow> m_RowPool = new List<RecycledRow>();

        // we keep this list in order to minimize temporary gc allocs
        List<RecycledRow> m_ScrollInsertionList = new List<RecycledRow>();

        // Persisted.
        float m_ScrollOffset;

        SelectionType m_SelectionType;

        Vector3 m_TouchDownPosition;

        long m_TouchDownTime;

        int m_VisibleRowCount;

        /// <summary>
        /// Creates a <see cref="GridView"/> with all default properties. The <see cref="GridView.itemsSource"/>,
        /// <see cref="GridView.itemHeight"/>, <see cref="GridView.makeItem"/> and <see cref="GridView.bindItem"/> properties
        /// must all be set for the GridView to function properly.
        /// </summary>
        public GridView()
        {
            AddStyleSheetPath("Packages/com.unity.2d.tilemap/Editor/UI/External/GridView.uss");
            AddToClassList(k_UssClassName);

            selectionType = SelectionType.Multiple;

            m_ScrollOffset = 0.0f;

            scrollView = new ScrollView { viewDataKey = "grid-view__scroll-view" };
            scrollView.StretchToParentSize();
            scrollView.verticalScroller.valueChanged += OnScroll;

            RegisterCallback<GeometryChangedEvent>(OnSizeChanged);
            RegisterCallback<CustomStyleResolvedEvent>(OnCustomStyleResolved);

            scrollView.contentContainer.RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            scrollView.contentContainer.RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            hierarchy.Add(scrollView);

            scrollView.contentContainer.focusable = true;
            scrollView.contentContainer.usageHints &= ~UsageHints.GroupTransform; // Scroll views with virtualized content shouldn't have the "view transform" optimization
        }

        void OnKeyPress(KeyDownEvent evt)
        {
            switch (evt.keyCode)
            {
                case KeyCode.A when evt.actionKey:
                    SelectAll();
                    evt.StopPropagation();
                    break;
                case KeyCode.Home:
                    ScrollToItem(0);
                    evt.StopPropagation();
                    break;
                case KeyCode.End:
                    ScrollToItem(-1);
                    evt.StopPropagation();
                    break;
                case KeyCode.Escape:
                    ClearSelection();
                    evt.StopPropagation();
                    break;
                case KeyCode.Return:
                    onItemsChosen?.Invoke(m_SelectedItems);
                    evt.StopPropagation();
                    break;
                case KeyCode.LeftArrow:
                    var firstIndexInRow = selectedIndex - selectedIndex % columnCount;
                    if (selectedIndex >= 0 && selectedIndex > firstIndexInRow)
                    {
                        var next = evt.actionKey ? firstIndexInRow :  selectedIndex - 1;
                        if (next < firstIndexInRow)
                            next = firstIndexInRow;

                        if (evt.shiftKey)
                            DoRangeSelection(next);
                        else
                            m_RangeSelectionOrigin = selectedIndex = next;

                        evt.StopPropagation();
                    }
                    break;
                case KeyCode.RightArrow:
                {
                    var currentRow = selectedIndex / columnCount;
                    var lastIndexInRow = Math.Min((currentRow + 1) * columnCount - 1, itemsSource.Count - 1);
                    if (selectedIndex >= 0 && selectedIndex < lastIndexInRow)
                    {
                        var next = evt.actionKey ? lastIndexInRow : selectedIndex + 1;
                        if (next > lastIndexInRow)
                            next = lastIndexInRow;

                        if (evt.shiftKey)
                            DoRangeSelection(next);
                        else
                            m_RangeSelectionOrigin = selectedIndex = next;

                        evt.StopPropagation();
                    }
                    break;
                }
                case KeyCode.UpArrow:
                    if (selectedIndex >= 0)
                    {
                        var next = evt.actionKey ?
                            selectedIndex % columnCount :
                            selectedIndex - columnCount;
                        if (next >= 0 && selectedIndex != next)
                        {
                            if (evt.shiftKey)
                                DoRangeSelection(next);
                            else
                                m_RangeSelectionOrigin = selectedIndex = next;

                            ScrollToItem(evt.actionKey ? 0 : selectedIndex);

                            evt.StopPropagation();
                        }
                    }
                    break;
                case KeyCode.DownArrow:
                {
                    if (selectedIndex >= 0)
                    {
                        var targetId = (Mathf.FloorToInt((float)itemsSource.Count / columnCount)) * columnCount + selectedIndex % columnCount;
                        var next = evt.actionKey ?
                             targetId >= itemsSource.Count ? targetId - columnCount : targetId :
                            selectedIndex + columnCount;
                        if (next < itemsSource.Count && selectedIndex != next)
                        {
                            if (evt.shiftKey)
                                DoRangeSelection(next);
                            else
                                m_RangeSelectionOrigin = selectedIndex = next;

                            ScrollToItem(evt.actionKey ? -1 : selectedIndex);

                            evt.StopPropagation();
                        }
                    }
                    break;
                }
            }
        }

        /// <summary>
        /// Constructs a <see cref="GridView"/>, with all required properties provided.
        /// </summary>
        /// <param name="itemsSource">The list of items to use as a data source.</param>
        /// <param name="itemHeight">The height of each item, in pixels.</param>
        /// <param name="makeItem">The factory method to call to create a display item. The method should return a
        /// VisualElement that can be bound to a data item.</param>
        /// <param name="bindItem">The method to call to bind a data item to a display item. The method
        /// receives as parameters the display item to bind, and the index of the data item to bind it to.</param>
        public GridView(IList itemsSource, int itemHeight, Func<VisualElement> makeItem, Action<VisualElement, int> bindItem)
            : this()
        {
            m_ItemsSource = itemsSource;
            m_ItemHeight = itemHeight;
            m_ItemHeightIsInline = true;

            m_MakeItem = makeItem;
            m_BindItem = bindItem;
        }

        /// <summary>
        /// Callback for binding a data item to the visual element.
        /// </summary>
        /// <remarks>
        /// The method called by this callback receives the VisualElement to bind, and the index of the
        /// element to bind it to.
        /// </remarks>
        public Action<VisualElement, int> bindItem
        {
            get { return m_BindItem; }
            set
            {
                m_BindItem = value;
                Refresh();
            }
        }

        /// <summary>
        /// The number of columns for this grid.
        /// </summary>
        public int columnCount
        {
            get => m_ColumnCount;

            set
            {
                if (m_ColumnCount != value && value > 0)
                {
                    m_ScrollOffset = 0;
                    m_ColumnCount = value;
                    Refresh();
                }
            }
        }

        /// <summary>
        /// Returns the content container for the <see cref="GridView"/>. Because the GridView control automatically manages
        /// its content, this always returns null.
        /// </summary>
        public override VisualElement contentContainer => null;

        /// <summary>
        /// The height of a single item in the list, in pixels.
        /// </summary>
        /// <remarks>
        /// GridView requires that all visual elements have the same height so that it can calculate the
        /// scroller size.
        ///
        /// This property must be set for the list view to function.
        /// </remarks>
        [UxmlAttribute]
        public int itemHeight
        {
            get { return m_ItemHeight; }
            set
            {
                if (m_ItemHeight != value && value > 0)
                {
                    m_ItemHeightIsInline = true;
                    m_ItemHeight = value;
                    Refresh();
                }
            }
        }

        /// <summary>
        ///
        /// </summary>
        public float itemWidth => (scrollView.contentViewport.layout.width / columnCount);

        /// <summary>
        /// The data source for list items.
        /// </summary>
        /// <remarks>
        /// This list contains the items that the <see cref="GridView"/> displays.
        ///
        /// This property must be set for the list view to function.
        /// </remarks>
        public IList itemsSource
        {
            get { return m_ItemsSource; }
            set
            {
                if (m_ItemsSource is INotifyCollectionChanged oldCollection)
                {
                    oldCollection.CollectionChanged -= OnItemsSourceCollectionChanged;
                }

                m_ItemsSource = value;
                if (m_ItemsSource is INotifyCollectionChanged newCollection)
                {
                    newCollection.CollectionChanged += OnItemsSourceCollectionChanged;
                }

                Refresh();
            }
        }

        /// <summary>
        /// Callback for constructing the VisualElement that is the template for each recycled and re-bound element in the list.
        /// </summary>
        /// <remarks>
        /// This callback needs to call a function that constructs a blank <see cref="VisualElement"/> that is
        /// bound to an element from the list.
        ///
        /// The GridView automatically creates enough elements to fill the visible area, and adds more if the area
        /// is expanded. As the user scrolls, the GridView cycles elements in and out as they appear or disappear.
        ///
        ///  This property must be set for the list view to function.
        /// </remarks>
        public Func<VisualElement> makeItem
        {
            get { return m_MakeItem; }
            set
            {
                if (m_MakeItem == value)
                    return;
                m_MakeItem = value;
                Refresh();
            }
        }

        /// <summary>
        /// The computed pixel-aligned height for the list elements.
        /// </summary>
        /// <remarks>
        /// This value changes depending on the current panel's DPI scaling.
        /// </remarks>
        /// <seealso cref="GridView.itemHeight"/>
        public float resolvedItemHeight
        {
            get
            {
                var dpiScaling = 1;//this.GetScaledPixelsPerPoint();
                return Mathf.Round(itemHeight * dpiScaling) / dpiScaling;
            }
        }

        /// <summary>
        ///
        /// </summary>
        public float resolvedItemWidth
        {
            get
            {
                var dpiScaling = 1;//this.GetScaledPixelsPerPoint();
                return Mathf.Round(itemWidth * dpiScaling) / dpiScaling;
            }
        }

        /// <summary>
        /// Returns or sets the selected item's index in the data source. If multiple items are selected, returns the
        /// first selected item's index. If multiple items are provided, sets them all as selected.
        /// </summary>
        public int selectedIndex
        {
            get { return m_SelectedIndices.Count == 0 ? -1 : m_SelectedIndices.First(); }
            set { SetSelection(value); }
        }

        /// <summary>
        /// Returns the indices of selected items in the data source. Always returns an enumerable, even if no item  is selected, or a
        /// single item is selected.
        /// </summary>
        public IEnumerable<int> selectedIndices => m_SelectedIndices;

        /// <summary>
        /// Returns the selected item from the data source. If multiple items are selected, returns the first selected item.
        /// </summary>
        public object selectedItem => m_SelectedItems.Count == 0 ? null : m_SelectedItems.First();

        /// <summary>
        /// Returns the selected items from the data source. Always returns an enumerable, even if no item is selected, or a single
        /// item is selected.
        /// </summary>
        public IEnumerable<object> selectedItems => m_SelectedItems;

        /// <summary>
        /// Returns the IDs of selected items in the data source. Always returns an enumerable, even if no item  is selected, or a
        /// single item is selected.
        /// </summary>
        public IEnumerable<int> selectedIds => m_SelectedIds;

        /// <summary>
        /// Controls the selection type.
        /// </summary>
        /// <remarks>
        /// You can set the GridView to make one item selectable at a time, make multiple items selectable, or disable selections completely.
        ///
        /// When you set the GridView to disable selections, any current selection is cleared.
        /// </remarks>
        [UxmlAttribute]
        public SelectionType selectionType
        {
            get { return m_SelectionType; }
            set
            {
                m_SelectionType = value;
                if (m_SelectionType == SelectionType.None || (m_SelectionType == SelectionType.Single && m_SelectedIndices.Count > 1))
                {
                    ClearSelection();
                }
            }
        }

        /// <summary>
        /// Enable this property to display a border around the GridView.
        /// </summary>
        /// <remarks>
        /// If set to true, a border appears around the ScrollView.
        /// </remarks>
        [UxmlAttribute]
        public bool showBorder
        {
            get => ClassListContains(k_BorderUssClassName);
            set => EnableInClassList(k_BorderUssClassName, value);
        }

        /// <summary>
        /// Callback for unbinding a data item from the VisualElement.
        /// </summary>
        /// <remarks>
        /// The method called by this callback receives the VisualElement to unbind, and the index of the
        /// element to unbind it from.
        /// </remarks>
        public Action<VisualElement, int> unbindItem { get; set; }

        internal Func<int, int> getItemId
        {
            get { return m_GetItemId; }
            set
            {
                m_GetItemId = value;
                Refresh();
            }
        }

        internal List<RecycledRow> rowPool
        {
            get { return m_RowPool; }
        }

        void ISerializationCallbackReceiver.OnAfterDeserialize()
        {
            Refresh();
        }

        void ISerializationCallbackReceiver.OnBeforeSerialize() {}

        /// <summary>
        /// Callback triggered when the user acts on a selection of one or more items, for example by double-clicking or pressing Enter.
        /// </summary>
        /// <remarks>
        /// This callback receives an enumerable that contains the item or items chosen.
        /// </remarks>
        public event Action<IEnumerable<object>> onItemsChosen;

        /// <summary>
        /// Callback triggered when the selection changes.
        /// </summary>
        /// <remarks>
        /// This callback receives an enumerable that contains the item or items selected.
        /// </remarks>
        public event Action<IEnumerable<object>> onSelectionChange;

        /// <summary>
        /// Adds an item to the collection of selected items.
        /// </summary>
        /// <param name="index">Item index.</param>
        public void AddToSelection(int index)
        {
            AddToSelection(new[] { index });
        }

        /// <summary>
        /// Deselects any selected items.
        /// </summary>
        public void ClearSelection()
        {
            if (!HasValidDataAndBindings() || m_SelectedIds.Count == 0)
                return;

            ClearSelectionWithoutValidation();
            NotifyOfSelectionChange();
        }

        /// <summary>
        /// Clears the GridView, recreates all visible visual elements, and rebinds all items.
        /// </summary>
        /// <remarks>
        /// Call this method whenever the data source changes.
        /// </remarks>
        public void Refresh()
        {
            foreach (var recycledRow in m_RowPool)
            {
                recycledRow.Clear();
            }

            m_RowPool.Clear();
            scrollView.Clear();
            m_VisibleRowCount = 0;

            m_SelectedIndices.Clear();
            m_SelectedItems.Clear();

            // O(n)
            if (m_SelectedIds.Count > 0)
            {
                // Add selected objects to working lists.
                for (var index = 0; index < m_ItemsSource.Count; ++index)
                {
                    if (!m_SelectedIds.Contains(GetIdFromIndex(index))) continue;

                    m_SelectedIndices.Add(index);
                    m_SelectedItems.Add(m_ItemsSource[index]);
                }
            }

            if (!HasValidDataAndBindings())
                return;

            m_LastHeight = scrollView.layout.height;

            if (float.IsNaN(m_LastHeight))
                return;

            m_FirstVisibleIndex = Math.Min((int)(m_ScrollOffset / resolvedItemHeight) * columnCount, m_ItemsSource.Count - 1);
            ResizeHeight(m_LastHeight);
        }

        /// <summary>
        /// Rebinds a single item if it is currently visible in the collection view.
        /// </summary>
        /// <param name="index">The item index.</param>
        internal void RefreshItem(int index)
        {
            foreach (var recycledRow in m_RowPool)
            {
                if (recycledRow.ContainsIndex(index, out var indexInRow))
                {
                    var item = makeItem != null && index < itemsSource.Count ? makeItem.Invoke() : CreateDummyItemElement();
                    SetupItemElement(item);

                    recycledRow.RemoveAt(indexInRow);
                    recycledRow.Insert(indexInRow, item);

                    bindItem.Invoke(item, recycledRow.indices[indexInRow]);
                    recycledRow.SetSelected(indexInRow, m_SelectedIds.Contains(recycledRow.ids[indexInRow]));
                    break;
                }
            }
        }

        /// <summary>
        /// Removes an item from the collection of selected items.
        /// </summary>
        /// <param name="index">The item index.</param>
        public void RemoveFromSelection(int index)
        {
            if (!HasValidDataAndBindings())
                return;

            RemoveFromSelectionWithoutValidation(index);
            NotifyOfSelectionChange();

            //SaveViewData();
        }

        /// <summary>
        /// Scrolls to a specific item index and makes it visible.
        /// </summary>
        /// <param name="index">Item index to scroll to. Specify -1 to make the last item visible.</param>
        public void ScrollToItem(int index)
        {
            if (!HasValidDataAndBindings())
                return;

            if (m_VisibleRowCount == 0 || index < -1)
                return;

            var pixelAlignedItemHeight = resolvedItemHeight;
            var actualCount = Math.Min(Mathf.FloorToInt(m_LastHeight / pixelAlignedItemHeight) * columnCount, itemsSource.Count);

            if (index == -1)
            {
                // Scroll to last item
                if (itemsSource.Count < actualCount)
                    scrollView.scrollOffset = new Vector2(0, 0);
                else
                    scrollView.scrollOffset = new Vector2(0, Mathf.FloorToInt(itemsSource.Count / (float)columnCount) * pixelAlignedItemHeight);
            }
            else if (m_FirstVisibleIndex >= index)
            {
                scrollView.scrollOffset = Vector2.up * (pixelAlignedItemHeight * Mathf.FloorToInt(index / (float)columnCount));
            }
            else // index > first
            {
                if (index < m_FirstVisibleIndex + actualCount)
                    return;

                var d = Mathf.FloorToInt(index - actualCount / (float)columnCount);
                var visibleOffset = pixelAlignedItemHeight - (m_LastHeight - Mathf.FloorToInt(actualCount / (float)columnCount) * pixelAlignedItemHeight);
                var yScrollOffset = pixelAlignedItemHeight * d + visibleOffset;

                scrollView.scrollOffset = new Vector2(scrollView.scrollOffset.x, yScrollOffset);
            }
        }

        /// <summary>
        /// Sets the currently selected item.
        /// </summary>
        /// <param name="index">The item index.</param>
        public void SetSelection(int index)
        {
            if (index < 0 || itemsSource == null || index >= itemsSource.Count)
            {
                ClearSelection();
                return;
            }

            SetSelection(new[] { index });
        }

        /// <summary>
        /// Sets a collection of selected items.
        /// </summary>
        /// <param name="indices">The collection of the indices of the items to be selected.</param>
        public void SetSelection(IEnumerable<int> indices)
        {
            switch (selectionType)
            {
                case SelectionType.None:
                    return;
                case SelectionType.Single:
                    if (indices != null)
                        indices = new[] { indices.Last() };
                    break;
                case SelectionType.Multiple:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            SetSelectionInternal(indices, true);
        }

        /// <summary>
        /// Sets a collection of selected items without triggering a selection change callback.
        /// </summary>
        /// <param name="indices">The collection of items to be selected.</param>
        public void SetSelectionWithoutNotify(IEnumerable<int> indices)
        {
            SetSelectionInternal(indices, false);
        }

        internal void AddToSelection(IList<int> indexes)
        {
            if (!HasValidDataAndBindings() || indexes == null || indexes.Count == 0)
                return;

            foreach (var index in indexes)
            {
                AddToSelectionWithoutValidation(index);
            }

            NotifyOfSelectionChange();

            //SaveViewData();
        }

        internal void SelectAll()
        {
            if (!HasValidDataAndBindings())
                return;

            if (selectionType != SelectionType.Multiple)
            {
                return;
            }

            for (var index = 0; index < itemsSource.Count; index++)
            {
                var id = GetIdFromIndex(index);
                var item = m_ItemsSource[index];

                foreach (var recycledRow in m_RowPool)
                {
                    if (recycledRow.ContainsId(id, out var indexInRow))
                        recycledRow.SetSelected(indexInRow, true);
                }

                if (!m_SelectedIds.Contains(id))
                {
                    m_SelectedIds.Add(id);
                    m_SelectedIndices.Add(index);
                    m_SelectedItems.Add(item);
                }
            }

            NotifyOfSelectionChange();

            //SaveViewData();
        }

        internal void SetSelectionInternal(IEnumerable<int> indices, bool sendNotification)
        {
            if (!HasValidDataAndBindings() || indices == null)
                return;

            ClearSelectionWithoutValidation();
            foreach (var index in indices.Where(index => index != -1))
            {
                AddToSelectionWithoutValidation(index);
            }

            if (sendNotification)
                NotifyOfSelectionChange();

            //SaveViewData();
        }

        void AddToSelectionWithoutValidation(int index)
        {
            if (m_SelectedIndices.Contains(index))
                return;

            var id = GetIdFromIndex(index);
            var item = m_ItemsSource[index];

            foreach (var recycledRow in m_RowPool)
            {
                if (recycledRow.ContainsId(id, out var indexInRow))
                    recycledRow.SetSelected(indexInRow, true);
            }

            m_SelectedIds.Add(id);
            m_SelectedIndices.Add(index);
            m_SelectedItems.Add(item);
        }

        void ClearSelectionWithoutValidation()
        {
            foreach (var recycledRow in m_RowPool)
            {
                recycledRow.ClearSelection();
            }

            m_SelectedIds.Clear();
            m_SelectedIndices.Clear();
            m_SelectedItems.Clear();
        }

        VisualElement CreateDummyItemElement()
        {
            var item = new VisualElement();
            SetupItemElement(item);
            return item;
        }

        void DoRangeSelection(int rangeSelectionFinalIndex)
        {
            ClearSelectionWithoutValidation();

            // Add range
            var range = new List<int>();
            if (rangeSelectionFinalIndex < m_RangeSelectionOrigin)
            {
                for (var i = rangeSelectionFinalIndex; i <= m_RangeSelectionOrigin; i++)
                {
                    range.Add(i);
                }
            }
            else
            {
                for (var i = rangeSelectionFinalIndex; i >= m_RangeSelectionOrigin; i--)
                {
                    range.Add(i);
                }
            }

            AddToSelection(range);
        }

        void DoSelect(Vector2 localPosition, int clickCount, bool actionKey, bool shiftKey)
        {
            var clickedIndex = GetIndexByPosition(localPosition);
            if (clickedIndex > m_ItemsSource.Count - 1)
                return;

            var clickedItemId = GetIdFromIndex(clickedIndex);
            switch (clickCount)
            {
                case 1:
                    if (selectionType == SelectionType.None)
                        return;

                    if (selectionType == SelectionType.Multiple && actionKey)
                    {
                        m_RangeSelectionOrigin = clickedIndex;

                        // Add/remove single clicked element
                        if (m_SelectedIds.Contains(clickedItemId))
                            RemoveFromSelection(clickedIndex);
                        else
                            AddToSelection(clickedIndex);
                    }
                    else if (selectionType == SelectionType.Multiple && shiftKey)
                    {
                        if (m_RangeSelectionOrigin == -1)
                        {
                            m_RangeSelectionOrigin = clickedIndex;
                            SetSelection(clickedIndex);
                        }
                        else
                        {
                            DoRangeSelection(clickedIndex);
                        }
                    }
                    else if (selectionType == SelectionType.Multiple && m_SelectedIndices.Contains(clickedIndex))
                    {
                        // Do noting, selection will be processed OnPointerUp.
                        // If drag and drop will be started GridViewDragger will capture the mouse and GridView will not receive the mouse up event.
                    }
                    else // single
                    {
                        m_RangeSelectionOrigin = clickedIndex;
                        SetSelection(clickedIndex);
                    }

                    break;
                case 2:
                    if (onItemsChosen != null)
                    {
                        ProcessSingleClick(clickedIndex);
                    }

                    onItemsChosen?.Invoke(m_SelectedItems);
                    break;
            }
        }

        int GetIdFromIndex(int index)
        {
            if (m_GetItemId == null)
                return index;
            return m_GetItemId(index);
        }

        bool HasValidDataAndBindings()
        {
            return itemsSource != null && makeItem != null && bindItem != null;
        }

        void NotifyOfSelectionChange()
        {
            if (!HasValidDataAndBindings())
                return;

            onSelectionChange?.Invoke(m_SelectedItems);
        }

        void OnAttachToPanel(AttachToPanelEvent evt)
        {
            if (evt.destinationPanel == null)
                return;

            scrollView.contentContainer.RegisterCallback<PointerDownEvent>(OnPointerDown);
            scrollView.contentContainer.RegisterCallback<PointerUpEvent>(OnPointerUp);
            scrollView.contentContainer.RegisterCallback<KeyDownEvent>(OnKeyPress);
        }

        void OnCustomStyleResolved(CustomStyleResolvedEvent e)
        {
            int height;
            if (!m_ItemHeightIsInline && e.customStyle.TryGetValue(s_ItemHeightProperty, out height))
            {
                if (m_ItemHeight != height)
                {
                    m_ItemHeight = height;
                    Refresh();
                }
            }
        }

        void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            if (evt.originPanel == null)
                return;

            scrollView.contentContainer.UnregisterCallback<PointerDownEvent>(OnPointerDown);
            scrollView.contentContainer.UnregisterCallback<PointerUpEvent>(OnPointerUp);
            scrollView.contentContainer.UnregisterCallback<KeyDownEvent>(OnKeyPress);
        }

        void OnPointerDown(PointerDownEvent evt)
        {
            if (!HasValidDataAndBindings())
                return;

            if (!evt.isPrimary)
                return;

            if (evt.button != (int)MouseButton.LeftMouse)
                return;

            if (evt.pointerType != "mouse")
            {
                m_TouchDownTime = evt.timestamp;
                m_TouchDownPosition = evt.position;
                return;
            }

            DoSelect(evt.localPosition, evt.clickCount, evt.actionKey, evt.shiftKey);
        }

        void OnPointerUp(PointerUpEvent evt)
        {
            if (!HasValidDataAndBindings())
                return;

            if (!evt.isPrimary)
                return;

            if (evt.button != (int)MouseButton.LeftMouse)
                return;

            if (evt.pointerType != "mouse")
            {
                var delay = evt.timestamp - m_TouchDownTime;
                var delta = evt.position - m_TouchDownPosition;
                if (delay < 500 && delta.sqrMagnitude <= 100)
                {
                    DoSelect(evt.localPosition, evt.clickCount, evt.actionKey, evt.shiftKey);
                }
            }
            else
            {
                var clickedIndex = GetIndexByPosition(evt.localPosition);
                if (selectionType == SelectionType.Multiple
                    && !evt.shiftKey
                    && !evt.actionKey
                    && m_SelectedIndices.Count > 1
                    && m_SelectedIndices.Contains(clickedIndex))
                {
                    ProcessSingleClick(clickedIndex);
                }
            }
        }

        int GetIndexByPosition(Vector2 localPosition)
        {
            return Mathf.FloorToInt(localPosition.y / resolvedItemHeight) * columnCount + Mathf.FloorToInt(localPosition.x / resolvedItemWidth);
        }

        internal VisualElement GetElementAt(int index)
        {
            foreach (var row in m_RowPool)
            {
                if (row.ContainsId(index, out var indexInRow))
                    return row[indexInRow];
            }

            return null;
        }

        void OnItemsSourceCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            Refresh();
        }

        void OnScroll(float offset)
        {
            if (!HasValidDataAndBindings())
                return;

            m_ScrollOffset = offset;
            var pixelAlignedItemHeight = resolvedItemHeight;
            var firstVisibleIndex = Mathf.FloorToInt(offset / pixelAlignedItemHeight) * columnCount;

            scrollView.contentContainer.style.paddingTop = Mathf.FloorToInt(firstVisibleIndex / (float)columnCount) * pixelAlignedItemHeight;
            scrollView.contentContainer.style.height = (Mathf.CeilToInt(itemsSource.Count / (float)columnCount) * pixelAlignedItemHeight);

            if (firstVisibleIndex != m_FirstVisibleIndex)
            {
                m_FirstVisibleIndex = firstVisibleIndex;

                if (m_RowPool.Count > 0)
                {
                    // we try to avoid rebinding a few items
                    if (m_FirstVisibleIndex < m_RowPool[0].firstIndex) //we're scrolling up
                    {
                        //How many do we have to swap back
                        var count = m_RowPool[0].firstIndex - m_FirstVisibleIndex;

                        var inserting = m_ScrollInsertionList;

                        for (var i = 0; i < count && m_RowPool.Count > 0; ++i)
                        {
                            var last = m_RowPool[m_RowPool.Count - 1];
                            inserting.Add(last);
                            m_RowPool.RemoveAt(m_RowPool.Count - 1); //we remove from the end

                            last.SendToBack(); //We send the element to the top of the list (back in z-order)
                        }

                        inserting.Reverse();

                        m_ScrollInsertionList = m_RowPool;
                        m_RowPool = inserting;
                        m_RowPool.AddRange(m_ScrollInsertionList);
                        m_ScrollInsertionList.Clear();
                    }
                    else if (m_FirstVisibleIndex > m_RowPool[0].firstIndex) //down
                    {
                        var inserting = m_ScrollInsertionList;

                        var checkIndex = 0;
                        while (checkIndex < m_RowPool.Count && m_FirstVisibleIndex > m_RowPool[checkIndex].firstIndex)
                        {
                            var first = m_RowPool[checkIndex];
                            inserting.Add(first);
                            first.BringToFront(); //We send the element to the bottom of the list (front in z-order)
                            checkIndex++;
                        }

                        m_RowPool.RemoveRange(0, checkIndex); //we remove them all at once
                        m_RowPool.AddRange(inserting); // add them back to the end
                        inserting.Clear();
                    }

                    //Let's rebind everything
                    for (var rowIndex = 0; rowIndex < m_RowPool.Count; rowIndex++)
                    {
                        for (var colIndex = 0; colIndex < columnCount; colIndex++)
                        {
                            var index = rowIndex * columnCount + colIndex + m_FirstVisibleIndex;

                            if (index < itemsSource.Count)
                            {
                                var item = m_RowPool[rowIndex].ElementAt(colIndex);
                                if (m_RowPool[rowIndex].indices[colIndex] == RecycledRow.kUndefinedIndex)
                                {
                                    var newItem = makeItem != null ? makeItem.Invoke() : CreateDummyItemElement();
                                    SetupItemElement(newItem);
                                    m_RowPool[rowIndex].RemoveAt(colIndex);
                                    m_RowPool[rowIndex].Insert(colIndex, newItem);
                                    item = newItem;
                                }

                                Setup(item, index);
                            }
                            else
                            {
                                var remainingOldItems = columnCount - colIndex;

                                while (remainingOldItems > 0)
                                {
                                    m_RowPool[rowIndex].RemoveAt(colIndex);
                                    m_RowPool[rowIndex].Insert(colIndex, CreateDummyItemElement());
                                    m_RowPool[rowIndex].ids.RemoveAt(colIndex);
                                    m_RowPool[rowIndex].ids.Insert(colIndex, RecycledRow.kUndefinedIndex);
                                    m_RowPool[rowIndex].indices.RemoveAt(colIndex);
                                    m_RowPool[rowIndex].indices.Insert(colIndex, RecycledRow.kUndefinedIndex);
                                    remainingOldItems--;
                                }
                            }
                        }
                    }
                }
            }
        }

        void OnSizeChanged(GeometryChangedEvent evt)
        {
            if (!HasValidDataAndBindings())
                return;

            if (Mathf.Approximately(evt.newRect.height, evt.oldRect.height))
                return;

            ResizeHeight(evt.newRect.height);
        }

        void ProcessSingleClick(int clickedIndex)
        {
            m_RangeSelectionOrigin = clickedIndex;
            SetSelection(clickedIndex);
        }

        void RemoveFromSelectionWithoutValidation(int index)
        {
            if (!m_SelectedIndices.Contains(index))
                return;

            var id = GetIdFromIndex(index);
            var item = m_ItemsSource[index];

            foreach (var recycledRow in m_RowPool)
            {
                if (recycledRow.ContainsId(id, out var indexInRow))
                    recycledRow.SetSelected(indexInRow, false);
            }

            m_SelectedIds.Remove(id);
            m_SelectedIndices.Remove(index);
            m_SelectedItems.Remove(item);
        }

        void ResizeHeight(float height)
        {
            if (!HasValidDataAndBindings())
                return;

            var pixelAlignedItemHeight = resolvedItemHeight;
            var rowCountForSource = Mathf.CeilToInt(itemsSource.Count / (float)columnCount);
            var contentHeight = rowCountForSource * pixelAlignedItemHeight;
            scrollView.contentContainer.style.height = contentHeight;

            var scrollableHeight = Mathf.Max(0, contentHeight - scrollView.contentViewport.layout.height);
            scrollView.verticalScroller.highValue = scrollableHeight;
            scrollView.verticalScroller.value = Mathf.Min(m_ScrollOffset, scrollView.verticalScroller.highValue);

            var rowCountForHeight = Mathf.FloorToInt(height / pixelAlignedItemHeight) + k_ExtraVisibleRows;
            var rowCount = Math.Min(rowCountForHeight, rowCountForSource);

            if (m_VisibleRowCount != rowCount)
            {
                if (m_VisibleRowCount > rowCount)
                {
                    // Shrink
                    var removeCount = m_VisibleRowCount - rowCount;
                    for (var i = 0; i < removeCount; i++)
                    {
                        var lastIndex = m_RowPool.Count - 1;
                        m_RowPool[lastIndex].Clear();
                        scrollView.Remove(m_RowPool[lastIndex]);
                        m_RowPool.RemoveAt(lastIndex);
                    }
                }
                else
                {
                    // Grow
                    var addCount = rowCount - m_VisibleRowCount;
                    for (var i = 0; i < addCount; i++)
                    {
                        var recycledRow = new RecycledRow(resolvedItemHeight);

                        for (var indexInRow = 0; indexInRow < columnCount; indexInRow++)
                        {
                            var index = m_FirstVisibleIndex + m_RowPool.Count * columnCount + indexInRow;
                            var item = makeItem != null && index < itemsSource.Count ? makeItem.Invoke() : CreateDummyItemElement();
                            SetupItemElement(item);

                            recycledRow.Add(item);

                            if (index < itemsSource.Count)
                            {
                                Setup(item, index);
                            }
                            else
                            {
                                recycledRow.ids.Add(RecycledRow.kUndefinedIndex);
                                recycledRow.indices.Add(RecycledRow.kUndefinedIndex);
                            }
                        }

                        m_RowPool.Add(recycledRow);
                        recycledRow.style.height = pixelAlignedItemHeight;

                        scrollView.Add(recycledRow);
                    }
                }

                m_VisibleRowCount = rowCount;
            }

            m_LastHeight = height;
        }

        void Setup(VisualElement item, int newIndex)
        {
            var newId = GetIdFromIndex(newIndex);

            if (!(item.parent is RecycledRow recycledRow))
                throw new Exception("The item to setup can't be orphan");

            var indexInRow = recycledRow.IndexOf(item);

            if (recycledRow.indices.Count <= indexInRow)
            {
                recycledRow.indices.Add(RecycledRow.kUndefinedIndex);
                recycledRow.ids.Add(RecycledRow.kUndefinedIndex);
            }

            if (recycledRow.indices[indexInRow] == newIndex)
                return;

            if (recycledRow.indices[indexInRow] != RecycledRow.kUndefinedIndex)
                unbindItem?.Invoke(item, recycledRow.indices[indexInRow]);

            recycledRow.indices[indexInRow] = newIndex;
            recycledRow.ids[indexInRow] = newId;

            bindItem.Invoke(item, recycledRow.indices[indexInRow]);

            recycledRow.SetSelected(indexInRow, m_SelectedIds.Contains(recycledRow.ids[indexInRow]));
        }

        void SetupItemElement(VisualElement item)
        {
            item.AddToClassList(k_ItemUssClassName);
            item.style.position = Position.Relative;
            item.style.height = itemHeight;
            item.style.width = itemWidth;
        }

        internal class RecycledRow : VisualElement
        {
            public const int kUndefinedIndex = -1;

            public readonly List<int> ids;

            public readonly List<int> indices;

            public RecycledRow(float height)
            {
                AddToClassList(k_RowUssClassName);
                style.height = height;

                indices = new List<int>();
                ids = new List<int>();
            }

            public int firstIndex => indices.Count > 0 ? indices[0] : kUndefinedIndex;
            public int lastIndex => indices.Count > 0 ? indices[indices.Count - 1] : kUndefinedIndex;

            public void ClearSelection()
            {
                for (var i = 0; i < childCount; i++)
                {
                    SetSelected(i, false);
                }
            }

            public bool ContainsId(int id, out int indexInRow)
            {
                indexInRow = ids.IndexOf(id);
                return indexInRow >= 0;
            }

            public bool ContainsIndex(int index, out int indexInRow)
            {
                indexInRow = indices.IndexOf(index);
                return indexInRow >= 0;
            }

            public void SetSelected(int indexInRow, bool selected)
            {
                if (childCount > indexInRow && indexInRow >= 0)
                {
                    if (selected)
                    {
                        AddElementToClass(ElementAt(indexInRow), itemSelectedVariantUssClassName, true);
                    }
                    else
                    {
                        RemoveElementFromClass(ElementAt(indexInRow), itemSelectedVariantUssClassName, true);
                    }
                }
            }

            static void AddElementToClass(VisualElement element, string className, bool includeChildren = false)
            {
                element.AddToClassList(className);
                if (includeChildren)
                {
                    foreach (var child in element.Children())
                        child.AddToClassList(className);
                }
            }
            static void RemoveElementFromClass(VisualElement element, string className, bool includeChildren = false)
            {
                element.RemoveFromClassList(className);
                if (includeChildren)
                {
                    foreach (var child in element.Children())
                        child.RemoveFromClassList(className);
                }
            }
        }
    }
}
