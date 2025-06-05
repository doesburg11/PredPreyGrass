using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using UnityEngine;
using UnityEngine.UIElements;
#if ENABLE_RUNTIME_DATA_BINDINGS
using Unity.Properties;
#endif

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    /// <summary>
    /// A view containing recycled rows with items inside.
    /// </summary>
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class GridView : BindableElement, ISerializationCallbackReceiver
    {
#if ENABLE_RUNTIME_DATA_BINDINGS
        internal static readonly BindingId itemHeightProperty = new BindingId(nameof(itemHeight));
        internal static readonly BindingId itemSquareProperty = new BindingId(nameof(itemSquare));
        internal static readonly BindingId selectionTypeProperty = new BindingId(nameof(selectionType));
        internal static readonly BindingId allowNoSelectionProperty = new BindingId(nameof(allowNoSelection));
#endif

        /// <summary>
        /// Available Operations.
        /// </summary>
        [Flags]
        public enum GridOperations
        {
            /// <summary>
            /// No operation.
            /// </summary>
            None = 0,

            /// <summary>
            /// Select all items.
            /// </summary>
            SelectAll = 1 << 0,

            /// <summary>
            /// Cancel selection.
            /// </summary>
            Cancel = 1 << 1,

            /// <summary>
            /// Move selection cursor left.
            /// </summary>
            Left = 1 << 2,

            /// <summary>
            /// Move selection cursor right.
            /// </summary>
            Right = 1 << 3,

            /// <summary>
            /// Move selection cursor up.
            /// </summary>
            Up = 1 << 4,

            /// <summary>
            /// Move selection cursor down.
            /// </summary>
            Down = 1 << 5,

            /// <summary>
            /// Move selection cursor to the beginning of the list.
            /// </summary>
            Begin = 1 << 6,

            /// <summary>
            /// Move selection cursor to the end of the list.
            /// </summary>
            End = 1 << 7,

            /// <summary>
            /// Choose selected items.
            /// </summary>
            Choose = 1 << 8,
        }

        const float k_PageSizeFactor = 0.25f;

        const int k_ExtraVisibleRows = 2;

        /// <summary>
        /// The USS class name for GridView elements.
        /// </summary>
        /// <remarks>
        /// Unity adds this USS class to every instance of the GridView element. Any styling applied to
        /// this class affects every GridView located beside, or below the stylesheet in the visual tree.
        /// </remarks>
        const string k_UssClassName = "unity2d-grid-view";

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
        /// The first column USS class name for GridView elements.
        /// </summary>
        static readonly string k_FirstColumnUssClassName = k_UssClassName + "__first-column";

        /// <summary>
        /// The last column USS class name for GridView elements.
        /// </summary>
        static readonly string k_LastColumnUssClassName = k_UssClassName + "__last-column";

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

        internal const int defaultItemHeight = 30;

        const float k_DpiScaling = 1f;

        const bool k_DefaultPreventScrollWithModifiers = true;

        static CustomStyleProperty<int> s_ItemHeightProperty = new CustomStyleProperty<int>("--unity-item-height");

        /// <summary>
        /// The <see cref="UnityEngine.UIElements.ScrollView"/> used by the GridView.
        /// </summary>
        public ScrollView scrollView { get; }

        Func<VisualElement> m_MakeItem;
        Action<VisualElement, int> m_BindItem;
        Func<int, int> m_GetItemId;

        readonly List<int> m_SelectedIds = new List<int>();
        readonly List<int> m_SelectedIndices = new List<int>();
        readonly List<object> m_SelectedItems = new List<object>();
        readonly List<int> m_PreviouslySelectedIndices = new List<int>();
        readonly List<int> m_OriginalSelection = new List<int>();

        float m_OriginalScrollOffset;

        int m_SoftSelectIndex = -1;

        int m_ColumnCount = 1;

        int m_FirstVisibleIndex;

        int m_ItemHeight = defaultItemHeight;
        float m_LastHeight;

        bool m_ItemHeightIsInline;
        bool m_ItemSquare;

        IList m_ItemsSource;

        int m_RangeSelectionOrigin = -1;

        bool m_IsRangeSelectionDirectionUp;

        List<RecycledRow> m_RowPool = new List<RecycledRow>();

        // We keep this list in order to minimize temporary gc allocations.
        List<RecycledRow> m_ScrollInsertionList = new List<RecycledRow>();

        // Persisted.
        float m_ScrollOffset;

        SelectionType m_SelectionType;

        bool m_AllowNoSelection = true;

        int m_VisibleRowCount;

        bool m_IsList;

        NavigationMoveEvent m_NavigationMoveAdapter;

        NavigationCancelEvent m_NavigationCancelAdapter;

        bool m_HasPointerMoved;

        bool m_SoftSelectIndexWasPreviouslySelected;

        /// <summary>
        /// Creates a <see cref="GridView"/> with all default properties. The <see cref="GridView.itemsSource"/>,
        /// <see cref="GridView.itemHeight"/>, <see cref="GridView.makeItem"/> and <see cref="GridView.bindItem"/> properties
        /// must all be set for the GridView to function properly.
        /// </summary>
        public GridView()
        {
            AddToClassList(k_UssClassName);

            styleSheets.Add(ResourceLoader.Load<StyleSheet>("SpriteLibraryEditor/GridView.uss"));

            selectionType = SelectionType.Single;
            m_ScrollOffset = 0.0f;

            scrollView = new ScrollView
            {
                viewDataKey = "grid-view__scroll-view",
                horizontalScrollerVisibility = ScrollerVisibility.Hidden
            };
            scrollView.StretchToParentSize();

            scrollView.verticalScroller.valueChanged += OnScroll;

            dragger = new Dragger(OnDraggerStarted, OnDraggerMoved, OnDraggerEnded, OnDraggerCanceled);

            RegisterCallback<GeometryChangedEvent>(OnSizeChanged);
            RegisterCallback<CustomStyleResolvedEvent>(OnCustomStyleResolved);

            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            hierarchy.Add(scrollView);

            scrollView.contentContainer.usageHints &= ~UsageHints.GroupTransform; // Scroll views with virtualized content shouldn't have the "view transform" optimization

            focusable = true;
            dragger.acceptStartDrag = DefaultAcceptStartDrag;
        }

        /// <summary>
        /// Constructs a <see cref="GridView"/>, with all required properties provided.
        /// </summary>
        /// <param name="itemsSource">The list of items to use as a data source.</param>
        /// <param name="makeItem">The factory method to call to create a display item. The method should return a
        /// VisualElement that can be bound to a data item.</param>
        /// <param name="bindItem">The method to call to bind a data item to a display item. The method
        /// receives as parameters the display item to bind, and the index of the data item to bind it to.</param>
        public GridView(IList itemsSource, Func<VisualElement> makeItem, Action<VisualElement, int> bindItem)
            : this()
        {
            m_ItemsSource = itemsSource;
            m_ItemHeightIsInline = true;

            m_MakeItem = makeItem;
            m_BindItem = bindItem;

            operationMask = ~GridOperations.None;
        }

        bool Apply(GridOperations operation, bool shiftKey)
        {
            if ((operation & operationMask) == 0)
                return false;

            void HandleSelectionAndScroll(int index)
            {
                if (selectionType == SelectionType.Multiple && shiftKey && m_SelectedIndices.Count != 0)
                    DoRangeSelection(index, true, true);
                else
                    selectedIndex = index;

                ScrollToItem(index);
            }

            switch (operation)
            {
                case GridOperations.None:
                    break;
                case GridOperations.SelectAll:
                    SelectAll();
                    return true;
                case GridOperations.Cancel:
                    ClearSelection();
                    return true;
                case GridOperations.Left:
                {
                    var newIndex = Mathf.Max(selectedIndex - 1, 0);
                    if (newIndex != selectedIndex)
                    {
                        HandleSelectionAndScroll(newIndex);
                        return true;
                    }
                }
                    break;
                case GridOperations.Right:
                {
                    var newIndex = Mathf.Min(selectedIndex + 1, itemsSource.Count - 1);
                    if (newIndex != selectedIndex)
                    {
                        HandleSelectionAndScroll(newIndex);
                        return true;
                    }
                }
                    break;
                case GridOperations.Up:
                {
                    var newIndex = Mathf.Max(selectedIndex - columnCount, 0);
                    if (newIndex != selectedIndex)
                    {
                        HandleSelectionAndScroll(newIndex);
                        return true;
                    }
                }
                    break;
                case GridOperations.Down:
                {
                    var newIndex = Mathf.Min(selectedIndex + columnCount, itemsSource.Count - 1);
                    if (newIndex != selectedIndex)
                    {
                        HandleSelectionAndScroll(newIndex);
                        return true;
                    }
                }
                    break;
                case GridOperations.Begin:
                    HandleSelectionAndScroll(0);
                    return true;
                case GridOperations.End:
                    HandleSelectionAndScroll(itemsSource.Count - 1);
                    return true;
                case GridOperations.Choose:
                    if (m_SelectedIndices.Count > 0)
                        itemsChosen?.Invoke(selectedItems);
                    return true;
                default:
                    throw new ArgumentOutOfRangeException(nameof(operation), operation, null);
            }

            return false;
        }

        void Apply(GridOperations operation, EventBase sourceEvent)
        {
            if ((operation & operationMask) != 0 && Apply(operation, (sourceEvent as IKeyboardEvent)?.shiftKey ?? false))
            {
                sourceEvent?.StopPropagation();
            }
        }

        /// <summary>
        /// Internal use only.
        /// </summary>
        /// <param name="operation"></param>
        internal void Apply(GridOperations operation) => Apply(operation, null);

        void OnDraggerStarted(PointerMoveEvent evt)
        {
            dragStarted?.Invoke(evt);
        }

        void OnDraggerMoved(PointerMoveEvent evt)
        {
            dragUpdated?.Invoke(evt);
        }

        void OnDraggerEnded(PointerUpEvent evt)
        {
            dragFinished?.Invoke(evt);
            CancelSoftSelect();
        }

        void OnDraggerCanceled()
        {
            dragCanceled?.Invoke();
            CancelSoftSelect();
        }

        /// <summary>
        /// Cancel drag operation.
        /// </summary>
        public void CancelDrag()
        {
            dragger?.Cancel();
        }

        void OnKeyDown(KeyDownEvent evt)
        {
            var operation = evt.keyCode switch
            {
                KeyCode.A when evt.actionKey => GridOperations.SelectAll,
                KeyCode.Escape => GridOperations.Cancel,
                KeyCode.Home => GridOperations.Begin,
                KeyCode.End => GridOperations.End,
                KeyCode.UpArrow => GridOperations.Up,
                KeyCode.DownArrow => GridOperations.Down,
                KeyCode.LeftArrow => GridOperations.Left,
                KeyCode.RightArrow => GridOperations.Right,
                KeyCode.KeypadEnter or KeyCode.Return => GridOperations.Choose,
                _ => GridOperations.None
            };

            Apply(operation, evt);
        }

        static void OnNavigationMove(NavigationMoveEvent evt) => evt.StopPropagation();
        static void OnNavigationCancel(NavigationCancelEvent evt) => evt.StopPropagation();

        /// <summary>
        /// Callback for binding a data item to the visual element.
        /// </summary>
        /// <remarks>
        /// The method called by this callback receives the VisualElement to bind, and the index of the
        /// element to bind it to.
        /// </remarks>
        public Action<VisualElement, int> bindItem
        {
            get => m_BindItem;
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
        /// The <see cref="Dragger"/> manipulator used by this <see cref="GridView"/>.
        /// </summary>
        public Dragger dragger { get; }

        /// <summary>
        /// A mask describing available operations in this <see cref="GridView"/> when the user interacts with it.
        /// </summary>
        public GridOperations operationMask { get; set; } =
            GridOperations.Choose |
            GridOperations.SelectAll | GridOperations.Cancel |
            GridOperations.Begin | GridOperations.End |
            GridOperations.Left | GridOperations.Right |
            GridOperations.Up | GridOperations.Down;

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
#if ENABLE_RUNTIME_DATA_BINDINGS
        [CreateProperty]
#endif
#if ENABLE_UXML_SERIALIZED_DATA
        [UxmlAttribute]
#endif
        public int itemHeight
        {
            get => m_ItemHeight;
            set
            {
                if (m_ItemHeight != value && value > 0)
                {
                    m_ItemHeightIsInline = true;
                    m_ItemHeight = value;
                    scrollView.verticalPageSize = m_ItemHeight * k_PageSizeFactor;
                    Refresh();

#if ENABLE_RUNTIME_DATA_BINDINGS
                    NotifyPropertyChanged(in itemHeightProperty);
#endif
                }
            }
        }

        /// <summary>
        /// Decides whether the item should have 1:1 aspect ratio.
        /// </summary>
#if ENABLE_RUNTIME_DATA_BINDINGS
        [CreateProperty]
#endif
#if ENABLE_UXML_SERIALIZED_DATA
        [UxmlAttribute]
#endif
        public bool itemSquare
        {
            get => m_ItemSquare;
            set
            {
                if (m_ItemSquare != value)
                {
                    m_ItemSquare = value;
                    Refresh();

#if ENABLE_RUNTIME_DATA_BINDINGS
                    NotifyPropertyChanged(in itemSquareProperty);
#endif
                }
            }
        }

        /// <summary>
        /// The width of the item.
        /// </summary>
        public float itemWidth => scrollView.contentViewport.layout.width / columnCount;

        /// <summary>
        /// The data source for list items.
        /// </summary>
        /// <remarks>
        /// This list contains the items that the <see cref="GridView"/> displays.
        /// This property must be set for the list view to function.
        /// </remarks>
        public IList itemsSource
        {
            get => m_ItemsSource;
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
            get => m_MakeItem;
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
        public float resolvedItemHeight => Mathf.Round(itemHeight * k_DpiScaling) / k_DpiScaling;

        /// <summary>
        /// The computed pixel-aligned width for the list elements.
        /// </summary>
        /// <remarks>
        /// This value changes depending on the current panel's DPI scaling.
        /// </remarks>
        /// <seealso cref="GridView.itemWidth"/>
        public float resolvedItemWidth => Mathf.Round(itemWidth * k_DpiScaling) / k_DpiScaling;

        /// <summary>
        /// Returns or sets the selected item's index in the data source. If multiple items are selected, returns the
        /// first selected item's index. If multiple items are provided, sets them all as selected.
        /// </summary>
        public int selectedIndex
        {
            get => m_SelectedIndices.Count == 0 ? -1 : m_SelectedIndices[0];
            set => SetSelection(value);
        }

        /// <summary>
        /// Returns the indices of selected items in the data source. Always returns an enumerable, even if no item  is selected, or a
        /// single item is selected.
        /// </summary>
        public IEnumerable<int> selectedIndices => m_SelectedIndices;

        /// <summary>
        /// Returns the selected item from the data source. If multiple items are selected, returns the first selected item.
        /// </summary>
        public object selectedItem => m_SelectedItems.Count == 0 ? null : m_SelectedItems[0];

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
#if ENABLE_RUNTIME_DATA_BINDINGS
        [CreateProperty]
#endif
#if ENABLE_UXML_SERIALIZED_DATA
        [UxmlAttribute]
#endif
        public SelectionType selectionType
        {
            get { return m_SelectionType; }
            set
            {
                var changed = m_SelectionType != value;
                m_SelectionType = value;

                if (m_SelectionType == SelectionType.None)
                {
                    ClearSelectionWithoutValidation();
                }
                else
                {
                    if (allowNoSelection)
                        ClearSelectionWithoutValidation();
                    else if (m_ItemsSource.Count > 0)
                        SetSelectionInternal(new[] { 0 }, false, false);
                    else
                        ClearSelectionWithoutValidation();
                }

                m_RangeSelectionOrigin = -1;
                PostSelection(updatePreviousSelection: true, sendNotification: true);

#if ENABLE_RUNTIME_DATA_BINDINGS
                if (changed)
                    NotifyPropertyChanged(in selectionTypeProperty);
#endif
            }
        }

        /// <summary>
        /// Whether the GridView allows to have no selection when the selection type is <see cref="SelectionType.Single"/> or <see cref="SelectionType.Multiple"/>.
        /// </summary>
#if ENABLE_RUNTIME_DATA_BINDINGS
        [CreateProperty]
#endif
#if ENABLE_UXML_SERIALIZED_DATA
        [UxmlAttribute]
#endif
        public bool allowNoSelection
        {
            get => m_AllowNoSelection;

            set
            {
                var changed = m_AllowNoSelection != value;
                m_AllowNoSelection = value;
                if (HasValidDataAndBindings() && !m_AllowNoSelection && m_SelectedIndices.Count == 0 && m_ItemsSource.Count > 0)
                    SetSelectionInternal(new[] { 0 }, true, true);

#if ENABLE_RUNTIME_DATA_BINDINGS
                if (changed)
                    NotifyPropertyChanged(in allowNoSelectionProperty);
#endif
            }
        }

        /// <summary>
        /// Returns true if the soft-selection is in progress.
        /// </summary>
        public bool isSelecting => m_SoftSelectIndex != -1;

        /// <summary>
        /// Enable this property to display a border around the GridView.
        /// </summary>
        /// <remarks>
        /// If set to true, a border appears around the ScrollView.
        /// </remarks>
        public bool showBorder
        {
            get => ClassListContains(k_BorderUssClassName);
            set => EnableInClassList(k_BorderUssClassName, value);
        }

        /// <summary>
        /// Prevents the grid view from scrolling when the user presses a modifier key at the same time as scrolling.
        /// </summary>
        public bool preventScrollWithModifiers { get; set; } = k_DefaultPreventScrollWithModifiers;

        /// <summary>
        /// Callback for unbinding a data item from the VisualElement.
        /// </summary>
        /// <remarks>
        /// The method called by this callback receives the VisualElement to unbind, and the index of the
        /// element to unbind it from.
        /// </remarks>
        public Action<VisualElement, int> unbindItem { get; set; }

        /// <summary>
        /// Callback for getting the ID of an item.
        /// </summary>
        /// <remarks>
        /// The method called by this callback receives the index of the item to get the ID from.
        /// </remarks>
        public Func<int, int> getItemId
        {
            get => m_GetItemId;
            set
            {
                m_GetItemId = value;
                Refresh();
            }
        }

        bool DefaultAcceptStartDrag(Vector2 worldPosition)
        {
            if (!HasValidDataAndBindings())
                return false;

            var idx = GetIndexByWorldPosition(worldPosition);
            return idx >= 0 && idx < itemsSource.Count;
        }

        internal List<RecycledRow> rowPool => m_RowPool;

        void ISerializationCallbackReceiver.OnAfterDeserialize()
        {
            Refresh();
        }

        void ISerializationCallbackReceiver.OnBeforeSerialize() { }

        /// <summary>
        /// Callback triggered when the user acts on a selection of one or more items, for example by double-clicking or pressing Enter.
        /// </summary>
        /// <remarks>
        /// This callback receives an enumerable that contains the item or items chosen.
        /// </remarks>
        public event Action<IEnumerable<object>> itemsChosen;

        /// <summary>
        /// Callback triggered when the selection changes.
        /// </summary>
        /// <remarks>
        /// This callback receives an enumerable that contains the item or items selected.
        /// </remarks>
        public event Action<IEnumerable<object>> selectionChanged;

        /// <summary>
        /// Callback triggered when the selection changes.
        /// </summary>
        /// <remarks>
        /// This callback receives an enumerable that contains the indices of selected items.
        /// </remarks>
        public event Action<IEnumerable<int>> selectedIndicesChanged;

        /// <summary>
        /// Callback triggered when the user right-clicks on an item.
        /// </summary>
        /// <remarks>
        /// This callback receives an enumerable that contains the item or items selected.
        /// </remarks>
        public event Action<PointerDownEvent> contextClicked;

        /// <summary>
        /// Callback triggered when the user double-clicks on an item.
        /// </summary>
        public event Action<int> doubleClicked;

        /// <summary>
        /// Callback triggered when drag has started.
        /// </summary>
        public event Action<PointerMoveEvent> dragStarted;

        /// <summary>
        /// Callback triggered when items are dragged.
        /// </summary>
        public event Action<PointerMoveEvent> dragUpdated;

        /// <summary>
        /// Callback triggered when drag has finished.
        /// </summary>
        public event Action<PointerUpEvent> dragFinished;

        /// <summary>
        /// Callback triggered when drag has been canceled.
        /// </summary>
        public event Action dragCanceled;

        /// <summary>
        /// Adds an item to the collection of selected items.
        /// </summary>
        /// <param name="index">Item index.</param>
        public void AddToSelection(int index)
        {
            AddToSelection(new[] { index }, true, true);
        }

        internal void AddToSelection(int index, bool updatePrevious, bool notify)
        {
            AddToSelection(new[] { index }, updatePrevious, notify);
        }

        /// <summary>
        /// Deselects any selected items.
        /// </summary>
        public void ClearSelection()
        {
            if (m_SelectedIndices.Count == 0 || !allowNoSelection)
                return;

            ClearSelectionWithoutValidation();
            PostSelection(true, true);
        }

        /// <summary>
        /// Clear the selection without triggering selection changed event.
        /// </summary>
        public void ClearSelectionWithoutNotify()
        {
            if (m_SelectedIndices.Count == 0 || !allowNoSelection)
                return;

            ClearSelectionWithoutValidation();
            m_RangeSelectionOrigin = -1;
            m_PreviouslySelectedIndices.Clear();
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
            m_SoftSelectIndex = -1;
            m_SoftSelectIndexWasPreviouslySelected = false;
            m_OriginalSelection.Clear();

            var newSelectedIds = new List<int>();

            // O(n)
            if (m_SelectedIds.Count > 0)
            {
                // Add selected objects to working lists.
                for (var index = 0; index < m_ItemsSource.Count; ++index)
                {
                    var id = GetIdFromIndex(index);
                    if (!m_SelectedIds.Contains(id))
                        continue;

                    m_SelectedIndices.Add(index);
                    m_SelectedItems.Add(m_ItemsSource[index]);
                    newSelectedIds.Add(id);
                }
            }

            m_SelectedIds.Clear();
            m_SelectedIds.AddRange(newSelectedIds);

            if (!HasValidDataAndBindings())
                return;

            m_LastHeight = scrollView.layout.height;

            if (float.IsNaN(m_LastHeight))
                return;

            m_FirstVisibleIndex = Math.Min((int)(m_ScrollOffset / resolvedItemHeight) * columnCount, m_ItemsSource.Count - 1);
            ResizeHeight(m_LastHeight);

            if (!allowNoSelection && m_SelectedIds.Count == 0)
            {
                if (m_ItemsSource.Count > 0)
                    SetSelectionInternal(new[]
                    {
                        m_FirstVisibleIndex >= 0 ? m_FirstVisibleIndex : 0
                    }, true, true);
            }
            else
            {
                PostSelection(true, true);
            }
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
            RemoveFromSelectionInternal(index, true, true);
        }

        internal void RemoveFromSelectionInternal(int index, bool updatePrevious, bool notify)
        {
            if (!HasValidDataAndBindings())
                return;

            if (m_SelectedIndices.Count == 1 && m_SelectedIndices[0] == index && !allowNoSelection)
                return;

            RemoveFromSelectionWithoutValidation(index);

            PostSelection(updatePrevious, notify);
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
            var lastRowIndex = Mathf.FloorToInt((itemsSource.Count - 1) / (float)columnCount);
            var maxOffset = Mathf.Max(0, lastRowIndex * pixelAlignedItemHeight - m_LastHeight + pixelAlignedItemHeight);
            var targetRowIndex = Mathf.FloorToInt(index / (float)columnCount);
            var targetOffset = targetRowIndex * pixelAlignedItemHeight;
            var currentOffset = scrollView.scrollOffset.y;
            var d = targetOffset - currentOffset;

            if (index == -1)
            {
                scrollView.scrollOffset = Vector2.up * maxOffset;
            }
            else if (d < 0)
            {
                scrollView.scrollOffset = Vector2.up * targetOffset;
            }
            else if (d > m_LastHeight - pixelAlignedItemHeight)
            {
                // need to scroll up so the item should be visible in last row
                targetOffset += pixelAlignedItemHeight - m_LastHeight;
                scrollView.scrollOffset = Vector2.up * Mathf.Min(maxOffset, targetOffset);
            }

            // else do nothing because the item is already entirely visible

            schedule.Execute(() => ResizeHeight(m_LastHeight)).ExecuteLater(2L);
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
            SetSelectionInternal(indices, true, true);
        }

        /// <summary>
        /// Sets a collection of selected items without triggering a selection change callback.
        /// </summary>
        /// <param name="indices">The collection of items to be selected.</param>
        public void SetSelectionWithoutNotify(IEnumerable<int> indices)
        {
            indices ??= Array.Empty<int>();

            switch (selectionType)
            {
                case SelectionType.None:
                    return;
                case SelectionType.Single:
                    var lastIndex = -1;
                    var count = 0;
                    foreach (var index in indices)
                    {
                        lastIndex = index;
                        count++;
                    }

                    if (count > 1)
                        indices = new[] { lastIndex };
                    break;
                case SelectionType.Multiple:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            SetSelectionInternal(indices, true, false);
        }

        internal void AddToSelection(IList<int> indexes, bool updatePrevious, bool notify)
        {
            if (!HasValidDataAndBindings() || indexes == null || indexes.Count == 0)
                return;

            foreach (var index in indexes)
                AddToSelectionWithoutValidation(index);

            PostSelection(updatePrevious, notify);
        }

        internal void SelectAll()
        {
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

            PostSelection(true, true);
        }

        internal void SetSelectionInternal(IEnumerable<int> indices, bool updatePrevious, bool sendNotification)
        {
            indices ??= Array.Empty<int>();

            switch (selectionType)
            {
                case SelectionType.None:
                    return;
                case SelectionType.Single:
                    var lastIndex = -1;
                    var count = 0;
                    foreach (var index in indices)
                    {
                        lastIndex = index;
                        count++;
                    }

                    if (count > 1)
                        indices = new[] { lastIndex };
                    break;
                case SelectionType.Multiple:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (!allowNoSelection)
            {
                // Check if empty.
                using var enumerator = indices.GetEnumerator();
                if (!enumerator.MoveNext())
                    return;
            }

            ClearSelectionWithoutValidation();
            foreach (var index in indices)
                AddToSelectionWithoutValidation(index);
            PostSelection(updatePrevious, sendNotification);
        }

        void AddToSelectionWithoutValidation(int index)
        {
            if (m_ItemsSource == null || index < 0 || index >= m_ItemsSource.Count || m_SelectedIndices.Contains(index))
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

        internal VisualElement GetVisualElement(int index)
        {
            if (index < 0 || index >= m_ItemsSource.Count || !m_SelectedIndices.Contains(index))
                return null;

            return GetVisualElementInternal(index);
        }

        internal VisualElement GetVisualElementWithoutSelection(int index)
        {
            if (index < 0 || index >= m_ItemsSource.Count)
                return null;

            return GetVisualElementInternal(index);
        }

        VisualElement GetVisualElementInternal(int index)
        {
            var id = GetIdFromIndex(index);

            foreach (var recycledRow in m_RowPool)
            {
                if (recycledRow.ContainsId(id, out var indexInRow))
                    return recycledRow.ElementAt(indexInRow);
            }

            return null;
        }

        void ClearSelectionWithoutValidation()
        {
            foreach (var recycledRow in m_RowPool)
                recycledRow.ClearSelection();

            m_SelectedIds.Clear();
            m_SelectedIndices.Clear();
            m_SelectedItems.Clear();
        }

        VisualElement CreateDummyItemElement()
        {
            var item = new VisualElement { pickingMode = PickingMode.Ignore };
            SetupItemElement(item);
            return item;
        }

        void DoRangeSelection(int rangeSelectionFinalIndex, bool updatePrevious, bool notify)
        {
            var max = -1;
            var min = -1;
            foreach (var i in m_SelectedIndices)
            {
                if (max == -1 || i > max)
                    max = i;
                if (min == -1 || i < min)
                    min = i;
            }

            m_RangeSelectionOrigin = m_IsRangeSelectionDirectionUp ? max : min;

            ClearSelectionWithoutValidation();

            // Add range
            var range = new List<int>();
            m_IsRangeSelectionDirectionUp = rangeSelectionFinalIndex < m_RangeSelectionOrigin;
            if (m_IsRangeSelectionDirectionUp)
            {
                for (var i = rangeSelectionFinalIndex; i <= m_RangeSelectionOrigin; i++)
                    range.Add(i);
            }
            else
            {
                for (var i = rangeSelectionFinalIndex; i >= m_RangeSelectionOrigin; i--)
                    range.Add(i);
            }

            AddToSelection(range, updatePrevious, notify);
        }

        void DoContextClickAfterSelect(PointerDownEvent evt)
        {
            contextClicked?.Invoke(evt);
        }

        void DoSoftSelect(PointerDownEvent evt, int clickCount)
        {
            var clickedIndex = GetIndexByWorldPosition(evt.position);
            if (clickedIndex > m_ItemsSource.Count - 1 || clickedIndex < 0)
            {
                if (evt.button == (int)MouseButton.LeftMouse && allowNoSelection)
                    ClearSelection();
                return;
            }

            m_SoftSelectIndex = clickedIndex;
            m_SoftSelectIndexWasPreviouslySelected = m_SelectedIndices.Contains(clickedIndex);

            if (clickCount == 1)
            {
                if (selectionType == SelectionType.None)
                    return;

                if (selectionType == SelectionType.Multiple && evt.actionKey)
                {
                    m_RangeSelectionOrigin = clickedIndex;

                    // Add/remove single clicked element
                    var clickedItemId = GetIdFromIndex(clickedIndex);
                    if (m_SelectedIds.Contains(clickedItemId))
                        RemoveFromSelectionInternal(clickedIndex, false, false);
                    else
                        AddToSelection(clickedIndex, false, false);
                }
                else if (selectionType == SelectionType.Multiple && evt.shiftKey)
                {
                    if (m_RangeSelectionOrigin == -1 || m_SelectedIndices.Count == 0)
                    {
                        m_RangeSelectionOrigin = clickedIndex;
                        SetSelectionInternal(new[] { clickedIndex }, false, false);
                    }
                    else
                    {
                        DoRangeSelection(clickedIndex, false, false);
                    }
                }
                else if (selectionType == SelectionType.Multiple && m_SoftSelectIndexWasPreviouslySelected)
                {
                    // Do noting, selection will be processed OnPointerUp.
                    // If drag and drop will be started GridViewDragger will capture the mouse and GridView will not receive the mouse up event.
                }
                else // single
                {
                    m_RangeSelectionOrigin = clickedIndex;
                    if (!(m_SelectedIndices.Count == 1 && m_SelectedIndices[0] == clickedIndex))
                    {
                        SetSelectionInternal(new[] { clickedIndex }, false, false);
                    }
                }
            }

            ScrollToItem(clickedIndex);
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

        void PostSelection(bool updatePreviousSelection, bool sendNotification)
        {
            if (AreSequencesEqual(m_PreviouslySelectedIndices, m_SelectedIndices))
                return;

            if (updatePreviousSelection)
            {
                m_PreviouslySelectedIndices.Clear();
                m_PreviouslySelectedIndices.AddRange(m_SelectedIndices);
            }

            if (sendNotification)
            {
                selectionChanged?.Invoke(m_SelectedItems);
                selectedIndicesChanged?.Invoke(m_SelectedIndices);
            }
        }

        void OnAttachToPanel(AttachToPanelEvent evt)
        {
            if (evt.destinationPanel == null)
                return;

            if (!UnityEngine.Device.Application.isMobilePlatform)
                scrollView.AddManipulator(dragger);

            scrollView.RegisterCallback<ClickEvent>(OnClick);
            scrollView.RegisterCallback<PointerDownEvent>(OnPointerDown);
            scrollView.RegisterCallback<PointerUpEvent>(OnPointerUp);
            scrollView.RegisterCallback<PointerMoveEvent>(OnPointerMove);
            RegisterCallback<KeyDownEvent>(OnKeyDown);
            RegisterCallback<NavigationMoveEvent>(OnNavigationMove);
            RegisterCallback<NavigationCancelEvent>(OnNavigationCancel);
            scrollView.RegisterCallback<WheelEvent>(OnWheel, TrickleDown.TrickleDown);
        }

        void OnCustomStyleResolved(CustomStyleResolvedEvent e)
        {
            if (!m_ItemHeightIsInline && e.customStyle.TryGetValue(s_ItemHeightProperty, out var height))
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

            scrollView.RemoveManipulator(dragger);

            scrollView.UnregisterCallback<ClickEvent>(OnClick);
            scrollView.UnregisterCallback<PointerDownEvent>(OnPointerDown);
            scrollView.UnregisterCallback<PointerUpEvent>(OnPointerUp);
            scrollView.UnregisterCallback<PointerMoveEvent>(OnPointerMove);
            UnregisterCallback<KeyDownEvent>(OnKeyDown);
            UnregisterCallback<NavigationMoveEvent>(OnNavigationMove);
            UnregisterCallback<NavigationCancelEvent>(OnNavigationCancel);
            scrollView.UnregisterCallback<WheelEvent>(OnWheel, TrickleDown.TrickleDown);
        }

        void OnWheel(WheelEvent evt)
        {
            if (preventScrollWithModifiers && evt.modifiers != EventModifiers.None)
                evt.StopImmediatePropagation();
        }

        void OnClick(ClickEvent evt)
        {
            if (!HasValidDataAndBindings())
                return;

            if (evt.clickCount == 2)
            {
                var clickedIndex = GetIndexByWorldPosition(evt.position);
                if (clickedIndex >= 0 && clickedIndex < m_ItemsSource.Count)
                {
                    doubleClicked?.Invoke(clickedIndex);
                    Apply(GridOperations.Choose, evt.shiftKey);
                }
            }
        }

        void OnPointerMove(PointerMoveEvent evt)
        {
            m_HasPointerMoved = true;
        }

        void OnPointerDown(PointerDownEvent evt)
        {
            evt.StopImmediatePropagation();
            if (!HasValidDataAndBindings())
                return;

            if (!evt.isPrimary)
                return;

            var capturingElement = panel?.GetCapturingElement(evt.pointerId);

            // if the pointer is captured by a child of the scroll view, abort any selection
            if (capturingElement is VisualElement ve &&
                ve != scrollView &&
                ve.FindCommonAncestor(scrollView) != null)
                return;

            m_OriginalSelection.Clear();
            m_OriginalSelection.AddRange(m_SelectedIndices);
            m_OriginalScrollOffset = m_ScrollOffset;
            m_SoftSelectIndex = -1;

            var clickCount = m_HasPointerMoved ? 1 : evt.clickCount;
            m_HasPointerMoved = false;

            DoSoftSelect(evt, clickCount);

            if (evt.button == (int)MouseButton.RightMouse)
                DoContextClickAfterSelect(evt);
        }

        void OnPointerUp(PointerUpEvent evt)
        {
            if (!HasValidDataAndBindings())
                return;

            if (!evt.isPrimary)
                return;

            if (m_SoftSelectIndex == -1)
                return;

            var index = m_SoftSelectIndex;
            m_SoftSelectIndex = -1;

            if (m_SoftSelectIndexWasPreviouslySelected &&
                evt.button == (int)MouseButton.LeftMouse &&
                evt.modifiers == EventModifiers.None)
            {
                ProcessSingleClick(index);
                return;
            }

            PostSelection(true, true);
        }

        void CancelSoftSelect()
        {
            if (m_SoftSelectIndex != -1)
            {
                SetSelectionInternal(m_OriginalSelection, false, false);
                scrollView.verticalScroller.value = m_OriginalScrollOffset;
            }

            m_SoftSelectIndex = -1;
        }

        /// <summary>
        /// Returns the index of the item at the given position.
        /// </summary>
        /// <remarks>
        /// The position is relative to the top left corner of the grid. No check is made to see if the index is valid.
        /// </remarks>
        /// <param name="worldPosition">The position of the item in the world-space.</param>
        /// <returns> The index of the item at the given position.</returns>
        public int GetIndexByWorldPosition(Vector2 worldPosition)
        {
            var localPosition = scrollView.contentContainer.WorldToLocal(worldPosition);
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
                            var last = m_RowPool[^1];
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

                            var isFirstColumn = colIndex == 0;
                            var isLastColumn = colIndex == columnCount - 1;

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
                                item.EnableInClassList(k_FirstColumnUssClassName, isFirstColumn);
                                item.EnableInClassList(k_LastColumnUssClassName, isLastColumn);
                            }
                            else
                            {
                                var remainingOldItems = columnCount - colIndex;

                                while (remainingOldItems > 0)
                                {
                                    m_RowPool[rowIndex].RemoveAt(colIndex);
                                    m_RowPool[rowIndex].Insert(colIndex, CreateDummyItemElement());
                                    m_RowPool[rowIndex][colIndex].EnableInClassList(k_FirstColumnUssClassName, isFirstColumn);
                                    m_RowPool[rowIndex][colIndex].EnableInClassList(k_LastColumnUssClassName, isLastColumn);
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
                            var index = m_RowPool.Count * columnCount + indexInRow + m_FirstVisibleIndex;
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

                            var isFirstColumn = indexInRow == 0;
                            var isLastColumn = indexInRow == columnCount - 1;
                            item.EnableInClassList(k_FirstColumnUssClassName, isFirstColumn);
                            item.EnableInClassList(k_LastColumnUssClassName, isLastColumn);
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

        static bool AreSequencesEqual<T>(IList<T> first, IList<T> second)
        {
            if (first == null || second == null || first.Count != second.Count)
                return false;

            for (var i = 0; i < first.Count; i++)
            {
                if (!first[i].Equals(second[i]))
                    return false;
            }

            return true;
        }

        void SetupItemElement(VisualElement item)
        {
            item.AddToClassList(k_ItemUssClassName);
            item.style.position = Position.Relative;
            if (itemSquare)
            {
                var itemSize = (float)itemHeight;
                item.style.height = item.style.width = itemSize;
            }
            else
            {
                item.style.flexBasis = 0;
                item.style.flexGrow = 1f;
                item.style.flexShrink = 1f;
            }
        }

#if ENABLE_UXML_TRAITS
        /// <summary>
        /// Instantiates a <see cref="GridView"/> using data from a UXML file.
        /// </summary>
        /// <remarks>
        /// This class is added to every <see cref="VisualElement"/> created from UXML.
        /// </remarks>
        public new class UxmlFactory : UxmlFactory<GridView, UxmlTraits> { }

        /// <summary>
        /// Defines <see cref="UxmlTraits"/> for the <see cref="GridView"/>.
        /// </summary>
        /// <remarks>
        /// This class defines the GridView element properties that you can use in a UI document asset (UXML file).
        /// </remarks>
        public new class UxmlTraits : BindableElement.UxmlTraits
        {
            readonly UxmlIntAttributeDescription m_ItemHeight = new UxmlIntAttributeDescription
            {
                name = "item-height",
                obsoleteNames = new[] { "itemHeight" },
                defaultValue = defaultItemHeight
            };

            readonly UxmlBoolAttributeDescription m_ItemSquare = new UxmlBoolAttributeDescription
            {
                name = "item-square",
                defaultValue = false
            };

            readonly UxmlEnumAttributeDescription<SelectionType> m_SelectionType = new UxmlEnumAttributeDescription<SelectionType>
            {
                name = "selection-type",
                defaultValue = SelectionType.Single
            };

            readonly UxmlBoolAttributeDescription m_ShowBorder = new UxmlBoolAttributeDescription
            {
                name = "show-border",
                defaultValue = false
            };

            readonly UxmlBoolAttributeDescription m_PreventScrollWithModifiers = new UxmlBoolAttributeDescription
            {
                name = "prevent-scroll-with-modifiers",
                defaultValue = k_DefaultPreventScrollWithModifiers
            };

            readonly UxmlBoolAttributeDescription m_AllowNoSelection = new UxmlBoolAttributeDescription
            {
                name = "allow-no-selection",
                defaultValue = true
            };

            /// <summary>
            /// Returns an empty enumerable, because list views usually do not have child elements.
            /// </summary>
            /// <returns>An empty enumerable.</returns>
            public override IEnumerable<UxmlChildElementDescription> uxmlChildElementsDescription
            {
                get { yield break; }
            }

            /// <summary>
            /// Initializes <see cref="GridView"/> properties using values from the attribute bag.
            /// </summary>
            /// <param name="ve">The object to initialize.</param>
            /// <param name="bag">The attribute bag.</param>
            /// <param name="cc">The creation context; unused.</param>
            public override void Init(VisualElement ve, IUxmlAttributes bag, CreationContext cc)
            {
                base.Init(ve, bag, cc);
                var view = (GridView)ve;

                // Avoid setting itemHeight unless it's explicitly defined.
                // Setting itemHeight property will activate inline property mode.
                var itemHeight = 0;
                if (m_ItemHeight.TryGetValueFromBag(bag, cc, ref itemHeight))
                    view.itemHeight = itemHeight;

                view.itemSquare = m_ItemSquare.GetValueFromBag(bag, cc);
                view.showBorder = m_ShowBorder.GetValueFromBag(bag, cc);
                view.preventScrollWithModifiers = m_PreventScrollWithModifiers.GetValueFromBag(bag, cc);
                view.selectionType = m_SelectionType.GetValueFromBag(bag, cc);
                view.allowNoSelection = m_AllowNoSelection.GetValueFromBag(bag, cc);
            }
        }
#endif

        internal class RecycledRow : VisualElement
        {
            public const int kUndefinedIndex = -1;

            public readonly List<int> ids;

            public readonly List<int> indices;

            public RecycledRow(float height)
            {
                AddToClassList(k_RowUssClassName);

                pickingMode = PickingMode.Ignore;

                style.height = height;

                indices = new List<int>();
                ids = new List<int>();
            }

            public int firstIndex => indices.Count > 0 ? indices[0] : kUndefinedIndex;
            public int lastIndex => indices.Count > 0 ? indices[^1] : kUndefinedIndex;

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
                        ElementAt(indexInRow).AddToClassList(itemSelectedVariantUssClassName);
                    else
                        ElementAt(indexInRow).RemoveFromClassList(itemSelectedVariantUssClassName);
                }
            }
        }
    }
}
