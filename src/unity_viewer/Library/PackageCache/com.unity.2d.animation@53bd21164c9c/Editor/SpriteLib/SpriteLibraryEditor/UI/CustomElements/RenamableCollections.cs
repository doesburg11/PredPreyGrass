using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    internal interface IRenamableCollection
    {
        void StartRename();
        void EndRename(bool cancelled = false);

        public Action<int, string> onRename { get; set; }
        public Action onItemsReordered { get; set; }

        /// <summary>
        /// Currently renamed index. If none is being renamed returns -1.
        /// </summary>
        int renamingIndex { get; }

        void SetSourceItems(IList sourceList);
        IList GetItemSource();
        void SetSelectionWithoutNotify(IEnumerable<int> indices);
        void SetElementSize(float newSize);
        void SetWidth(float newWidth);
        void ScrollToItem(int id);

        public const string labelElementName = "SpriteLibraryLabel";
        public const string textFieldElementName = "SpriteLibraryTextField";
    }

    internal class RenamableListView : ListView, IRenamableCollection
    {
        public Action<int, string> onRename { get; set; }
        public Action onItemsReordered { get; set; }

        public delegate bool CanRenameDelegate(int index);

        public CanRenameDelegate CanRenameAtIndex;

        public int renamingIndex { get; private set; } = -1;

        public RenamableListView()
        {
            name = "RenamableCollection";
            reorderable = false;

            pickingMode = PickingMode.Ignore;

            var scrollView = this.Q<ScrollView>();
            scrollView.pickingMode = PickingMode.Ignore;
            scrollView.contentContainer.pickingMode = PickingMode.Ignore;
            scrollView.contentViewport.pickingMode = PickingMode.Ignore;
            scrollView.contentViewport.parent.pickingMode = PickingMode.Ignore;

            itemsChosen += OnItemChosen;
            itemIndexChanged += OnItemIndexChanged;

            RegisterCallback<KeyDownEvent>(OnCollectionKeyDown);

            this.StretchToParentSize();
            fixedItemHeight = 20;
        }

        void OnItemIndexChanged(int oldIndex, int newIndex)
        {
            onItemsReordered?.Invoke();
        }

        void OnCollectionKeyDown(KeyDownEvent evt)
        {
            if (evt.keyCode == KeyCode.Escape)
            {
                evt.StopPropagation();
                EndRename(true);
            }

#if UNITY_EDITOR_WIN || UNITY_EDITOR_LINUX
            if (evt.keyCode == KeyCode.F2)
            {
                evt.StopPropagation();
                schedule.Execute(StartRename);
            }
#endif
        }

        public void SetSourceItems(IList sourceList)
        {
            itemsSource = sourceList;
            RefreshItems();
        }

        public IList GetItemSource() => itemsSource;

        public void SetElementSize(float newSize)
        {
            fixedItemHeight = newSize;
            Rebuild();
        }

        public void SetWidth(float newWidth) { }

        void OnItemChosen(IEnumerable<object> obj)
        {
            schedule.Execute(StartRename);
        }

        public void StartRename()
        {
            if (renamingIndex != -1)
                EndRename(true);

            if (CanRename(selectedIndex))
            {
                renamingIndex = selectedIndex;
                RefreshItem(renamingIndex);
            }
        }

        public void EndRename(bool cancelled = false)
        {
            if (renamingIndex == -1)
                return;

            var changedId = renamingIndex;
            renamingIndex = -1;
            if (changedId >= 0)
            {
                var ve = GetRootElementForIndex(changedId);
                if (ve != null)
                {
                    var text = ve.Q<TextField>();
                    text.Blur();

                    if (!cancelled)
                        onRename?.Invoke(changedId, text.value);

                    RefreshItem(changedId);
                    schedule.Execute(() => FocusOnItem(changedId));
                }
            }
        }

        void FocusOnItem(int index)
        {
            GetRootElementForIndex(index)?.Focus();
        }

        bool CanRename(int index)
        {
            if (index >= 0 && index < itemsSource.Count)
                return CanRenameAtIndex == null || CanRenameAtIndex(index);
            return false;
        }
    }

    internal class RenamableGridView : GridView, IRenamableCollection
    {
        public Action<int, string> onRename { get; set; }
        public Action onItemsReordered { get; set; }

        public int renamingIndex { get; private set; } = -1;

        float m_Width;

        public RenamableGridView()
        {
            name = "RenamableCollection";

            itemSquare = true;

            itemsChosen += OnItemChosen;

            pickingMode = PickingMode.Ignore;

            scrollView.pickingMode = PickingMode.Ignore;
            scrollView.contentContainer.pickingMode = PickingMode.Ignore;
            scrollView.contentViewport.pickingMode = PickingMode.Ignore;
            scrollView.contentViewport.parent.pickingMode = PickingMode.Ignore;

            scrollView.horizontalScrollerVisibility = ScrollerVisibility.Hidden;
            scrollView.verticalScrollerVisibility = ScrollerVisibility.Auto;

            RegisterCallback<KeyDownEvent>(OnCollectionKeyDown);
        }

        void OnCollectionKeyDown(KeyDownEvent evt)
        {
            if (evt.keyCode == KeyCode.Escape)
            {
                evt.StopPropagation();
                EndRename(true);
            }

#if UNITY_EDITOR_WIN || UNITY_EDITOR_LINUX
            if (evt.keyCode == KeyCode.F2)
            {
                evt.StopPropagation();
                schedule.Execute(StartRename);
            }
#endif
        }

        void OnItemChosen(IEnumerable<object> obj)
        {
            schedule.Execute(StartRename);
        }

        public void StartRename()
        {
            if (renamingIndex != -1)
                EndRename(true);

            if (CanRename(selectedIndex))
            {
                renamingIndex = selectedIndex;
                RefreshItem(renamingIndex);
            }
        }

        public void EndRename(bool cancelled = false)
        {
            if (renamingIndex == -1)
                return;

            var changedId = renamingIndex;
            renamingIndex = -1;
            if (changedId >= 0)
            {
                var ve = GetElementAt(changedId);
                if (ve != null)
                {
                    var text = ve.Q<TextField>();
                    text.Blur();

                    if (!cancelled)
                        onRename?.Invoke(changedId, text.value);

                    RefreshItem(changedId);
                    schedule.Execute(() => FocusOnItem(changedId));
                }
            }
        }

        void FocusOnItem(int index)
        {
            GetElementAt(index)?.Focus();
        }

        public delegate bool CanRenameDelegate(int index);

        public CanRenameDelegate CanRenameAtIndex;

        bool CanRename(int index)
        {
            if (index >= 0 && index < itemsSource.Count)
                return CanRenameAtIndex == null || CanRenameAtIndex(index);
            return false;
        }

        public void SetSourceItems(IList sourceList)
        {
            itemsSource = sourceList;
            UpdateColumnCount();

            var scroll = scrollView.scrollOffset;
            scrollView.scrollOffset = Vector2.zero;
            scrollView.scrollOffset = scroll;
        }

        public IList GetItemSource() => itemsSource;

        public void SetElementSize(float newSize)
        {
            itemHeight = (int)newSize;
            UpdateColumnCount();
        }

        public void SetWidth(float newWidth)
        {
            m_Width = newWidth;
            UpdateColumnCount();
        }

        void UpdateColumnCount()
        {
            var newColumnCount = (int)(m_Width / itemHeight);

            if (columnCount != newColumnCount)
            {
                columnCount = newColumnCount;
                Refresh();
            }
        }
    }
}
