using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    /// <summary>
    /// Events triggerred from the controller class to notify different UI elements about changes.
    /// </summary>
    internal class ControllerEvents
    {
        /// <summary>
        /// Category list changed. Boolean is true when the provided list is filtered.
        /// </summary>
        public UnityEvent<List<CategoryData>, bool> onModifiedCategories { get; } = new();

        /// <summary>
        /// Label list changed. Boolean is true when the provided list is filtered.
        /// </summary>
        public UnityEvent<List<LabelData>, bool> onModifiedLabels { get; } = new();

        /// <summary>
        /// New Sprite Library Asset has been selected in the Project folder.
        /// </summary>
        public UnityEvent<SpriteLibraryAsset> onSelectedLibrary { get; } = new();

        /// <summary>
        /// List of selected category names.
        /// </summary>
        public UnityEvent<List<string>> onSelectedCategories { get; } = new();

        /// <summary>
        /// List of selected label names.
        /// </summary>
        public UnityEvent<List<string>> onSelectedLabels { get; } = new();

        /// <summary>
        /// View parameters have changed.
        /// </summary>
        public UnityEvent<ViewData> onViewChanged { get; } = new();

        /// <summary>
        /// Main Library is set.
        /// </summary>
        public UnityEvent<SpriteLibraryAsset> onMainLibraryChanged { get; } = new();

        /// <summary>
        /// Library data has been modified.
        /// </summary>
        public UnityEvent<bool> onLibraryDataChanged { get; } = new();
    }

    /// <summary>
    /// Events that notify controller of changes in the UI.
    /// </summary>
    internal class ViewEvents
    {
        /// <summary>
        /// Triggerred to create a new Sprite Library Asset at given location.
        /// </summary>
        public UnityEvent<string> onCreateNewSpriteLibraryAsset { get; } = new();

        /// <summary>
        /// Main UI Split Pane View size changed.
        /// </summary>
        public UnityEvent<float> onMainUISplitPaneSizeChanged { get; } = new();

        /// <summary>
        /// On triggerred Save action.
        /// </summary>
        public UnityEvent onSave { get; } = new();

        /// <summary>
        /// On triggerred Revert action.
        /// </summary>
        public UnityEvent onRevert { get; } = new();

        /// <summary>
        /// Auto-save has changed.
        /// </summary>
        public UnityEvent<bool> onToggleAutoSave { get; } = new();

        /// <summary>
        /// View size slider value changed.
        /// </summary>
        public UnityEvent<float> onViewSizeUpdate { get; } = new();

        /// <summary>
        /// View type changed.
        /// </summary>
        public UnityEvent<ViewType> onViewTypeUpdate { get; } = new();

        /// <summary>
        /// Triggerred on filter string changed.
        /// </summary>
        public UnityEvent<string> onSelectedFilter { get; } = new();

        /// <summary>
        /// Triggerred on selected filter type.
        /// </summary>
        public UnityEvent<SearchType> onSelectedFilterType { get; } = new();

        /// <summary>
        /// Triggerred when Main Library Asset is set.
        /// </summary>
        public UnityEvent<SpriteLibraryAsset> onSetMainAsset { get; } = new();

        /// <summary>
        /// Triggerred on new Categories selected.
        /// </summary>
        public UnityEvent<IList<string>> onSelectCategories { get; } = new();

        /// <summary>
        /// Triggerred on new Labels selected.
        /// </summary>
        public UnityEvent<IList<string>> onSelectLabels { get; } = new();

        /// <summary>
        /// Create new Category.
        /// </summary>
        public UnityEvent<string, IList<Sprite>> onCreateNewCategory { get; } = new();

        /// <summary>
        /// Rename selected Category.
        /// </summary>
        public UnityEvent<string> onRenameCategory { get; } = new();

        /// <summary>
        /// Triggerred when Categories are reordered in the list.
        /// </summary>
        public UnityEvent<IList<string>> onReorderCategories { get; } = new();

        /// <summary>
        /// Triggerred when selected categories are to be deleted.
        /// </summary>
        public UnityEvent onDeleteCategories { get; } = new();

        /// <summary>
        /// Create a new Label.
        /// </summary>
        public UnityEvent<string> onCreateNewLabel { get; } = new();

        /// <summary>
        /// Rename selected Label.
        /// </summary>
        public UnityEvent<string> onRenameLabel { get; } = new();

        /// <summary>
        /// Triggerred when Labels are reordered in the list.
        /// </summary>
        public UnityEvent<IList<string>> onReorderLabels { get; } = new();

        /// <summary>
        /// Delete selected Labels.
        /// </summary>
        public UnityEvent onDeleteLabels { get; } = new();

        /// <summary>
        /// Set Label's Sprite.
        /// </summary>
        public UnityEvent<string, Sprite> onSetLabelSprite { get; } = new();

        /// <summary>
        /// Add data to Categories. Triggerred when data is dragged and dropped into Categories.
        /// </summary>
        public UnityEvent<IList<DragAndDropData>, bool, string> onAddDataToCategories { get; } = new();

        /// <summary>
        /// Add data to Labels. Triggerred when data is dragged and dropped into labels.
        /// </summary>
        public UnityEvent<IList<DragAndDropData>, bool, string> onAddDataToLabels { get; } = new();

        /// <summary>
        /// Revert Labels in the collection.
        /// </summary>
        public UnityEvent<IList<string>> onRevertOverridenLabels { get; } = new();

        /// <summary>
        /// Selection lock value changed.
        /// </summary>
        public UnityEvent<bool> onToggleSelectionLock = new();
    }
}