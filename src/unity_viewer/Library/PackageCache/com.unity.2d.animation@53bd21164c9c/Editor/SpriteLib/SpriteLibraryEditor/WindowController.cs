using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    internal enum ActionType
    {
        SelectCategory,
        SelectLabels,
        RenameCategory,
        RenameLabel,
        CreateCategory,
        CreateLabel,
        DeleteCategories,
        DeleteLabels,
        ReorderCategories,
        ReorderLabels,
        ModifiedCategories,
        ModifiedLabels,

        SetMainLibrary,
        SetLabelSprite,

        None
    }

    internal enum ViewType
    {
        List,
        Grid
    }

    internal struct ViewData
    {
        /// <summary>
        /// View type (List / Grid).
        /// </summary>
        public ViewType viewType;

        /// <summary>
        /// Relative size for selected View Type in range 0-1.
        /// </summary>
        public float relativeSize;

        /// <summary>
        /// Absolute size based on the slider position in range 0-1.
        /// </summary>
        public float absoluteSize;
    }

    internal enum SearchType
    {
        CategoryAndLabel,
        Category,
        Label
    }

    internal class WindowController
    {
        ISpriteLibraryEditorWindow m_Window;
        SpriteLibraryEditorModel m_Model;

        List<string> selectedCategories => m_Model.GetSelectedCategories();
        List<string> selectedLabels => m_Model.GetSelectedLabels();

        public SpriteLibraryAsset GetSelectedAsset() => m_Model.selectedAsset;

        ControllerEvents m_ControllerEvents;
        ViewEvents m_ViewEvents;

        bool hasSelectedLibrary => m_Model.selectedAsset != null;

        const float k_ViewSizeDivision = 0.1f;
        float m_ViewSize;
        ViewType m_ViewType = ViewType.List;

        string m_FilterString = "";
        SearchType m_FilterType = SearchType.CategoryAndLabel;

        string m_SelectedAssetPath;

        const string k_DefaultLabelName = "New Label";
        const string k_DefaultCategoryName = "New Category";

        bool m_AutoSave;
        bool m_SelectionLocked;

        public WindowController(ISpriteLibraryEditorWindow window, SpriteLibraryEditorModel model, ControllerEvents controllerEvents, ViewEvents viewEvents)
        {
            m_Window = window;
            m_ControllerEvents = controllerEvents;
            m_ViewEvents = viewEvents;

            m_Model = model;

            AddAssetPostprocessorListeners();
            AddViewEventListeners();

            Selection.selectionChanged += SelectionChanged;
            Undo.undoRedoPerformed += PropagateLastAction;

            m_AutoSave = SpriteLibraryEditorWindow.Settings.autoSave;
        }

        public void Destroy()
        {
            RemoveAssetPostprocessorListeners();

            Selection.selectionChanged -= SelectionChanged;
            Undo.undoRedoPerformed -= PropagateLastAction;

            if (m_Model != null)
                m_Model.Destroy();
        }

        public void SaveChanges()
        {
            m_Model.SaveLibrary(m_SelectedAssetPath);

            m_ControllerEvents.onSelectedLibrary?.Invoke(m_Model.selectedAsset);
        }

        public void RevertChanges()
        {
            m_Model.SelectLabels(new List<string>());
            m_Model.SelectCategories(new List<string>());
            m_Model.SelectAsset(m_Model.selectedAsset);

            m_ControllerEvents.onSelectedLibrary?.Invoke(m_Model.selectedAsset);
            RefreshView();
            RefreshSelection();
        }

        public void SelectAsset(SpriteLibraryAsset asset, bool modifiedExternally = false)
        {
            // Only allow .spriteLib files to be used in the editor.
            // This is to prevent the user from opening the editor with .psd or .psb containing spriteLibs.
            if (!AssetDatabase.GetAssetPath(asset).EndsWith(".spriteLib"))
                return;
            
            if (!modifiedExternally)
            {
                if (asset == null || asset == m_Model.selectedAsset)
                    return;

                if (m_Window.hasUnsavedChanges)
                    m_Window.HandleUnsavedChanges();
            }

            m_SelectedAssetPath = asset != null ? AssetDatabase.GetAssetPath(asset) : null;

            m_Model.SelectAsset(asset);

            m_ControllerEvents.onSelectedLibrary?.Invoke(asset);

            RefreshView();
            RefreshSelection();
        }

        void AddViewEventListeners()
        {
            m_ViewEvents.onCreateNewSpriteLibraryAsset.AddListener(CreateNewSpriteLibraryAsset);
            m_ViewEvents.onSave.AddListener(OnSave);
            m_ViewEvents.onRevert.AddListener(OnRevert);
            m_ViewEvents.onToggleAutoSave.AddListener(ToggleAutoSave);
            m_ViewEvents.onToggleSelectionLock.AddListener(ToggleSelectionLock);

            m_ViewEvents.onViewSizeUpdate.AddListener(ChangeViewSize);
            m_ViewEvents.onViewTypeUpdate.AddListener(ChangeViewType);

            m_ViewEvents.onSelectedFilter.AddListener(SelectedFilter);
            m_ViewEvents.onSelectedFilterType.AddListener(SelectedFilterType);

            m_ViewEvents.onSetMainAsset.AddListener(SetMainAsset);
            m_ViewEvents.onSelectCategories.AddListener(SelectCategories);
            m_ViewEvents.onSelectLabels.AddListener(SelectLabels);

            m_ViewEvents.onCreateNewCategory.AddListener(CreateNewCategory);
            m_ViewEvents.onRenameCategory.AddListener(RenameSelectedCategory);
            m_ViewEvents.onReorderCategories.AddListener(ReorderCategories);
            m_ViewEvents.onDeleteCategories.AddListener(DeleteSelectedCategories);

            m_ViewEvents.onCreateNewLabel.AddListener(CreateNewLabel);
            m_ViewEvents.onRenameLabel.AddListener(RenameSelectedLabel);
            m_ViewEvents.onReorderLabels.AddListener(ReorderLabels);
            m_ViewEvents.onDeleteLabels.AddListener(DeleteSelectedLabels);
            m_ViewEvents.onSetLabelSprite.AddListener(SetLabelSprite);

            m_ViewEvents.onAddDataToCategories.AddListener(AddDataToCategories);
            m_ViewEvents.onAddDataToLabels.AddListener(AddDataToLabels);
            m_ViewEvents.onRevertOverridenLabels.AddListener(RevertOverridenLabels);
        }

        void AddAssetPostprocessorListeners()
        {
            SpriteLibraryAssetPostprocessor.OnImported += OnAssetModified;
            SpriteLibraryAssetPostprocessor.OnDeleted += OnAssetModified;
            SpriteLibraryAssetPostprocessor.OnMovedAssetFromTo += OnAssetMoved;
        }

        void RemoveAssetPostprocessorListeners()
        {
            SpriteLibraryAssetPostprocessor.OnImported -= OnAssetModified;
            SpriteLibraryAssetPostprocessor.OnDeleted -= OnAssetModified;
            SpriteLibraryAssetPostprocessor.OnMovedAssetFromTo -= OnAssetMoved;
        }

        static void CreateNewSpriteLibraryAsset(string newAssetPath)
        {
            if (string.IsNullOrEmpty(newAssetPath) || !string.Equals(Path.GetExtension(newAssetPath), SpriteLibrarySourceAsset.extension, StringComparison.OrdinalIgnoreCase))
                return;

            // Make sure that the extension is exactly the same
            if (Path.GetExtension(newAssetPath) != SpriteLibrarySourceAsset.extension)
                newAssetPath = newAssetPath.Replace(Path.GetExtension(newAssetPath), SpriteLibrarySourceAsset.extension);

            var assetToSave = ScriptableObject.CreateInstance<SpriteLibrarySourceAsset>();
            SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(assetToSave, newAssetPath);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);

            var newAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(newAssetPath);
            Selection.objects = new UnityEngine.Object[] { newAsset };
        }

        void ChangeViewSize(float newSize)
        {
            m_ViewSize = newSize;
            if (m_ViewSize > k_ViewSizeDivision && m_ViewType == ViewType.List)
                m_ViewType = ViewType.Grid;
            if (m_ViewSize < k_ViewSizeDivision && m_ViewType == ViewType.Grid)
                m_ViewType = ViewType.List;

            m_ControllerEvents.onViewChanged?.Invoke(new ViewData { viewType = m_ViewType, relativeSize = GetAdjustedViewSize(m_ViewSize, m_ViewType), absoluteSize = m_ViewSize });
        }

        void ChangeViewType(ViewType viewType)
        {
            if (m_ViewType == viewType)
                return;

            m_ViewType = viewType;
            m_ViewSize = m_ViewType == ViewType.List ? 0.0f : k_ViewSizeDivision;

            m_ControllerEvents.onViewChanged?.Invoke(new ViewData { viewType = m_ViewType, relativeSize = GetAdjustedViewSize(m_ViewSize, m_ViewType), absoluteSize = m_ViewSize });
        }

        static float GetAdjustedViewSize(float size, ViewType viewType)
        {
            if (viewType == ViewType.List)
                return size / k_ViewSizeDivision;
            return (size - k_ViewSizeDivision) / (1 - k_ViewSizeDivision);
        }

        void SelectedFilterType(SearchType newFilterType)
        {
            if (m_FilterType == newFilterType)
                return;

            m_FilterType = newFilterType;

            RefreshView();
        }

        void SelectedFilter(string newFilterString)
        {
            if (string.Equals(m_FilterString, newFilterString, StringComparison.OrdinalIgnoreCase))
                return;

            m_FilterString = newFilterString;

            RefreshView();
        }

        void OnAssetModified(string modifiedAssetPath)
        {
            var isModifiedExternally = !(m_SelectedAssetPath != modifiedAssetPath || m_Model.isSaving);
            if (!isModifiedExternally)
                return;

            SelectAsset(AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(m_SelectedAssetPath), true);
        }

        void OnAssetMoved(string sourcePath, string destinationPath)
        {
            if (sourcePath == m_SelectedAssetPath)
            {
                if (string.IsNullOrEmpty(destinationPath))
                {
                    SelectAsset(null, true);
                }
                else
                {
                    m_Model.SetAssetPath(destinationPath);
                    m_SelectedAssetPath = destinationPath;
                }
            }
        }

        void SelectionChanged()
        {
            if (!m_SelectionLocked)
                SelectAsset(SpriteLibrarySourceAssetImporter.GetAssetFromSelection());
        }

        void RefreshView()
        {
            var areCategoriesFiltered = m_FilterType is SearchType.CategoryAndLabel or SearchType.Category && !string.IsNullOrEmpty(m_FilterString);
            var filteredCategories = m_Model.GetFilteredCategories(m_FilterString, m_FilterType);
            m_ControllerEvents.onModifiedCategories?.Invoke(filteredCategories, areCategoriesFiltered);

            var areLabelsFiltered = m_FilterType is SearchType.CategoryAndLabel or SearchType.Label && !string.IsNullOrEmpty(m_FilterString);
            var filteredLabels = m_Model.GetFilteredLabels(m_FilterString, m_FilterType);
            m_ControllerEvents.onModifiedLabels?.Invoke(filteredLabels, areLabelsFiltered);

            m_ControllerEvents.onLibraryDataChanged?.Invoke(m_Model.isModified);
        }

        void RefreshSelection()
        {
            m_ControllerEvents.onSelectedCategories?.Invoke(selectedCategories);
            m_ControllerEvents.onSelectedLabels?.Invoke(selectedLabels);
        }

        void PropagateLastAction()
        {
            if (SpriteLibraryEditorModel.IsActionModifyingAssets(m_Model.lastActionType))
                AutoSave();

            if (m_Model.lastActionType == ActionType.SetMainLibrary)
                m_ControllerEvents.onMainLibraryChanged?.Invoke(m_Model.GetMainLibrary());

            RefreshView();
            RefreshSelection();

            m_Model.lastActionType = ActionType.None;
        }

        void SetMainAsset(SpriteLibraryAsset libraryAsset)
        {
            if (!hasSelectedLibrary)
                return;

            var validAsset = true;
            if (libraryAsset != null)
            {
                if (libraryAsset == m_Model.selectedAsset || SpriteLibrarySourceAssetImporter.GetAssetParentChain(libraryAsset).Contains(GetSelectedAsset()))
                {
                    Debug.LogWarning(TextContent.spriteLibraryCircularDependency);
                    validAsset = false;
                }
            }

            if (validAsset)
            {
                m_Model.BeginUndo(ActionType.SetMainLibrary, TextContent.spriteLibrarySetMainLibrary);
                m_Model.SetMainLibrary(libraryAsset);
                m_Model.SelectCategories(new List<string>());
                m_Model.SelectLabels(new List<string>());
                m_Model.EndUndo();

                AutoSave();
                m_ControllerEvents.onMainLibraryChanged?.Invoke(libraryAsset);
                RefreshView();
                RefreshSelection();
            }
            else
            {
                m_ControllerEvents.onMainLibraryChanged?.Invoke(m_Model.GetMainLibrary());
            }
        }

        void SelectCategories(IList<string> newSelection)
        {
            if (!hasSelectedLibrary)
                return;

            newSelection ??= new List<string>();

            if (Equals(m_Model.GetSelectedCategories(), newSelection))
                return;

            m_Model.BeginUndo(ActionType.SelectCategory, TextContent.spriteLibrarySelectCategories);
            m_Model.SelectCategories(newSelection);
            m_Model.SelectLabels(new List<string>());
            m_Model.EndUndo();

            RefreshView();
            RefreshSelection();
        }

        void SelectLabels(IList<string> newSelection)
        {
            newSelection ??= new List<string>();

            if (AreSequencesEqual(m_Model.GetSelectedLabels(), newSelection))
                return;

            m_Model.BeginUndo(ActionType.SelectLabels, TextContent.spriteLibrarySelectLabels);
            m_Model.SelectLabels(newSelection);
            m_Model.EndUndo();

            RefreshSelection();
        }

        void CreateNewCategory(string categoryName = null, IList<Sprite> sprites = null)
        {
            if (!hasSelectedLibrary)
                return;

            if (string.IsNullOrEmpty(categoryName))
                categoryName = k_DefaultCategoryName;

            m_Model.BeginUndo(ActionType.CreateCategory, TextContent.spriteLibraryCreateCategory);
            m_Model.CreateNewCategory(categoryName, sprites);
            m_Model.SelectCategories(new List<string> { m_Model.GetAllCategories()[^1].name });
            m_Model.SelectLabels(new List<string>());
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void RenameSelectedCategory(string newName)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            newName = newName?.Trim();
            if (string.IsNullOrEmpty(newName))
                return;

            var categoryData = GetCategoryData(selectedCategories[0]);
            if (categoryData == null || categoryData.fromMain)
                return;

            m_Model.BeginUndo(ActionType.RenameCategory, TextContent.spriteLibraryRenameCategory);
            m_Model.RenameSelectedCategory(newName);
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void ReorderCategories(IList<string> reorderedCategories)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            if (m_Model.CompareCategories(reorderedCategories))
                return;

            var categoriesToReorder = selectedCategories;
            m_Model.BeginUndo(ActionType.ReorderCategories, TextContent.spriteLibraryReorderCategories);
            m_Model.ReorderCategories(reorderedCategories);
            m_Model.SelectCategories(categoriesToReorder);
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void DeleteSelectedCategories()
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            var validCategories = false;
            foreach (var category in selectedCategories)
            {
                var categoryData = GetCategoryData(category);
                if (categoryData != null && !categoryData.fromMain)
                {
                    validCategories = true;
                    break;
                }
            }

            if (!validCategories)
                return;

            m_Model.BeginUndo(ActionType.DeleteCategories, TextContent.spriteLibraryDeleteCategories);
            m_Model.DeleteSelectedCategories();
            m_Model.SelectCategories(new List<string>());
            m_Model.SelectLabels(new List<string>());
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void CreateNewLabel(string labelName = null)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            if (string.IsNullOrEmpty(labelName))
                labelName = k_DefaultLabelName;

            m_Model.BeginUndo(ActionType.CreateLabel, TextContent.spriteLibraryCreateLabel);
            m_Model.CreateNewLabel(labelName);
            m_Model.SelectLabels(new List<string> { m_Model.GetAllLabels()[^1].name });
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void RenameSelectedLabel(string newName)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories || !m_Model.hasSelectedLabels)
                return;

            var labelData = GetLabelData(selectedLabels[0]);
            if (labelData == null || labelData.fromMain)
                return;

            newName = newName?.Trim();
            if (newName == string.Empty)
                return;

            m_Model.BeginUndo(ActionType.RenameLabel, TextContent.spriteLibraryRenameLabel);
            m_Model.RenameSelectedLabel(newName);
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void ReorderLabels(IList<string> reorderedLabels)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            if (m_Model.CompareLabels(reorderedLabels))
                return;

            var labelsToReorder = selectedLabels;
            m_Model.BeginUndo(ActionType.ReorderLabels, TextContent.spriteLibraryReorderLabels);
            m_Model.ReorderLabels(reorderedLabels);
            m_Model.SelectLabels(labelsToReorder);
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
        }

        void DeleteSelectedLabels()
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories || !m_Model.hasSelectedLabels)
                return;

            var canAnyLabelBeDeleted = false;
            foreach (var label in selectedLabels)
            {
                if (!GetLabelData(label).fromMain)
                {
                    canAnyLabelBeDeleted = true;
                    break;
                }
            }

            if (!canAnyLabelBeDeleted)
                return;

            m_Model.BeginUndo(ActionType.DeleteLabels, TextContent.spriteLibraryDeleteLabels);
            m_Model.DeleteSelectedLabels();
            m_Model.SelectLabels(new List<string>());
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void SetLabelSprite(string labelName, Sprite newSprite)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            var labelData = GetLabelData(labelName);
            if (labelData == null || labelData.sprite == newSprite)
                return;

            m_Model.BeginUndo(ActionType.SetLabelSprite, TextContent.spriteLibrarySetLabelSprite);
            m_Model.SetLabelSprite(labelName, newSprite);
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void ToggleAutoSave(bool newAutoSaveValue)
        {
            m_AutoSave = newAutoSaveValue;
            if (m_AutoSave)
                OnSave();
        }

        void OnSave()
        {
            m_Window.SaveChanges();
        }

        void OnRevert()
        {
            if (m_Window.hasUnsavedChanges)
                m_Window.HandleRevertChanges();
        }

        void AutoSave()
        {
            if (m_AutoSave)
                m_Window.SaveChanges();
        }

        void ToggleSelectionLock(bool isSelectionLocked)
        {
            m_SelectionLocked = isSelectionLocked;

            SelectionChanged();
        }

        void AddDataToCategories(IList<DragAndDropData> spritesData, bool alt, string category)
        {
            if (!hasSelectedLibrary)
                return;

            if (spritesData == null || spritesData.Count == 0)
                return;

            m_Model.BeginUndo(ActionType.ModifiedCategories, TextContent.spriteLibraryAddDataToCategories);

            foreach (var data in spritesData)
            {
                var isDroppedIntoEmptyArea = string.IsNullOrEmpty(category);
                if (isDroppedIntoEmptyArea)
                {
                    if (data.spriteSourceType == SpriteSourceType.Psb)
                        HandlePsdData(data);
                    else if (data.spriteSourceType == SpriteSourceType.Sprite)
                        m_Model.CreateNewCategory(data.name, data.sprites);
                }
                else
                    m_Model.AddLabelsToCategory(category, data.sprites, true);
            }

            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void AddDataToLabels(IList<DragAndDropData> spritesData, bool alt, string label)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            var sprites = new List<Sprite>();
            foreach (var data in spritesData)
                sprites.AddRange(data.sprites);

            if (sprites.Count == 0)
                return;

            m_Model.BeginUndo(ActionType.ModifiedLabels, TextContent.spriteLibraryAddDataToLabels);
            if (!string.IsNullOrEmpty(label))
                m_Model.SetLabelSprite(label, sprites[0]);
            else // empty area
                m_Model.AddLabelsToCategory(selectedCategories[0], sprites, false);
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        void HandlePsdData(DragAndDropData data)
        {
            var psdFilePath = AssetDatabase.GetAssetPath(data.sprites[0]);
            var characterObj = AssetDatabase.LoadAssetAtPath<GameObject>(psdFilePath);
            var categoryDictionary = new Dictionary<string, List<Sprite>>();
            var objectList = new Queue<Transform>();
            objectList.Enqueue(characterObj.transform);
            while (objectList.Count > 0)
            {
                var goTransform = objectList.Dequeue();
                var spriteList = new List<Sprite>();
                for (var i = 0; i < goTransform.childCount; i++)
                {
                    var childTransform = goTransform.GetChild(i);

                    var spriteRenderer = childTransform.GetComponent<SpriteRenderer>();
                    if (spriteRenderer != null)
                        spriteList.Add(spriteRenderer.sprite);
                    else
                        objectList.Enqueue(childTransform);
                }

                if (spriteList.Count > 0)
                {
                    if (goTransform == characterObj.transform)
                    {
                        foreach (var sprite in spriteList)
                            categoryDictionary[sprite.name] = new List<Sprite> { sprite };
                    }
                    else
                    {
                        categoryDictionary[goTransform.name] = spriteList;
                    }
                }
            }

            foreach (var (categoryName, sprites) in categoryDictionary)
            {
                var addedToCategory = false;
                foreach (var cat in m_Model.GetAllCategories())
                {
                    if (cat.name == categoryName)
                    {
                        m_Model.AddLabelsToCategory(categoryName, sprites, true);
                        addedToCategory = true;
                        break;
                    }
                }

                if (!addedToCategory)
                    m_Model.CreateNewCategory(categoryName, sprites);
            }
        }

        void RevertOverridenLabels(IList<string> labels)
        {
            if (!hasSelectedLibrary || !m_Model.hasSelectedCategories)
                return;

            var canRevertChanges = false;
            foreach (var label in labels)
            {
                if (string.IsNullOrEmpty(label))
                    continue;

                var data = GetLabelData(label);
                if (data != null && (data.spriteOverride || data.categoryFromMain && !data.fromMain))
                {
                    canRevertChanges = true;
                    break;
                }
            }

            if (!canRevertChanges)
                return;

            m_Model.BeginUndo(ActionType.SetLabelSprite, TextContent.spriteLibrarySetLabelSprite);
            m_Model.RevertLabels(labels);
            m_Model.SelectLabels(new List<string>());
            m_Model.EndUndo();

            AutoSave();
            RefreshView();
            RefreshSelection();
        }

        CategoryData GetCategoryData(string categoryName)
        {
            if (string.IsNullOrEmpty(categoryName))
                return null;

            foreach (var category in m_Model.GetAllCategories())
            {
                if (category.name == categoryName)
                    return category;
            }

            return null;
        }

        LabelData GetLabelData(string labelName)
        {
            if (string.IsNullOrEmpty(labelName))
                return null;

            foreach (var label in m_Model.GetAllLabels())
            {
                if (label.name == labelName)
                    return label;
            }

            return null;
        }

        static bool AreSequencesEqual(IList<string> first, IList<string> second)
        {
            if (first == null || second == null || first.Count != second.Count)
                return false;

            for (var i = 0; i < first.Count; i++)
            {
                if (first[i] != second[i])
                    return false;
            }

            return true;
        }
    }
}