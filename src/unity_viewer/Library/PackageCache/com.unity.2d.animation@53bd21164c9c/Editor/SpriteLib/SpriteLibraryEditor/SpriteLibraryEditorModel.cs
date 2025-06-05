using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    internal class CategoryData
    {
        public readonly string name;
        public readonly bool fromMain;
        public readonly bool isOverride;

        public CategoryData(string name, bool fromMain, bool isOverride)
        {
            this.name = name;
            this.fromMain = fromMain;
            this.isOverride = isOverride;
        }
    }

    internal class LabelData
    {
        public readonly string name;
        public readonly Sprite sprite;
        public readonly bool fromMain;
        public readonly bool spriteOverride;
        public readonly bool categoryFromMain;

        public LabelData(string name, Sprite sprite, bool fromMain, bool spriteOverride, bool categoryFromMain)
        {
            this.name = name;
            this.sprite = sprite;
            this.fromMain = fromMain;
            this.spriteOverride = spriteOverride;
            this.categoryFromMain = categoryFromMain;
        }
    }

    internal class SpriteLibraryEditorModel : ScriptableObject
    {
        [SerializeField]
        ActionType m_LastActionType;

        public ActionType lastActionType
        {
            get => m_LastActionType;
            set => m_LastActionType = value;
        }

        public SpriteLibraryAsset selectedAsset { get; private set; }
        public bool hasSelectedCategories => m_SelectedCategoryIndices != null && m_SelectedCategoryIndices.Count > 0;
        public bool hasSelectedLabels => m_SelectedLabelIndices != null && m_SelectedLabelIndices.Count > 0;
        public bool isModified => m_CurrentVersion != m_SavedVersion;
        public bool isSaving { get; private set; }

        [SerializeField]
        List<int> m_SelectedCategoryIndices = new();
        [SerializeField]
        List<int> m_SelectedLabelIndices = new();

        [SerializeField]
        List<SpriteLibCategoryOverride> m_CurrentLibrary;

        [SerializeField]
        uint m_CurrentVersion;
        uint m_SavedVersion;

        string m_SelectedAssetPath;

        [SerializeField]
        string m_PrimaryLibraryGUID;

        public List<string> GetSelectedCategories()
        {
            var selection = new List<string>();
            if (m_CurrentLibrary == null)
                return selection;

            foreach (var selectedCategoryIndex in m_SelectedCategoryIndices)
                selection.Add(m_CurrentLibrary[selectedCategoryIndex].name);

            return selection;
        }

        public void SelectCategories(IList<string> newSelection)
        {
            if (newSelection == null || newSelection.Count == 0)
            {
                m_SelectedCategoryIndices = new List<int>();
                return;
            }

            if (m_CurrentLibrary == null)
                return;

            m_SelectedCategoryIndices = new List<int>(newSelection.Count);
            for (var selection = 0; selection < newSelection.Count; selection++)
            {
                var index = -1;
                for (var i = 0; i < m_CurrentLibrary.Count; i++)
                {
                    if (m_CurrentLibrary[i].name == newSelection[selection])
                    {
                        index = i;
                        break;
                    }
                }

                if (index != -1)
                    m_SelectedCategoryIndices.Add(index);
            }
        }

        public List<string> GetSelectedLabels()
        {
            var selectedCategory = GetSelectedCategory();
            if (selectedCategory == null)
                return new List<string>();

            var labels = selectedCategory.overrideEntries;
            var selectedLabels = new List<string>(m_SelectedLabelIndices.Count);
            for (var i = 0; i < m_SelectedLabelIndices.Count; i++)
                selectedLabels.Add(labels[m_SelectedLabelIndices[i]].name);

            return selectedLabels;
        }

        public void SelectLabels(IList<string> labels)
        {
            if (labels == null || labels.Count == 0)
            {
                m_SelectedLabelIndices = new List<int>();
                return;
            }

            var category = GetSelectedCategory();
            if (category == null)
                return;

            m_SelectedLabelIndices = new List<int>(labels.Count);
            for (var i = 0; i < labels.Count; i++)
                m_SelectedLabelIndices.Add(category.overrideEntries.FindIndex(label => label.name == labels[i]));
        }

        public List<CategoryData> GetAllCategories()
        {
            var allCategories = new List<CategoryData>();
            if (m_CurrentLibrary == null)
                return allCategories;

            for (var i = 0; i < m_CurrentLibrary.Count; i++)
                allCategories.Add(CreateCategoryData(m_CurrentLibrary[i]));

            return allCategories;
        }

        public List<CategoryData> GetFilteredCategories(string filterString, SearchType searchType, StringComparison comparison = StringComparison.OrdinalIgnoreCase)
        {
            var categories = new List<CategoryData>();
            if (m_CurrentLibrary != null)
            {
                var filter = searchType is SearchType.CategoryAndLabel or SearchType.Category && !string.IsNullOrEmpty(filterString);
                foreach (var spriteLibCategoryOverride in m_CurrentLibrary)
                {
                    if (!filter || spriteLibCategoryOverride.name.Contains(filterString, comparison))
                        categories.Add(CreateCategoryData(spriteLibCategoryOverride));
                }
            }

            return categories;
        }

        public List<LabelData> GetAllLabels()
        {
            var category = GetSelectedCategory();
            if (category == null)
                return new List<LabelData>();

            var categoryLabels = category.overrideEntries;
            var labels = new List<LabelData>(category.entryOverrideCount);
            for (var i = 0; i < categoryLabels.Count; i++)
                labels.Add(CreateLabelData(category, categoryLabels[i]));

            return labels;
        }

        public List<LabelData> GetFilteredLabels(string filterString, SearchType searchType, StringComparison comparison = StringComparison.OrdinalIgnoreCase)
        {
            var category = GetSelectedCategory();
            if (category == null)
                return new List<LabelData>();

            var labels = new List<LabelData>();
            var filter = searchType is SearchType.CategoryAndLabel or SearchType.Label && !string.IsNullOrEmpty(filterString);
            foreach (var categoryEntryOverride in category.overrideEntries)
            {
                if (!filter || categoryEntryOverride.name.Contains(filterString, comparison))
                    labels.Add(CreateLabelData(category, categoryEntryOverride));
            }

            return labels;
        }

        public void BeginUndo(ActionType actionType, string actionName)
        {
            lastActionType = actionType;

            Undo.IncrementCurrentGroup();
            Undo.RegisterCompleteObjectUndo(this, actionName);

            if (IsActionModifyingAssets(actionType))
                m_CurrentVersion++;
        }

        public void EndUndo()
        {
            Undo.CollapseUndoOperations(Undo.GetCurrentGroup());

            lastActionType = ActionType.None;
        }

        public static bool IsActionModifyingAssets(ActionType actionType)
        {
            var isNotModifying = actionType is
                ActionType.None or
                ActionType.SelectCategory or
                ActionType.SelectLabels;

            return !isNotModifying;
        }

        public void CreateNewLabel(string labelName)
        {
            var category = GetSelectedCategory();
            var newLabel = new SpriteCategoryEntryOverride
            {
                name = labelName,
                sprite = null,
                spriteOverride = null,
                fromMain = false,
            };
            category.overrideEntries.Add(newLabel);
            category.RenameDuplicateOverrideEntries();
            category.UpdateOverrideCount();
        }

        public void SetLabelSprite(string labelName, Sprite newSprite)
        {
            var category = GetSelectedCategory();
            var label = GetLabel(category, labelName);
            if (label != null)
            {
                label.spriteOverride = newSprite;
                if (!label.fromMain)
                    label.sprite = newSprite;
            }

            category.UpdateOverrideCount();
        }

        public void DeleteSelectedLabels()
        {
            var category = GetSelectedCategory();
            if (category == null)
                return;

            var labelsToRemove = new List<SpriteCategoryEntryOverride>();
            for (var i = 0; i < m_SelectedLabelIndices.Count; i++)
            {
                var labelToRemove = category.overrideEntries[m_SelectedLabelIndices[i]];
                if (!labelToRemove.fromMain)
                    labelsToRemove.Add(labelToRemove);
            }

            foreach (var label in labelsToRemove)
                category.overrideEntries.Remove(label);

            category.UpdateOverrideCount();
        }

        public void RenameSelectedLabel(string newName)
        {
            var category = GetSelectedCategory();
            var label = GetSelectedLabel();
            label.name = newName;
            category.RenameDuplicateOverrideEntries();
        }

        public void AddLabelsToCategory(string categoryName, IEnumerable<Sprite> sprites, bool replaceConflicting)
        {
            var category = GetCategory(categoryName);
            var conflictingLabels = new List<Sprite>();
            var newLabels = new List<Sprite>();
            foreach (var sprite in sprites)
            {
                if (GetLabel(category, sprite.name) != null)
                    conflictingLabels.Add(sprite);
                else
                    newLabels.Add(sprite);
            }

            foreach (var newLabel in newLabels)
                category.overrideEntries.Add(new SpriteCategoryEntryOverride
                {
                    name = newLabel.name,
                    sprite = newLabel,
                    spriteOverride = newLabel,
                    fromMain = false
                });

            if (replaceConflicting)
            {
                foreach (var conflictingLabel in conflictingLabels)
                {
                    var label = GetLabel(category, conflictingLabel.name);
                    label.spriteOverride = conflictingLabel;
                    if (!label.fromMain)
                        label.sprite = conflictingLabel;
                }
            }
            else
            {
                foreach (var conflictingLabel in conflictingLabels)
                {
                    category.overrideEntries.Add(new SpriteCategoryEntryOverride
                    {
                        name = conflictingLabel.name,
                        sprite = conflictingLabel,
                        spriteOverride = conflictingLabel,
                        fromMain = false
                    });
                }
            }

            category.RenameDuplicateOverrideEntries();
            category.UpdateOverrideCount();
        }

        public void DeleteSelectedCategories()
        {
            if (m_CurrentLibrary == null || m_SelectedCategoryIndices == null || m_SelectedCategoryIndices.Count == 0)
                return;

            var categoriesToRemove = new List<SpriteLibCategoryOverride>(m_SelectedCategoryIndices.Count);
            foreach (var selectedCategoryIndex in m_SelectedCategoryIndices)
                categoriesToRemove.Add(m_CurrentLibrary[selectedCategoryIndex]);

            foreach (var categoryToRemove in categoriesToRemove)
                m_CurrentLibrary.Remove(categoryToRemove);
        }

        public void ReorderCategories(IList<string> reorderedCategories)
        {
            var categories = new List<SpriteLibCategoryOverride>(reorderedCategories.Count);
            for (var i = 0; i < reorderedCategories.Count; i++)
            {
                var reorderedCategory = GetCategory(reorderedCategories[i]);
                categories.Add(reorderedCategory);
            }

            m_CurrentLibrary = categories;
        }

        public void RenameSelectedCategory(string newName)
        {
            var category = GetSelectedCategory();
            category.name = newName;
            RenameDuplicatedCategories();
        }

        public void SelectAsset(SpriteLibraryAsset asset)
        {
            ClearUndo();

            m_SavedVersion = m_CurrentVersion;

            selectedAsset = asset;

            m_CurrentLibrary = new List<SpriteLibCategoryOverride>();
            m_PrimaryLibraryGUID = null;

            m_SelectedAssetPath = selectedAsset != null ? AssetDatabase.GetAssetPath(selectedAsset) : null;
            m_SelectedCategoryIndices = new List<int>();
            m_SelectedLabelIndices = new List<int>();

            var sourceAsset = selectedAsset != null ? SpriteLibrarySourceAssetImporter.LoadSpriteLibrarySourceAsset(m_SelectedAssetPath) : null;
            if (sourceAsset != null)
            {
                m_CurrentLibrary = new List<SpriteLibCategoryOverride>(sourceAsset.library);

                // Update hashes to make sure name hashes are correct.
                foreach (var categoryOverride in m_CurrentLibrary)
                    categoryOverride.UpdateHash();

                m_PrimaryLibraryGUID = sourceAsset.primaryLibraryGUID;

                var mainLibrary = GetMainLibrary();
                if (mainLibrary)
                    SetMainLibrary(mainLibrary);
            }
        }

        public void ReorderLabels(IList<string> reorderedLabels)
        {
            var category = GetSelectedCategory();
            var labels = new List<SpriteCategoryEntryOverride>(reorderedLabels.Count);
            for (var i = 0; i < reorderedLabels.Count; i++)
                labels.Add(GetLabel(category, reorderedLabels[i]));

            for (var i = 0; i < labels.Count; i++)
            {
                var label = labels[i];
                if (label.fromMain)
                    continue;

                var index = category.overrideEntries.IndexOf(label);

                (category.overrideEntries[i], category.overrideEntries[index]) = (category.overrideEntries[index], category.overrideEntries[i]);
            }

            category.overrideEntries = labels;
        }

        public void CreateNewCategory(string categoryName, IList<Sprite> sprites)
        {
            var newLabelCount = sprites?.Count ?? 0;
            var newLabels = new List<SpriteCategoryEntryOverride>(newLabelCount);
            if (sprites != null)
            {
                for (var i = 0; i < newLabelCount; i++)
                    newLabels.Add(new SpriteCategoryEntryOverride
                    {
                        name = sprites[i].name,
                        fromMain = false,
                        sprite = sprites[i],
                        spriteOverride = sprites[i]
                    });
            }

            var category = new SpriteLibCategoryOverride
            {
                name = categoryName,
                fromMain = false,
                entryOverrideCount = newLabelCount,
                overrideEntries = newLabels
            };

            m_CurrentLibrary.Add(category);
            RenameDuplicatedCategories();
        }

        public void SaveLibrary(string path)
        {
            if (m_CurrentLibrary == null)
                return;

            Debug.Assert(!string.IsNullOrEmpty(path), "Asset path cannot be empty.");

            m_SavedVersion = m_CurrentVersion;
            isSaving = true;
            var assetToSave = CreateInstance<SpriteLibrarySourceAsset>();
            assetToSave.SetLibrary(m_CurrentLibrary);
            assetToSave.SetPrimaryLibraryGUID(m_PrimaryLibraryGUID);
            SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(assetToSave, path);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            isSaving = false;
        }

        public SpriteLibraryAsset GetMainLibrary()
        {
            if (string.IsNullOrEmpty(m_PrimaryLibraryGUID))
                return null;
            var assetPath = AssetDatabase.GUIDToAssetPath(m_PrimaryLibraryGUID);
            var asset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(assetPath);
            return asset;
        }

        public void SetMainLibrary(SpriteLibraryAsset newMainLibrary)
        {
            m_PrimaryLibraryGUID = newMainLibrary != null ? AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(newMainLibrary)) : null;
            var newCategories = newMainLibrary != null ? newMainLibrary.categories : new List<SpriteLibCategory>();

            var existingCategories = new List<SpriteLibCategoryOverride>(m_CurrentLibrary);
            m_CurrentLibrary.Clear();

            // populate new primary
            foreach (var newCategory in newCategories)
            {
                var labels = new List<SpriteCategoryEntryOverride>();
                SpriteLibCategoryOverride existingCategory = null;
                for (var i = 0; i < existingCategories.Count; i++)
                {
                    var category = existingCategories[i];
                    if (category.name == newCategory.name)
                    {
                        existingCategory = category;
                        existingCategory.fromMain = true;
                        existingCategories.RemoveAt(i);
                        break;
                    }
                }

                var newEntries = newCategory.categoryList;
                foreach (var newEntry in newEntries)
                {
                    var sprite = newEntry.sprite;

                    labels.Add(new SpriteCategoryEntryOverride
                    {
                        name = newEntry.name,
                        sprite = sprite,
                        spriteOverride = sprite,
                        fromMain = true
                    });
                }

                var overrideCount = 0;
                if (existingCategory != null)
                {
                    foreach (var existingLabel in existingCategory.overrideEntries)
                    {
                        var foundLabel = false;
                        foreach (var newLabel in labels)
                        {
                            if (existingLabel.name == newLabel.name)
                            {
                                if (newLabel.spriteOverride != existingLabel.spriteOverride)
                                {
                                    newLabel.spriteOverride = existingLabel.spriteOverride;
                                    overrideCount++;
                                }

                                foundLabel = true;
                                break;
                            }
                        }

                        if (!foundLabel)
                        {
                            overrideCount++;
                            labels.Add(new SpriteCategoryEntryOverride
                            {
                                name = existingLabel.name,
                                sprite = existingLabel.sprite,
                                spriteOverride = existingLabel.spriteOverride,
                                fromMain = false
                            });
                        }
                    }
                }

                m_CurrentLibrary.Add(new SpriteLibCategoryOverride
                {
                    name = newCategory.name,
                    overrideEntries = labels,
                    fromMain = true,
                    entryOverrideCount = overrideCount
                });
            }

            foreach (var existingCategory in existingCategories)
            {
                var keepCategory = false;
                if (existingCategory.fromMain)
                {
                    for (var i = existingCategory.overrideEntries.Count; i-- > 0;)
                    {
                        var entry = existingCategory.overrideEntries[i];
                        if (!entry.fromMain || entry.sprite != entry.spriteOverride)
                        {
                            entry.fromMain = false;
                            entry.sprite = entry.spriteOverride;
                            keepCategory = true;
                        }
                        else
                            existingCategory.overrideEntries.RemoveAt(i);
                    }
                }

                if (!existingCategory.fromMain || keepCategory)
                {
                    existingCategory.fromMain = false;
                    existingCategory.entryOverrideCount = 0;
                    m_CurrentLibrary.Add(existingCategory);
                }
            }
        }

        public void RevertLabels(IList<string> labels)
        {
            var category = GetSelectedCategory();
            foreach (var labelName in labels)
            {
                var label = GetLabel(category, labelName);
                if (label.fromMain && label.sprite != label.spriteOverride)
                {
                    label.spriteOverride = label.sprite;
                }
                else if (category.fromMain && !label.fromMain)
                {
                    category.overrideEntries.Remove(label);
                }
            }

            category.UpdateOverrideCount();
        }

        public void SetAssetPath(string assetPath)
        {
            m_SelectedAssetPath = assetPath;
        }

        SpriteLibCategoryOverride GetSelectedCategory()
        {
            if (m_CurrentLibrary == null || m_CurrentLibrary.Count == 0 || !hasSelectedCategories)
                return null;

            var selectedCategoryIndex = m_SelectedCategoryIndices[0];
            if (selectedCategoryIndex < 0 || selectedCategoryIndex >= m_CurrentLibrary.Count)
                return null;

            return m_CurrentLibrary[selectedCategoryIndex];
        }

        SpriteLibCategoryOverride GetCategory(string categoryName)
        {
            if (m_CurrentLibrary == null || m_CurrentLibrary.Count == 0)
                return null;

            foreach (var category in m_CurrentLibrary)
            {
                if (category.name == categoryName)
                    return category;
            }

            return null;
        }

        SpriteCategoryEntryOverride GetSelectedLabel()
        {
            var selectedCategory = GetSelectedCategory();
            if (selectedCategory == null || !hasSelectedLabels)
                return null;

            var selectedLabelIndex = m_SelectedLabelIndices[0];
            if (selectedLabelIndex < 0 || selectedLabelIndex >= selectedCategory.overrideEntries.Count)
                return null;

            return selectedCategory.overrideEntries[selectedLabelIndex];
        }

        static SpriteCategoryEntryOverride GetLabel(SpriteLibCategoryOverride category, string labelName)
        {
            if (category == null)
                return null;

            foreach (var spriteCategoryEntryOverride in category.overrideEntries)
            {
                if (spriteCategoryEntryOverride.name == labelName)
                    return spriteCategoryEntryOverride;
            }

            return null;
        }

        public List<LabelData> GetLabels(string categoryName)
        {
            if (m_CurrentLibrary != null && m_CurrentLibrary != null)
            {
                foreach (var spriteLibCategoryOverride in m_CurrentLibrary)
                {
                    if (spriteLibCategoryOverride.name == categoryName)
                    {
                        var labelsCache = spriteLibCategoryOverride.overrideEntries;
                        var labels = new List<LabelData>(labelsCache.Count);
                        foreach (var categoryEntryOverride in labelsCache)
                            labels.Add(CreateLabelData(spriteLibCategoryOverride, categoryEntryOverride));
                        return labels;
                    }
                }
            }

            return new List<LabelData>();
        }

        public bool CompareLabels(IList<string> labelsToCompare)
        {
            if (labelsToCompare == null)
                return false;

            var category = GetSelectedCategory();
            if (category == null || category.entryOverrideCount != labelsToCompare.Count)
                return false;

            for (var i = 0; i < labelsToCompare.Count; i++)
            {
                if (category.overrideEntries[i].name != labelsToCompare[i])
                    return false;
            }

            return true;
        }

        public bool CompareCategories(IList<string> categoriesToCompare)
        {
            if (categoriesToCompare == null)
                return false;

            if (m_CurrentLibrary.Count != categoriesToCompare.Count)
                return false;

            for (var i = 0; i < categoriesToCompare.Count; i++)
            {
                if (m_CurrentLibrary[i].name != categoriesToCompare[i])
                    return false;
            }

            return true;
        }

        void RenameDuplicatedCategories()
        {
            if (m_CurrentLibrary != null)
                SpriteLibraryAsset.RenameDuplicate(m_CurrentLibrary, (_, _) => { });
        }

        void ClearUndo() => Undo.ClearUndo(this);

        public void Destroy()
        {
            ClearUndo();
            DestroyImmediate(this);
        }

        static CategoryData CreateCategoryData(SpriteLibCategoryOverride category)
        {
            return new CategoryData(
                name: category.name,
                fromMain: category.fromMain,
                isOverride: category.fromMain && category.entryOverrideCount > 0);
        }

        static LabelData CreateLabelData(SpriteLibCategoryOverride category, SpriteCategoryEntryOverride label)
        {
            return new LabelData(
                name: label.name,
                sprite: label.spriteOverride,
                fromMain: label.fromMain,
                spriteOverride: label.sprite != label.spriteOverride,
                categoryFromMain: category.fromMain);
        }
    }
}