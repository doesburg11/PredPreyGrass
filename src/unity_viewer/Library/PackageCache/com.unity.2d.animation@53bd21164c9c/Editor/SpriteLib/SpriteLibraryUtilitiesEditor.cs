using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.U2D.Animation;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Animation
{
    internal static class SpriteLibraryUtilitiesEditor
    {
        public static void ExportSpriteLibraryToAssetFile(SpriteLibrary spriteLibrary, string savePath)
        {
            Debug.Assert(!string.IsNullOrEmpty(savePath) && !string.IsNullOrEmpty(Path.GetFileName(savePath)));

            var serializedObject = new SerializedObject(spriteLibrary);
            var masterLibraryProperty = serializedObject.FindProperty(SpriteLibraryComponentPropertyString.spriteLibraryAsset);
            var libraryProperty = serializedObject.FindProperty(SpriteLibraryComponentPropertyString.library);

            var masterLibraryPath = masterLibraryProperty.objectReferenceValue != null ? AssetDatabase.GetAssetPath(masterLibraryProperty.objectReferenceValue) : "";

            var overrides = new List<SpriteLibCategoryOverride>();
            CopySpriteLibraryToOverride(overrides, libraryProperty);
            var assetToSave = ScriptableObject.CreateInstance<SpriteLibrarySourceAsset>();
            assetToSave.SetLibrary(overrides);
            if (!string.IsNullOrEmpty(masterLibraryPath))
                assetToSave.SetPrimaryLibraryGUID(AssetDatabase.AssetPathToGUID(masterLibraryPath));
            SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(assetToSave, savePath);
            Object.DestroyImmediate(assetToSave);

            AssetDatabase.ImportAsset(savePath);
            var savedAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(savePath);
            if (savedAsset == null)
            {
                Debug.LogError($"Failed to export Sprite Library Asset to {savePath} asset.");
                return;
            }

            libraryProperty.ClearArray();
            masterLibraryProperty.objectReferenceValue = savedAsset;
            serializedObject.ApplyModifiedProperties();
            serializedObject.Update();
        }

        public static void CopySpriteLibraryToOverride(IList<SpriteLibCategoryOverride> destination, SerializedProperty library)
        {
            if (destination == null || library == null || library.arraySize == 0)
                return;

            destination.Clear();

            var categoryEntries = library.GetArrayElementAtIndex(0);
            for (var i = 0; i < library.arraySize; ++i)
            {
                var overrideCategory = new SpriteLibCategoryOverride()
                {
                    categoryList = new List<SpriteCategoryEntry>(),
                    entryOverrideCount = 0,
                    fromMain = false,
                    name = categoryEntries.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue,
                    overrideEntries = new List<SpriteCategoryEntryOverride>()
                };
                var entries = categoryEntries.FindPropertyRelative(SpriteLibraryPropertyString.categoryList);
                var overrideCategoryEntries = overrideCategory.overrideEntries;
                if (entries.arraySize > 0)
                {
                    var entry = entries.GetArrayElementAtIndex(0);
                    for (var j = 0; j < entries.arraySize; ++j)
                    {
                        overrideCategoryEntries.Add(new SpriteCategoryEntryOverride()
                        {
                            fromMain = false,
                            name = entry.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue,
                            sprite = (Sprite)entry.FindPropertyRelative(SpriteLibraryPropertyString.sprite).objectReferenceValue,
                            spriteOverride = (Sprite)entry.FindPropertyRelative(SpriteLibraryPropertyString.sprite).objectReferenceValue
                        });
                        entry.Next(false);
                    }
                }

                destination.Add(overrideCategory);
                categoryEntries.Next(false);
            }
        }

        public static void UpdateLibraryWithNewMainLibrary(SpriteLibraryAsset newMainLibrary, SerializedProperty destLibrary)
        {
            var emptyStringArray = Array.Empty<string>();
            var newCategories = newMainLibrary != null ? newMainLibrary.GetCategoryNames().ToArray() : emptyStringArray;

            // populate new primary
            var newCategoryIndex = 0;
            foreach (var newCategory in newCategories)
            {
                SerializedProperty existingCategory = null;
                if (destLibrary.arraySize > 0)
                {
                    var cat = destLibrary.GetArrayElementAtIndex(0);
                    for (var i = 0; i < destLibrary.arraySize; ++i)
                    {
                        if (cat.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue == newCategory)
                        {
                            existingCategory = cat;
                            if (i != newCategoryIndex)
                                destLibrary.MoveArrayElement(i, newCategoryIndex);
                            break;
                        }

                        cat.Next(false);
                    }
                }

                if (existingCategory != null)
                {
                    if (!existingCategory.FindPropertyRelative(SpriteLibraryPropertyString.fromMain).boolValue)
                        existingCategory.FindPropertyRelative(SpriteLibraryPropertyString.fromMain).boolValue = true;
                }
                else
                {
                    destLibrary.InsertArrayElementAtIndex(newCategoryIndex);
                    existingCategory = destLibrary.GetArrayElementAtIndex(newCategoryIndex);
                    SetPropertyName(existingCategory, newCategory);
                    existingCategory.FindPropertyRelative(SpriteLibraryPropertyString.fromMain).boolValue = true;
                    existingCategory.FindPropertyRelative(SpriteLibraryPropertyString.overrideEntryCount).intValue = 0;
                    existingCategory.FindPropertyRelative(SpriteLibraryPropertyString.overrideEntries).arraySize = 0;
                }

                newCategoryIndex++;

                var newEntries = newMainLibrary.GetCategoryLabelNames(newCategory);
                var entries = existingCategory.FindPropertyRelative(SpriteLibraryPropertyString.overrideEntries);
                var newEntryIndex = 0;
                foreach (var newEntry in newEntries)
                {
                    SerializedProperty cacheEntry = null;
                    if (entries.arraySize > 0)
                    {
                        var ent = entries.GetArrayElementAtIndex(0);
                        for (int j = 0; j < entries.arraySize; ++j)
                        {
                            if (ent.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue == newEntry)
                            {
                                cacheEntry = ent;
                                if (j != newEntryIndex)
                                    entries.MoveArrayElement(j, newEntryIndex);
                                break;
                            }

                            ent.Next(false);
                        }
                    }

                    var mainSprite = newMainLibrary.GetSprite(newCategory, newEntry);
                    if (cacheEntry == null)
                    {
                        entries.InsertArrayElementAtIndex(newEntryIndex);
                        cacheEntry = entries.GetArrayElementAtIndex(newEntryIndex);
                        SetPropertyName(cacheEntry, newEntry);
                        cacheEntry.FindPropertyRelative(SpriteLibraryPropertyString.spriteOverride)
                            .objectReferenceValue = mainSprite;
                    }

                    ++newEntryIndex;
                    if (!cacheEntry.FindPropertyRelative(SpriteLibraryPropertyString.fromMain).boolValue)
                        cacheEntry.FindPropertyRelative(SpriteLibraryPropertyString.fromMain).boolValue = true;
                    if (cacheEntry.FindPropertyRelative(SpriteLibraryPropertyString.sprite).objectReferenceValue != mainSprite)
                        cacheEntry.FindPropertyRelative(SpriteLibraryPropertyString.sprite).objectReferenceValue = mainSprite;
                }
            }

            // Remove any library or entry that is not in primary and not overridden
            for (var i = 0; i < destLibrary.arraySize; ++i)
            {
                var categoryProperty = destLibrary.GetArrayElementAtIndex(i);
                var categoryEntriesProperty = categoryProperty.FindPropertyRelative(SpriteLibraryPropertyString.overrideEntries);
                var categoryFromMainProperty = categoryProperty.FindPropertyRelative(SpriteLibraryPropertyString.fromMain);

                var categoryName = categoryProperty.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue;
                var categoryInPrimary = newCategories.Contains(categoryName);
                var entriesInPrimary = categoryInPrimary ? newMainLibrary.GetCategoryLabelNames(categoryName) : emptyStringArray;

                var categoryOverride = 0;
                for (var j = 0; j < categoryEntriesProperty.arraySize; ++j)
                {
                    var entry = categoryEntriesProperty.GetArrayElementAtIndex(j);
                    var entryName = entry.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue;
                    var entryInPrimary = entriesInPrimary.Contains(entryName);
                    var entryFromMainProperty = entry.FindPropertyRelative(SpriteLibraryPropertyString.fromMain);
                    var overrideSpriteProperty = entry.FindPropertyRelative(SpriteLibraryPropertyString.spriteOverride);
                    var spriteProperty = entry.FindPropertyRelative(SpriteLibraryPropertyString.sprite);
                    if (!entryInPrimary)
                    {
                        // Entry no longer in new primary.
                        // Check for override and set it to us
                        if (entryFromMainProperty.boolValue)
                        {
                            if (overrideSpriteProperty.objectReferenceValue == spriteProperty.objectReferenceValue)
                            {
                                categoryEntriesProperty.DeleteArrayElementAtIndex(j);
                                --j;
                                continue;
                            }
                        }

                        if (entryFromMainProperty.boolValue)
                            entryFromMainProperty.boolValue = false;
                        if (spriteProperty.objectReferenceValue != overrideSpriteProperty.objectReferenceValue)
                            spriteProperty.objectReferenceValue = overrideSpriteProperty.objectReferenceValue;
                        ++categoryOverride;
                    }
                    else
                    {
                        // Check if sprite has been override
                        if (spriteProperty.objectReferenceValue != overrideSpriteProperty.objectReferenceValue)
                            ++categoryOverride;
                    }
                }

                if (!categoryInPrimary && categoryEntriesProperty.arraySize == 0 && categoryFromMainProperty.boolValue)
                {
                    destLibrary.DeleteArrayElementAtIndex(i);
                    --i;
                    continue;
                }

                // since there is override, and we removed the main. This category now
                // belows to the library
                if (!categoryInPrimary)
                {
                    if (categoryFromMainProperty.boolValue)
                        categoryFromMainProperty.boolValue = false;
                }
                else
                {
                    if (categoryProperty.FindPropertyRelative(SpriteLibraryPropertyString.overrideEntryCount).intValue != categoryOverride)
                        categoryProperty.FindPropertyRelative(SpriteLibraryPropertyString.overrideEntryCount).intValue = categoryOverride;
                }
            }
        }

        static void SetPropertyName(SerializedProperty sp, string newName)
        {
            sp.FindPropertyRelative(SpriteLibraryPropertyString.name).stringValue = newName;
            sp.FindPropertyRelative(SpriteLibraryPropertyString.hash).intValue = SpriteLibraryUtility.GetStringHash(newName);
        }
    }
}