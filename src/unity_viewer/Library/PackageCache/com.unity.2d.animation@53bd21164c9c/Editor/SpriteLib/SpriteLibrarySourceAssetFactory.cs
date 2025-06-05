using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.U2D.Animation;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Animation
{
    /// <summary>
    /// Represents a Sprite Library's label.
    /// </summary>
    [Serializable]
    public class SpriteLibraryLabel : ISpriteLibraryLabel
    {
        /// <summary>
        /// Label's name.
        /// </summary>
        public string name => m_Name;

        /// <summary>
        /// Sprite associated with the label.
        /// </summary>
        public Sprite sprite => m_Sprite;

        [SerializeField]
        string m_Name;

        [SerializeField]
        Sprite m_Sprite;

        /// <summary>
        /// Constructs a new label.
        /// </summary>
        /// <param name="labelName">Label's name.</param>
        /// <param name="labelSprite">Label's Sprite.</param>
        public SpriteLibraryLabel(string labelName, Sprite labelSprite)
        {
            m_Name = labelName;
            m_Sprite = labelSprite;
        }
    }

    /// <summary>
    /// Represents a Sprite Library's category.
    /// </summary>
    [Serializable]
    public class SpriteLibraryCategory : ISpriteLibraryCategory
    {
        /// <summary>
        /// Category's name.
        /// </summary>
        public string name => m_Name;

        /// <summary>
        /// List of labels in category.
        /// </summary>
        public IEnumerable<ISpriteLibraryLabel> labels => m_Labels;

        [SerializeField]
        List<SpriteLibraryLabel> m_Labels;

        [SerializeField]
        string m_Name;

        /// <summary>
        /// Constructs a new category.
        /// </summary>
        /// <param name="categoryName">Category's name.</param>
        /// <param name="categoryLabels">Collection of labels in a category.</param>
        public SpriteLibraryCategory(string categoryName, IEnumerable<SpriteLibraryLabel> categoryLabels)
        {
            m_Name = categoryName;
            m_Labels = new List<SpriteLibraryLabel>(categoryLabels);
        }
    }

    /// <summary>
    /// Class used for creating new Sprite Library Source Assets.
    /// </summary>
    public static class SpriteLibrarySourceAssetFactory
    {
        /// <summary>
        /// Sprite Library Source Asset's extension.
        /// </summary>
        public const string extension = SpriteLibrarySourceAsset.extension;

        /// <summary>
        /// Creates a new Sprite Library Source Asset at a given path.
        /// </summary>
        /// <param name="path">Save path. Must be within the Assets folder.</param>
        /// <param name="categories">Collection of categories in the library.</param>
        /// <param name="mainLibraryPath">A path to the main library. Null if there is no main library.</param>
        /// <returns>A relative path to the Project with correct extension.</returns>
        /// <exception cref="InvalidOperationException">Throws when the save path is invalid/</exception>
        public static string Create(string path, IEnumerable<ISpriteLibraryCategory> categories, string mainLibraryPath = null)
        {
            if (string.IsNullOrEmpty(path))
                throw new InvalidOperationException("Save path cannot be null or empty.");

            var relativePath = GetRelativePath(path);
            if (string.IsNullOrEmpty(relativePath))
                throw new InvalidOperationException($"{nameof(LoadSpriteLibrarySourceAsset)} can only be saved in the Assets folder.");

            relativePath = Path.ChangeExtension(relativePath, extension);

            SpriteLibraryAsset mainLibrary = null;
            if (!string.IsNullOrEmpty(mainLibraryPath))
            {
                mainLibrary = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(mainLibraryPath);
                if (mainLibrary == null)
                    throw new InvalidOperationException($"No {nameof(SpriteLibraryAsset)} found at path: '{mainLibraryPath}'");
            }

            var asset = ScriptableObject.CreateInstance<SpriteLibrarySourceAsset>();
            var categoryList = new List<SpriteLibCategoryOverride>();
            if (categories != null)
            {
                foreach (var category in categories)
                {
                    var spriteLibCategory = new SpriteLibCategoryOverride
                    {
                        name = category.name,
                        overrideEntries = new List<SpriteCategoryEntryOverride>()
                    };
                    foreach (var label in category.labels)
                    {
                        var spriteCategoryEntryOverride = new SpriteCategoryEntryOverride
                        {
                            name = label.name,
                            spriteOverride = label.sprite
                        };
                        spriteLibCategory.overrideEntries.Add(spriteCategoryEntryOverride);
                    }

                    categoryList.Add(spriteLibCategory);
                }
            }

            if (mainLibrary != null)
            {
                asset.SetPrimaryLibraryGUID(AssetDatabase.GUIDFromAssetPath(mainLibraryPath).ToString());

                var newCategories = mainLibrary.categories ?? new List<SpriteLibCategory>();

                var existingCategories = new List<SpriteLibCategoryOverride>(categoryList);
                categoryList.Clear();

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

                    categoryList.Add(new SpriteLibCategoryOverride
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
                        categoryList.Add(existingCategory);
                    }
                }
            }

            asset.SetLibrary(categoryList);

            SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(asset, relativePath);
            Object.DestroyImmediate(asset);
            return relativePath;
        }

        /// <summary>
        /// Creates a new Sprite Library Source Asset at a given path.
        /// </summary>
        /// <param name="path">Save path. Must be within the Assets folder.</param>
        /// <param name="spriteLibraryAsset">Sprite Library Asset to be saved.</param>
        /// <param name="mainLibraryPath">A path to the main library. Null if there is no main library.</param>
        /// <returns>A relative path to the Project with correct extension.</returns>
        /// <exception cref="InvalidOperationException">Throws when the save path is invalid/</exception>
        public static string Create(string path, SpriteLibraryAsset spriteLibraryAsset, string mainLibraryPath = null)
        {
            return Create(path, spriteLibraryAsset.categories, mainLibraryPath);
        }

        /// <summary>
        /// Creates a new Sprite Library Source Asset at a given path.
        /// </summary>
        /// <param name="spriteLibraryAsset">Sprite Library Asset to be saved.</param>
        /// <param name="path">Save path. Must be within the Assets folder.</param>
        /// <param name="mainLibraryPath">A path to the main library. Null if there is no main library.</param>
        /// <returns>A relative path to the Project with correct extension.</returns>
        /// <exception cref="InvalidOperationException">Throws when the save path is invalid/</exception>
        public static string SaveAsSourceAsset(this SpriteLibraryAsset spriteLibraryAsset, string path, string mainLibraryPath = null)
        {
            return Create(path, spriteLibraryAsset, mainLibraryPath);
        }

        internal static SpriteLibrarySourceAsset LoadSpriteLibrarySourceAsset(string path)
        {
            var loadedObjects = UnityEditorInternal.InternalEditorUtility.LoadSerializedFileAndForget(path);
            foreach (var obj in loadedObjects)
            {
                if (obj is SpriteLibrarySourceAsset asset)
                    return asset;
            }

            return null;
        }

        static string GetRelativePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return null;

            if (!path.StartsWith("Assets/") && !path.StartsWith(Application.dataPath))
                return null;

            var pathStartIndex = path.IndexOf("Assets");
            return pathStartIndex == -1 ? null : path.Substring(pathStartIndex);
        }
    }
}
