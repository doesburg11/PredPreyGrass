using System.Collections.Generic;
using System.IO;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation
{
    /// <summary>
    /// A ScriptedImporter that imports .spriteLib extension file to generate
    /// SpriteLibraryAsset
    /// </summary>
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.2d.animation@latest/index.html?subfolder=/manual/SL-Asset.html")]
    [ScriptedImporter(22100000, "spriteLib", AllowCaching = true)]
    public class SpriteLibrarySourceAssetImporter : ScriptedImporter
    {
        /// <summary>
        /// Implementation of ScriptedImporter.OnImportAsset
        /// </summary>
        /// <param name="ctx">
        /// This argument contains all the contextual information needed to process the import
        /// event and is also used by the custom importer to store the resulting Unity Asset.
        /// </param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var spriteLib = ScriptableObject.CreateInstance<SpriteLibraryAsset>();
            spriteLib.name = Path.GetFileNameWithoutExtension(assetPath);
            var sourceAsset = UnityEditorInternal.InternalEditorUtility.LoadSerializedFileAndForget(assetPath);
            if (sourceAsset?.Length > 0)
            {
                var sourceLibraryAsset = sourceAsset[0] as SpriteLibrarySourceAsset;
                if (sourceLibraryAsset != null)
                {
                    if (!HasValidMainLibrary(sourceLibraryAsset, assetPath))
                        sourceLibraryAsset.SetPrimaryLibraryGUID(string.Empty);

                    UpdateSpriteLibrarySourceAssetLibraryWithMainAsset(sourceLibraryAsset);

                    foreach (var cat in sourceLibraryAsset.library)
                    {
                        spriteLib.AddCategoryLabel(null, cat.name, null);
                        foreach (var entry in cat.overrideEntries)
                        {
                            spriteLib.AddCategoryLabel(entry.spriteOverride, cat.name, entry.name);
                        }
                    }

                    spriteLib.modificationHash = sourceLibraryAsset.modificationHash;
                    spriteLib.version = sourceLibraryAsset.version;

                    if (!string.IsNullOrEmpty(sourceLibraryAsset.primaryLibraryGUID))
                        ctx.DependsOnArtifact(AssetDatabase.GUIDToAssetPath(sourceLibraryAsset.primaryLibraryGUID));
                }
            }

            ctx.AddObjectToAsset("SpriteLib", spriteLib, EditorIconUtility.LoadIconResource("Animation.SpriteLibrary", "ComponentIcons", "ComponentIcons"));
        }

        internal static void UpdateSpriteLibrarySourceAssetLibraryWithMainAsset(SpriteLibrarySourceAsset sourceLibraryAsset)
        {
            var so = new SerializedObject(sourceLibraryAsset);
            var library = so.FindProperty(SpriteLibrarySourceAssetPropertyString.library);
            var mainLibraryAssetAssetPath = AssetDatabase.GUIDToAssetPath(sourceLibraryAsset.primaryLibraryGUID);
            var mainLibraryAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(mainLibraryAssetAssetPath);
            SpriteLibraryUtilitiesEditor.UpdateLibraryWithNewMainLibrary(mainLibraryAsset, library);
            if (so.hasModifiedProperties)
            {
                var modHashProperty = so.FindProperty(SpriteLibrarySourceAssetPropertyString.modificationHash);
                modHashProperty.longValue = mainLibraryAsset != null ? mainLibraryAsset.modificationHash : SpriteLibraryUtility.GenerateHash();
                so.ApplyModifiedPropertiesWithoutUndo();
            }
        }

        internal static bool HasValidMainLibrary(SpriteLibrarySourceAsset sourceLibraryAsset, string assetPath)
        {
            if (string.IsNullOrEmpty(sourceLibraryAsset.primaryLibraryGUID))
                return false;

            var primaryLibraryPath = AssetDatabase.GUIDToAssetPath(sourceLibraryAsset.primaryLibraryGUID);
            if (assetPath == primaryLibraryPath)
                return false;

            var primaryAssetParentChain = GetAssetParentChain(AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(primaryLibraryPath));
            foreach (var parentLibrary in primaryAssetParentChain)
            {
                var parentPath = AssetDatabase.GetAssetPath(parentLibrary);
                if (parentPath == assetPath)
                    return false;
            }

            return true;
        }

        internal static SpriteLibrarySourceAsset LoadSpriteLibrarySourceAsset(string path)
        {
            var loadedObjects = UnityEditorInternal.InternalEditorUtility.LoadSerializedFileAndForget(path);
            foreach (var obj in loadedObjects)
            {
                if (obj is SpriteLibrarySourceAsset)
                    return (SpriteLibrarySourceAsset)obj;
            }

            return null;
        }

        internal static void SaveSpriteLibrarySourceAsset(SpriteLibrarySourceAsset obj, string path)
        {
            if (!HasValidMainLibrary(obj, path))
                obj.SetPrimaryLibraryGUID(string.Empty);

            UnityEditorInternal.InternalEditorUtility.SaveToSerializedFileAndForget(new[] { obj }, path, true);
        }

        [MenuItem("internal:Assets/Convert to SpriteLibrarySourceAsset", true)]
        static bool ConvertToSpriteLibrarySourceAssetValidate()
        {
            foreach (var obj in Selection.objects)
            {
                if (obj is SpriteLibraryAsset)
                    return true;
            }

            return false;
        }

        [MenuItem("internal:Assets/Convert to SpriteLibrarySourceAsset")]
        static void ConvertToSourceAsset()
        {
            foreach (var obj in Selection.objects)
            {
                if (obj is SpriteLibraryAsset)
                {
                    var asset = (SpriteLibraryAsset)obj;
                    var path = AssetDatabase.GetAssetPath(asset);
                    var currentAssetPath = Path.GetDirectoryName(path);
                    var fileName = Path.GetFileNameWithoutExtension(path);
                    var convertFileName = fileName + SpriteLibrarySourceAsset.extension;
                    convertFileName = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(currentAssetPath, convertFileName));
                    var convertAsset = ScriptableObject.CreateInstance<SpriteLibrarySourceAsset>();
                    convertAsset.SetLibrary(new List<SpriteLibCategoryOverride>(asset.categories.Count));
                    for (var i = 0; i < asset.categories.Count; ++i)
                    {
                        var category = asset.categories[i];
                        var newCategory = new SpriteLibCategoryOverride()
                        {
                            overrideEntries = new List<SpriteCategoryEntryOverride>(category.categoryList.Count),
                            name = category.name,
                            entryOverrideCount = 0,
                            fromMain = false
                        };
                        convertAsset.AddCategory(newCategory);
                        for (var j = 0; j < category.categoryList.Count; ++j)
                        {
                            newCategory.overrideEntries.Add(new SpriteCategoryEntryOverride()
                            {
                                name = category.categoryList[j].name,
                                sprite = null,
                                fromMain = false,
                                spriteOverride = category.categoryList[j].sprite
                            });
                        }
                    }

                    SaveSpriteLibrarySourceAsset(convertAsset, convertFileName);
                }
            }

            AssetDatabase.Refresh();
        }

        internal static SpriteLibraryAsset GetAssetParent(SpriteLibraryAsset asset)
        {
            var currentAssetPath = AssetDatabase.GetAssetPath(asset);
            if (AssetImporter.GetAtPath(currentAssetPath) is SpriteLibrarySourceAssetImporter)
            {
                var sourceAsset = LoadSpriteLibrarySourceAsset(currentAssetPath);
                var primaryLibraryId = sourceAsset != null ? sourceAsset.primaryLibraryGUID : null;
                if (primaryLibraryId != null)
                {
                    var primaryLibraryAssetAssetPath = AssetDatabase.GUIDToAssetPath(primaryLibraryId);
                    return AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(primaryLibraryAssetAssetPath);
                }
            }

            return null;
        }

        internal static List<SpriteLibraryAsset> GetAssetParentChain(SpriteLibraryAsset asset)
        {
            var chain = new List<SpriteLibraryAsset>();
            if (asset != null)
            {
                var parent = GetAssetParent(asset);
                while (parent != null && !chain.Contains(parent))
                {
                    chain.Add(parent);
                    parent = GetAssetParent(parent);
                }
            }

            return chain;
        }

        internal static SpriteLibraryAsset GetAssetFromSelection()
        {
            foreach (var selectedObject in Selection.objects)
            {
                var selectedAsset = selectedObject as SpriteLibraryAsset;
                if (selectedAsset != null)
                    return selectedAsset;
            }

            return null;
        }
    }
}
