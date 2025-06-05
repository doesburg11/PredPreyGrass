using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Reflection;
using UnityEditor.SceneManagement;
using UnityEditor.U2D.Sprites;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.U2D.Animation;

using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Animation.Upgrading
{
    internal class SpriteLibUpgrader : BaseUpgrader
    {
        static class Contents
        {
            public static readonly string ProgressBarTitle = L10n.Tr("Upgrading Sprite Library Assets");
            public static readonly string VerifyingSelection = L10n.Tr("Verifying the selection");
            public static readonly string CreatingNewLibraries = L10n.Tr("Creating new Sprite Library Assets");
            public static readonly string ReassignAssetsInComponents = L10n.Tr("Re-assigning assets in components");
            public static readonly string RemoveOldSpriteLibraries = L10n.Tr("Removing old Sprite Library Assets");
        }

        const string k_SpriteLibTypeId = "t:SpriteLibraryAsset";
        const string k_PsbImporterCategoriesId = "m_SpriteCategoryList.m_Categories";
        static readonly Dictionary<Type, List<FieldInfo>> k_SpriteLibraryReferenceLookup;

        bool m_OnlyFindOldAssets;
        bool m_OnlySearchInAssets;

        HashSet<int> m_IndicesWithAssetBundleConnection = new HashSet<int>();
        HashSet<string> m_AssetBundlesNeedingUpgrade = new HashSet<string>();

        static SpriteLibUpgrader()
        {
            k_SpriteLibraryReferenceLookup = GetSpriteLibReferenceLookup();
        }

        /// <param name="onlyFindOldAssets">Set this to true if you only want to find .asset SpriteLibraries</param>
        /// <param name="onlySearchInAssets">Set this to true if you only want to find SpriteLibraries in the Assets/ folder or its children</param>
        public SpriteLibUpgrader(bool onlyFindOldAssets = true, bool onlySearchInAssets = true)
        {
            m_OnlyFindOldAssets = onlyFindOldAssets;
            m_OnlySearchInAssets = onlySearchInAssets;
        }

        static Dictionary<Type, List<FieldInfo>> GetSpriteLibReferenceLookup()
        {
            var result = new Dictionary<Type, List<FieldInfo>>();

            var allObjectsWithSpriteLibProperties = TypeCache
                .GetTypesDerivedFrom<Component>()
                .Where(type => type
                    .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .Any(HasSpriteLibField));

            foreach (var property in allObjectsWithSpriteLibProperties)
            {
                if (!result.ContainsKey(property))
                    result.Add(property, new List<FieldInfo>());

                var libraryFields = property
                    .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .Where(HasSpriteLibField)
                    .ToList();

                foreach (var field in libraryFields)
                {
                    result[property].Add(field);
                }
            }

            return result;
        }

        static bool HasSpriteLibField(FieldInfo field)
        {
            return (field.FieldType == typeof(SpriteLibraryAsset) ||
                    field.FieldType == typeof(SpriteLibraryAsset[]) ||
                    field.FieldType == typeof(List<SpriteLibraryAsset>)) &&
                (field.IsPublic || field.IsDefined(typeof(SerializeField)));
        }

        internal override List<Object> GetUpgradableAssets()
        {
            var rootFolder = m_OnlySearchInAssets ? new[] { "Assets" } : null;
            var assetPaths = AssetDatabase.FindAssets(k_SpriteLibTypeId, rootFolder)
                .Select(AssetDatabase.GUIDToAssetPath).ToArray();

            if (m_OnlyFindOldAssets)
            {
                assetPaths = assetPaths
                    .Where(x => x.EndsWith(".asset") || UpgradeUtilities.IsPsbImportedFile(x))
                    .ToArray();
            }

            var assets = assetPaths
                .Select(AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>)
                .Cast<Object>()
                .ToList();
            return assets;
        }

        internal override UpgradeReport UpgradeSelection(List<ObjectIndexPair> objects)
        {
            EditorUtility.DisplayProgressBar(Contents.ProgressBarTitle, Contents.VerifyingSelection, 0f);

            var entries = new List<UpgradeEntry>();

            var libraryIndexPairs = new Dictionary<int, SpriteLibraryAsset>();
            string msg;
            foreach (var obj in objects)
            {
                if (obj.Target == null)
                {
                    msg = "The upgrade failed. Invalid selection.";
                    m_Logger.Add(msg);
                    m_Logger.AddLineBreak();
                    entries.Add(new UpgradeEntry()
                    {
                        Result = UpgradeResult.Error,
                        Target = obj.Target,
                        Index = obj.Index,
                        Message = msg
                    });
                }
                else if (obj.Target is SpriteLibraryAsset lib)
                {
                    libraryIndexPairs.Add(obj.Index, lib);
                }
                else
                {
                    msg = $"The upgrade failed. {obj.Target.name} is not a SpriteLibraryAsset.";
                    m_Logger.Add(msg);
                    m_Logger.AddLineBreak();
                    entries.Add(new UpgradeEntry()
                    {
                        Result = UpgradeResult.Error,
                        Target = obj.Target,
                        Index = obj.Index,
                        Message = msg
                    });
                }
            }
            var oldLibraries = libraryIndexPairs.Values.ToList();

            EditorUtility.DisplayProgressBar(Contents.ProgressBarTitle, Contents.CreatingNewLibraries, 0.2f);
            m_Logger.AddLineBreak();
            m_Logger.Add(Contents.CreatingNewLibraries);
            var newSourceAssetPaths = CreateNewAssetLibraries(libraryIndexPairs);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            var newLibraries = LoadNewLibraries(newSourceAssetPaths);

            EditorUtility.DisplayProgressBar(Contents.ProgressBarTitle, Contents.ReassignAssetsInComponents, 0.4f);
            m_Logger.AddLineBreak();
            m_Logger.Add(Contents.ReassignAssetsInComponents);
            var assetPaths = GetAllAssetPaths(libraryIndexPairs);
            if (assetPaths.Length > 0)
                ReassignAssets(assetPaths, oldLibraries, newLibraries);

            EditorUtility.DisplayProgressBar(Contents.ProgressBarTitle, Contents.RemoveOldSpriteLibraries, 0.8f);
            m_Logger.AddLineBreak();
            m_Logger.Add(Contents.RemoveOldSpriteLibraries);
            RemoveOldLibraries(oldLibraries);
            AssetDatabase.Refresh();

            for (var i = 0; i < newLibraries.Count; ++i)
            {
                UpgradeResult result;
                if (m_IndicesWithAssetBundleConnection.Contains(newSourceAssetPaths[i].Item1))
                {
                    msg = $"Successfully replaced {newLibraries[i].name}. Note that the asset is connected to an AssetBundle. Make sure to rebuild this AssetBundle to complete the upgrade. See more info in the upgrade log.";
                    result = UpgradeResult.Warning;
                }
                else
                {
                    msg = $"Successfully replaced {newLibraries[i].name}";
                    result = UpgradeResult.Successful;
                }

                m_Logger.Add(msg);
                entries.Add(new UpgradeEntry()
                {
                    Result = result,
                    Target = newLibraries[i],
                    Index = newSourceAssetPaths[i].Item1,
                    Message = msg
                });
            }

            EditorUtility.ClearProgressBar();

            if (m_AssetBundlesNeedingUpgrade.Count > 0)
                AddAssetBundlesToLog();

            var report = new UpgradeReport()
            {
                UpgradeEntries = entries,
                Log = m_Logger.GetLog()
            };

            m_Logger.Clear();
            m_AssetBundlesNeedingUpgrade.Clear();
            return report;
        }

        static string[] GetAllAssetPaths(Dictionary<int, SpriteLibraryAsset> spriteLibraries)
        {
            var ids = new List<string>();
            foreach(var lib in spriteLibraries.Values)
                ids.Add(GetObjectIDString(lib));
            var spriteLibIds = ids.ToArray();

            var assetPaths = new List<string>();
            var allAssetPaths = AssetDatabase.GetAllAssetPaths();
            foreach (var path in allAssetPaths)
            {
                if (!IsPrefabOrScenePath(path, spriteLibIds))
                    continue;

                assetPaths.Add(path);
            }

            return assetPaths.ToArray();
        }

        static string GetObjectIDString(Object obj)
        {
            if (AssetDatabase.TryGetGUIDAndLocalFileIdentifier(obj.GetInstanceID(), out string guid, out long localId))
                return "fileID: " + localId + ", guid: " + guid;

            return null;
        }

        static bool IsPrefabOrScenePath(string path, string[] ids)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));

            if (path.StartsWith("Packages"))
                return false;

            if (path.EndsWith(".prefab", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".unity", StringComparison.OrdinalIgnoreCase))
                return DoesFileContainString(path, ids);

            return false;
        }

        static bool DoesFileContainString(string path, string[] strings)
        {
            if (strings != null && strings.Length > 0)
            {
                using (var file = File.OpenText(path))
                {
                    string line;
                    while ((line = file.ReadLine()) != null)
                    {
                        for (var i = 0; i < strings.Length; i++)
                        {
                            if (line.Contains(strings[i]))
                                return true;
                        }
                    }
                }
            }

            return false;
        }

        List<(int, string)> CreateNewAssetLibraries(Dictionary<int, SpriteLibraryAsset> spriteLibraries)
        {
            var newLibraryPaths = new List<(int, string)>();
            foreach (var pair in spriteLibraries)
            {
                var sourceAsset = pair.Value;

                var path = AssetDatabase.GetAssetPath(sourceAsset);
                var currentAssetPath = Path.GetDirectoryName(path);
                var fileName = Path.GetFileNameWithoutExtension(path);
                var convertFileName = fileName + SpriteLibrarySourceAsset.extension;
                convertFileName = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(currentAssetPath, convertFileName));

                var destAsset = ScriptableObject.CreateInstance<SpriteLibrarySourceAsset>();
                destAsset.SetLibrary(new List<SpriteLibCategoryOverride>(sourceAsset.categories.Count));
                foreach (var sourceCat in sourceAsset.categories)
                {
                    var destCat = new SpriteLibCategoryOverride()
                    {
                        overrideEntries = new List<SpriteCategoryEntryOverride>(sourceCat.categoryList.Count),
                        name = sourceCat.name,
                        entryOverrideCount = sourceCat.categoryList.Count,
                        fromMain = false
                    };
                    destAsset.AddCategory(destCat);
                    foreach (var entry in sourceCat.categoryList)
                    {
                        destCat.overrideEntries.Add(new SpriteCategoryEntryOverride()
                        {
                            name = entry.name,
                            sprite = entry.sprite,
                            fromMain = false,
                            spriteOverride = entry.sprite
                        });
                    }
                }

                var assetBundle = AssetDatabase.GetImplicitAssetBundleName(path);
                if (!string.IsNullOrEmpty(assetBundle))
                {
                    m_AssetBundlesNeedingUpgrade.Add(assetBundle);
                    m_IndicesWithAssetBundleConnection.Add(pair.Key);
                    m_Logger.Add($"{sourceAsset.name} is connected with the following AssetBundle: {assetBundle}.");
                }

                SpriteLibrarySourceAssetImporter.SaveSpriteLibrarySourceAsset(destAsset, convertFileName);
                newLibraryPaths.Add(new ValueTuple<int, string>(pair.Key, convertFileName));
                m_Logger.Add($"Created a new SpriteLibrary with the data of {sourceAsset.name}. The new SpriteLibrary is located at: {convertFileName}");
            }

            return newLibraryPaths;
        }

        static List<SpriteLibraryAsset> LoadNewLibraries(List<(int, string)> sourceAssetPaths)
        {
            var newLibraries = new List<SpriteLibraryAsset>();
            foreach (var path in sourceAssetPaths)
            {
                var newLibraryAsset = AssetDatabase.LoadAssetAtPath<SpriteLibraryAsset>(path.Item2);
                newLibraries.Add(newLibraryAsset);
            }

            return newLibraries;
        }

        void ReassignAssets(string[] assetPaths, List<SpriteLibraryAsset> oldLibraries, List<SpriteLibraryAsset> newLibraries)
        {
            var index = 0;
            foreach (var assetPath in assetPaths)
            {
                m_Logger.Add($"Scanning {assetPath} for components with SpriteLibraryAsset references in need of reassignment.");
                var ext = Path.GetExtension(assetPath);
                if (ext == ".prefab")
                    UpgradePrefab(assetPath, oldLibraries, newLibraries, UpgradeGameObject);
                else if (ext == ".unity")
                    UpgradeScene(assetPath, oldLibraries, newLibraries, UpgradeGameObject);

                var assetBundle = AssetDatabase.GetImplicitAssetBundleName(assetPath);
                if (!string.IsNullOrEmpty(assetBundle))
                {
                    m_AssetBundlesNeedingUpgrade.Add(assetBundle);
                    m_IndicesWithAssetBundleConnection.Add(index);
                }

                index++;
            }
        }

        static void UpgradePrefab(string path, List<SpriteLibraryAsset> oldLibraries, List<SpriteLibraryAsset> newLibraries,
            Action<GameObject, List<SpriteLibraryAsset>, List<SpriteLibraryAsset>> objectUpgrader)
        {
            var objects = AssetDatabase.LoadAllAssetsAtPath(path);

            var firstIndex = 0;
            for (var i = 0; i < objects.Length; i++)
            {
                if (objects[i] as GameObject)
                {
                    firstIndex = i;
                    break;
                }
            }

            if (!PrefabUtility.IsPartOfImmutablePrefab(objects[firstIndex]))
            {
                foreach (var obj in objects)
                {
                    var go = obj as GameObject;
                    if (go != null)
                    {
                        objectUpgrader(go, oldLibraries, newLibraries);
                    }
                }

                var asset = objects[firstIndex] as GameObject;
                PrefabUtility.SavePrefabAsset(asset.transform.root.gameObject);
            }
        }

        static void UpgradeScene(string path, List<SpriteLibraryAsset> oldLibraries, List<SpriteLibraryAsset> newLibraries,
            Action<GameObject, List<SpriteLibraryAsset>, List<SpriteLibraryAsset>> objectUpgrader)
        {
            var scene = default(Scene);
            var openedByUser = false;
            for (var i = 0; i < SceneManager.sceneCount && !openedByUser; i++)
            {
                scene = SceneManager.GetSceneAt(i);
                if (path == scene.path)
                    openedByUser = true;
            }

            if (!openedByUser)
                scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Additive);

            var gameObjects = scene.GetRootGameObjects();
            foreach (var go in gameObjects)
                objectUpgrader(go, oldLibraries, newLibraries);

            EditorSceneManager.SaveScene(scene);
            if (!openedByUser)
                EditorSceneManager.CloseScene(scene, true);
        }

        void UpgradeGameObject(GameObject go, List<SpriteLibraryAsset> oldLibraries, List<SpriteLibraryAsset> newLibraries)
        {
            var types = k_SpriteLibraryReferenceLookup.Keys;
            foreach (var referenceType in types)
            {
                var components = go.GetComponentsInChildren(referenceType);
                foreach (var component in components)
                {
                    if (PrefabUtility.IsPartOfPrefabInstance(component))
                        continue;

                    var fieldInfos = k_SpriteLibraryReferenceLookup[referenceType];
                    foreach (var field in fieldInfos)
                    {
                        var asset = field.GetValue(component);
                        if (asset is SpriteLibraryAsset spriteLibAsset)
                        {
                            var index = oldLibraries.FindIndex(x => x.GetHashCode() == spriteLibAsset.GetHashCode());
                            if (index == -1)
                                continue;

                            field.SetValue(component, newLibraries[index]);
                            m_Logger.Add($"Updated the SpriteLibraryAsset reference in {component.GetType()} on the GameObject {component.name}");
                        }
                        else if (asset is SpriteLibraryAsset[] spriteLibArray)
                        {
                            for (var i = 0; i < spriteLibArray.Length; ++i)
                            {
                                var index = oldLibraries.FindIndex(x => x.GetHashCode() == spriteLibArray[i].GetHashCode());
                                if (index == -1)
                                    continue;

                                spriteLibArray[i] = newLibraries[index];
                            }

                            field.SetValue(component, spriteLibArray);
                            m_Logger.Add($"Updated the SpriteLibraryAsset[] reference in {component.GetType()} on the GameObject {component.name}");
                        }
                        else if (asset is List<SpriteLibraryAsset> spriteLibList)
                        {
                            for (var i = 0; i < spriteLibList.Count; ++i)
                            {
                                var index = oldLibraries.FindIndex(x => x.GetHashCode() == spriteLibList[i].GetHashCode());
                                if (index == -1)
                                    continue;

                                spriteLibList[i] = newLibraries[index];
                            }

                            field.SetValue(component, spriteLibList);
                            m_Logger.Add($"Updated the List<SpriteLibraryAsset> reference in {component.GetType()} on the GameObject {component.name}");
                        }
                    }
                }
            }
        }

        void RemoveOldLibraries(List<SpriteLibraryAsset> oldLibraries)
        {
            foreach (var library in oldLibraries)
            {
                var path = AssetDatabase.GetAssetPath(library);
                var isPsbFile = UpgradeUtilities.IsPsbImportedFile(path);
                if (!string.IsNullOrEmpty(path) && !isPsbFile)
                {
                    m_Logger.Add($"Deleting {path} from project");
                    AssetDatabase.DeleteAsset(path);
                }
            }
        }

        // Leaving this in if we want to cleanup .psbs in the future
        void RemoveSpriteLibFromPsb(string path)
        {
            var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);

            var factory = new SpriteDataProviderFactories();
            factory.Init();
            var dataProvider = factory.GetSpriteEditorDataProviderFromObject(texture);
            dataProvider.InitSpriteEditorDataProvider();
            if (dataProvider.targetObject == null)
            {
                m_Logger.Add($"Could not load the PSDImporter from the path: {path}. Aborting the Sprite Library cleanup inside the .psb.");
                return;
            }

            var so = new SerializedObject(dataProvider.targetObject);
            var property = so.FindProperty(k_PsbImporterCategoriesId);
            if (property != null && property.isArray)
            {
                property.arraySize = 0;
                so.ApplyModifiedPropertiesWithoutUndo();
                dataProvider.Apply();
                m_Logger.Add($"Removed the Sprite Library asset inside {path}");

                var assetImporter = dataProvider.targetObject as AssetImporter;
                assetImporter.SaveAndReimport();
                m_Logger.Add($"Saved and re-imported the file.");
            }
            else
            {
                m_Logger.Add($"Could not find any Sprite Library asset inside the .psb to cleanup.");
            }
        }

        void AddAssetBundlesToLog()
        {
            m_Logger.AddLineBreak();
            m_Logger.Add("[NOTE] The following AssetBundles need to be rebuilt:");
            foreach(var assetBundle in m_AssetBundlesNeedingUpgrade)
                m_Logger.Add(assetBundle);
        }
    }
}
