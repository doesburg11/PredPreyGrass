using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor.U2D;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.U2D;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Class containing utility functions for TileSet operations
    /// </summary>
    internal static class TileSetUtility
    {
        static internal Action<int, ProjectWindowCallback.EndNameEditAction, string, Texture2D, string> StartNewAssetNameEditingDelegate = ProjectWindowUtil.StartNameEditingIfProjectWindowExists;

        private static bool HasTileSetExtension(string path)
        {
            return FileUtil.GetPathExtension(path).ToLower().Equals("tileset");
        }

        private static SpriteAtlas CreateDefaultSpriteAtlas()
        {
            var spriteAtlas = new SpriteAtlas();
            spriteAtlas.SetV2();
            var sats = new SpriteAtlasTextureSettings()
            {
                filterMode = FilterMode.Point,
                anisoLevel = 0,
                generateMipMaps = false,
                sRGB = true,
            };
            spriteAtlas.SetTextureSettings(sats);
            return spriteAtlas;
        }

        public static bool IsPalettePrefabFromTileSet(GameObject gameObject)
        {
            if (PrefabUtility.IsPartOfAnyPrefab(gameObject))
            {
                var prefab = PrefabUtility.GetPrefabAssetRootGameObject(gameObject);
                var path = AssetDatabase.GetAssetPath(prefab);
                return HasTileSetExtension(path);
            }
            return false;
        }

        internal static void CreateTileSetAssetInProjectWindow(string defaultAssetName)
        {
            var assetSelectionPath = AssetDatabase.GetAssetPath(Selection.activeObject);
            var isFolder = false;
            if (!string.IsNullOrEmpty(assetSelectionPath))
                isFolder = File.GetAttributes(assetSelectionPath).HasFlag(FileAttributes.Directory);
            var path = ProjectWindowUtil.GetActiveFolderPath();
            if (isFolder)
            {
                path = assetSelectionPath;
            }

            var destName = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(path, defaultAssetName));
            var icon = EditorGUIUtility.IconContent<GameObject>().image as Texture2D;

            CreateAssetEndNameEditAction action = ScriptableObject.CreateInstance<CreateAssetEndNameEditAction>();
            StartNewAssetNameEditingDelegate(0, action, destName, icon, "");
        }

        internal static void CreateTileSetAsset(string pathName)
        {
            var tileSet = ScriptableObject.CreateInstance<TileSet>();
            var spriteAtlas = CreateDefaultSpriteAtlas();

            SaveTileSetAsset(pathName, tileSet, spriteAtlas);
        }

        internal class CreateAssetEndNameEditAction : ProjectWindowCallback.EndNameEditAction
        {
            public override void Action(int instanceId, string pathName, string resourceFile)
            {
                CreateTileSetAsset(pathName);
            }
        }

        internal static TileSet LoadTileSetFromPalettePrefab(GameObject gameObject)
        {
            return LoadTileSetAsset(AssetDatabase.GetAssetPath(gameObject));
        }

        internal static TileSet LoadTileSetAsset(string pathName)
        {
            if (String.IsNullOrEmpty(pathName))
                return null;

            var objects = InternalEditorUtility.LoadSerializedFileAndForget(pathName);
            TileSet tileSet = default;
            foreach (var obj in objects)
            {
                if (obj is TileSet tileSetObj)
                {
                    tileSet = tileSetObj;
                    break;
                }
            }
            if (tileSet == null)
            {
                Debug.LogError("Unable to load TileSet asset");
                return null;
            }
            return tileSet;
        }

        internal static SpriteAtlas LoadSpriteAtlasAsset(string pathName)
        {
            if (String.IsNullOrEmpty(pathName))
                return null;

            var objects = InternalEditorUtility.LoadSerializedFileAndForget(pathName);
            SpriteAtlas spriteAtlas = default;
            foreach (var obj in objects)
            {
                if (obj is SpriteAtlas spriteAtlasObj)
                {
                    spriteAtlas = spriteAtlasObj;
                    break;
                }
            }
            if (spriteAtlas == null)
            {
                Debug.LogError("Unable to load SpriteAtlas asset");
                return null;
            }
            return spriteAtlas;
        }

        internal static void SaveTileSetAsset(string pathName, TileSet tileSet)
        {
            if (tileSet == null)
                return;
            if (!HasTileSetExtension(pathName))
                return;

            SpriteAtlas spriteAtlas = default;
            if (String.IsNullOrEmpty(pathName))
            {
                spriteAtlas = CreateDefaultSpriteAtlas();
            }
            else
            {
                var objects = InternalEditorUtility.LoadSerializedFileAndForget(pathName);
                foreach (var obj in objects)
                {
                    if (obj is SpriteAtlas spriteAtlasObj)
                    {
                        spriteAtlas = spriteAtlasObj;
                        break;
                    }
                }
            }
            if (spriteAtlas == null)
            {
                Debug.LogError("Unable to save TileSet asset", tileSet);
                return;
            }
            SaveTileSetAsset(pathName, tileSet, spriteAtlas);
        }

        internal static void SaveTileSetAsset(string pathName, TileSet tileSet, SpriteAtlas spriteAtlas)
        {
            if (tileSet == null || spriteAtlas == null)
                return;
            if (!HasTileSetExtension(pathName))
                return;

            // Do this so that asset change save dialog will not show
            var originalValue = EditorPrefs.GetBool("VerifySavingAssets", false);
            EditorPrefs.SetBool("VerifySavingAssets", false);
            InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] {
                tileSet, spriteAtlas
            }, pathName, EditorSettings.serializationMode != SerializationMode.ForceBinary);
            EditorPrefs.SetBool("VerifySavingAssets", originalValue);
            AssetDatabase.ImportAsset(pathName, ImportAssetOptions.ForceSynchronousImport);
        }
    }
}
