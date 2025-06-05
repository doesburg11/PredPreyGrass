using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor.PackageManager.UI;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    internal static class DefaultAssetCreation
    {
        enum TilePaletteAssetMenuPriority : int
        {
            Rectangular = 4,
            HexagonalFlatTop,
            HexagonalPointTop,
            Isometric,
            TileSet
        }

        private const int k_TilePaletteWhiteboxAssetMenuPriority = (int)TilePaletteAssetMenuPriority.Rectangular + 100;

        static internal Action<int, ProjectWindowCallback.EndNameEditAction, string, Texture2D, string> StartNewAssetNameEditingDelegate = ProjectWindowUtil.StartNameEditingIfProjectWindowExists;

        [MenuItem("Assets/Create/2D/Tile Palette/Rectangular", priority = (int)TilePaletteAssetMenuPriority.Rectangular)]
        static void MenuItem_AssetsCreate2DTilePaletteRectangular(MenuCommand menuCommand)
        {
            CreateAssetObject("Rectangular Palette", GridLayout.CellLayout.Rectangle, GridLayout.CellSwizzle.XYZ, GridPalette.CellSizing.Automatic, new Vector3(1, 1, 0));
        }

        [MenuItem("Assets/Create/2D/Tile Palette/Hexagonal Flat Top", priority = (int)TilePaletteAssetMenuPriority.HexagonalFlatTop)]
        static void MenuItem_AssetsCreate2DTilePaletteHexagonalFlatTop(MenuCommand menuCommand)
        {
            CreateAssetObject("Hexagonal Flat Top Palette", GridLayout.CellLayout.Hexagon, GridLayout.CellSwizzle.YXZ, GridPalette.CellSizing.Manual, new Vector3(0.8659766f, 1, 0));
        }

        [MenuItem("Assets/Create/2D/Tile Palette/Hexagonal Point Top", priority = (int)TilePaletteAssetMenuPriority.HexagonalPointTop)]
        static void MenuItem_AssetsCreate2DTilePaletteHexagonalPointedTop(MenuCommand menuCommand)
        {
            CreateAssetObject("Hexagonal Point Top Palette", GridLayout.CellLayout.Hexagon, GridLayout.CellSwizzle.XYZ, GridPalette.CellSizing.Manual, new Vector3(0.8659766f, 1, 0));
        }

        [MenuItem("Assets/Create/2D/Tile Palette/Isometric", priority = (int)TilePaletteAssetMenuPriority.Isometric)]
        static void MenuItem_AssetsCreate2DTilePaletteIsometric(MenuCommand menuCommand)
        {
            CreateAssetObject("Isometric Palette", GridLayout.CellLayout.Isometric, GridLayout.CellSwizzle.XYZ, GridPalette.CellSizing.Manual, new Vector3(1, 0.5f, 0));
        }

        [MenuItem("Assets/Create/2D/Tile Palette/New Tile Set", priority = (int)TilePaletteAssetMenuPriority.TileSet)]
        public static void CreateTileSetAsset()
        {
            TileSetUtility.CreateTileSetAssetInProjectWindow("New TileSet.tileset");
        }

        [InitializeOnLoadMethod]
        internal static void InitializeWhiteboxSampleMenuItems()
        {
            // Defer addition of whitebox sample menu items
            Menu.menuChanged += ValidateWhiteboxSampleMenuItems;
        }

        private static string GenerateWhiteboxMenuName(Sample sample)
        {
            var name = $"Assets/Create/2D/Tile Palette/{sample.displayName}";
            name = name.Replace(" Tile Palette", String.Empty);
            return name;
        }

        private static void ValidateWhiteboxSampleMenuItems()
        {
            Menu.menuChanged -= ValidateWhiteboxSampleMenuItems;
            CreateWhiteboxSampleMenuItems();
        }

        private static void CreateWhiteboxSampleMenuItems()
        {
            var samples = TilePaletteWhiteboxSamplesUtility.whiteboxSamples;
            foreach (var sample in samples)
            {
                var name = GenerateWhiteboxMenuName(sample);
                if (Menu.MenuItemExists(name))
                {
                    continue;
                }
                Menu.AddMenuItem(name, String.Empty, false, k_TilePaletteWhiteboxAssetMenuPriority,
                    () => { DuplicateWhiteboxSample(sample); }, null);
            }
        }

        private static void DuplicateWhiteboxSample(Sample sample)
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

            var destName = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(path, sample.displayName));
            var icon = EditorGUIUtility.IconContent<GameObject>().image as Texture2D;

            DuplicateWhiteboxPaletteEndNameEditAction action = ScriptableObject.CreateInstance<DuplicateWhiteboxPaletteEndNameEditAction>();
            action.sample = sample;
            action.path = path;
            StartNewAssetNameEditingDelegate(0, action, destName, icon, "");
        }

        static void CreateAssetObject(string defaultAssetName, GridLayout.CellLayout layout, GridLayout.CellSwizzle swizzle, GridPalette.CellSizing cellSizing, Vector3 cellSize)
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

            Dictionary<Vector2Int, TileDragAndDropHoverData> hoverData = null;
            if (Selection.objects != null && Selection.objects.Length > 0)
            {
                var sheets = TileDragAndDrop.GetValidSpritesheets(Selection.objects);
                var sprites = TileDragAndDrop.GetValidSingleSprites(Selection.objects);
                var tiles = TileDragAndDrop.GetValidTiles(Selection.objects);
                var gos = TileDragAndDrop.GetValidGameObjects(Selection.objects);
                hoverData = TileDragAndDrop.CreateHoverData(sheets, sprites, tiles, gos, layout);
            }

            CreateAssetEndNameEditAction action = ScriptableObject.CreateInstance<CreateAssetEndNameEditAction>();
            action.swizzle = swizzle;
            action.layout = layout;
            action.cellSize = cellSize;
            action.cellSizing = cellSizing;
            action.hoverData = hoverData;

            StartNewAssetNameEditingDelegate(0, action, destName, icon, "");
        }

        internal class DuplicateWhiteboxPaletteEndNameEditAction : ProjectWindowCallback.EndNameEditAction
        {
            public string path;
            public Sample sample;

            public override void Action(int instanceId, string pathName, string resourceFile)
            {
                TilePaletteWhiteboxSamplesUtility.DuplicateWhiteboxSample(sample, path, FileUtil.UnityGetFileName(pathName));
            }
        }

        internal class CreateAssetEndNameEditAction : ProjectWindowCallback.EndNameEditAction
        {
            public GridLayout.CellLayout layout { get; set; }
            public GridLayout.CellSwizzle swizzle { get; set; }
            public Vector3 cellSize { get; set; }
            public GridPalette.CellSizing cellSizing { get; set; }

            public Dictionary<Vector2Int, TileDragAndDropHoverData> hoverData { get; set; }

            public override void Action(int instanceId, string pathName, string resourceFile)
            {
                var directoryName = Path.GetDirectoryName(pathName);
                var go = GridPaletteUtility.CreateNewPalette(directoryName, Path.GetFileName(pathName), layout,
                    cellSizing, cellSize, swizzle);

                if (hoverData == null)
                    return;

                var tilemap = go.GetComponentInChildren<Tilemap>();
                var tileSheet = TileDragAndDrop.ConvertToTileSheet(hoverData, directoryName);
                var i = 0;
                foreach (var item in hoverData)
                {
                    if (i >= tileSheet.Count)
                        break;

                    var offset = Vector3.zero;
                    if (item.Value.hasOffset)
                    {
                        offset = item.Value.positionOffset - tilemap.tileAnchor;
                        offset.x *= item.Value.scaleFactor.x;
                        offset.y *= item.Value.scaleFactor.y;
                        offset.z *= item.Value.scaleFactor.z;
                    }
                    tilemap.SetTile(new Vector3Int(item.Key.x, item.Key.y), tileSheet[i++]);
                }
                EditorUtility.SetDirty(go);
                AssetDatabase.SaveAssetIfDirty(go);

                Selection.activeObject = go;
            }
        }
    }
}
