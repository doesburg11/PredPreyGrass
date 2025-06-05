using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Utility Class for creating Palettes
    /// </summary>
    public static class GridPaletteUtility
    {
        internal enum GridPaletteType
        {
            Rectangle,
            HexagonalPointTop,
            HexagonalFlatTop,
            Isometric,
            IsometricZAsY,
        };

        internal static readonly Vector3 defaultSortAxis = new Vector3(0f, 0f, 1f);

        internal static GridLayout.CellLayout GetCellLayoutFromGridPaletteType(GridPaletteType paletteType)
        {
            switch (paletteType)
            {
                case GridPaletteType.HexagonalPointTop:
                case GridPaletteType.HexagonalFlatTop:
                {
                    return GridLayout.CellLayout.Hexagon;
                }
                case GridPaletteType.Isometric:
                {
                    return GridLayout.CellLayout.Isometric;
                }
                case GridPaletteType.IsometricZAsY:
                {
                    return GridLayout.CellLayout.IsometricZAsY;
                }
            }
            return GridLayout.CellLayout.Rectangle;
        }

        internal static RectInt GetBounds(GameObject palette)
        {
            if (palette == null)
                return new RectInt();

            Vector2Int min = new Vector2Int(int.MaxValue, int.MaxValue);
            Vector2Int max = new Vector2Int(int.MinValue, int.MinValue);

            foreach (var tilemap in palette.GetComponentsInChildren<Tilemap>())
            {
                Vector3Int p1 = tilemap.editorPreviewOrigin;
                Vector3Int p2 = p1 + tilemap.editorPreviewSize;
                Vector2Int tilemapMin = new Vector2Int(Mathf.Min(p1.x, p2.x), Mathf.Min(p1.y, p2.y));
                Vector2Int tilemapMax = new Vector2Int(Mathf.Max(p1.x, p2.x), Mathf.Max(p1.y, p2.y));
                min = new Vector2Int(Mathf.Min(min.x, tilemapMin.x), Mathf.Min(min.y, tilemapMin.y));
                max = new Vector2Int(Mathf.Max(max.x, tilemapMax.x), Mathf.Max(max.y, tilemapMax.y));
            }

            return GridEditorUtility.GetMarqueeRect(min, max);
        }

        /// <summary>
        /// Creates a Palette Asset at the current selected folder path. This will show a popup allowing you to choose
        /// a different folder path for saving the Palette Asset if required.
        /// </summary>
        /// <param name="name">Name of the Palette Asset.</param>
        /// <param name="layout">Grid Layout of the Palette Asset.</param>
        /// <param name="cellSizing">Cell Sizing of the Palette Asset.</param>
        /// <param name="cellSize">Cell Size of the Palette Asset.</param>
        /// <param name="swizzle">Cell Swizzle of the Palette.</param>
        /// <returns>The created Palette Asset if successful.</returns>
        public static GameObject CreateNewPaletteAtCurrentFolder(string name, GridLayout.CellLayout layout, GridPalette.CellSizing cellSizing, Vector3 cellSize, GridLayout.CellSwizzle swizzle)
        {
            return CreateNewPaletteAtCurrentFolder(name, layout, cellSizing, cellSize, swizzle
                , TransparencySortMode.Default, defaultSortAxis);
        }

        /// <summary>
        /// Creates a Palette Asset at the current selected folder path. This will show a popup allowing you to choose
        /// a different folder path for saving the Palette Asset if required.
        /// </summary>
        /// <param name="name">Name of the Palette Asset.</param>
        /// <param name="layout">Grid Layout of the Palette Asset.</param>
        /// <param name="cellSizing">Cell Sizing of the Palette Asset.</param>
        /// <param name="cellSize">Cell Size of the Palette Asset.</param>
        /// <param name="swizzle">Cell Swizzle of the Palette.</param>
        /// <param name="sortMode">Transparency Sort Mode for the Palette</param>
        /// <param name="sortAxis">Transparency Sort Axis for the Palette</param>
        /// <returns>The created Palette Asset if successful.</returns>
        public static GameObject CreateNewPaletteAtCurrentFolder(string name
            , GridLayout.CellLayout layout
            , GridPalette.CellSizing cellSizing
            , Vector3 cellSize
            , GridLayout.CellSwizzle swizzle
            , TransparencySortMode sortMode
            , Vector3 sortAxis)
        {
            string defaultPath = ProjectBrowser.s_LastInteractedProjectBrowser ? ProjectBrowser.s_LastInteractedProjectBrowser.GetActiveFolderPath() : "Assets";
            string folderPath = EditorUtility.SaveFolderPanel("Create palette into folder ", defaultPath, "");
            folderPath = FileUtil.GetProjectRelativePath(folderPath);
            if (!TilePaletteSaveUtility.ValidateSaveFolder(folderPath))
                return null;
            return CreateNewPalette(folderPath, name, layout, cellSizing, cellSize, swizzle, sortMode, sortAxis);
        }

        /// <summary>
        /// Creates a Palette Asset at the given folder path.
        /// </summary>
        /// <param name="folderPath">Folder Path of the Palette Asset.</param>
        /// <param name="name">Name of the Palette Asset.</param>
        /// <param name="layout">Grid Layout of the Palette Asset.</param>
        /// <param name="cellSizing">Cell Sizing of the Palette Asset.</param>
        /// <param name="cellSize">Cell Size of the Palette Asset.</param>
        /// <param name="swizzle">Cell Swizzle of the Palette.</param>
        /// <returns>The created Palette Asset if successful.</returns>
        public static GameObject CreateNewPalette(string folderPath
            , string name
            , GridLayout.CellLayout layout
            , GridPalette.CellSizing cellSizing
            , Vector3 cellSize
            , GridLayout.CellSwizzle swizzle)
        {
            return CreateNewPalette(folderPath, name, layout, cellSizing, cellSize, swizzle,
                TransparencySortMode.Default, defaultSortAxis);
        }

        /// <summary>
        /// Creates a Palette Asset at the given folder path.
        /// </summary>
        /// <param name="folderPath">Folder Path of the Palette Asset.</param>
        /// <param name="name">Name of the Palette Asset.</param>
        /// <param name="layout">Grid Layout of the Palette Asset.</param>
        /// <param name="cellSizing">Cell Sizing of the Palette Asset.</param>
        /// <param name="cellSize">Cell Size of the Palette Asset.</param>
        /// <param name="swizzle">Cell Swizzle of the Palette.</param>
        /// <param name="sortMode">Transparency Sort Mode for the Palette</param>
        /// <param name="sortAxis">Transparency Sort Axis for the Palette</param>
        /// <returns>The created Palette Asset if successful.</returns>
        public static GameObject CreateNewPalette(string folderPath
            , string name
            , GridLayout.CellLayout layout
            , GridPalette.CellSizing cellSizing
            , Vector3 cellSize
            , GridLayout.CellSwizzle swizzle
            , TransparencySortMode sortMode
            , Vector3 sortAxis)
        {
            var temporaryGO = CreateNewPaletteGameObject(name, layout, cellSize, swizzle);
            var path = AssetDatabase.GenerateUniqueAssetPath(folderPath + "/" + name + ".prefab");
            var prefab = PrefabUtility.SaveAsPrefabAssetAndConnect(temporaryGO, path, InteractionMode.AutomatedAction);
            var gridPalette = CreateGridPalette(cellSizing, sortMode, sortAxis);
            AssetDatabase.AddObjectToAsset(gridPalette, prefab);
            PrefabUtility.ApplyPrefabInstance(temporaryGO, InteractionMode.AutomatedAction);
            AssetDatabase.Refresh();

            Object.DestroyImmediate(temporaryGO);
            return AssetDatabase.LoadAssetAtPath<GameObject>(path);
        }

        internal static GameObject CreateNewPaletteGameObject(string name
            , GridLayout.CellLayout layout
            , Vector3 cellSize
            , GridLayout.CellSwizzle swizzle)
        {
            var temporaryGO = new GameObject(name);
            var grid = temporaryGO.AddComponent<Grid>();
            grid.cellSize = cellSize;
            grid.cellLayout = layout;
            grid.cellSwizzle = swizzle;
            CreateNewLayer(temporaryGO, "Layer1", layout);
            return temporaryGO;
        }

        internal static IEnumerable<(T1, T2, T3)> MultipleEnumerate<T1, T2, T3>(IEnumerable<T1> t1s, IEnumerable<T2> t2s, IEnumerable<T3> t3s)
        {
            using IEnumerator<T1> enum1 = t1s.GetEnumerator();
            using IEnumerator<T2> enum2 = t2s.GetEnumerator();
            using IEnumerator<T3> enum3 = t3s.GetEnumerator();
            while (enum1.MoveNext() && enum2.MoveNext() && enum3.MoveNext())
                yield return (enum1.Current, enum2.Current, enum3.Current);
        }

        /// <summary>
        /// Creates a Palette Asset to be used as a sub-asset.
        /// </summary>
        /// <param name="name">Name of the Palette Asset.</param>
        /// <param name="layout">Grid Layout of the Palette Asset.</param>
        /// <param name="cellSizing">Cell Sizing of the Palette Asset.</param>
        /// <param name="cellSize">Cell Size of the Palette Asset.</param>
        /// <param name="swizzle">Cell Swizzle of the Palette.</param>
        /// <param name="sortMode">Transparency Sort Mode for the Palette</param>
        /// <param name="sortAxis">Transparency Sort Axis for the Palette</param>
        /// <param name="texture2Ds">Texture Sources of Sprites</param>
        /// <param name="textureSprites">Sprites to add to the Palette, organised by containing Texture</param>
        /// <param name="palette">Palette Settings asset created if successful</param>
        /// <param name="tiles">Tile assets created if successful</param>
        /// <returns>The created Palette Asset if successful.</returns>
        internal static GameObject CreateNewPaletteAsSubAsset(string name
            , GridLayout.CellLayout layout
            , GridPalette.CellSizing cellSizing
            , Vector3 cellSize
            , GridLayout.CellSwizzle swizzle
            , TransparencySortMode sortMode
            , Vector3 sortAxis
            , IEnumerable<Texture2D> texture2Ds
            , IEnumerable<IEnumerable<Sprite>> textureSprites
            , IEnumerable<TileTemplate> templates
            , out GridPalette palette
            , out List<TileBase> tiles)
        {
            palette = null;
            tiles = null;
            if (texture2Ds == null || textureSprites == null || templates == null)
                return null;

            var createTileMethod = GridPaintActiveTargetsPreferences.GetCreateTileFromPaletteUsingPreferences();
            var paletteGO = new GameObject(name);
            var grid = paletteGO.AddComponent<Grid>();

            grid.cellSize = cellSize;
            grid.cellLayout = layout;
            grid.cellSwizzle = swizzle;
            var layer = CreateNewLayer(paletteGO, "Layer1", layout);
            var tilemap = layer.GetComponent<Tilemap>();

            palette = CreateGridPalette(cellSizing, sortMode, sortAxis);
            var tilesSet = new HashSet<TileBase>();

            var paletteOffset = Vector3Int.zero;
            var uniqueNames = new HashSet<string>();

            var itemCount = 0;
            foreach (var (_, _, _) in MultipleEnumerate(texture2Ds, textureSprites, templates))
            {
                itemCount++;
            }

            var square = Mathf.CeilToInt(Mathf.Sqrt(itemCount));
            var height = 0;

            foreach (var (texture2D, textureSprite, tileTemplate) in MultipleEnumerate(texture2Ds, textureSprites, templates))
            {
                if (texture2D == null || textureSprite == null)
                    continue;

                var tileChangeData = new List<TileChangeData>();
                if (tileTemplate != null)
                {
                    try
                    {
                        tileTemplate.CreateTileAssets(texture2D, textureSprite, ref tileChangeData);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Unable to create Tile Assets for {tileTemplate} using {texture2D} with error: {e.ToString()}", tileTemplate);
                    }

                    var x = Int32.MaxValue;
                    var y = Int32.MinValue;
                    foreach (var tileChange in tileChangeData)
                    {
                        var position = tileChange.position;
                        x = Math.Min(x, position.x);
                        y = Math.Max(y, position.y);
                    }
                    var localOffset = new Vector3Int(-x, -y, 0);
                    for (var i = 0; i < tileChangeData.Count; ++i)
                    {
                        var tileChange = tileChangeData[i];
                        tileChange.position += paletteOffset + localOffset;
                        tileChangeData[i] = tileChange;
                        tilesSet.Add(tileChange.tile);
                    }
                }
                else
                {
                    if (createTileMethod == null)
                        continue;

                    var sprites = new List<Sprite>();
                    foreach (var sprite in textureSprite)
                    {
                        sprites.Add(sprite);
                    }

                    var ratio = TileDragAndDrop.GetRatioOfSameSizedSprites(sprites);
                    Dictionary<Vector2Int, TileDragAndDropHoverData> hoverData = null;
                    var tileMap = new Dictionary<Sprite, TileBase>();

                    // Use Texture position to place sprites if they are same sized
                    if (GridPalette.CellSizing.Automatic == cellSizing && ratio >= 0.66f)
                    {
                        hoverData = TileDragAndDrop.CreateHoverData(sprites, layout);
                    }
                    else
                    {
                        hoverData = new Dictionary<Vector2Int, TileDragAndDropHoverData>();
                        var width = Mathf.RoundToInt(Mathf.Sqrt(sprites.Count));
                        var currentPosition = Vector2Int.zero;
                        foreach (Sprite sprite in sprites)
                        {
                            hoverData.Add(currentPosition, new TileDragAndDropHoverData(sprite, tilemap.tileAnchor, Vector3.one, true));
                            currentPosition += new Vector2Int(1, 0);
                            if (currentPosition.x >= width)
                                currentPosition = new Vector2Int(0, currentPosition.y - 1);
                        }
                    }

                    var y = Int32.MinValue;
                    foreach (var key in hoverData.Keys)
                    {
                        y = Math.Max(y, key.y);
                    }
                    foreach (var item in hoverData)
                    {
                        var i = 0;
                        if (item.Value.hoverObject is Sprite sprite)
                        {
                            if (!tileMap.TryGetValue(sprite, out TileBase tile))
                            {
                                tile = createTileMethod.Invoke(null, new object[] { sprite }) as TileBase;
                                tileMap.Add(sprite, tile);
                            }
                            if (tile == null)
                                continue;

                            var tileName = tile.name;
                            if (string.IsNullOrEmpty(tileName) || uniqueNames.Contains(tileName))
                            {
                                tileName = TileDragAndDrop.GenerateUniqueNameForNamelessSprite(sprite, uniqueNames, ref i);
                                tile.name = tileName;
                            }
                            uniqueNames.Add(tileName);

                            tileChangeData.Add(new TileChangeData()
                            {
                                position = new Vector3Int(item.Key.x, item.Key.y - y, 0) + paletteOffset,
                                tile = tile,
                                transform = Matrix4x4.TRS(item.Value.positionOffset - tilemap.tileAnchor, Quaternion.identity, Vector3.one),
                                color = Color.white
                            });
                            tilesSet.Add(tile);
                        }
                    }
                }
                if (tileChangeData.Count > 0)
                    tilemap.SetTiles(tileChangeData.ToArray(), true);

                if (++height == square)
                {
                    height = 0;
                    paletteOffset.y = 0;
                    paletteOffset.x = tilemap.size.x;
                }
                else
                {
                    var y = 0;
                    foreach (var changeData in tileChangeData)
                    {
                        y = Math.Min(y, changeData.position.y);
                    }
                    paletteOffset.y = y - 1;
                }
            }

            tilemap.CompressBounds();
            tiles = new List<TileBase>(tilesSet);

            return paletteGO;
        }

        private static GameObject CreateNewLayer(GameObject paletteGO, string name, GridLayout.CellLayout layout)
        {
            GameObject newLayerGO = new GameObject(name);
            var tilemap = newLayerGO.AddComponent<Tilemap>();
            var renderer = newLayerGO.AddComponent<TilemapRenderer>();
            newLayerGO.transform.parent = paletteGO.transform;
            newLayerGO.layer = paletteGO.layer;

            // Set defaults for certain layouts
            switch (layout)
            {
                case GridLayout.CellLayout.Hexagon:
                {
                    tilemap.tileAnchor = Vector3.zero;
                    break;
                }
                case GridLayout.CellLayout.Isometric:
                {
                    renderer.sortOrder = TilemapRenderer.SortOrder.TopRight;
                    break;
                }
                case GridLayout.CellLayout.IsometricZAsY:
                {
                    renderer.sortOrder = TilemapRenderer.SortOrder.TopRight;
                    renderer.mode = TilemapRenderer.Mode.Individual;
                    break;
                }
            }

            return newLayerGO;
        }

        internal static GridPalette GetGridPaletteFromPaletteAsset(Object palette)
        {
            string assetPath = AssetDatabase.GetAssetPath(palette);
            GridPalette paletteAsset = AssetDatabase.LoadAssetAtPath<GridPalette>(assetPath);
            return paletteAsset;
        }

        internal static GridPalette CreateGridPalette(GridPalette.CellSizing cellSizing)
        {
            return CreateGridPalette(cellSizing, TransparencySortMode.Default, defaultSortAxis);
        }

        internal static GridPalette CreateGridPalette(GridPalette.CellSizing cellSizing
            , TransparencySortMode sortMode
            , Vector3 sortAxis
        )
        {
            var palette = ScriptableObject.CreateInstance<GridPalette>();
            palette.name = "Palette Settings";
            palette.cellSizing = cellSizing;
            palette.transparencySortMode = sortMode;
            palette.transparencySortAxis = sortAxis;
            return palette;
        }

        internal static Vector3 CalculateAutoCellSize(Grid grid, Vector3 defaultValue)
        {
            Tilemap[] tilemaps = grid.GetComponentsInChildren<Tilemap>();
            Sprite[] sprites = null;
            var maxSize = Vector2.negativeInfinity;
            var minSize = Vector2.positiveInfinity;

            // Get minimum and maximum sizes for Sprites
            foreach (var tilemap in tilemaps)
            {
                var spriteCount = tilemap.GetUsedSpritesCount();
                if (sprites == null || sprites.Length < spriteCount)
                    sprites = new Sprite[spriteCount];
                tilemap.GetUsedSpritesNonAlloc(sprites);
                for (int i = 0; i < spriteCount; ++i)
                {
                    Sprite sprite = sprites[i];
                    if (sprite != null)
                    {
                        var cellSize = new Vector3(sprite.rect.width, sprite.rect.height, 0f) / sprite.pixelsPerUnit;
                        if (tilemap.cellSwizzle == GridLayout.CellSwizzle.YXZ)
                        {
                            (cellSize.x, cellSize.y) = (cellSize.y, cellSize.x);
                        }
                        minSize.x = Mathf.Min(cellSize.x, minSize.x);
                        minSize.y = Mathf.Min(cellSize.y, minSize.y);
                        maxSize.x = Mathf.Max(cellSize.x, maxSize.x);
                        maxSize.y = Mathf.Max(cellSize.y, maxSize.y);
                    }
                }
            }
            // Validate that Sprites are in multiples of sizes
            foreach (var tilemap in tilemaps)
            {
                var spriteCount = tilemap.GetUsedSpritesCount();
                if (sprites == null || sprites.Length < spriteCount)
                    sprites = new Sprite[spriteCount];
                tilemap.GetUsedSpritesNonAlloc(sprites);
                for (int i = 0; i < spriteCount; ++i)
                {
                    Sprite sprite = sprites[i];
                    if (sprite != null)
                    {
                        var cellSize = new Vector3(sprite.rect.width, sprite.rect.height, 0f) / sprite.pixelsPerUnit;
                        if (tilemap.cellSwizzle == GridLayout.CellSwizzle.YXZ)
                        {
                            var swap = cellSize.x;
                            cellSize.x = cellSize.y;
                            cellSize.y = swap;
                        }
                        // Return maximum size if sprites are not multiples of the smallest size
                        if (cellSize.x % minSize.x > 0)
                            return maxSize.x * maxSize.y <= 0f ? defaultValue : new Vector3(maxSize.x, maxSize.y, 0f);
                        if (cellSize.y % minSize.y > 0)
                            return maxSize.x * maxSize.y <= 0f ? defaultValue : new Vector3(maxSize.x, maxSize.y, 0f);
                    }
                }
            }
            return minSize.x * minSize.y <= 0f || minSize == Vector2.positiveInfinity ? defaultValue : new Vector3(minSize.x, minSize.y, 0f);
        }
    }
}
