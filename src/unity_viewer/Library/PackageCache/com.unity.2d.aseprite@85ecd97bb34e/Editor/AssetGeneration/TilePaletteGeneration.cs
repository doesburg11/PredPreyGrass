using System.Collections.Generic;
using UnityEditor.AssetImporters;
using UnityEditor.Tilemaps;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.U2D.Aseprite
{
    internal static class TilePaletteGeneration
    {
        public static void Generate(
            AssetImportContext ctx,
            List<Layer> layers,
            List<TileSet> tileSets,
            Sprite[] sprites,
            float ppu,
            ref UnityEngine.Object mainAsset)
        {
            if (tileSets.Count == 0 || sprites.Length == 0)
                return;

            var tileLayers = GetTileLayers(layers);
            var textures = GetTextures(ctx);

            TileTemplate[] tileTemplates;
            List<Sprite>[] tileSprites;

#if ENABLE_TILEMAP_API
            if (tileLayers.Count > 0)
            {
                tileTemplates = GetLayerOrderedTileTemplate(tileLayers, tileSets, sprites, out tileSprites);
            }
            else
#endif            
            {
                tileSprites = GetTileSprites(tileSets, sprites);
                tileTemplates = GetDefaultTileTemplate(textures.Count);
            }

            var tileSet = GetFirstUsedTileSet(tileSets, tileLayers);
            
            var paletteName = tileSet.name;
            var cellLayout = GridLayout.CellLayout.Rectangle;
            var cellSizing = GridPalette.CellSizing.Manual;
            var cellSize = new Vector3(tileSet.tileSize.x / ppu, tileSet.tileSize.y / ppu, 0);
            var cellSwizzle = GridLayout.CellSwizzle.XYZ;
            var sortMode = TransparencySortMode.Default;
            var sortAxis = Vector3.forward;
            
            var paletteGameObject = GridPaletteUtility.CreateNewPaletteAsSubAsset(
                paletteName,
                cellLayout,
                cellSizing,
                cellSize,
                cellSwizzle,
                sortMode,
                sortAxis,
                textures,
                tileSprites,
                tileTemplates,
                out var palette,
                out var tileAssets
            );
            
            foreach (var tile in tileAssets)
                ctx.AddObjectToAsset(tile.name, tile);

            ctx.AddObjectToAsset($"{paletteName}_PaletteSettings", palette);
            ctx.AddObjectToAsset(paletteGameObject.name, paletteGameObject);
            mainAsset = paletteGameObject;
        }
        
        static List<Texture2D> GetTextures(AssetImportContext ctx)
        {
            var assetObjects = new List<Object>();
            ctx.GetObjects(assetObjects);

            var textures = new List<Texture2D>();
            foreach (var obj in assetObjects)
            {
                if (obj is Texture2D texture)
                    textures.Add(texture);
            }
            return textures;            
        }

        static List<Layer> GetTileLayers(List<Layer> layers)
        {
            var tileLayers = new List<Layer>();
            foreach (var layer in layers)
            {
                if (layer.layerType == LayerTypes.Tilemap && layer.tileCells.Count > 0)
                    tileLayers.Add(layer);
            }
            return tileLayers;
        }

        static TileSet GetFirstUsedTileSet(List<TileSet> tileSets, List<Layer> layers)
        {
            foreach (var layer in layers)
            {
                var index = layer.tileSetIndex;
                var tileSet = tileSets.Find(x => x.id == index);
                if (tileSet != null)
                    return tileSet;
            }
            return tileSets[0];
        }
        
#if ENABLE_TILEMAP_API
        static TileTemplate[] GetLayerOrderedTileTemplate(List<Layer> layers, List<TileSet> tileSets, Sprite[] sprites, out List<Sprite>[] spritesPerTileSet)
        {
            var tileSprites = new List<Sprite>();
            var tilePositions = new List<Vector3Int>();

            for (var i = 0; i < layers.Count; ++i)
                GetSpriteAndPositionFromLayer(layers[i], i, tileSets, sprites, ref tileSprites, ref tilePositions);

            var tileTemplate = ScriptableObject.CreateInstance<PositionTileTemplate>();
            tileTemplate.positions = tilePositions;
            var tileTemplates = new TileTemplate[]
            {
                tileTemplate
            };

            spritesPerTileSet = new[] { tileSprites };
            return tileTemplates;
        }
#endif

        static void GetSpriteAndPositionFromLayer(Layer layer, int layerIndex, List<TileSet> tileSets, Sprite[] sprites, ref List<Sprite> tileSprites, ref List<Vector3Int> tilePositions)
        {
            var tileSet = tileSets.Find(x => x.id == layer.tileSetIndex);
            var tiles = tileSet.tiles;
            
            // We only support single cell tileMaps right now.
            var cell = layer.tileCells[0];
            
            var width = cell.cellRect.width;
            var xPos = cell.cellRect.x / tileSet.tileSize.x;
            var yPos = cell.cellRect.y / tileSet.tileSize.y;
            var tileIndices = cell.tileIndices;

            var spriteIdCache = new GUID[sprites.Length];
            for (var i = 0; i < spriteIdCache.Length; ++i)
                spriteIdCache[i] = sprites[i].GetSpriteID();
                
            for (var i = 0; i < tileIndices.Length; ++i)
            {
                if (tileIndices[i] == 0)
                    continue;

                var tileIndex = tiles.FindIndex(x => x.tileId == tileIndices[i]);
                var spriteId = tiles[tileIndex].spriteId;
                var spriteIndex = spriteIdCache.FindIndex(x => x == spriteId);
                
                tileSprites.Add(sprites[spriteIndex]);
                var tilePos = new Vector3Int(xPos + (i % width), yPos + (i / width), layerIndex);
                tilePos.y *= -1;
                tilePositions.Add(tilePos);
            }
        }

        static TileTemplate[] GetDefaultTileTemplate(int noOfTextures)
        {
            return new TileTemplate[noOfTextures];
        }

        static List<Sprite>[] GetTileSprites(List<TileSet> tileSets, Sprite[] sprites)
        {
            var tileSprites = new List<Sprite>(sprites.Length);
            for (var i = 0; i < tileSets.Count; ++i)
            {
                var tiles = tileSets[i].tiles;
                for (var m = 0; m < tiles.Count; ++m)
                {
                    var tile = tiles[m];
                    var spriteIndex = System.Array.FindIndex(sprites, x => x.GetSpriteID() == tile.spriteId);
                    if (spriteIndex == -1)
                        continue;
                    
                    tileSprites.Add(sprites[spriteIndex]);
                }
            }

            return new []{ tileSprites };
        }
    }
}
