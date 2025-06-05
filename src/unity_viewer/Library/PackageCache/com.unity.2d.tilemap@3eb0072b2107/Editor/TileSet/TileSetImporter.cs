using System.Collections.Generic;
using System.IO;
using UnityEditor.AssetImporters;
using UnityEditor.U2D;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.U2D;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Importer for the TileSet
    /// </summary>
    [HelpURL("tilemaps/tile-palettes/tile-set-properties")]
    [ScriptedImporter(1, "tileset", importQueueOffset: 4, AllowCaching = true)]
    public class TileSetImporter : ScriptedImporter
    {
        [SerializeField]
        internal bool paletteGridFoldout = true;
        [SerializeField]
        internal bool textureSourcesFoldout = true;
        [SerializeField]
        internal bool spriteAtlasSettingsFoldout;

        /// <summary>
        /// This method is called by the Asset pipeline to import files as a TileSet asset.
        /// </summary>
        /// <param name="ctx">
        /// This argument contains all the contextual information needed to process the import event
        /// and is also used by the custom importer to store the resulting Unity Asset.
        /// </param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var loadedObjects = InternalEditorUtility.LoadSerializedFileAndForget(ctx.assetPath);
            TileSet tileSet = default;
            SpriteAtlas spriteAtlas = default;
            foreach (var loadedObject in loadedObjects)
            {
                if (loadedObject is TileSet loadedTileSet)
                    tileSet = loadedTileSet;
                if (loadedObject is SpriteAtlas loadedSpriteAtlas)
                    spriteAtlas = loadedSpriteAtlas;
            }
            if (tileSet == null)
            {
                Debug.LogError("Unable to load TileSet asset");
                return;
            }

            var tileSetName = Path.GetFileNameWithoutExtension(ctx.assetPath);

            // Gather Sprites from Textures
            var textureList = new List<Texture2D>();
            var spritesList = new List<List<Sprite>>();
            var templateList = new List<TileTemplate>();
            foreach (var textureSource in tileSet.textureSources)
            {
                if (textureSource.texture == null)
                    continue;

                var texturePath = AssetDatabase.GetAssetPath(textureSource.texture);
                ctx.DependsOnArtifact(texturePath);
                var textureObjects = AssetDatabase.LoadAllAssetsAtPath(texturePath);

                var textureSprites = new List<Sprite>();
                foreach (var textureObject in textureObjects)
                {
                    var sprite = textureObject as Sprite;
                    if (sprite != null)
                        textureSprites.Add(sprite);
                }
                if (textureSprites.Count > 0)
                {
                    textureList.Add(textureSource.texture);
                    spritesList.Add(textureSprites);
                    TileTemplate tileTemplate = null;
                    if (textureSource.tileTemplate != null)
                    {
                        var templatePath = AssetDatabase.GetAssetPath(textureSource.tileTemplate);
                        ctx.DependsOnArtifact(templatePath);
                        tileTemplate = AssetDatabase.LoadAssetAtPath<TileTemplate>(templatePath);
                    }
                    templateList.Add(tileTemplate);
                }

                // Register textures to SpriteAtlas
                if (tileSet.createAtlas && spriteAtlas != null)
                {
                    spriteAtlas.Remove(new Object[] { textureSource.texture });
                    spriteAtlas.Add(new Object[] { textureSource.texture });
                }
            }

            // Create Palette
            var paletteGO = GridPaletteUtility.CreateNewPaletteAsSubAsset(tileSetName
                , tileSet.cellLayout
                , tileSet.cellSizing
                , tileSet.cellSize
                , tileSet.hexagonLayout == TileSet.HexagonLayout.PointTop
                    ? GridLayout.CellSwizzle.XYZ
                    : GridLayout.CellSwizzle.YXZ
                , tileSet.sortMode
                , tileSet.sortAxis
                , textureList
                , spritesList
                , templateList
                , out var gridPalette
                , out var tileList);

            ctx.AddObjectToAsset(gridPalette.name, gridPalette);
            ctx.AddObjectToAsset(paletteGO.name, paletteGO);
            foreach (var tile in tileList)
            {
                // Use Sprite ID + Tile for pure Tile
                if (tile is Tile t && t.sprite != null)
                {
                    ctx.AddObjectToAsset($"{t.sprite.GetSpriteID()} Tile", tile);
                }
                else
                {
                    ctx.AddObjectToAsset(tile.name, tile);
                }
            }

            // Sprite Atlas
            if (tileSet.createAtlas && spriteAtlas != null)
            {
                spriteAtlas.name = $"{tileSetName} Atlas";
                spriteAtlas.RegisterAndPackAtlas(ctx, this, tileSet.scriptablePacker);
                ctx.AddObjectToAsset(spriteAtlas.name, spriteAtlas);
            }

            ctx.SetMainObject(paletteGO);
        }
    }
}

