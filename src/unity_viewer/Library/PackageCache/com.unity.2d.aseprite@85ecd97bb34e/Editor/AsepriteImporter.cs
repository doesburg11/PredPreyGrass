using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEditor.AssetImporters;
using UnityEditor.U2D.Aseprite.Common;
using UnityEditor.U2D.Sprites;
using UnityEngine.Serialization;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// ScriptedImporter to import Aseprite files
    /// </summary>
    // Version using unity release + 5 digit padding for future upgrade. Eg 2021.2 -> 21200000
    [ScriptedImporter(21300004, new string[] { "aseprite", "ase" }, AllowCaching = true)]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.2d.aseprite@latest")]
    public partial class AsepriteImporter : ScriptedImporter, ISpriteEditorDataProvider
    {
        [SerializeField] int m_ImporterVersion = 0;
        
        [SerializeField]
        TextureImporterSettings m_TextureImporterSettings = new TextureImporterSettings()
        {
            mipmapEnabled = false,
            mipmapFilter = TextureImporterMipFilter.BoxFilter,
            sRGBTexture = true,
            borderMipmap = false,
            mipMapsPreserveCoverage = false,
            alphaTestReferenceValue = 0.5f,
            readable = false,

#if ENABLE_TEXTURE_STREAMING
            streamingMipmaps = false,
            streamingMipmapsPriority = 0,
#endif

            fadeOut = false,
            mipmapFadeDistanceStart = 1,
            mipmapFadeDistanceEnd = 3,

            convertToNormalMap = false,
            heightmapScale = 0.25F,
            normalMapFilter = 0,

            generateCubemap = TextureImporterGenerateCubemap.AutoCubemap,
            cubemapConvolution = 0,

            seamlessCubemap = false,

            npotScale = TextureImporterNPOTScale.ToNearest,

            spriteMode = (int)SpriteImportMode.Multiple,
            spriteExtrude = 1,
            spriteMeshType = SpriteMeshType.Tight,
            spriteAlignment = (int)SpriteAlignment.Center,
            spritePivot = new Vector2(0.5f, 0.5f),
            spritePixelsPerUnit = 100.0f,
            spriteBorder = new Vector4(0.0f, 0.0f, 0.0f, 0.0f),

            alphaSource = TextureImporterAlphaSource.FromInput,
            alphaIsTransparency = true,
            spriteTessellationDetail = -1.0f,

            textureType = TextureImporterType.Sprite,
            textureShape = TextureImporterShape.Texture2D,

            filterMode = FilterMode.Point,
            aniso = 1,
            mipmapBias = 0.0f,
            wrapModeU = TextureWrapMode.Clamp,
            wrapModeV = TextureWrapMode.Clamp,
            wrapModeW = TextureWrapMode.Clamp
        };

        [SerializeField] AsepriteImporterSettings m_PreviousAsepriteImporterSettings;
        [SerializeField]
        AsepriteImporterSettings m_AsepriteImporterSettings = new AsepriteImporterSettings()
        {
            fileImportMode = FileImportModes.AnimatedSprite,
            importHiddenLayers = false,
            layerImportMode = LayerImportModes.MergeFrame,
            defaultPivotAlignment = SpriteAlignment.BottomCenter,
            defaultPivotSpace = PivotSpaces.Canvas,
            customPivotPosition = new Vector2(0.5f, 0.5f),
            mosaicPadding = 4,
            spritePadding = 0,
            generateAnimationClips = true,
            generateModelPrefab = true,
            addSortingGroup = true,
            addShadowCasters = false,
            generateIndividualEvents = true,
            generateSpriteAtlas = true
        };

        // Use for inspector to check if the file node is checked
        [SerializeField]
#pragma warning disable 169, 414
        bool m_ImportFileNodeState = true;

        // Used by platform settings to mark it dirty so that it will trigger a reimport
        [SerializeField]
#pragma warning disable 169, 414
        long m_PlatformSettingsDirtyTick;

        [SerializeField] string m_TextureAssetName = null;

        [SerializeField] List<SpriteMetaData> m_SingleSpriteImportData = new List<SpriteMetaData>(1) { new SpriteMetaData() };
        [FormerlySerializedAs("m_MultiSpriteImportData")]
        [SerializeField] List<SpriteMetaData> m_AnimatedSpriteImportData = new List<SpriteMetaData>();
        [SerializeField] List<SpriteMetaData> m_SpriteSheetImportData = new List<SpriteMetaData>();
        [SerializeField] List<SpriteMetaData> m_TileSetImportData = new List<SpriteMetaData>();

        [SerializeField] List<Layer> m_AsepriteLayers = new List<Layer>();
        [SerializeField] List<TileSet> m_TileSets = new List<TileSet>();

        [SerializeField] List<TextureImporterPlatformSettings> m_PlatformSettings = new List<TextureImporterPlatformSettings>();

        [SerializeField] bool m_GeneratePhysicsShape = false;
        [SerializeField] SecondarySpriteTexture[] m_SecondarySpriteTextures;
        [SerializeField] string m_SpritePackingTag = "";

        SpriteImportMode spriteImportModeToUse => m_TextureImporterSettings.textureType != TextureImporterType.Sprite ?
            SpriteImportMode.None : (SpriteImportMode)m_TextureImporterSettings.spriteMode;

        AsepriteImportData m_ImportData;
        AsepriteFile m_AsepriteFile;
        List<Tag> m_Tags = new List<Tag>();
        List<Frame> m_Frames = new List<Frame>();

        [SerializeField] Vector2Int m_CanvasSize;
        [SerializeField] Vector2 m_PreviousTextureSize;

        GameObject m_RootGameObject;
        readonly Dictionary<int, GameObject> m_LayerIdToGameObject = new Dictionary<int, GameObject>();

        AsepriteImportData importData
        {
            get
            {
                var returnValue = m_ImportData;
                if (returnValue == null)
                {
                    // Using LoadAllAssetsAtPath because AsepriteImportData is hidden
                    var assets = AssetDatabase.LoadAllAssetsAtPath(assetPath);
                    foreach (var asset in assets)
                    {
                        if (asset is AsepriteImportData data)
                            returnValue = data;
                    }
                }


                if (returnValue == null)
                    returnValue = ScriptableObject.CreateInstance<AsepriteImportData>();

                m_ImportData = returnValue;
                return returnValue;
            }
        }

        internal bool isNPOT => Mathf.IsPowerOfTwo(importData.textureActualWidth) && Mathf.IsPowerOfTwo(importData.textureActualHeight);

        internal int textureActualWidth
        {
            get => importData.textureActualWidth;
            private set => importData.textureActualWidth = value;
        }

        internal int textureActualHeight
        {
            get => importData.textureActualHeight;
            private set => importData.textureActualHeight = value;
        }

        float definitionScale
        {
            get
            {
                var definitionScaleW = importData.importedTextureWidth / (float)textureActualWidth;
                var definitionScaleH = importData.importedTextureHeight / (float)textureActualHeight;
                return Mathf.Min(definitionScaleW, definitionScaleH);
            }
        }

        internal SecondarySpriteTexture[] secondaryTextures
        {
            get => m_SecondarySpriteTextures;
            set => m_SecondarySpriteTextures = value;
        }

        /// <inheritdoc />
        public override void OnImportAsset(AssetImportContext ctx)
        {
            if (m_ImportData == null)
                m_ImportData = ScriptableObject.CreateInstance<AsepriteImportData>();
            m_ImportData.hideFlags = HideFlags.HideInHierarchy;

            try
            {
                var isSuccessful = ParseAsepriteFile(ctx.assetPath);
                if (!isSuccessful)
                    return;

                UpdateImporterDataToNewVersion();
                
                var layersFromFile = FetchLayersFromFile(asepriteFile, m_CanvasSize, includeHiddenLayers, layerImportMode == LayerImportModes.MergeFrame);

                FetchImageDataFromLayers(layersFromFile, out var imageBuffers, out var imageSizes);
                FetchImageDataFromTilemaps(imageBuffers, imageSizes);

                var mosaicPad = m_AsepriteImporterSettings.mosaicPadding;
                var spritePad = m_AsepriteImporterSettings.fileImportMode == FileImportModes.AnimatedSprite ? m_AsepriteImporterSettings.spritePadding : 0;
                var requireSquarePotTexture = IsRequiringSquarePotTexture(ctx);
                ImagePacker.Pack(imageBuffers.ToArray(), imageSizes.ToArray(), (int)mosaicPad, spritePad, requireSquarePotTexture, out var outputImageBuffer, out var packedTextureWidth, out var packedTextureHeight, out var spriteRects, out var uvTransforms);

                var packOffsets = new Vector2Int[spriteRects.Length];
                for (var i = 0; i < packOffsets.Length; ++i)
                {
                    packOffsets[i] = new Vector2Int(uvTransforms[i].x - spriteRects[i].position.x, uvTransforms[i].y - spriteRects[i].position.y);
                    packOffsets[i] *= -1;
                }

                SpriteMetaData[] spriteImportData;
                if (m_AsepriteImporterSettings.fileImportMode == FileImportModes.SpriteSheet)
                    spriteImportData = GetSpriteImportData().ToArray();
                else
                {
                    CellTasks.GetCellsFromLayers(m_AsepriteLayers, out var cells);

                    var newSpriteMeta = new List<SpriteMetaData>(cells.Count);

                    // Create SpriteMetaData for each cell
                    var importedRectsHaveChanged = false;
                    for (var i = 0; i < cells.Count; ++i)
                    {
                        var cell = cells[i];
                        var dataIndex = newSpriteMeta.Count;
                        var spriteData = CreateNewSpriteMetaData(
                            cell.name,
                            cell.spriteId,
                            cell.cellRect.position,
                            spriteRects[dataIndex],
                            packOffsets[dataIndex],
                            uvTransforms[dataIndex],
                            new int2(m_CanvasSize.x, m_CanvasSize.y));
                        newSpriteMeta.Add(spriteData);

                        if (cell.updatedCellRect)
                            importedRectsHaveChanged = true;
                    }

                    // Create SpriteMetaData for each tile
                    for (var i = 0; i < m_TileSets.Count; ++i)
                    {
                        var tiles = m_TileSets[i].tiles;
                        for (var m = 0; m < tiles.Count; ++m)
                        {
                            var tile = tiles[m];
                            var dataIndex = newSpriteMeta.Count;
                            var spriteData = CreateNewSpriteMetaData(
                                tile.name,
                                tile.spriteId,
                                Vector2Int.one,
                                spriteRects[dataIndex],
                                packOffsets[dataIndex],
                                uvTransforms[dataIndex],
                                tile.size);
                            newSpriteMeta.Add(spriteData);
                        }
                    }

                    // If the packing texture size has changed, the rects will have a new position which must be taken into account.
                    // OR
                    // The import mode is set to Tile Set. 
                    if (Math.Abs(m_PreviousTextureSize.x - packedTextureWidth) > Mathf.Epsilon 
                        || Math.Abs(m_PreviousTextureSize.y - packedTextureHeight) > Mathf.Epsilon
                        || importMode == FileImportModes.TileSet)
                    {
                        importedRectsHaveChanged = true;
                    }
                    
                    spriteImportData = UpdateSpriteImportData(newSpriteMeta, spriteRects, uvTransforms, importedRectsHaveChanged);
                }

                // These two values should be set before we generate the Texture (and the Sprites)
                // as the values are used in OnPostprocessSprites by 2D Animation.
                textureActualHeight = packedTextureHeight;
                textureActualWidth = packedTextureWidth;
                m_PreviousTextureSize = new Vector2(textureActualWidth, textureActualHeight);
                
                var output = TextureGeneration.Generate(
                    ctx,
                    outputImageBuffer,
                    packedTextureWidth,
                    packedTextureHeight,
                    spriteImportData,
                    in m_PlatformSettings,
                    in m_TextureImporterSettings,
                    m_SpritePackingTag,
                    in m_SecondarySpriteTextures);

                if (output.texture)
                {
                    importData.importedTextureHeight = output.texture.height;
                    importData.importedTextureWidth = output.texture.width;
                }
                else
                {
                    importData.importedTextureHeight = textureActualHeight;
                    importData.importedTextureWidth = textureActualWidth;
                }

                if (output.texture != null && output.sprites != null)
                    SetPhysicsOutline(GetDataProvider<ISpritePhysicsOutlineDataProvider>(), output.sprites, definitionScale, pixelsPerUnit, m_GeneratePhysicsShape);

                RegisterAssets(ctx, output);
                OnPostAsepriteImport?.Invoke(new ImportEventArgs(this, ctx));

                outputImageBuffer.DisposeIfCreated();
                foreach (var cellBuffer in imageBuffers)
                    cellBuffer.DisposeIfCreated();
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to import file {assetPath}. Error: {e.Message} \n{e.StackTrace}");
            }
            finally
            {
                m_PreviousAsepriteImporterSettings = m_AsepriteImporterSettings;
                EditorUtility.SetDirty(this);
                m_AsepriteFile?.Dispose();
            }
        }

        void UpdateImporterDataToNewVersion()
        {
            // Upgrade from version 0 -> 1
            // This upgrade updates all layer GUIDs.
            // In the past, we made use of layer name & layer index, which caused issues when new layers were added.
            // The new GUID uses the name of the layer and its parents name. 
            if (m_ImporterVersion == 0)
            {
                foreach (var layer in m_AsepriteLayers)
                    layer.guid = Layer.GenerateGuid(layer, m_AsepriteLayers);
                m_ImporterVersion++;
            }
            // Upgrade from version 1 -> 2
            // This upgrade populates the UUID field for each layer.
            // The UUID is a unique identifier for each layer, which is also used by Aseprite (if the setting has been enabled in Aseprite).
            if (m_ImporterVersion == 1)
            {
                foreach (var layer in m_AsepriteLayers)
                    layer.uuid = new UUID((uint)layer.guid, 0, 0, 0);
                m_ImporterVersion++;
            }
        }

        bool ParseAsepriteFile(string path)
        {
            m_AsepriteFile = AsepriteReader.ReadFile(path);
            if (m_AsepriteFile == null)
                return false;

            m_CanvasSize = new Vector2Int(m_AsepriteFile.width, m_AsepriteFile.height);

            m_Frames = ExtractFrameData(in m_AsepriteFile);
            m_Tags = ExtractTagsData(in m_AsepriteFile);
            var newTileData = ExtractTileData(in m_AsepriteFile);
            m_TileSets = UpdateTileData(newTileData, m_TileSets);

            return true;
        }

        static List<TileSet> UpdateTileData(List<TileSet> newTileSet, List<TileSet> oldTileSet)
        {
            if (oldTileSet.Count == 0)
                return new List<TileSet>(newTileSet);

            var finalSets = new List<TileSet>(oldTileSet);

            // Remove old tile sets
            for (var i = 0; i < oldTileSet.Count; ++i)
            {
                var oldSet = oldTileSet[i];
                if (newTileSet.FindIndex(x => x.guid == oldSet.guid) == -1)
                    finalSets.Remove(oldSet);
            }

            // Add new tile sets
            for (var i = 0; i < newTileSet.Count; ++i)
            {
                var newLayer = newTileSet[i];
                var setIndex = finalSets.FindIndex(x => x.guid == newLayer.guid);
                if (setIndex == -1)
                    finalSets.Add(newLayer);
            }

            // Update tile set data
            for (var i = 0; i < finalSets.Count; ++i)
            {
                var finalSet = finalSets[i];
                var setIndex = newTileSet.FindIndex(x => x.guid == finalSet.guid);
                if (setIndex != -1)
                {
                    var oldTiles = finalSet.tiles;
                    var newTiles = newTileSet[setIndex].tiles;
                    for (var m = 0; m < newTiles.Count; ++m)
                    {
                        if (m < oldTiles.Count)
                        {
                            var oldTile = oldTiles[m];
                            var newTile = newTiles[m];
                            newTile.spriteId = oldTile.spriteId;
                            newTiles[m] = newTile;
                        }
                    }
                    finalSet.tiles = new List<Tile>(newTiles);
                    finalSet.name = newTileSet[setIndex].name;
                    finalSet.tileSize = newTileSet[setIndex].tileSize;
                }
            }

            return finalSets;
        }

        static List<Layer> FetchLayersFromFile(in AsepriteFile asepriteFile, Vector2Int canvasSize, bool includeHiddenLayers, bool isMerged)
        {
            var newLayers = RestructureLayerAndCellData(in asepriteFile, canvasSize);
            FilterOutLayers(newLayers, includeHiddenLayers);
            UpdateCellNames(newLayers, isMerged);
            return newLayers;
        }

        static List<Layer> RestructureLayerAndCellData(in AsepriteFile file, Vector2Int canvasSize)
        {
            var frameData = file.frameData;

            var nameGenerator = new UniqueNameGenerator();
            var layers = new List<Layer>();
            var parentTable = new Dictionary<int, Layer>();
            
            for (var i = 0; i < frameData.Count; ++i)
            {
                var chunks = frameData[i].chunks;
                for (var m = 0; m < chunks.Count; ++m)
                {
                    if (chunks[m].chunkType == ChunkTypes.Layer)
                    {
                        var layerChunk = chunks[m] as LayerChunk;

                        var layer = new Layer();
                        var childLevel = layerChunk.childLevel;
                        parentTable[childLevel] = layer;

                        layer.parentIndex = childLevel == 0 ? -1 : parentTable[childLevel - 1].index;

                        layer.name = nameGenerator.GetUniqueName(layerChunk.name, layer.parentIndex);
                        layer.layerFlags = layerChunk.flags;
                        layer.layerType = layerChunk.layerType;
                        layer.blendMode = layerChunk.blendMode;
                        layer.opacity = layerChunk.opacity / 255f;
                        layer.tileSetIndex = layerChunk.tileSetIndex;
                        layer.index = layers.Count;
                        layer.uuid = layerChunk.uuid;
                        if (layer.uuid == UUID.zero)
                        {
                            var guid = (uint)Layer.GenerateGuid(layer, layers);
                            layer.uuid = new UUID(guid, 0, 0, 0);
                        }

                        layers.Add(layer);
                    }
                }
            }

            for (var i = 0; i < frameData.Count; ++i)
            {
                var chunks = frameData[i].chunks;
                for (var m = 0; m < chunks.Count; ++m)
                {
                    if (chunks[m].chunkType == ChunkTypes.Cell)
                    {
                        var cellChunk = chunks[m] as CellChunk;
                        var layer = layers.Find(x => x.index == cellChunk.layerIndex);
                        if (layer == null)
                        {
                            Debug.LogWarning($"Could not find the layer for one of the cells. Frame Index={i}, Chunk Index={m}.");
                            continue;
                        }

                        var cellType = cellChunk.cellType;
                        if (cellType == CellTypes.LinkedCell)
                        {
                            var cell = new LinkedCell
                            {
                                frameIndex = i,
                                linkedToFrame = cellChunk.linkedToFrame
                            };
                            layer.linkedCells.Add(cell);
                        }
                        else if (cellType == CellTypes.CompressedTileMap)
                        {
                            var tileCell = new TileCell
                            {
                                layerIndex = cellChunk.layerIndex,
                                frameIndex = i,
                                cellRect = new RectInt(cellChunk.posX, cellChunk.posY, cellChunk.width, cellChunk.height)
                            };

                            var chunkTileIndices = cellChunk.tileIndices;
                            var tileIndices = new uint[chunkTileIndices.Length];
                            for (var n = 0; n < tileIndices.Length; ++n)
                                tileIndices[n] = chunkTileIndices[n];
                            tileCell.tileIndices = tileIndices;
                            layer.tileCells.Add(tileCell);
                        }
                        else
                        {
                            var cell = new Cell
                            {
                                frameIndex = i,
                                updatedCellRect = false
                            };

                            // Flip Y. Aseprite 0,0 is at Top Left. Unity 0,0 is at Bottom Left.
                            var cellY = (canvasSize.y - cellChunk.posY) - cellChunk.height;
                            cell.cellRect = new RectInt(cellChunk.posX, cellY, cellChunk.width, cellChunk.height);
                            cell.opacity = cellChunk.opacity / 255f;
                            cell.blendMode = layer.blendMode;
                            cell.image = cellChunk.image;
                            cell.additiveSortOrder = cellChunk.zIndex;
                            cell.name = layer.name;
                            cell.spriteId = GUID.Generate();

                            var opacity = cell.opacity * layer.opacity;
                            if ((1f - opacity) > Mathf.Epsilon)
                                TextureTasks.AddOpacity(ref cell.image, opacity);

                            layer.cells.Add(cell);
                        }
                    }
                }
            }

            return layers;
        }

        static void FilterOutLayers(List<Layer> layers, bool includeHiddenLayers)
        {
            for (var i = layers.Count - 1; i >= 0; --i)
            {
                var layer = layers[i];
                if (!includeHiddenLayers && !ImportUtilities.IsLayerVisible(layer.index, in layers))
                {
                    DisposeCellsInLayer(layer);
                    layers.RemoveAt(i);
                    continue;
                }

                var cells = layer.cells;
                for (var m = cells.Count - 1; m >= 0; --m)
                {
                    var width = cells[m].cellRect.width;
                    var height = cells[m].cellRect.width;
                    if (width == 0 || height == 0)
                        cells.RemoveAt(m);
                    else if (cells[m].image == default || !cells[m].image.IsCreated)
                        cells.RemoveAt(m);
                    else if (ImportUtilities.IsEmptyImage(cells[m].image))
                        cells.RemoveAt(m);
                }
            }
        }

        static void DisposeCellsInLayer(Layer layer)
        {
            foreach (var cell in layer.cells)
            {
                var image = cell.image;
                image.DisposeIfCreated();
            }
        }

        static void UpdateCellNames(List<Layer> layers, bool isMerged)
        {
            for (var i = 0; i < layers.Count; ++i)
            {
                var cells = layers[i].cells;
                for (var m = 0; m < cells.Count; ++m)
                {
                    var cell = cells[m];
                    cell.name = ImportUtilities.GetCellName(cell.name, cell.frameIndex, cells.Count, isMerged);
                    cells[m] = cell;
                }
            }
        }

        void FetchImageDataFromLayers(List<Layer> newLayers, out List<NativeArray<Color32>> imageBuffers, out List<int2> imageSizes)
        {
            // TileSet will always be in individual layers, since each layer can refer to a different tileSet.
            if (layerImportMode == LayerImportModes.IndividualLayers || importMode == FileImportModes.TileSet)
            {
                m_AsepriteLayers = UpdateLayers(newLayers, m_AsepriteLayers, true);

                CellTasks.GetCellsFromLayers(m_AsepriteLayers, out var cells);
                CellTasks.CollectDataFromCells(cells, out imageBuffers, out imageSizes);
                CellTasks.FlipCellBuffers(imageBuffers, imageSizes);
            }
            else
            {
                var assetName = System.IO.Path.GetFileNameWithoutExtension(assetPath);
                ImportMergedLayers.Import(assetName, newLayers, out imageBuffers, out imageSizes);

                // Update layers after merged, since merged import creates new layers.
                // The new layers should be compared and merged together with the ones existing in the meta file.
                m_AsepriteLayers = UpdateLayers(newLayers, m_AsepriteLayers, false);
            }
        }

        static List<Layer> UpdateLayers(List<Layer> newLayers, List<Layer> oldLayers, bool isIndividual)
        {
            if (oldLayers.Count == 0)
                return new List<Layer>(newLayers);

            var finalLayers = new List<Layer>(oldLayers);

            // Remove old layers & Add new layers if: 
            // - We are using Individual layer import mode
            // OR
            // - There are more than one old layer. This path is for when going from Individual mode to Merged mode.
            if (isIndividual || oldLayers.Count > 1)
            {
                // Remove old layers
                foreach (var oldLayer in oldLayers)
                {
                    if (newLayers.FindIndex(x => x.uuid == oldLayer.uuid) == -1)
                        finalLayers.Remove(oldLayer);
                }

                // Add new layers
                foreach (var newLayer in newLayers)
                {
                    var layerIndex = finalLayers.FindIndex(x => x.uuid == newLayer.uuid);
                    if (layerIndex == -1)
                        finalLayers.Add(newLayer);
                }
            }

            // Update layer data
            foreach (var finalLayer in finalLayers)
            {
                var layerIndex = isIndividual ? newLayers.FindIndex(x => x.uuid == finalLayer.uuid) : 0;
                if (layerIndex != -1)
                {
                    var oldCells = finalLayer.cells;
                    var newCells = newLayers[layerIndex].cells;
                    for (var m = 0; m < newCells.Count; ++m)
                    {
                        if (m < oldCells.Count)
                        {
                            var oldCell = oldCells[m];
                            var newCell = newCells[m];
                            newCell.spriteId = oldCell.spriteId;
                            newCell.updatedCellRect = newCell.cellRect != oldCell.cellRect;
                            newCells[m] = newCell;
                        }
                    }
                    finalLayer.cells = new List<Cell>(newCells);
                    finalLayer.linkedCells = new List<LinkedCell>(newLayers[layerIndex].linkedCells);
                    finalLayer.tileCells = new List<TileCell>(newLayers[layerIndex].tileCells);
                    finalLayer.tileSetIndex = newLayers[layerIndex].tileSetIndex;
                    finalLayer.index = newLayers[layerIndex].index;
                    finalLayer.opacity = newLayers[layerIndex].opacity;
                    finalLayer.parentIndex = newLayers[layerIndex].parentIndex;
                    finalLayer.layerType = newLayers[layerIndex].layerType;
                }
            }

            return finalLayers;
        }

        void FetchImageDataFromTilemaps(List<NativeArray<Color32>> imageBuffers, List<int2> imageSizes)
        {
            var images = new List<NativeArray<Color32>>();
            var sizes = new List<int2>();
            for (var i = 0; i < m_TileSets.Count; ++i)
            {
                var tiles = m_TileSets[i].tiles;
                for (var m = 0; m < tiles.Count; ++m)
                {
                    images.Add(tiles[m].image);
                    sizes.Add(tiles[m].size);
                }
            }

            CellTasks.FlipCellBuffers(images, sizes);
            imageBuffers.AddRange(images);
            imageSizes.AddRange(sizes);
        }

        bool IsRequiringSquarePotTexture(AssetImportContext ctx)
        {
#pragma warning disable CS0618
            var platformSettings = PlatformSettingsUtilities.GetPlatformTextureSettings(ctx.selectedBuildTarget, in m_PlatformSettings);
            return platformSettings.format is >= TextureImporterFormat.PVRTC_RGB2 and <= TextureImporterFormat.PVRTC_RGBA4; 
#pragma warning restore CS0618
        }

        static List<Frame> ExtractFrameData(in AsepriteFile file)
        {
            var noOfFrames = file.noOfFrames;
            var frames = new List<Frame>(noOfFrames);
            for (var i = 0; i < noOfFrames; ++i)
            {
                var frameData = file.frameData[i];
                var eventData = ExtractEventDataFromCells(frameData);

                var frame = new Frame()
                {
                    duration = frameData.frameDuration,
                    eventData = eventData
                };
                frames.Add(frame);
            }

            return frames;
        }

        static (string, object)[] ExtractEventDataFromCells(FrameData frameData)
        {
            var chunks = frameData.chunks;
            var eventData = new HashSet<(string, object)>();
            for (var i = 0; i < chunks.Count; ++i)
            {
                if (chunks[i].chunkType != ChunkTypes.Cell)
                    continue;
                var cellChunk = (CellChunk)chunks[i];
                if (cellChunk.dataChunk == null)
                    continue;
                var dataText = cellChunk.dataChunk.text;
                if (string.IsNullOrEmpty(dataText) || !dataText.StartsWith("event:"))
                    continue;
                var eventString = dataText.Remove(0, "event:".Length);
                var eventParts = eventString.Split(',');
                if (eventParts.Length == 0)
                    continue;

                for(var m = 0; m < eventParts.Length; ++m)
                    eventParts[m] = eventParts[m].Trim(' ');
                
                if (eventParts.Length == 1)
                    eventData.Add((eventParts[0], null));
                else
                {
                    var eventParam = eventParts[1];
                    if (int.TryParse(eventParam, out var intParam))
                        eventData.Add((eventParts[0], intParam));
                    else if (float.TryParse(eventParam, out var floatParam))
                        eventData.Add((eventParts[0], floatParam));
                    else if (eventParam.StartsWith("\"") && eventParam.EndsWith("\""))
                    {
                        eventParam = eventParam.Trim('"');
                        eventData.Add((eventParts[0], eventParam));   
                    }
                    else 
                        eventData.Add((eventParts[0], null));
                }
            }

            var eventArr = new (string, object)[eventData.Count];
            eventData.CopyTo(eventArr);
            return eventArr;
        }

        static List<Tag> ExtractTagsData(in AsepriteFile file)
        {
            var tags = new List<Tag>();

            var noOfFrames = file.noOfFrames;
            for (var i = 0; i < noOfFrames; ++i)
            {
                var frame = file.frameData[i];
                var noOfChunks = frame.chunkCount;
                for (var m = 0; m < noOfChunks; ++m)
                {
                    var chunk = frame.chunks[m];
                    if (chunk.chunkType != ChunkTypes.Tags)
                        continue;

                    var tagChunk = chunk as TagsChunk;
                    var noOfTags = tagChunk.noOfTags;
                    for (var n = 0; n < noOfTags; ++n)
                    {
                        var data = tagChunk.tagData[n];
                        var tag = new Tag();
                        tag.name = data.name;
                        tag.noOfRepeats = data.noOfRepeats;
                        tag.fromFrame = data.fromFrame;
                        // Adding one more frame as Aseprite's tags seems to always be 1 short.
                        tag.toFrame = data.toFrame + 1;

                        tags.Add(tag);
                    }
                }
            }

            return tags;
        }

        static List<TileSet> ExtractTileData(in AsepriteFile file)
        {
            var tilemaps = new List<TileSet>();
            var tileSetNames = new HashSet<string>();
            
            var noOfFrames = file.noOfFrames;
            for (var i = 0; i < noOfFrames; ++i)
            {
                var frame = file.frameData[i];
                var noOfChunks = frame.chunkCount;
                for (var m = 0; m < noOfChunks; ++m)
                {
                    var chunk = frame.chunks[m];
                    if (chunk.chunkType != ChunkTypes.Tileset)
                        continue;

                    var tileSetChunk = chunk as TilesetChunk;
                    var noOfTiles = tileSetChunk.noOfTiles;
                    var tileWidth = tileSetChunk.width;
                    var tileHeight = tileSetChunk.height;
                    var tileSize = new int2(tileWidth, tileHeight);
                    var tileSetName = tileSetChunk.tileSetName;
                    var tileSetId = tileSetChunk.tileSetId;
                    
                    // Ensure the tileSets have unique names
                    if (tileSetNames.Contains(tileSetName))
                        tileSetName = $"{tileSetName}_{tileSetId}";
                    tileSetNames.Add(tileSetName);

                    var tiles = new List<Tile>();
                    for (uint j = 0; j < noOfTiles; ++j)
                    {
                        var tileImage = tileSetChunk.tileImages[j];
                        if (ImportUtilities.IsEmptyImage(in tileImage))
                            continue;

                        var tile = new Tile()
                        {
                            tileId = j,
                            image = tileImage,
                            size = tileSize,
                            spriteId = GUID.Generate(),
                            name = $"{tileSetName}_{j}"
                        };
                        tiles.Add(tile);
                    }

                    var tilemap = new TileSet()
                    {
                        id = tileSetId,
                        name = tileSetName,
                        tileSize = tileSize,
                        tiles = tiles,
                    };
                    tilemap.guid = TileSet.GenerateGuid(tilemap);
                    tilemaps.Add(tilemap);
                }
            }

            return tilemaps;
        }

        SpriteMetaData[] UpdateSpriteImportData(List<SpriteMetaData> newSpriteMeta, RectInt[] newSpriteRects, Vector2Int[] uvTransforms, bool importedRectsHaveChanged)
        {
            var finalSpriteMeta = GetSpriteImportData();
            if (finalSpriteMeta.Count <= 0)
            {
                finalSpriteMeta.Clear();
                finalSpriteMeta.AddRange(newSpriteMeta);
            }
            else
            {
                // Remove old SpriteMeta.
                for (var i = finalSpriteMeta.Count - 1; i >= 0; --i)
                {
                    var spriteData = finalSpriteMeta[i];
                    if (newSpriteMeta.FindIndex(x => x.spriteID == spriteData.spriteID) == -1)
                    {
                        finalSpriteMeta.Remove(spriteData);
                        
                        // The spriteRect, UV, etc. are all based on the latest order of cells. When removing an item from the finalSpriteMeta, there is a chance
                        // that the order is no longer the same. We therefore need to update all spriteRects, UVs and pivots to match the new order.
                        importedRectsHaveChanged = true;
                    }
                }

                // Add new SpriteMeta.
                for (var i = 0; i < newSpriteMeta.Count; ++i)
                {
                    var newMeta = newSpriteMeta[i];
                    if (finalSpriteMeta.FindIndex(x => x.spriteID == newMeta.spriteID) == -1)
                        finalSpriteMeta.Add(newMeta);
                }

                // Update with new pack data
                for (var i = 0; i < newSpriteMeta.Count; ++i)
                {
                    var newMeta = newSpriteMeta[i];
                    var finalMeta = finalSpriteMeta.Find(x => x.spriteID == newMeta.spriteID);
                    
                    // Override previous pivot and sprite rect if:
                    // - Importer settings have been updated
                    // - OR
                    // - The cell's size has changed in DCC
                    // - OR
                    // - The packing texture's size has changed
                    // - OR
                    // - A layer has been removed / renamed
                    // - OR
                    // - The import mode is set to TileSet
                    if (finalMeta != null && (AreSettingsUpdated() || importedRectsHaveChanged))
                    {
                        finalMeta.alignment = newMeta.alignment;
                        finalMeta.pivot = newMeta.pivot;
                        finalMeta.rect = new Rect(newSpriteRects[i].x, newSpriteRects[i].y, newSpriteRects[i].width, newSpriteRects[i].height);
                        finalMeta.uvTransform = uvTransforms[i];
                    }
                }
            }

            return finalSpriteMeta.ToArray();
        }

        bool AreSettingsUpdated()
        {
            return !m_PreviousAsepriteImporterSettings.IsDefault() &&
                   (pivotAlignment != m_PreviousAsepriteImporterSettings.defaultPivotAlignment ||
                    pivotSpace != m_PreviousAsepriteImporterSettings.defaultPivotSpace ||
                    customPivotPosition != m_PreviousAsepriteImporterSettings.customPivotPosition ||
                    spritePadding != m_PreviousAsepriteImporterSettings.spritePadding);
        }

        SpriteMetaData CreateNewSpriteMetaData(
            string spriteName,
            GUID spriteID,
            Vector2Int position,
            RectInt spriteRect,
            Vector2Int packOffset,
            Vector2Int uvTransform,
            int2 canvasSize)
        {
            var spriteData = new SpriteMetaData();
            spriteData.border = Vector4.zero;

            if (importMode == FileImportModes.TileSet)
            {
                // Get the position of the sprite in the tile
                var spritePosition = (Vector2)uvTransform;
                // Remove the mosaic padding
                spritePosition.x -= spriteRect.x;
                spritePosition.y -= spriteRect.y;
                // Calculate how many times the sprite can go into the tile
                var scaleX = canvasSize.x / (float)spriteRect.width;
                var scaleY = canvasSize.y / (float)spriteRect.height;
                
                var pivot = new float2(spritePosition.x / (float)canvasSize.x, spritePosition.y / (float)canvasSize.y);
                
                // Sprites are stored at the center of a Tile asset, so add center alignment to the pivot.
                var alignmentPos = ImportUtilities.PivotAlignmentToVector(SpriteAlignment.Center);
                pivot.x += alignmentPos.x;
                pivot.y += alignmentPos.y;

                pivot.x *= scaleX;
                pivot.y *= scaleY;
                
                spriteData.alignment = SpriteAlignment.Custom;
                spriteData.pivot = pivot;
            }
            else if (pivotSpace == PivotSpaces.Canvas)
            {
                spriteData.alignment = SpriteAlignment.Custom;

                var cellRect = new RectInt(position.x, position.y, spriteRect.width, spriteRect.height);
                cellRect.x += packOffset.x;
                cellRect.y += packOffset.y;

                spriteData.pivot = ImportUtilities.CalculateCellPivot(cellRect, spritePadding, canvasSize, pivotAlignment, customPivotPosition);
            }
            else
            {
                spriteData.alignment = pivotAlignment;
                spriteData.pivot = customPivotPosition;
            }

            spriteData.rect = new Rect(spriteRect.x, spriteRect.y, spriteRect.width, spriteRect.height);
            spriteData.spriteID = spriteID;
            spriteData.name = spriteName;
            spriteData.uvTransform = uvTransform;
            return spriteData;
        }

        static void SetPhysicsOutline(ISpritePhysicsOutlineDataProvider physicsOutlineDataProvider, Sprite[] sprites, float definitionScale, float pixelsPerUnit, bool generatePhysicsShape)
        {
            foreach (var sprite in sprites)
            {
                var guid = sprite.GetSpriteID();
                var outline = physicsOutlineDataProvider.GetOutlines(guid);

                var generated = false;
                if ((outline == null || outline.Count == 0) && generatePhysicsShape)
                {
                    InternalEditorBridge.GenerateOutlineFromSprite(sprite, 0.25f, 200, true, out var defaultOutline);
                    outline = new List<Vector2[]>(defaultOutline.Length);
                    for (var i = 0; i < defaultOutline.Length; ++i)
                    {
                        outline.Add(defaultOutline[i]);
                    }

                    generated = true;
                }
                if (outline != null && outline.Count > 0)
                {
                    // Ensure that outlines are all valid.
                    var validOutlineCount = 0;
                    for (var i = 0; i < outline.Count; ++i)
                        validOutlineCount += ((outline[i].Length > 2) ? 1 : 0);

                    var index = 0;
                    var convertedOutline = new Vector2[validOutlineCount][];
                    var useScale = generated ? pixelsPerUnit * definitionScale : definitionScale;

                    var outlineOffset = Vector2.zero;
                    outlineOffset.x = sprite.rect.width * 0.5f;
                    outlineOffset.y = sprite.rect.height * 0.5f;

                    for (var i = 0; i < outline.Count; ++i)
                    {
                        if (outline[i].Length > 2)
                        {
                            convertedOutline[index] = new Vector2[outline[i].Length];
                            for (var j = 0; j < outline[i].Length; ++j)
                                convertedOutline[index][j] = outline[i][j] * useScale + outlineOffset;
                            index++;
                        }
                    }
                    sprite.OverridePhysicsShape(convertedOutline);
                }
            }
        }

        void RegisterAssets(AssetImportContext ctx, TextureGenerationOutput output)
        {
            if ((output.sprites == null || output.sprites.Length == 0) && output.texture == null)
            {
                Debug.LogWarning(TextContent.noSpriteOrTextureImportWarning, this);
                return;
            }

            var assetNameGenerator = new UniqueNameGenerator();
            if (!string.IsNullOrEmpty(output.importInspectorWarnings))
            {
                Debug.LogWarning(output.importInspectorWarnings);
            }
            if (output.importWarnings != null && output.importWarnings.Length != 0)
            {
                foreach (var warning in output.importWarnings)
                    Debug.LogWarning(warning);
            }
            if (output.thumbNail == null)
                Debug.LogWarning("Thumbnail generation fail");
            if (output.texture == null)
            {
                throw new Exception("Texture import fail");
            }

            var assetName = assetNameGenerator.GetUniqueName(System.IO.Path.GetFileNameWithoutExtension(ctx.assetPath), -1, true, this);
            UnityEngine.Object mainAsset = null;

            RegisterTextureAsset(ctx, output, assetName, ref mainAsset);
            RegisterSprites(ctx, output, assetNameGenerator);

            if (m_AsepriteImporterSettings.fileImportMode == FileImportModes.AnimatedSprite)
            {
                RegisterGameObjects(ctx, output, ref mainAsset);
                RegisterAnimationClip(ctx, assetName, output);
                RegisterAnimatorController(ctx, assetName);
            }
            else if (m_AsepriteImporterSettings.fileImportMode == FileImportModes.TileSet)
            {
                RegisterTilePalette(ctx, output, ref mainAsset);
                // RegisterSpriteAtlas(ctx, assetName);
            }

            ctx.AddObjectToAsset("AsepriteImportData", m_ImportData);
            ctx.SetMainObject(mainAsset);
        }

        void RegisterTextureAsset(AssetImportContext ctx, TextureGenerationOutput output, string assetName, ref UnityEngine.Object mainAsset)
        {
            var registerTextureNameId = string.IsNullOrEmpty(m_TextureAssetName) ? "Texture" : m_TextureAssetName;

            output.texture.name = assetName;
            ctx.AddObjectToAsset(registerTextureNameId, output.texture, output.thumbNail);
            mainAsset = output.texture;
        }

        static void RegisterSprites(AssetImportContext ctx, TextureGenerationOutput output, UniqueNameGenerator assetNameGenerator)
        {
            if (output.sprites == null)
                return;

            foreach (var sprite in output.sprites)
            {
                var spriteGuid = sprite.GetSpriteID().ToString();
                var spriteAssetName = assetNameGenerator.GetUniqueName(spriteGuid, -1, false, sprite);
                ctx.AddObjectToAsset(spriteAssetName, sprite);
            }
        }

        void RegisterGameObjects(AssetImportContext ctx, TextureGenerationOutput output, ref UnityEngine.Object mainAsset)
        {
            if (output.sprites.Length == 0)
                return;
            if (m_AsepriteImporterSettings.fileImportMode != FileImportModes.AnimatedSprite)
                return;

            PrefabGeneration.Generate(
                ctx,
                output,
                m_AsepriteLayers,
                m_LayerIdToGameObject,
                m_CanvasSize,
                m_AsepriteImporterSettings,
                ref mainAsset,
                out m_RootGameObject);
        }

        void RegisterAnimationClip(AssetImportContext ctx, string assetName, TextureGenerationOutput output)
        {
            if (output.sprites.Length == 0)
                return;
            if (m_AsepriteImporterSettings.fileImportMode != FileImportModes.AnimatedSprite)
                return;
            if (!generateAnimationClips)
                return;
            var noOfFrames = m_AsepriteFile.noOfFrames;
            if (noOfFrames == 1)
                return;

            var sprites = output.sprites;
            var clips = AnimationClipGeneration.Generate(
                assetName,
                sprites,
                m_AsepriteFile,
                m_AsepriteLayers,
                m_Frames,
                m_Tags,
                m_LayerIdToGameObject,
                m_AsepriteImporterSettings.generateIndividualEvents);

            for (var i = 0; i < clips.Length; ++i)
                ctx.AddObjectToAsset(clips[i].name, clips[i]);
        }

        void RegisterAnimatorController(AssetImportContext ctx, string assetName)
        {
            if (m_AsepriteImporterSettings.fileImportMode != FileImportModes.AnimatedSprite)
                return;

            AnimatorControllerGeneration.Generate(ctx, assetName, m_RootGameObject, generateModelPrefab);
        }

        void RegisterTilePalette(AssetImportContext ctx, TextureGenerationOutput output, ref UnityEngine.Object mainAsset)
        {
            if (m_TileSets.Count == 0 || output.sprites.Length == 0)
                return;

            var sprites = output.sprites;
            TilePaletteGeneration.Generate(
                ctx,
                m_AsepriteLayers,
                m_TileSets,
                sprites,
                pixelsPerUnit,
                ref mainAsset);
        }

        void RegisterSpriteAtlas(AssetImportContext ctx, string assetName)
        {
            if (!m_AsepriteImporterSettings.generateSpriteAtlas)
                return;
            
            SpriteAtlasGeneration.Generate(ctx, this, assetName);
        }

        internal void Apply()
        {
            // Do this so that asset change save dialog will not show
            var originalValue = EditorPrefs.GetBool("VerifySavingAssets", false);
            EditorPrefs.SetBool("VerifySavingAssets", false);
            AssetDatabase.ForceReserializeAssets(new string[] { assetPath }, ForceReserializeAssetsOptions.ReserializeMetadata);
            EditorPrefs.SetBool("VerifySavingAssets", originalValue);
        }

        /// <inheritdoc />
        public override bool SupportsRemappedAssetType(Type type)
        {
            if (type == typeof(AnimationClip))
                return true;
            return base.SupportsRemappedAssetType(type);
        }

        void SetPlatformTextureSettings(TextureImporterPlatformSettings platformSettings)
        {
            var index = m_PlatformSettings.FindIndex(x => x.name == platformSettings.name);
            if (index < 0)
                m_PlatformSettings.Add(platformSettings);
            else
                m_PlatformSettings[index] = platformSettings;
        }

        void SetDirty()
        {
            EditorUtility.SetDirty(this);
        }

        List<SpriteMetaData> GetSpriteImportData()
        {
            if (spriteImportModeToUse == SpriteImportMode.Multiple)
            {
                switch (m_AsepriteImporterSettings.fileImportMode)
                {
                    case FileImportModes.SpriteSheet:
                        return m_SpriteSheetImportData;
                    case FileImportModes.TileSet:
                        return m_TileSetImportData;
                    case FileImportModes.AnimatedSprite:
                    default:
                        return m_AnimatedSpriteImportData;
                }
            }
            return m_SingleSpriteImportData;
        }

        internal SpriteRect GetSpriteData(GUID guid)
        {
            if (spriteImportModeToUse != SpriteImportMode.Multiple)
                return m_SingleSpriteImportData[0];

            switch (m_AsepriteImporterSettings.fileImportMode)
            {
                case FileImportModes.SpriteSheet:
                    {
                        foreach (var metaData in m_SpriteSheetImportData)
                        {
                            if (metaData.spriteID == guid)
                                return metaData;
                        }
                        return default;
                    }
                case FileImportModes.TileSet:
                    {
                        foreach (var metaData in m_TileSetImportData)
                        {
                            if (metaData.spriteID == guid)
                                return metaData;
                        }
                        return default;
                    }                
                case FileImportModes.AnimatedSprite:
                default:
                    {
                        foreach (var metaData in m_AnimatedSpriteImportData)
                        {
                            if (metaData.spriteID == guid)
                                return metaData;
                        }
                        return default;
                    }
            }
        }

        internal TextureImporterPlatformSettings[] GetAllPlatformSettings()
        {
            return m_PlatformSettings.ToArray();
        }

        internal void ReadTextureSettings(TextureImporterSettings dest)
        {
            m_TextureImporterSettings.CopyTo(dest);
        }
    }
}
