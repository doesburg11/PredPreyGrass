using System.Collections.ObjectModel;
using System.IO;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Aseprite cell types.
    /// </summary>
    public enum CellTypes
    {
        /// <summary>Raw pixel data</summary>
        RawImage = 0,
        /// <summary>Cell linked to another cell</summary>
        LinkedCell = 1,
        /// <summary>Compressed pixel data</summary>
        CompressedImage = 2,
        /// <summary>Compressed tilemap data</summary>
        CompressedTileMap = 3
    }

    /// <summary>
    /// Parsed representation of an Aseprite Cell chunk.
    /// </summary>
    public class CellChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.Cell;

        internal CellChunk(uint chunkSize, ushort colorDepth, ReadOnlyCollection<PaletteEntry> paletteEntries, byte alphaPaletteEntry) : base(chunkSize)
        {
            m_ColorDepth = colorDepth;
            m_PaletteEntries = paletteEntries;
            m_AlphaPaletteEntry = alphaPaletteEntry;
        }

        readonly ushort m_ColorDepth;
        readonly ReadOnlyCollection<PaletteEntry> m_PaletteEntries;
        readonly byte m_AlphaPaletteEntry;

        /// <summary>
        /// The layer index is a number to identify a layer in the sprite.
        /// Layers are numbered in the same order as Layer Chunks appear in the file.
        /// </summary>
        public ushort layerIndex { get; private set; }
        /// <summary>
        /// The Cell's X position on the canvas.
        /// </summary>
        public short posX { get; private set; }
        /// <summary>
        /// The Cell's Y position on the canvas.
        /// </summary>
        public short posY { get; private set; }
        /// <summary>
        /// Opacity level of the cell (0 = transparent, 255 = opaque).
        /// </summary>
        public byte opacity { get; private set; }
        /// <summary>
        /// The type of cell.
        /// </summary>
        public CellTypes cellType { get; private set; }
        /// <summary>
        /// A cell's draw order. Higher number means towards the front.
        /// </summary>
        internal short zIndex { get; private set; }
        /// <summary>
        /// The frame index of the cell (Only available for Linked Cells).
        /// </summary>
        public int linkedToFrame { get; private set; } = -1;
        /// <summary>
        /// The width of the cell in pixels.
        /// </summary>
        public ushort width { get; private set; }
        /// <summary>
        /// The height of the cell in pixels.
        /// </summary>
        public ushort height { get; private set; }
        /// <summary>
        /// The image data of the cell.
        /// </summary>
        public NativeArray<Color32> image { get; private set; }
        /// <summary>
        /// User data associated with the cell.
        /// </summary>
        public UserDataChunk dataChunk { get; set; }
        /// <summary>
        /// Indices to the tiles in the cell. Note, only populated if the cell is of type CompressedTileMap.
        /// </summary>
        internal NativeArray<uint> tileIndices  { get; private set; }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            layerIndex = reader.ReadUInt16();
            posX = reader.ReadInt16();
            posY = reader.ReadInt16();
            opacity = reader.ReadByte();
            cellType = (CellTypes)reader.ReadUInt16();
            zIndex = reader.ReadInt16();

            // Not in use bytes
            for (var i = 0; i < 5; ++i)
            {
                var miscVal = reader.ReadByte();
                Assert.IsTrue(miscVal == 0);
            }

            if (cellType == CellTypes.RawImage)
            {
                width = reader.ReadUInt16();
                height = reader.ReadUInt16();

                byte[] imageData = null;
                if (m_ColorDepth == 32)
                    imageData = reader.ReadBytes(width * height * 4);
                else if (m_ColorDepth == 16)
                    imageData = reader.ReadBytes(width * height * 2);
                else if (m_ColorDepth == 8)
                    imageData = reader.ReadBytes(width * height);

                if (imageData != null)
                    image = AsepriteUtilities.GenerateImageData(m_ColorDepth, imageData, m_PaletteEntries, m_AlphaPaletteEntry);

            }
            else if (cellType == CellTypes.LinkedCell)
            {
                linkedToFrame = reader.ReadUInt16();
            }
            else if (cellType == CellTypes.CompressedImage)
            {
                width = reader.ReadUInt16();
                height = reader.ReadUInt16();

                var dataSize = (int)m_ChunkSize - ChunkHeader.stride - 20;
                var decompressedData = AsepriteUtilities.ReadAndDecompressedData(reader, dataSize);

                image = AsepriteUtilities.GenerateImageData(m_ColorDepth, decompressedData, m_PaletteEntries, m_AlphaPaletteEntry);
            }
            else if (cellType == CellTypes.CompressedTileMap) 
            {
                // Width and height in number of tiles
                width = reader.ReadUInt16();
                height = reader.ReadUInt16();
                
                var bitsPerTile = reader.ReadUInt16();
                var tileIdMask = reader.ReadUInt32();
                var xFlipMask = reader.ReadUInt32();
                var yFlipMask = reader.ReadUInt32();
                var rotation90Mask = reader.ReadUInt32();

                // Not in use bytes
                for (var i = 0; i < 10; ++i)
                    reader.ReadByte();

                var dataSize = (int)m_ChunkSize - ChunkHeader.stride - 48;
                var decompressedData = AsepriteUtilities.ReadAndDecompressedData(reader, dataSize);

                var bytesPerTile = bitsPerTile / 8;
                var noOfTiles = decompressedData.Length / bytesPerTile;

                var indices = new NativeArray<uint>(noOfTiles, Allocator.Persistent);

                using var memoryStream = new MemoryStream(decompressedData);
                using var binaryReader = new BinaryReader(memoryStream);
                for (var i = 0; i < noOfTiles; ++i)
                {
                    uint tileData = 0;
                    if (bitsPerTile == 32)
                        tileData = binaryReader.ReadUInt32();
                    else if (bitsPerTile == 16)
                        tileData = binaryReader.ReadUInt16();
                    else if (bitsPerTile == 8)
                        tileData = binaryReader.ReadByte();
                    
                    var tileId = tileData & tileIdMask;
                    indices[i] = tileId;
                }

                tileIndices = indices;
            }
        }

        /// <summary>
        /// Dispose of the image and tile data.
        /// </summary>
        public override void Dispose()
        {
            image.DisposeIfCreated();
            tileIndices.DisposeIfCreated();
        }
    }
}
