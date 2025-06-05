using System;
using System.Collections.ObjectModel;
using System.IO;
using Unity.Collections;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Flags to define where data for a tileset is stored.
    /// </summary>
    [Flags]
    public enum TileSetFlags
    {
        /// <summary>Includes a link to an external file</summary>
        IncludesLinkToExternal = 1,
        /// <summary>Includes a link to files in this file</summary>
        IncludesTilesInFile = 2,
        /// <summary>Misc</summary>
        Misc = 4,
    }

    /// <summary>
    /// Parsed representation of an Aseprite Tileset chunk.
    /// Not supported yet.
    /// </summary>
    public class TilesetChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.Tileset;

        /// <summary>
        /// The ID of the tileset.
        /// </summary>
        public uint tileSetId { get; private set; }
        /// <summary>
        /// Flags to define where data for a tileset is stored.
        /// </summary>
        public TileSetFlags tileSetFlags { get; private set; }
        /// <summary>
        /// The number of tiles in the tileset.
        /// </summary>
        public uint noOfTiles { get; private set; }
        /// <summary>
        /// Tile width in pixels.
        /// </summary>
        public ushort width { get; private set; }
        /// <summary>
        /// Tile height in pixels.
        /// </summary>
        public ushort height { get; private set; }
        /// <summary>
        /// The name of the tileset.
        /// </summary>
        public string tileSetName { get; private set; }
        /// <summary>
        /// The image data of the tiles.
        /// </summary>
        public NativeArray<Color32>[] tileImages { get; private set; }

        readonly ushort m_ColorDepth;
        readonly ReadOnlyCollection<PaletteEntry> m_PaletteEntries;
        readonly byte m_AlphaPaletteEntry;

        internal TilesetChunk(uint chunkSize, ushort colorDepth, ReadOnlyCollection<PaletteEntry> paletteEntries, byte alphaPaletteEntry) : base(chunkSize)
        {
            m_ColorDepth = colorDepth;
            m_PaletteEntries = paletteEntries;
            m_AlphaPaletteEntry = alphaPaletteEntry;
        }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            tileSetId = reader.ReadUInt32();
            tileSetFlags = (TileSetFlags)reader.ReadUInt32();
            noOfTiles = reader.ReadUInt32();
            width = reader.ReadUInt16();
            height = reader.ReadUInt16();

            var baseIndex = reader.ReadInt16();
            var reservedBytes = reader.ReadBytes(14);

            tileSetName = AsepriteUtilities.ReadString(reader);
            if (string.IsNullOrEmpty(tileSetName))
                tileSetName = $"Tileset_{tileSetId}";

            // Not supported yet.
            if ((tileSetFlags & TileSetFlags.IncludesLinkToExternal) != 0)
            {
                var idOfExternalFile = reader.ReadUInt32();
                var tileSetIdInExternal = reader.ReadUInt32();
            }
            if ((tileSetFlags & TileSetFlags.IncludesTilesInFile) != 0)
            {
                var compressedDataLength = (int)reader.ReadUInt32();
                var decompressedData = AsepriteUtilities.ReadAndDecompressedData(reader, compressedDataLength);

                using var tilemapImage = AsepriteUtilities.GenerateImageData(m_ColorDepth, decompressedData, m_PaletteEntries, m_AlphaPaletteEntry);

                tileImages = new NativeArray<Color32>[noOfTiles];
                for (var i = 0; i < noOfTiles; ++i)
                {
                    var tileImage = new NativeArray<Color32>(width * height, Allocator.Persistent);
                    var tileStartHeight = i * height;

                    for (var m = 0; m < height; ++m)
                    {
                        var sourceHeight = (tileStartHeight + m) * width;
                        var destHeight = m * width;
                        NativeArray<Color32>.Copy(tilemapImage, sourceHeight, tileImage, destHeight, width);
                    }

                    tileImages[i] = tileImage;
                }
            }
        }

        /// <summary>
        /// Dispose of the image data.
        /// </summary>
        public override void Dispose()
        {
            if (tileImages == null || tileImages.Length == 0)
                return;

            for (var i = 0; i < tileImages.Length; ++i)
                tileImages[i].DisposeIfCreated();
        }
    }
}
