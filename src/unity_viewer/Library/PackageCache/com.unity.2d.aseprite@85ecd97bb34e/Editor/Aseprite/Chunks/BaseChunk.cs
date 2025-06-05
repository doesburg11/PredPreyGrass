using System;
using System.IO;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Aseprite Chunk Types.
    /// </summary>
    public enum ChunkTypes
    {
        /// <summary>Default type</summary>
        None = 0,
        /// <summary>Old palette data</summary>
        OldPalette = 0x0004,
        /// <summary>Second old palette data - Not in use</summary>
        OldPalette2 = 0x0011,
        /// <summary>Layer data</summary>
        Layer = 0x2004,
        /// <summary>Cell data</summary>
        Cell = 0x2005,
        /// <summary>Extra cell data - Not in use</summary>
        CellExtra = 0x2006,
        /// <summary>Color profile data</summary>
        ColorProfile = 0x2007,
        /// <summary>External files data - Not in use</summary>
        ExternalFiles = 0x2008,
        /// <summary>Mask data - Not in use</summary>
        Mask = 0x2016,
        /// <summary>Path data - Not in use</summary>
        Path = 0x2017,
        /// <summary>Tag data</summary>
        Tags = 0x2018,
        /// <summary>Palette data</summary>
        Palette = 0x2019,
        /// <summary>User data</summary>
        UserData = 0x2020,
        /// <summary>Slice data - Not in use</summary>
        Slice = 0x2022,
        /// <summary>Tile set data - Not in use</summary>
        Tileset = 0x2023
    }

    /// <summary>
    /// The header of each chunk.
    /// </summary>
    public class ChunkHeader
    {
        /// <summary>
        /// The stride of the chunk header in bytes.
        /// </summary>
        public const int stride = 6;
        /// <summary>
        /// The size of the chunk in bytes.
        /// </summary>
        public uint chunkSize { get; private set; }
        /// <summary>
        /// The type of the chunk.
        /// </summary>
        public ChunkTypes chunkType { get; private set; }

        internal void Read(BinaryReader reader)
        {
            chunkSize = reader.ReadUInt32();
            chunkType = (ChunkTypes)reader.ReadUInt16();
        }
    }

    /// <summary>
    /// Base class for all chunks.
    /// </summary>
    public abstract class BaseChunk : IDisposable
    {
        /// <summary>
        /// The type of the chunk.
        /// </summary>
        public virtual ChunkTypes chunkType => ChunkTypes.None;

        /// <summary>
        /// The size of the chunk in bytes.
        /// </summary>
        protected readonly uint m_ChunkSize;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="chunkSize">The size of the chunk in bytes.</param>
        protected BaseChunk(uint chunkSize)
        {
            m_ChunkSize = chunkSize;
        }

        internal bool Read(BinaryReader reader)
        {
            var bytes = reader.ReadBytes((int)m_ChunkSize - ChunkHeader.stride);
            using var memoryStream = new MemoryStream(bytes);
            using var chunkReader = new BinaryReader(memoryStream);

            try
            {
                InternalRead(chunkReader);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to read a chunk of type: {chunkType}. Skipping the chunk. \nException: {e}");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Implement this method to read the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected abstract void InternalRead(BinaryReader reader);

        /// <summary>
        /// Implement this method to dispose of the chunk data.
        /// </summary>
        public virtual void Dispose() { }
    }
}
