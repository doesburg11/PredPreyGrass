using System;
using System.Collections.ObjectModel;
using System.IO;
using UnityEngine.Assertions;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Parsed representation of an Aseprite frame.
    /// </summary>
    public class FrameData : IDisposable
    {
        /// <summary>
        /// Bytes in this frame.
        /// </summary>
        public uint noOfBytes { get; private set; }
        /// <summary>
        /// Frame duration in milliseconds.
        /// </summary>
        public ushort frameDuration { get; private set; }
        /// <summary>
        /// Number of chunks.
        /// </summary>
        public uint chunkCount { get; private set; }
        /// <summary>
        /// Chunk data for this frame.
        /// </summary>
        public ReadOnlyCollection<BaseChunk> chunks => Array.AsReadOnly(m_Chunks);
        BaseChunk[] m_Chunks;

        internal void Read(BinaryReader reader)
        {
            noOfBytes = reader.ReadUInt32();
            var misc0 = reader.ReadUInt16();
            var legacyChunkCount = reader.ReadUInt16();
            frameDuration = reader.ReadUInt16();
            var misc1 = reader.ReadByte();
            var misc2 = reader.ReadByte();
            var chunkCount = reader.ReadUInt32();

            Assert.IsTrue(misc0 == 0xF1FA, "Reading mismatch.");

            this.chunkCount = chunkCount != 0 ? chunkCount : legacyChunkCount;
            m_Chunks = new BaseChunk[this.chunkCount];
        }

        internal void SetChunkData(int index, BaseChunk data)
        {
            if (index < 0 || index >= m_Chunks.Length)
                return;
            m_Chunks[index] = data;
        }

        /// <summary>
        /// Dispose of the frame data.
        /// </summary>
        public void Dispose()
        {
            for (var i = 0; i < m_Chunks.Length; ++i)
                m_Chunks[i].Dispose();
        }
    }
}
