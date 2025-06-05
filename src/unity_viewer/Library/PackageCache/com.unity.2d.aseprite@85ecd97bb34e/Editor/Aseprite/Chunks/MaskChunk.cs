using System.IO;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Parsed representation of an Aseprite Mask chunk.
    /// Not supported yet.
    /// </summary>
    internal class MaskChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.Mask;

        internal MaskChunk(uint chunkSize) : base(chunkSize) { }
        
        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>        
        protected override void InternalRead(BinaryReader reader) { }
    }
}
