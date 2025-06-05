using System.IO;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Empty default chunk.
    /// </summary>
    internal class NoneChunk : BaseChunk
    {
        internal NoneChunk(uint chunkSize) : base(chunkSize) { }
        
        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader) { }
    }
}
