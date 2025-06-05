using System.IO;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Parsed representation of an Aseprite Old Palette (no. 2) chunk.
    /// Not supported yet.
    /// </summary>
    internal class OldPaletteChunk2 : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.OldPalette2;

        internal OldPaletteChunk2(uint chunkSize) : base(chunkSize) { }
        
        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>        
        protected override void InternalRead(BinaryReader reader) { }
    }
}
