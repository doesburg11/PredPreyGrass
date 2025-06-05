using System.Collections.Generic;
using System.IO;
using System.Collections.ObjectModel;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Parsed representation of an Aseprite Old Palette chunk.
    /// Not supported yet.
    /// </summary>
    internal class OldPaletteChunk : BaseChunk, IPaletteProvider
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.OldPalette;

        /// <summary>
        /// Array of palette entries.
        /// </summary>
        public ReadOnlyCollection<PaletteEntry> entries => System.Array.AsReadOnly(m_Entries);
        PaletteEntry[] m_Entries;

        internal OldPaletteChunk(uint chunkSize) : base(chunkSize) { }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            var noOfPackets = reader.ReadUInt16();
            var colorEntries = new List<PaletteEntry>();

            var colorIndex = 0;
            for (var i = 0; i < noOfPackets; ++i)
            {
                var noOfColorsToSkip = reader.ReadByte();
                colorIndex += noOfColorsToSkip;

                int noOfColorsInEntry = reader.ReadByte();
                if (noOfColorsInEntry == 0)
                    noOfColorsInEntry = 256;

                // If j + colorIndex >= 256 it means that the color chunk is invalid, so we stop reading.
                for (var j = 0; j < noOfColorsInEntry && (j + colorIndex < 256); ++j)
                {
                    var r = reader.ReadByte();
                    var g = reader.ReadByte();
                    var b = reader.ReadByte();

                    colorEntries.Add(new PaletteEntry("", new Color32(r, g, b, 255)));
                }
            }

            m_Entries = colorEntries.ToArray();
        }
    }
}
