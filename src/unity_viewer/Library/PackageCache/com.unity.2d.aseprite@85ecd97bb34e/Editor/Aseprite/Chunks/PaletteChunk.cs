using System.Collections.ObjectModel;
using System.IO;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Structure for an entry in the color palette.
    /// </summary>
    public struct PaletteEntry
    {
        internal PaletteEntry(string name, Color32 color)
        {
            this.name = name;
            this.color = color;
        }

        /// <summary>
        /// Name of the color.
        /// </summary>
        public string name { get; private set; }
        /// <summary>
        /// Color value.
        /// </summary>
        public Color32 color { get; private set; }
    }

    /// <summary>
    /// Parsed representation of an Aseprite Palette chunk.
    /// </summary>
    public class PaletteChunk : BaseChunk, IPaletteProvider
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.Palette;

        /// <summary>
        /// Number of entries in the palette.
        /// </summary>
        public uint noOfEntries { get; private set; }
        /// <summary>
        /// Index of the first color to change.
        /// </summary>
        public uint firstColorIndex { get; private set; }
        /// <summary>
        /// Index of the last color to change.
        /// </summary>
        public uint lastColorIndex { get; private set; }
        /// <summary>
        /// Array of palette entries.
        /// </summary>
        public ReadOnlyCollection<PaletteEntry> entries => System.Array.AsReadOnly(m_Entries);
        PaletteEntry[] m_Entries;

        internal PaletteChunk(uint chunkSize) : base(chunkSize) { }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            noOfEntries = reader.ReadUInt32();
            firstColorIndex = reader.ReadUInt32();
            lastColorIndex = reader.ReadUInt32();

            // Reserved bytes
            for (var i = 0; i < 8; ++i)
                reader.ReadByte();

            m_Entries = new PaletteEntry[noOfEntries];
            for (var i = 0; i < noOfEntries; ++i)
            {
                var entryFlag = reader.ReadUInt16();
                var red = reader.ReadByte();
                var green = reader.ReadByte();
                var blue = reader.ReadByte();
                var alpha = reader.ReadByte();

                var color = new Color32(red, green, blue, alpha);
                var name = "";

                var hasName = entryFlag == 1;
                if (hasName)
                    name = AsepriteUtilities.ReadString(reader);

                m_Entries[i] = new PaletteEntry(name, color);
            }
        }
    }
}
