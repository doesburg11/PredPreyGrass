using System.IO;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Supported color profiles.
    /// </summary>
    public enum ColorProfileTypes
    {
        /// <summary>
        /// No color profile (as in old .aseprite files).
        /// </summary>
        NoProfile = 0,
        /// <summary>
        /// use sRGB.
        /// </summary>
        sRGB = 1,
        /// <summary>
        /// use the embedded ICC profile.
        /// </summary>
        ICC = 2
    }

    /// <summary>
    /// Parsed representation of an Aseprite ColorProfile chunk.
    /// </summary>
    public class ColorProfileChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.ColorProfile;

        /// <summary>
        /// The color profile used in this Aseprite file.
        /// </summary>
        public ColorProfileTypes profileType { get; private set; }
        /// <summary>
        /// Flags for this color profile. <br />
        /// 1 - use special fixed gamma.
        /// </summary>
        public ushort flags { get; private set; }
        /// <summary>
        /// Fixed gamma (1.0 = linear).
        /// </summary>
        public float gamma { get; private set; }

        internal ColorProfileChunk(uint chunkSize) : base(chunkSize) { }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            profileType = (ColorProfileTypes)reader.ReadUInt16();
            flags = reader.ReadUInt16();
            gamma = reader.ReadSingle();

            // Reserved bytes
            for (var i = 0; i < 8; ++i)
                reader.ReadByte();

            if (profileType == ColorProfileTypes.ICC)
            {
                var iccProfileLength = reader.ReadUInt32();
                for (var i = 0; i < iccProfileLength; ++i)
                    reader.ReadByte();
            }
        }
    }
}
