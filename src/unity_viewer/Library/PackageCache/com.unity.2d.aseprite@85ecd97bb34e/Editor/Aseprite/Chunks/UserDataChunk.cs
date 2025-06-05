using System;
using System.IO;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    [Flags]
    internal enum UserDataFlags
    {
        HasText = 1,
        HasColor = 2
    }

    /// <summary>
    /// Parsed representation of an Aseprite UserData chunk.
    /// </summary>
    public class UserDataChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.UserData;

        /// <summary>
        /// Text data.
        /// </summary>
        public string text { get; private set; }
        /// <summary>
        /// Color data.
        /// </summary>
        public Color32 color { get; private set; }

        internal UserDataChunk(uint chunkSize) : base(chunkSize) { }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            var flag = (UserDataFlags)reader.ReadUInt32();

            if ((flag & UserDataFlags.HasText) != 0)
                text = AsepriteUtilities.ReadString(reader);
            if ((flag & UserDataFlags.HasColor) != 0)
            {
                var red = reader.ReadByte();
                var green = reader.ReadByte();
                var blue = reader.ReadByte();
                var alpha = reader.ReadByte();
                color = new Color32(red, green, blue, alpha);
            }
        }
    }
}
