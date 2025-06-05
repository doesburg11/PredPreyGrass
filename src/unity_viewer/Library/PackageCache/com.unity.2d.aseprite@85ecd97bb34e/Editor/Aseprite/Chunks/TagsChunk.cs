using System.Collections.ObjectModel;
using System.IO;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// The different loop animation directions.
    /// </summary>
    public enum LoopAnimationDirection
    {
        /// <summary>Loop direction - Forward</summary>
        Forward = 0,
        /// <summary>Loop direction - Reverse</summary>
        Reverse = 1,
        /// <summary>Loop direction - PingPong</summary>
        PingPong = 2,
        /// <summary>Loop direction - PingPongReverse</summary>
        PingPongReverse = 3,
    }

    /// <summary>
    /// Parsed representation of an Aseprite Tag data.
    /// </summary>
    public class TagData
    {
        /// <summary>
        /// The starting frame of the tag.
        /// </summary>
        public ushort fromFrame { get; set; }
        /// <summary>
        /// The ending frame of the tag.
        /// </summary>
        public ushort toFrame { get; set; }
        /// <summary>
        /// The loop animation direction.
        /// </summary>
        public LoopAnimationDirection loopDirection { get; set; }
        /// <summary>
        /// The number of times the animation should loop.<br />
        /// 0 = Doesn't specify (plays infinite in UI, once on export, for ping-pong it plays once in each direction).<br />
        /// 1 = Plays once (for ping-pong, it plays just in one direction).<br />
        /// 2 = Plays twice (for ping-pong, it plays once in one direction, and once in reverse).<br />
        /// n = Plays N times.
        /// </summary>
        public ushort noOfRepeats { get; set; }
        /// <summary>
        /// The name of the tag.
        /// </summary>
        public string name { get; set; }
    }

    /// <summary>
    /// Parsed representation of an Aseprite Tags chunk.
    /// </summary>
    public class TagsChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.Tags;

        /// <summary>
        /// The number of tags in the chunk.
        /// </summary>
        public int noOfTags { get; private set; }

        /// <summary>
        /// The tag data stored in the chunk.
        /// </summary>
        public ReadOnlyCollection<TagData> tagData => System.Array.AsReadOnly(m_TagData);
        TagData[] m_TagData;

        internal TagsChunk(uint chunkSize) : base(chunkSize) { }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            noOfTags = reader.ReadUInt16();

            // Not in use bytes
            for (var i = 0; i < 8; ++i)
                reader.ReadByte();

            m_TagData = new TagData[noOfTags];
            for (var i = 0; i < noOfTags; ++i)
            {
                m_TagData[i] = new TagData();
                m_TagData[i].fromFrame = reader.ReadUInt16();
                m_TagData[i].toFrame = reader.ReadUInt16();
                m_TagData[i].loopDirection = (LoopAnimationDirection)reader.ReadByte();
                m_TagData[i].noOfRepeats = reader.ReadUInt16();

                // Not in use bytes
                for (var m = 0; m < 6; ++m)
                    reader.ReadByte();
                // Tag color. Deprecated.
                for (var m = 0; m < 3; ++m)
                    reader.ReadByte();
                reader.ReadByte();

                m_TagData[i].name = AsepriteUtilities.ReadString(reader);
            }
        }
    }
}
