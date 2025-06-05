using System;
using System.IO;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Flags for layer options.
    /// </summary>
    [Flags]
    public enum LayerFlags
    {
        /// <summary>Flag set if the layer is visible</summary>
        Visible = 1,
        /// <summary>Editable - not in use flag</summary>
        Editable = 2,
        /// <summary>LockMovement - not in use flag</summary>
        LockMovement = 4,
        /// <summary>Background - not in use flag</summary>
        Background = 8,
        /// <summary>PreferLinkedCels - not in use flag</summary>
        PreferLinkedCels = 16,
        /// <summary>DisplayAsCollapsed - not in use flag</summary>
        DisplayAsCollapsed = 32,
        /// <summary>ReferenceLayer - not in use flag</summary>
        ReferenceLayer = 64
    }

    /// <summary>
    /// Layer types.
    /// </summary>
    public enum LayerTypes
    {
        /// <summary>Normal layer</summary>
        Normal = 0,
        /// <summary>Group layer</summary>
        Group = 1,
        /// <summary>Tilemap layer</summary>
        Tilemap = 2
    }

    /// <summary>
    /// Layer blend modes.
    /// </summary>
    public enum BlendModes
    {
        /// <summary>Normal blend mode</summary>
        Normal = 0,
        /// <summary>Multiply blend mode</summary>
        Multiply = 1,
        /// <summary>Screen blend mode</summary>
        Screen = 2,
        /// <summary>Overlay blend mode</summary>
        Overlay = 3,
        /// <summary>Darken blend mode</summary>
        Darken = 4,
        /// <summary>Lighten blend mode</summary>
        Lighten = 5,
        /// <summary>ColorDodge blend mode</summary>
        ColorDodge = 6,
        /// <summary>ColorBurn blend mode</summary>
        ColorBurn = 7,
        /// <summary>HardLight blend mode</summary>
        HardLight = 8,
        /// <summary>SoftLight blend mode</summary>
        SoftLight = 9,
        /// <summary>Difference blend mode</summary>
        Difference = 10,
        /// <summary>Exclusion blend mode</summary>
        Exclusion = 11,
        /// <summary>Hue blend mode</summary>
        Hue = 12,
        /// <summary>Saturation blend mode</summary>
        Saturation = 13,
        /// <summary>Color blend mode</summary>
        Color = 14,
        /// <summary>Luminosity blend mode</summary>
        Luminosity = 15,
        /// <summary>Addition blend mode</summary>
        Addition = 16,
        /// <summary>Subtract blend mode</summary>
        Subtract = 17,
        /// <summary>Divide blend mode</summary>
        Divide = 18
    }

    /// <summary>
    /// Parsed representation of an Aseprite Layer chunk.
    /// </summary>
    public class LayerChunk : BaseChunk
    {
        /// <inheritdoc />
        public override ChunkTypes chunkType => ChunkTypes.Layer;
        
        /// <summary>
        /// Layer UUID (Only available if the user has enabled UUID for layers in Aseprite).
        /// </summary>
        internal UUID uuid { get; private set; }
        /// <summary>
        /// Layer option flags.
        /// </summary>
        public LayerFlags flags { get; private set; }
        /// <summary>
        /// Type of layer.
        /// </summary>
        public LayerTypes layerType { get; private set; }
        /// <summary>
        /// The child level is used to show the relationship of this layer with the last one read.
        /// </summary>
        public ushort childLevel { get; private set; }
        /// <summary>
        /// Layer blend mode.
        /// </summary>
        public BlendModes blendMode { get; private set; }
        /// <summary>Layer opacity (0 = transparent, 255 = opaque).</summary>
        public byte opacity { get; private set; }
        /// <summary>
        /// Layer name.
        /// </summary>
        public string name { get; private set; }
        /// <summary>
        /// Tileset index (Only available for Tilemap layers).
        /// </summary>
        public uint tileSetIndex { get; private set; }

        bool m_HasUuid;

        internal LayerChunk(uint chunkSize, bool hasUuid) : base(chunkSize)
        {
            m_HasUuid = hasUuid;
        }

        /// <summary>
        /// Read and store the chunk data.
        /// </summary>
        /// <param name="reader">The active binary reader of the file.</param>
        protected override void InternalRead(BinaryReader reader)
        {
            flags = (LayerFlags)reader.ReadUInt16();
            layerType = (LayerTypes)reader.ReadUInt16();
            childLevel = reader.ReadUInt16();
            var defaultLayerWidth = reader.ReadUInt16();
            var defaultLayerHeight = reader.ReadUInt16();
            blendMode = (BlendModes)reader.ReadUInt16();
            opacity = reader.ReadByte();

            // Not in use bytes
            for (var i = 0; i < 3; ++i)
                reader.ReadByte();

            name = AsepriteUtilities.ReadString(reader);
            if (layerType == LayerTypes.Tilemap)
                tileSetIndex = reader.ReadUInt32();
            
            if (m_HasUuid)
            {
                var value0 = reader.ReadUInt32();
                var value1 = reader.ReadUInt32();
                var value2 = reader.ReadUInt32();
                var value3 = reader.ReadUInt32();
                uuid = new UUID(value0, value1, value2, value3);
            }
        }
    }
}
