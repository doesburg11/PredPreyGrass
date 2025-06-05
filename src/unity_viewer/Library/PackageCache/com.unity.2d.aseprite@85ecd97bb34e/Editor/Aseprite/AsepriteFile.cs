/// .ase & .aseprite file format specs:
/// https://github.com/aseprite/aseprite/blob/main/docs/ase-file-specs.md

using System;
using System.Collections.ObjectModel;
using System.IO;
using UnityEngine.Assertions;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// Parsed representation of an Aseprite file.
    /// Should be disposed after use.
    /// </summary>
    public class AsepriteFile : IDisposable
    {
        [Flags]
        internal enum HeaderFlags : uint
        {
            None = 0,
            LayerOpacityInvalidValue = 1,
            LayerBlendModeValidForGroups = 2,
            LayersHaveUuid = 4
        }
        
        /// <summary>
        /// File size in bytes.
        /// </summary>
        public uint fileSize { get; private set; }
        /// <summary>
        /// Number of frames in the file.
        /// </summary>
        public ushort noOfFrames { get; private set; }
        /// <summary>
        /// Canvas width in pixels.
        /// </summary>
        public ushort width { get; private set; }
        /// <summary>
        /// Canvas height in pixels.
        /// </summary>
        public ushort height { get; private set; }
        /// <summary>
        /// Color depth (bits per pixel).
        /// </summary>
        public ushort colorDepth { get; private set; }
        /// <summary>
        /// Flags are used to highlight behavior changes in this particular file
        /// </summary>
        internal HeaderFlags flags { get; private set; }
        /// <summary>
        /// Time per frame in milliseconds.
        /// </summary>
        public ushort animSpeed { get; private set; }
        /// <summary>
        /// Palette entry (index) which represent transparent color
        /// in all non-background layers (only for Indexed sprites).
        /// </summary>
        public byte alphaPaletteEntry { get; private set; }
        /// <summary>
        /// Number of colors (0 means 256 for old sprites).
        /// </summary>
        public ushort noOfColors { get; private set; }
        /// <summary>
        /// Pixel width (pixel ratio is "pixel width/pixel height").
        /// If this or pixel height field is zero, pixel ratio is 1:1.
        /// </summary>
        public byte pixelWidth { get; private set; }
        /// <summary>
        /// Pixel height (pixel ratio is "pixel width/pixel height").
        /// If this or pixel width field is zero, pixel ratio is 1:1.
        /// </summary>
        public byte pixelHeight { get; private set; }
        /// <summary>
        /// X position of the grid.
        /// </summary>
        public short gridPosX { get; private set; }
        /// <summary>
        /// Y position of the grid.
        /// </summary>
        public short gridPosY { get; private set; }
        /// <summary>
        /// Grid width (zero if there is no grid, grid size is 16x16 on Aseprite by default).
        /// </summary>
        public ushort gridWidth { get; private set; }
        /// <summary>
        /// Grid height (zero if there is no grid).
        /// </summary>
        public ushort gridHeight { get; private set; }
        /// <summary>
        /// Parsed data of each frame.
        /// </summary>
        public ReadOnlyCollection<FrameData> frameData => Array.AsReadOnly(m_FrameData);
        FrameData[] m_FrameData;

        internal void Read(BinaryReader reader)
        {
            var streamLength = reader.BaseStream.Length;
            Assert.IsTrue(streamLength >= 128, "File is too small to be a valid Aseprite file.");

            fileSize = reader.ReadUInt32();
            var misc0 = reader.ReadUInt16();
            noOfFrames = reader.ReadUInt16();
            width = reader.ReadUInt16();
            height = reader.ReadUInt16();
            colorDepth = reader.ReadUInt16();
            flags = (HeaderFlags)reader.ReadUInt32();
            animSpeed = reader.ReadUInt16();
            var misc1 = reader.ReadUInt32();
            var misc2 = reader.ReadUInt32();
            alphaPaletteEntry = reader.ReadByte();
            var miscByte0 = reader.ReadByte();
            var miscByte1 = reader.ReadByte();
            var miscByte2 = reader.ReadByte();
            noOfColors = reader.ReadUInt16();
            pixelWidth = reader.ReadByte();
            pixelHeight = reader.ReadByte();
            gridPosX = reader.ReadInt16();
            gridPosY = reader.ReadInt16();
            gridWidth = reader.ReadUInt16();
            gridHeight = reader.ReadUInt16();

            Assert.IsTrue(misc0 == 0xA5E0, "Unexpected file content. The file is most likely corrupt.");

            // Unused 84 bytes
            for (var i = 0; i < 84; ++i)
                reader.ReadByte();

            m_FrameData = new FrameData[noOfFrames];
        }

        internal void SetFrameData(int index, FrameData data)
        {
            if (index < 0 || index >= m_FrameData.Length)
                return;
            m_FrameData[index] = data;
        }

        /// <summary>
        /// Disposes the loaded Aseprite file.
        /// </summary>
        public void Dispose()
        {
            for (var i = 0; i < m_FrameData.Length; ++i)
                m_FrameData[i].Dispose();
        }
    }
}
