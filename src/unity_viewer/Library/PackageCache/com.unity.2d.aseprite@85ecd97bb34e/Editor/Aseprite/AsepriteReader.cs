using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    /// <summary>
    /// File parsing utility for Aseprite files.
    /// </summary>
    public static class AsepriteReader
    {
        /// <summary>
        /// Reads an Aseprite file from the given path.
        /// </summary>
        /// <param name="assetPath">Path to the file.</param>
        /// <returns>Returns a parsed representation of the file.</returns>
        public static AsepriteFile ReadFile(string assetPath)
        {
            var doesFileExist = File.Exists(assetPath);
            if (!doesFileExist)
            {
                Debug.LogError($"File does not exist at path: {assetPath}");
                return null;
            }

            var fileStream = new FileStream(assetPath, FileMode.Open, FileAccess.Read);
            var binaryReader = new BinaryReader(fileStream);

            var file = new AsepriteFile();
            try
            {
                file.Read(binaryReader);
                ReadFrames(in binaryReader, ref file);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to import file: {assetPath}. Error: {e.Message} \n{e.StackTrace}");
                file = null;
            }
            finally
            {
                binaryReader.Close();
                fileStream.Close();
            }

            return file;
        }

        static void ReadFrames(in BinaryReader reader, ref AsepriteFile file)
        {
            var paletteProvider = default(IPaletteProvider);

            for (var i = 0; i < file.noOfFrames; ++i)
            {
                var frame = new FrameData();
                frame.Read(reader);
                file.SetFrameData(i, frame);

                var noOfChunks = frame.chunkCount;
                for (var m = 0; m < noOfChunks; ++m)
                {
                    var chunkHeader = new ChunkHeader();
                    chunkHeader.Read(reader);

                    BaseChunk chunk = null;
                    switch (chunkHeader.chunkType)
                    {
                        case ChunkTypes.Cell:
                            chunk = new CellChunk(chunkHeader.chunkSize, file.colorDepth, paletteProvider?.entries, file.alphaPaletteEntry);
                            break;
                        case ChunkTypes.CellExtra:
                            chunk = new CellExtra(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.ColorProfile:
                            chunk = new ColorProfileChunk(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.ExternalFiles:
                            chunk = new ExternalFilesChunk(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.Layer:
                            chunk = new LayerChunk(chunkHeader.chunkSize, (file.flags & AsepriteFile.HeaderFlags.LayersHaveUuid) != 0);
                            break;
                        case ChunkTypes.Mask:
                            chunk = new MaskChunk(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.OldPalette:
                            chunk = new OldPaletteChunk(chunkHeader.chunkSize);
                            paletteProvider = ((IPaletteProvider)chunk);
                            break;
                        case ChunkTypes.OldPalette2:
                            chunk = new OldPaletteChunk2(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.Palette:
                            chunk = new PaletteChunk(chunkHeader.chunkSize);
                            paletteProvider = ((IPaletteProvider)chunk);
                            break;
                        case ChunkTypes.Path:
                            chunk = new PathChunk(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.Slice:
                            chunk = new SliceChunk(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.Tags:
                            chunk = new TagsChunk(chunkHeader.chunkSize);
                            break;
                        case ChunkTypes.Tileset:
                            chunk = new TilesetChunk(chunkHeader.chunkSize, file.colorDepth, paletteProvider?.entries, file.alphaPaletteEntry);
                            break;
                        case ChunkTypes.UserData:
                            chunk = new UserDataChunk(chunkHeader.chunkSize);
                            break;
                        default:
                            Debug.LogWarning($"Could not read chunk data with ID: {chunkHeader.chunkType}. Aborting.");
                            return;
                    }

                    var successful = chunk.Read(reader);
                    if (!successful)
                    {
                        frame.SetChunkData(m, new NoneChunk(0));
                        continue;
                    }

                    frame.SetChunkData(m, chunk);

                    if (chunk.chunkType == ChunkTypes.UserData)
                        AssociateUserDataWithChunk(frame.chunks, m, (UserDataChunk)chunk);
                }
            }
        }

        static void AssociateUserDataWithChunk(IReadOnlyList<BaseChunk> chunks, int index, UserDataChunk userData)
        {
            BaseChunk firstNonDataChunk = null;
            for (var i = index; i >= 0; --i)
            {
                if (chunks[i] != null && chunks[i] is not UserDataChunk)
                {
                    firstNonDataChunk = chunks[i];
                    break;
                }
            }

            if (firstNonDataChunk == null)
                return;

            switch (firstNonDataChunk.chunkType)
            {
                case ChunkTypes.Cell:
                    ((CellChunk)firstNonDataChunk).dataChunk = userData;
                    break;
            }
        }
    }
}
