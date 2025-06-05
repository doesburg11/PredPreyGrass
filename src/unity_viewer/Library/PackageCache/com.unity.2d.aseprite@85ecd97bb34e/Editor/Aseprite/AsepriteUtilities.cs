using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal static class AsepriteUtilities
    {
        public static string ReadString(BinaryReader reader)
        {
            var strLength = reader.ReadUInt16();
            var text = "";
            for (var m = 0; m < strLength; ++m)
            {
                var character = (char)reader.ReadByte();
                text += character;
            }
            return text;
        }

        public static byte[] ReadAndDecompressedData(BinaryReader reader, int dataLength)
        {
            // 2 bytes of Rfc1950Header that we do not want
            var magicBytes = reader.ReadBytes(2);

            var compressedData = reader.ReadBytes(dataLength - 2);
            var decompressedData = Zlib.Decompress(compressedData);
            return decompressedData;
        }

        public static NativeArray<Color32> GenerateImageData(ushort colorDepth, byte[] imageData, ReadOnlyCollection<PaletteEntry> paletteEntries, byte alphaPaletteEntry)
        {
            if (colorDepth == 32 || colorDepth == 16)
                return ByteToColorArray(imageData, colorDepth);
            if (colorDepth == 8)
                return ByteToColorArrayUsingPalette(imageData, paletteEntries, alphaPaletteEntry);
            return default;
        }

        static NativeArray<Color32> ByteToColorArray(in byte[] data, ushort colorDepth)
        {
            NativeArray<Color32> image = default;
            if (colorDepth == 32)
            {
                image = new NativeArray<Color32>(data.Length / 4, Allocator.Persistent);
                var imageLength = image.Length;
                unsafe
                {
                    var imageArr = (Color32*)image.GetUnsafePtr();
                    for (var i = 0; i < imageLength; ++i)
                    {
                        var dataIndex = i * 4;
                        imageArr[i] = new Color32(
                            data[dataIndex],
                            data[dataIndex + 1],
                            data[dataIndex + 2],
                            data[dataIndex + 3]);
                    }
                }
            }
            else if (colorDepth == 16)
            {
                image = new NativeArray<Color32>(data.Length / 2, Allocator.Persistent);
                var imageLength = image.Length;
                unsafe
                {
                    var imageArr = (Color32*)image.GetUnsafePtr();
                    for (var i = 0; i < imageLength; ++i)
                    {
                        var dataIndex = i * 2;
                        var value = data[dataIndex];
                        var alpha = data[dataIndex + 1];
                        imageArr[i] = new Color32(value, value, value, alpha);
                    }
                }
            }
            return image;
        }

        static NativeArray<Color32> ByteToColorArrayUsingPalette(in byte[] data, ReadOnlyCollection<PaletteEntry> paletteEntries, byte alphaPaletteEntry)
        {
            NativeArray<Color32> image = default;
            if (paletteEntries == null || paletteEntries.Count == 0)
                return default;

            var alphaColor = new Color32(0, 0, 0, 0);

            image = new NativeArray<Color32>(data.Length, Allocator.Persistent);
            for (var i = 0; i < image.Length; ++i)
            {
                var paletteIndex = data[i];
                if (paletteIndex != alphaPaletteEntry && paletteIndex < paletteEntries.Count)
                {
                    var entry = paletteEntries[paletteIndex];
                    image[i] = entry.color;
                }
                else
                    image[i] = alphaColor;
            }

            return image;
        }
        
        public static bool Contains<T>(this IReadOnlyCollection<T> list, T item)
        {
            foreach (var listItem in list)
            {
                if (listItem.Equals(item))
                    return true;
            }
            return false;
        }     
        
        public static int FindIndex<T>(this IReadOnlyCollection<T> list, Predicate<T> match)
        {
            var index = 0;
            foreach (var listItem in list)
            {
                if (match(listItem))
                    return index;
                index++;
            }
            return -1;
        }
        
        public static T Find<T>(this IReadOnlyCollection<T> list, Predicate<T> match)
        {
            foreach (var listItem in list)
            {
                if (match(listItem))
                    return listItem;
            }
            return default;
        }
    }
}
