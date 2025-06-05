using System.IO;
using System.IO.Compression;

namespace UnityEditor.U2D.Aseprite
{
    internal static class Zlib
    {
        public static byte[] Decompress(byte[] compressedData)
        {
            byte[] decompressedData;
            using (var decompressedStream = new MemoryStream())
            {
                using (var compressStream = new MemoryStream(compressedData))
                {
                    using (var deflateStream = new DeflateStream(compressStream, CompressionMode.Decompress))
                    {
                        deflateStream.CopyTo(decompressedStream);
                    }
                }

                decompressedData = decompressedStream.ToArray();
            }
            return decompressedData;
        }
    }
}
