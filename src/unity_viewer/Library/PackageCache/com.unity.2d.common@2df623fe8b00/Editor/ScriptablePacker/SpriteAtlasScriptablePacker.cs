using System.IO;
using UnityEngine;
using UnityEditor.U2D;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace UnityEditor.U2D.Common.SpriteAtlasPacker
{

    // Pixel Mask. Stores Rasterized Sprite Pixels.
    internal struct PixelMask
    {
        // Actual Size
        internal int2 size;
        // Border to minize search criteria.
        internal int4 rect;
        // Intermediate MinMax.
        internal int4 minmax;
        // Input Rect.
        internal int4 texrect;
        // Rasterized Texture Data.
        [NativeDisableContainerSafetyRestriction]
        internal NativeArray<byte> pixels;
    };

    // Atlas Masks. Stores Multiple Rasterized Sprite Pixels.
    internal struct AtlasMask
    {
        // Actual Size
        internal int2 size;
        // Border to minize search criteria.
        internal int4 rect;
        // Intermediate MinMax.
        internal int4 minmax;
        // Rasterized Texture Data.
        [NativeDisableContainerSafetyRestriction]
        internal NativeArray<byte> pixels;
    };

    // Internal Config params.
    internal struct UPackConfig
    {
        // Padding
        internal int padding;
        // Is Tight Packing. 1 for TIght.
        internal int packing;
        // Enable Rotation.
        internal int rotates;
        // Max Texture Size.
        internal int maxSize;
        // Block Offset.
        internal int bOffset;
        // Reserved.
        internal int freeBox;
        // Reserved.
        internal int jobSize;
        // Reserved.
        internal int sprSize;
    }

    [BurstCompile]
    internal struct UPack
    {

        ////////////////////////////////////////////////////////////////
        // Pixel Fetch.
        ////////////////////////////////////////////////////////////////

        internal static unsafe Color32* GetPixelOffsetBuffer(int offset, Color32* pixels)
        {
            return pixels + offset;
        }

        internal static unsafe Color32 GetPixel(Color32* pixels, ref int2 textureCfg, int x, int y)
        {
            int offset = x + (y * textureCfg.x);
            return *(pixels + offset);
        }

        internal static float Min3(float a, float b, float c)
        {
            var bc = math.min(b, c);
            return math.min(a, bc);
        }

        internal static byte Color32ToByte(Color32 rgba)
        {
            var r = (int)(rgba.r / 32);
            var g = (int)(rgba.g / 64);
            var b = (int)(rgba.b / 32);
            return (byte)(r | (g << 3) | (b << 5));
        }

        internal static Color32 ByteToColor32(byte rgb)
        {
            Color32 c = new Color32();
            int rgba = (int)rgb;
            c.r = (byte)((rgba & 0x00000007) * 32);
            c.g = (byte)(((rgba >> 3) & 0x00000003) * 64);
            c.b = (byte)(((rgba >> 5) & 0x00000007) * 32);
            c.a = ((int)c.r != 0 || (int)c.g != 0 || (int)c.b != 0) ? (byte)255 : (byte)0;
            return c;
        }

        ////////////////////////////////////////////////////////////////
        // Rasterization.
        ////////////////////////////////////////////////////////////////

        internal static float Max3(float a, float b, float c)
        {
            var bc = math.max(b, c);
            return math.max(a, bc);
        }

        internal static int Orient2d(float2 a, float2 b, float2 c)
        {
            return (int)((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
        }

        [BurstCompile]
        internal static unsafe void Pixelate(ref PixelMask pixelMask, ref int2 textureCfg, Color32* pixels, ref Color32 target, byte targetColor, int sx, int sy, int x, int y)
        {
            int _x = x - pixelMask.texrect.x;
            int _y = y - pixelMask.texrect.y;

#if DEBUGIMAGE
            Color32 src = GetPixel(pixels, ref textureCfg, sx, sy); // To debug with real colors.
            src = ( src.a != 0 ) ? src : target;
            pixelMask.pixels[(_y * pixelMask.size.x) + _x] = Color32ToByte(src);
#else
            pixelMask.pixels[(_y * pixelMask.size.x) + _x] = targetColor;
#endif

            pixelMask.minmax.x = math.min(_x, pixelMask.minmax.x);
            pixelMask.minmax.y = math.min(_y, pixelMask.minmax.y);
            pixelMask.minmax.z = math.max(_x, pixelMask.minmax.z);
            pixelMask.minmax.w = math.max(_y, pixelMask.minmax.w);
        }

        [BurstCompile]
        internal static unsafe void Pad(ref PixelMask pixelMask, ref Color32 tgtColor, byte tgtColorByte, int dx, int dy, int padx, int pady)
        {
            for (int y = -pady; y < pady; ++y)
            {
                for (int x = -padx; x < padx; ++x)
                {
                    int _x = math.min(math.max(dx + x, 0), pixelMask.size.x) - pixelMask.texrect.x;
                    int _y = math.min(math.max(dy + y, 0), pixelMask.size.y) - pixelMask.texrect.y;
                    if (_x < 0 || _y < 0 || _x > pixelMask.size.x || _y > pixelMask.size.y)
                        continue;

                    if (pixelMask.pixels[(_y * pixelMask.size.x) + _x] == 0)
                    {
#if DEBUGIMAGE
                        pixelMask.pixels[(_y * pixelMask.size.x) + _x] = Color32ToByte(tgtColor);
#else
                        pixelMask.pixels[(_y * pixelMask.size.x) + _x] = tgtColorByte;
#endif
                        pixelMask.minmax.x = math.min(_x, pixelMask.minmax.x);
                        pixelMask.minmax.y = math.min(_y, pixelMask.minmax.y);
                        pixelMask.minmax.z = math.max(_x, pixelMask.minmax.z);
                        pixelMask.minmax.w = math.max(_y, pixelMask.minmax.w);
                    }
                }
            }
        }

        [BurstCompile]
        internal static unsafe void RasterizeTriangle(ref UPackConfig cfg, ref PixelMask pixelMask, Color32* pixels, ref int2 textureCfg, ref Color32 srcColor, byte srcColorByte, ref float2 v0, ref float2 v1, ref float2 v2, int padx, int pady)
        {
            // Compute triangle bounding box
            int minX = (int)Min3(v0.x, v1.x, v2.x);
            int minY = (int)Min3(v0.y, v1.y, v2.y);
            int maxX = (int)Max3(v0.x, v1.x, v2.x);
            int maxY = (int)Max3(v0.y, v1.y, v2.y);
            var padColor = new Color32(64, 254, 64, 254);
            var padColorByte = Color32ToByte(padColor);

            // Clip against bounds
            minX = math.max(minX, 0);
            minY = math.max(minY, 0);
            maxX = math.min(maxX, pixelMask.rect.x - 1);
            maxY = math.min(maxY, pixelMask.rect.y - 1);

            // Triangle setup
            int A01 = (int)(v0.y - v1.y), B01 = (int)(v1.x - v0.x);
            int A12 = (int)(v1.y - v2.y), B12 = (int)(v2.x - v1.x);
            int A20 = (int)(v2.y - v0.y), B20 = (int)(v0.x - v2.x);

            // Barycentric coordinates at minX/minY corner
            float2 p = new float2(minX, minY);
            int w0_row = Orient2d(v1, v2, p);
            int w1_row = Orient2d(v2, v0, p);
            int w2_row = Orient2d(v0, v1, p);

            // Rasterize
            for (int py = minY; py <= maxY; ++py)
            {
                // Barycentric coordinates at start of row
                int w0 = w0_row;
                int w1 = w1_row;
                int w2 = w2_row;

                for (int px = minX; px <= maxX; ++px)
                {
                    // If p is on or inside all edges, render pixel.
                    if ((w0 | w1 | w2) >= 0)
                    {
                        int _padx = px + padx;
                        int _pady = py + pady;
                        Pixelate(ref pixelMask, ref textureCfg, pixels, ref srcColor, srcColorByte, px, py, _padx, _pady);
                        Pad(ref pixelMask, ref padColor, padColorByte, _padx, _pady, padx, pady);
                    }

                    // One step to the right
                    w0 += A12;
                    w1 += A20;
                    w2 += A01;
                }

                // One row step
                w0_row += B12;
                w1_row += B20;
                w2_row += B01;
            }
        }

        [BurstCompile]
        internal static unsafe bool Rasterize(ref UPackConfig cfg, Color32* pixels, ref int2 textureCfg, Vector2* vertices, int vertexCount, int* indices, int indexCount, ref PixelMask pixelMask, int padx, int pady)
        {
            var _v = float2.zero;
            var srcColor = new Color32(64, 64, 254, 254);
            var srcColorByte = Color32ToByte(srcColor);

            for (int i = 0; i < indexCount; i = i + 3)
            {
                int i1 = indices[i + 0];
                int i2 = indices[i + 1];
                int i3 = indices[i + 2];

                float2 v1 = vertices[i1];
                float2 v2 = vertices[i2];
                float2 v3 = vertices[i3];

                if (Orient2d(v1, v2, v3) < 0)
                {
                    _v = v1;
                    v1 = v2;
                    v2 = _v;
                }

                RasterizeTriangle(ref cfg, ref pixelMask, pixels, ref textureCfg, ref srcColor, srcColorByte, ref v1, ref v2, ref v3, padx, pady);
            }

            return true;
        }

        ////////////////////////////////////////////////////////////////
        // Rasterization.
        ////////////////////////////////////////////////////////////////

        [BurstCompile]
        internal unsafe struct SpriteRaster : IJob
        {
            // Pack Config
            public UPackConfig cfg;
            // Index to process.
            public int index;
            // Texture Input
            public int2 textureCfg;
            // Input Pixels
            [NativeDisableUnsafePtrRestriction]
            public Color32* pixels;
            // Vector2 positions.
            [NativeDisableUnsafePtrRestriction]
            public Vector2* vertices;
            // Vertex Count
            public int vertexCount;
            // Indices
            [NativeDisableUnsafePtrRestriction]
            public int* indices;
            // Index Count;
            public int indexCount;
            // SpriteRaster
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<PixelMask> spriteMasks;

            public void Execute()
            {

                // Rasterize Source Sprite.
                var spriteMask = spriteMasks[index];
                spriteMask.rect.z = spriteMask.rect.w = spriteMask.minmax.z = spriteMask.minmax.w = 0;
                spriteMask.rect.x = spriteMask.rect.y = spriteMask.minmax.x = spriteMask.minmax.y = cfg.sprSize;
                UPack.Rasterize(ref cfg, pixels, ref textureCfg, vertices, vertexCount, indices, indexCount, ref spriteMask, cfg.padding, cfg.padding);
                spriteMask.rect.x = math.max(0, spriteMask.minmax.x - cfg.padding);
                spriteMask.rect.y = math.max(0, spriteMask.minmax.y - cfg.padding);
                spriteMask.rect.z = math.min(cfg.maxSize, spriteMask.minmax.z + cfg.padding);
                spriteMask.rect.w = math.min(cfg.maxSize, spriteMask.minmax.w + cfg.padding);
                byte color = Color32ToByte(new Color32(254, 64, 64, 254));

                // If Tight packing fill Rect.
                if (0 == cfg.packing)
                {
                    for (int y = spriteMask.rect.y; y <= spriteMask.rect.w; ++y)
                    {
                        for (int x = spriteMask.rect.x; x <= spriteMask.rect.z; ++x)
                        {
                            spriteMask.pixels[y * spriteMask.size.x + x] = (spriteMask.pixels[y * spriteMask.size.x + x] != 0) ? spriteMask.pixels[y * spriteMask.size.x + x] : color;
                        }
                    }
                }

                spriteMasks[index] = spriteMask;

            }
        }

        ////////////////////////////////////////////////////////////////
        // Atlas Packing.
        ////////////////////////////////////////////////////////////////

        [BurstCompile]
        internal static bool TestMask(ref AtlasMask atlasMask, ref PixelMask spriteMask, int ax, int ay, int sx, int sy)
        {
            var satlasPixel = atlasMask.pixels[ay * atlasMask.size.x + ax];
            var spritePixel = spriteMask.pixels[sy * spriteMask.size.x + sx];
            return (spritePixel > 0 && satlasPixel > 0);
        }

        [BurstCompile]
        internal static unsafe bool TestMask(ref AtlasMask atlasMask, ref PixelMask spriteMask, int x, int y)
        {

            var spriteRect = spriteMask.rect;

            if (TestMask(ref atlasMask, ref spriteMask, (x), (y), spriteRect.x, spriteRect.y))
                return false;
            if (TestMask(ref atlasMask, ref spriteMask, (x), (y + (spriteRect.w - spriteRect.y)), spriteRect.x, spriteRect.y))
                return false;
            if (TestMask(ref atlasMask, ref spriteMask, (x + (spriteRect.z - spriteRect.x)), (y), spriteRect.z, spriteRect.w))
                return false;
            if (TestMask(ref atlasMask, ref spriteMask, (x + (spriteRect.z - spriteRect.x)), (y + (spriteRect.w - spriteRect.y)), spriteRect.z, spriteRect.w))
                return false;
            if (TestMask(ref atlasMask, ref spriteMask, (x), (y), spriteRect.z / 2, spriteRect.y / 2))
                return false;

            for (int j = spriteRect.y, _j = 0; j < spriteRect.w; ++j, ++_j)
            {
                for (int i = spriteRect.x, _i = 0; i < spriteRect.z; ++i, ++_i)
                {
                    if (TestMask(ref atlasMask, ref spriteMask, (_i + x), (_j + y), i, j))
                        return false;
                }
            }

            return true;
        }

        [BurstCompile]
        internal static void ApplyMask(ref UPackConfig cfg, ref AtlasMask atlasMask, ref PixelMask spriteMask, int ax, int ay, int sx, int sy)
        {
            var pixel = spriteMask.pixels[sy * spriteMask.size.x + sx];
            if (pixel != 0)
            {
                atlasMask.pixels[ay * atlasMask.size.x + ax] = pixel;
                atlasMask.minmax.x = math.min(ax, atlasMask.minmax.x);
                atlasMask.minmax.y = math.min(ay, atlasMask.minmax.y);
                atlasMask.minmax.z = math.max(ax, atlasMask.minmax.z);
                atlasMask.minmax.w = math.max(ay, atlasMask.minmax.w);
            }
        }

        [BurstCompile]
        internal static unsafe void ApplyMask(ref UPackConfig cfg, ref AtlasMask atlasMask, ref PixelMask spriteMask, int x, int y)
        {
            var spriteRect = spriteMask.rect;

            for (int j = spriteRect.y, _j = 0; j < spriteRect.w; ++j, ++_j)
            {
                for (int i = spriteRect.x, _i = 0; i < spriteRect.z; ++i, ++_i)
                {
                    ApplyMask(ref cfg, ref atlasMask, ref spriteMask, (_i + x), (_j + y), i, j);
                }
            }
        }

        ////////////////////////////////////////////////////////////////
        // Fit Sprite in a given RECT for Best Fit
        ////////////////////////////////////////////////////////////////

        [BurstCompile]
        internal struct SpriteFitter : IJob
        {

            // Cfg
            public UPackConfig config;
            // Test Inc
            public int4 atlasXInc;
            // Test Inc.
            public int4 atlasYInc;
            // Result Index.
            public int resultIndex;
            // AtlasMask
            public AtlasMask atlasMask;
            // SpriteMask
            public PixelMask spriteMask;
            // ResultSet
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int4> resultSet;

            public void Execute()
            {
                bool more = true;

                for (int y = atlasYInc.x; (more && y <= atlasYInc.y); y += atlasYInc.z)
                {
                    if (y + spriteMask.rect.w >= atlasMask.rect.y)
                        break;

                    for (int x = atlasXInc.x; (more && x <= atlasXInc.y); x += atlasXInc.z)
                    {
                        if (x + spriteMask.rect.z >= atlasMask.rect.x)
                            continue;

                        more = TestMask(ref atlasMask, ref spriteMask, x, y) == false;

                        if (!more)
                            resultSet[resultIndex] = new int4(x, y, more ? 0 : 1, 0);
                    }
                }
            }
        }

        ////////////////////////////////////////////////////////////////
        // Random Fit Only for Testing.
        ////////////////////////////////////////////////////////////////

        internal static unsafe bool RandomFit(ref UPackConfig cfg, ref NativeArray<SpriteFitter> fitterJob, ref NativeArray<JobHandle> fitterJobHandles, ref NativeArray<int4> resultArray, ref AtlasMask atlasMask, ref PixelMask spriteMask, ref int4 output)

        {
            bool more = true;
            int inc = math.min(atlasMask.rect.x, atlasMask.rect.y), rx = -1, ry = -1;

            int jobCount = 32;
            for (int i = 0; i < jobCount; ++i)
                fitterJobHandles[i] = default(JobHandle);

            System.Random rnd = new System.Random();
            while (more && (atlasMask.rect.x <= cfg.maxSize || atlasMask.rect.y <= cfg.maxSize))
            {
                int index = 0;
                int xrmax = atlasMask.minmax.z;
                int yrmax = atlasMask.minmax.w;

                // Random Search.
                {
                    int ix = atlasMask.minmax.z;
                    int iy = atlasMask.minmax.w;
                    UnsafeUtility.MemClear(resultArray.GetUnsafePtr(), resultArray.Length * sizeof(int4));
                    fitterJob[0] = new SpriteFitter() { atlasMask = atlasMask, spriteMask = spriteMask, atlasXInc = new int4(ix, ix, 1, 0), atlasYInc = new int4(iy, iy, 1, 0), resultSet = resultArray, resultIndex = 0 };
                    fitterJobHandles[0] = fitterJob[0].Schedule();

                    for (int i = 1; i < jobCount; ++i)
                    {
                        int x = atlasMask.minmax.z - rnd.Next(0, xrmax);
                        int y = atlasMask.minmax.w - rnd.Next(0, yrmax);
                        fitterJob[index] = new SpriteFitter() { atlasMask = atlasMask, spriteMask = spriteMask, atlasXInc = new int4(x, x, 1, 0), atlasYInc = new int4(y, y, 1, 0), resultSet = resultArray, resultIndex = i };
                        fitterJobHandles[index] = fitterJob[index].Schedule();
                        index++;
                    }
                    JobHandle.ScheduleBatchedJobs();
                    var jobHandle = JobHandle.CombineDependencies(fitterJobHandles);
                    jobHandle.Complete();

                    int area = atlasMask.size.x * atlasMask.size.y;
                    for (int j = 0; j < index; ++j)
                    {
                        if (resultArray[j].z == 1 && area > (rx * ry))
                        {
                            more = false;
                            area = rx * ry;
                            rx = resultArray[j].x;
                            ry = resultArray[j].y;
                        }
                    }

                    if (false == more)
                    {
                        ApplyMask(ref cfg, ref atlasMask, ref spriteMask, rx, ry);
                        break;
                    }

                }

                if (atlasMask.rect.x >= cfg.maxSize || atlasMask.rect.y >= cfg.maxSize)
                    break;
                atlasMask.rect.x = atlasMask.rect.y = math.min(cfg.maxSize, atlasMask.rect.y + inc);
            }

            output = new int4(rx, ry, 0, 0);
            return (rx != -1 && ry != -1);

        }

        ////////////////////////////////////////////////////////////////
        // Best Fit.
        ////////////////////////////////////////////////////////////////

        internal static unsafe bool BestFit(ref UPackConfig cfg, ref NativeArray<SpriteFitter> fitterJob, ref NativeArray<JobHandle> fitterJobHandles, ref NativeArray<int4> resultArray, ref AtlasMask atlasMask, ref PixelMask spriteMask, ref int4 output)

        {
            bool more = true;
            int inc = math.min(atlasMask.rect.x, atlasMask.rect.y), rx = -1, ry = -1;
            for (int i = 0; i < cfg.jobSize; ++i)
                fitterJobHandles[i] = default(JobHandle);

            while (more)
            {

                int index = 0;
                UnsafeUtility.MemClear(resultArray.GetUnsafePtr(), resultArray.Length * sizeof(int4));

                // Small Search.
                for (int y = 0; (y < atlasMask.rect.y); y += inc)
                {
                    fitterJob[index] = new SpriteFitter() { config = cfg, atlasMask = atlasMask, spriteMask = spriteMask, atlasXInc = new int4(0, atlasMask.rect.x, atlasMask.rect.z, 0), atlasYInc = new int4(y, y + inc, atlasMask.rect.w, 0), resultSet = resultArray, resultIndex = index };
                    fitterJobHandles[index] = fitterJob[index].Schedule();
                    index++;
                }
                JobHandle.ScheduleBatchedJobs();
                var jobHandle = JobHandle.CombineDependencies(fitterJobHandles);
                jobHandle.Complete();

                int area = atlasMask.size.x * atlasMask.size.y;
                for (int j = 0; j < index; ++j)
                {
                    if (resultArray[j].z == 1 && area > (resultArray[j].x * resultArray[j].y))
                    {
                        more = false;
                        rx = resultArray[j].x;
                        ry = resultArray[j].y;
                        area = rx * ry;
                    }
                }

                if (false == more)
                {
                    ApplyMask(ref cfg, ref atlasMask, ref spriteMask, rx, ry);
                    break;
                }

                if (atlasMask.rect.x >= cfg.maxSize && atlasMask.rect.y >= cfg.maxSize)
                {
                    // Either successful or need another page.
                    break;
                }
                else
                {
#if SQUAREINCR
                    atlasMask.rect.x = math.min(cfg.maxSize, atlasMask.rect.x * 2);
                    atlasMask.rect.y = math.min(cfg.maxSize, atlasMask.rect.y * 2);
#else
                    // Row Expansion first.
                    bool incY = (atlasMask.rect.y < atlasMask.rect.x);
                    atlasMask.rect.x = incY ? atlasMask.rect.x : math.min(cfg.maxSize, atlasMask.rect.x * 2);
                    atlasMask.rect.y = incY ? math.min(cfg.maxSize, atlasMask.rect.y * 2) : atlasMask.rect.y;
#endif
                }
            }

            output = new int4(rx, ry, 0, 0);
            return (rx != -1 && ry != -1);

        }

    }

    internal class SpriteAtlasScriptablePacker : UnityEditor.U2D.ScriptablePacker
    {

        static void DebugImage(NativeArray<byte> image, int w, int h, string path)
        {
#if DEBUGIMAGE
            var t = new Texture2D(w, h, UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SRGB, 0);
            var p = new NativeArray<Color32>(image.Length, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < image.Length; ++i)
                p[i] = UPack.ByteToColor32(image[i]);
            t.SetPixelData<Color32>(p, 0);
            byte[] _bytes = t.EncodeToPNG();
            System.IO.File.WriteAllBytes(path, _bytes);
#endif
        }

        static unsafe bool PrepareInput(UPackConfig cfg, int2 spriteSize, PackerData input)
        {

            for (int i = 0; i < input.spriteData.Length; ++i)
            {

                Color32* pixels = (Color32*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(input.colorData);
                var tsize = new Vector2Int(cfg.maxSize, cfg.maxSize);
                var inputSpriteC = input.spriteData[i];
                var textureDataC = input.textureData[inputSpriteC.texIndex];
                var spritePixels = UPack.GetPixelOffsetBuffer(textureDataC.bufferOffset, pixels);

                if (inputSpriteC.rect.x + inputSpriteC.rect.width > spriteSize.x || inputSpriteC.rect.y + inputSpriteC.rect.height > spriteSize.y)
                {
                    return false;
                }

                if (inputSpriteC.rect.width + (2 * cfg.padding) > cfg.maxSize || inputSpriteC.rect.height + (2 * cfg.padding) > cfg.maxSize)
                {
                    return false;
                }

#if DEBUGIMAGE
                var outputCoordX = 0;
                var outputCoordY = 0;
                var spriteOutput = new SpriteData();

                spriteOutput.texIndex = i;
                spriteOutput.guid = inputSpriteC.guid;
                spriteOutput.rect = new RectInt() { x = outputCoordX, y = outputCoordY, width = inputSpriteC.rect.width, height = inputSpriteC.rect.height };
                spriteOutput.output.x = 0;
                spriteOutput.output.y = 0;

                var atlasTexture = new NativeArray<Color32>(tsize.x * tsize.y, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

                for (int y = inputSpriteC.rect.y; y < (inputSpriteC.rect.y + inputSpriteC.rect.height); ++y)
                {
                    outputCoordX = 0;
                    var textureCfg = new int2(textureDataC.width, textureDataC.height);
                    for (int x = inputSpriteC.rect.x; x < (inputSpriteC.rect.x + inputSpriteC.rect.width); ++x)
                    {
                        Color32 color = UPack.GetPixel(spritePixels, ref textureCfg, x, y);
                        int outOffset = outputCoordX + (outputCoordY * tsize.y);
                        atlasTexture[outOffset] = color;
                        outputCoordX++;
                    }
                    outputCoordY++;
                }

                {
                    Texture2D t = new Texture2D(cfg.maxSize, cfg.maxSize, UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SRGB, 0);
                    t.SetPixelData<Color32>(atlasTexture, 0);
                    byte[] _bytes = UnityEngine.ImageConversion.EncodeToPNG(t);
                    System.IO.File.WriteAllBytes(Path.Combine(Application.dataPath, "../") + "Temp/" + "Input" + i + ".png", _bytes);
                }

                atlasTexture.Dispose();
#endif

            }

            return true;

        }

        public override bool Pack(SpriteAtlasPackingSettings config, SpriteAtlasTextureSettings setting, PackerData input)
        {

            var cfg = new UPackConfig();
            var quality = 3;
            var startRect = 128;

            var c = new Color32(32, 64, 128, 255);
            var b = UPack.Color32ToByte(c);
            var d = UPack.ByteToColor32(b);

            cfg.padding = config.padding;
            cfg.bOffset = config.blockOffset * (1 << (int)quality);
            cfg.maxSize = setting.maxTextureSize;
            cfg.rotates = config.enableRotation ? 1 : 0;
            cfg.packing = config.enableTightPacking ? 1 : 0;
            cfg.freeBox = cfg.bOffset;
            cfg.jobSize = 1024;
            cfg.sprSize = 2048;

            var spriteCount = input.spriteData.Length;
            var spriteBatch = math.min(spriteCount, SystemInfo.processorCount);

            // Because Atlas Masks are Serial / Raster in Jobs.
            var atlasCount = 0;
            var spriteSize = new int2(cfg.sprSize, cfg.sprSize);
            var validAtlas = true;

            // Rasterization.
            NativeArray<AtlasMask> atlasMasks = new NativeArray<AtlasMask>(spriteCount, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            NativeArray<PixelMask> spriteMasks = new NativeArray<PixelMask>(spriteBatch, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            var rasterJobHandles = new NativeArray<JobHandle>(spriteBatch, Allocator.Persistent);
            var rasterJob = new NativeArray<UPack.SpriteRaster>(spriteBatch, Allocator.Persistent);

            // PolygonFitting
            var fitterJobHandles = new NativeArray<JobHandle>(cfg.jobSize, Allocator.Persistent);
            var fitterJob = new NativeArray<UPack.SpriteFitter>(cfg.jobSize, Allocator.Persistent);
            var fitterResult = new NativeArray<int4>(cfg.jobSize, Allocator.Persistent);

            // Initialize Batch Sprite Masks.
            for (int i = 0; i < spriteBatch; ++i)
            {

                PixelMask spriteMask = new PixelMask();
                spriteMask.pixels = new NativeArray<byte>(spriteSize.x * spriteSize.y, Allocator.Persistent, NativeArrayOptions.ClearMemory);
                spriteMask.size = spriteSize;
                spriteMask.rect = int4.zero;
                spriteMask.minmax = new int4(spriteSize.x, spriteSize.y, 0, 0);
                spriteMasks[i] = spriteMask;

            }

            unsafe
            {

                // Prepare.
                bool v = PrepareInput(cfg, spriteSize, input);
                if (!v)
                    return false;

                // Copy back to Processing Data
                for (int batch = 0; batch < spriteCount; batch += spriteBatch)
                {

                    var spriteBgn = batch;
                    var spriteEnd = math.min(spriteCount, spriteBgn + spriteBatch);
                    int index = 0;

                    for (int i = spriteBgn; i < spriteEnd; ++i)
                    {
                        var inputSprite = input.spriteData[i];
                        var textureData = input.textureData[inputSprite.texIndex];

                        // Clear Mem of SpriteMask.
                        var spriteMask = spriteMasks[index];
                        UnsafeUtility.MemClear(spriteMask.pixels.GetUnsafePtr(), ((spriteMask.rect.w * spriteMask.size.x) + spriteMask.rect.z) * UnsafeUtility.SizeOf<Color32>());
                        spriteMask.size = spriteSize;
                        spriteMask.rect = int4.zero;
                        spriteMask.minmax = new int4(spriteSize.x, spriteSize.y, 0, 0);
                        spriteMask.texrect = new int4(inputSprite.rect.x, inputSprite.rect.y, inputSprite.rect.width, inputSprite.rect.height);
                        spriteMasks[index] = spriteMask;

                        unsafe
                        {
                            rasterJob[index] = new UPack.SpriteRaster()
                            {
                                cfg = cfg,
                                index = index,
                                pixels = (Color32*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(input.colorData) + textureData.bufferOffset,
                                vertices = (Vector2*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(input.vertexData) + inputSprite.vertexOffset,
                                vertexCount = inputSprite.vertexCount,
                                indices = (int*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(input.indexData) + inputSprite.indexOffset,
                                indexCount = inputSprite.indexCount,
                                textureCfg = new int2(textureData.width, textureData.height),
                                spriteMasks = spriteMasks
                            };
                        }
                        rasterJobHandles[index] = rasterJob[index].Schedule();
                        index++;
                    }

                    JobHandle.ScheduleBatchedJobs();
                    var jobHandle = JobHandle.CombineDependencies(rasterJobHandles);
                    jobHandle.Complete();
                    index = 0;

                    for (int sprite = spriteBgn; sprite < spriteEnd; ++sprite)
                    {

                        var inputSpriteC = input.spriteData[sprite];
                        // Rasterize Source Sprite.
                        var spriteMask = spriteMasks[index];

                        int page = -1;
                        validAtlas = false;
                        var result = int4.zero;
                        for (int i = (atlasCount - 1); i >= 0 && false == validAtlas; --i)
                        {
                            var atlasMask = atlasMasks[i];
                            validAtlas = UPack.BestFit(ref cfg, ref fitterJob, ref fitterJobHandles, ref fitterResult, ref atlasMask, ref spriteMask, ref result);
                            if (validAtlas)
                            {
                                atlasMasks[i] = atlasMask;
                                page = i;
                            }
                        }

                        // Test
                        if (!validAtlas)
                        {
                            page = atlasCount;
                            AtlasMask atlasMask = new AtlasMask();
                            atlasMask.pixels = new NativeArray<byte>(cfg.maxSize * cfg.maxSize, Allocator.Persistent, NativeArrayOptions.ClearMemory);
                            atlasMask.size = new int2(cfg.maxSize, cfg.maxSize);
                            atlasMask.rect.x = atlasMask.rect.y = startRect;
                            atlasMask.rect.z = atlasMask.rect.w = cfg.bOffset;
                            validAtlas = UPack.BestFit(ref cfg, ref fitterJob, ref fitterJobHandles, ref fitterResult, ref atlasMask, ref spriteMask, ref result);
                            atlasMasks[atlasCount] = atlasMask;
                            atlasCount++;
                        }

                        if (!validAtlas)
                        {
                            break;
                        }

                        // Clear Mem of SpriteMask.
                        DebugImage(spriteMask.pixels, cfg.maxSize, cfg.maxSize, Path.Combine(Application.dataPath, "../") + "Temp/" + "Input" + sprite + ".png");
                        UnsafeUtility.MemClear(spriteMask.pixels.GetUnsafePtr(), ((spriteMask.rect.w * spriteMask.size.x) + spriteMask.rect.z) * UnsafeUtility.SizeOf<Color32>());

                        inputSpriteC.output.x = result.x;
                        inputSpriteC.output.y = result.y;
                        inputSpriteC.output.page = validAtlas ? page : -1;
                        input.spriteData[sprite] = inputSpriteC;
                        index++;
                    }

                    if (!validAtlas)
                    {
                        break;
                    }

                }
                for (int j = 0; j < atlasCount; ++j)
                    DebugImage(atlasMasks[j].pixels, cfg.maxSize, cfg.maxSize, Path.Combine(Application.dataPath, "../") + "Temp/" + "Packer" + j + ".png");

                // If there is an error fallback
                if (!validAtlas)
                {
                    for (int i = 0; i < spriteCount; ++i)
                    {
                        var inputSpriteC = input.spriteData[i];
                        inputSpriteC.output.x = inputSpriteC.output.y = 0;
                        inputSpriteC.output.page = -1;
                        input.spriteData[i] = inputSpriteC;
                    }
                }

                for (int j = 0; j < spriteBatch; ++j)
                    spriteMasks[j].pixels.Dispose();
                for (int j = 0; j < atlasCount; ++j)
                    atlasMasks[j].pixels.Dispose();
                atlasMasks.Dispose();
                spriteMasks.Dispose();

                rasterJob.Dispose();
                rasterJobHandles.Dispose();

                fitterJob.Dispose();
                fitterJobHandles.Dispose();

            }
            return true;

        }

    }

}