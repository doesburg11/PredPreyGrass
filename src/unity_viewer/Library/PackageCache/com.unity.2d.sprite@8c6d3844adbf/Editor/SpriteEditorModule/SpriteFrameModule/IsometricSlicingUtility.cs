using System.Collections.Generic;
using UnityEngine;
using UnityTexture2D = UnityEngine.Texture2D;

namespace UnityEditor.U2D.Sprites
{
    internal class IsometricSlicingUtility
    {
        private static bool PixelHasAlpha(int x, int y, int width, bool[] alphaPixelCache)
        {
            var index = y * width + x;
            return alphaPixelCache[index];
        }

        public static IEnumerable<Rect> GetIsometricRects(UnityTexture2D textureToUse, Vector2 size, Vector2 offset, bool isAlternate, bool keepEmptyRects)
        {
            var alphaPixelCache = new bool[textureToUse.width * textureToUse.height];
            Color32[] pixels = textureToUse.GetPixels32();
            for (int i = 0; i < pixels.Length; i++)
                alphaPixelCache[i] = pixels[i].a != 0;

            var gradient = (size.x / 2) / (size.y / 2);
            bool isAlt = isAlternate;
            float x = offset.x;
            if (isAlt)
                x += size.x / 2;
            float y = textureToUse.height - offset.y;
            while (y - size.y >= 0)
            {
                while (x + size.x <= textureToUse.width)
                {
                    var rect = new Rect(x, y - size.y, size.x, size.y);
                    if (!keepEmptyRects)
                    {
                        int sx = (int)rect.x;
                        int sy = (int)rect.y;
                        int width = (int)size.x;
                        int odd = ((int)size.y) % 2;
                        int topY = ((int)size.y / 2) - 1;
                        int bottomY = topY + odd;
                        int totalPixels = 0;
                        int alphaPixels = 0;
                        {
                            for (int ry = 0; ry <= topY; ry++)
                            {
                                var pixelOffset = Mathf.CeilToInt(gradient * ry);
                                for (int rx = pixelOffset; rx < width - pixelOffset; ++rx)
                                {
                                    if (PixelHasAlpha(sx + rx, sy + topY - ry, textureToUse.width, alphaPixelCache))
                                        alphaPixels++;
                                    if (PixelHasAlpha(sx + rx, sy + bottomY + ry, textureToUse.width, alphaPixelCache))
                                        alphaPixels++;
                                    totalPixels += 2;
                                }
                            }
                        }
                        if (odd > 0)
                        {
                            int ry = topY + 1;
                            for (int rx = 0; rx < size.x; ++rx)
                            {
                                if (PixelHasAlpha(sx + rx, sy + ry, textureToUse.width, alphaPixelCache))
                                    alphaPixels++;
                                totalPixels++;
                            }
                        }
                        if (totalPixels > 0 && ((float)alphaPixels) / totalPixels > 0.01f)
                            yield return rect;
                    }
                    else
                        yield return rect;
                    x += size.x;
                }
                isAlt = !isAlt;
                x = offset.x;
                if (isAlt)
                    x += size.x / 2;
                y -= size.y / 2;
            }
        }
    }
}
