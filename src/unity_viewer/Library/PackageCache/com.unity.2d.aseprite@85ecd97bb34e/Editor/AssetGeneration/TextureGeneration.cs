using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEditor.AssetImporters;
using UnityEditor.U2D.Aseprite.Common;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal static class TextureGeneration
    {
        public static TextureGenerationOutput Generate(AssetImportContext ctx,
            NativeArray<Color32> imageData,
            int textureWidth,
            int textureHeight,
            in SpriteMetaData[] sprites,
            in List<TextureImporterPlatformSettings> allPlatformSettings,
            in TextureImporterSettings textureImporterSettings,
            string spritePackingTag,
            in SecondarySpriteTexture[] secondarySpriteTextures)
        {
            if (!imageData.IsCreated || imageData.Length == 0)
                return new TextureGenerationOutput();

            var output = new TextureGenerationOutput();
            UnityEngine.Profiling.Profiler.BeginSample("ImportTexture");
            try
            {
                var platformSettings = PlatformSettingsUtilities.GetPlatformTextureSettings(ctx.selectedBuildTarget, in allPlatformSettings);

                var textureSettings = textureImporterSettings.ExtractTextureSettings();
                textureSettings.assetPath = ctx.assetPath;
                textureSettings.enablePostProcessor = true;
                textureSettings.containsAlpha = true;
                textureSettings.hdr = false;

                var textureAlphaSettings = textureImporterSettings.ExtractTextureAlphaSettings();
                var textureMipmapSettings = textureImporterSettings.ExtractTextureMipmapSettings();
                var textureCubemapSettings = textureImporterSettings.ExtractTextureCubemapSettings();
                var textureWrapSettings = textureImporterSettings.ExtractTextureWrapSettings();

                switch (textureImporterSettings.textureType)
                {
                    case TextureImporterType.Default:
                        output = TextureGeneratorHelper.GenerateTextureDefault(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureAlphaSettings, textureMipmapSettings, textureCubemapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.NormalMap:
                        var textureNormalSettings = textureImporterSettings.ExtractTextureNormalSettings();
                        output = TextureGeneratorHelper.GenerateNormalMap(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureNormalSettings, textureMipmapSettings, textureCubemapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.GUI:
                        output = TextureGeneratorHelper.GenerateTextureGUI(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureAlphaSettings, textureMipmapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.Sprite:
                        var textureSpriteSettings = textureImporterSettings.ExtractTextureSpriteSettings();
                        textureSpriteSettings.packingTag = spritePackingTag;
                        textureSpriteSettings.qualifyForPacking = !string.IsNullOrEmpty(spritePackingTag);
                        textureSpriteSettings.spriteSheetData = new SpriteImportData[sprites.Length];
                        textureSettings.npotScale = TextureImporterNPOTScale.None;
                        textureSettings.secondaryTextures = secondarySpriteTextures;

                        for (var i = 0; i < sprites.Length; ++i)
                            textureSpriteSettings.spriteSheetData[i] = sprites[i];

                        output = TextureGeneratorHelper.GenerateTextureSprite(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureSpriteSettings, textureAlphaSettings, textureMipmapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.Cursor:
                        output = TextureGeneratorHelper.GenerateTextureCursor(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureAlphaSettings, textureMipmapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.Cookie:
                        output = TextureGeneratorHelper.GenerateCookie(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureAlphaSettings, textureMipmapSettings, textureCubemapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.Lightmap:
                        output = TextureGeneratorHelper.GenerateLightmap(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureMipmapSettings, textureWrapSettings);
                        break;
                    case TextureImporterType.SingleChannel:
                        output = TextureGeneratorHelper.GenerateTextureSingleChannel(imageData, textureWidth, textureHeight, textureSettings, platformSettings, textureAlphaSettings, textureMipmapSettings, textureCubemapSettings, textureWrapSettings);
                        break;
                    default:
                        Debug.LogAssertion("Unknown texture type for import");
                        output = default(TextureGenerationOutput);
                        break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Unable to generate Texture2D. Possibly texture size is too big to be generated. Error: {e}", ctx.mainObject);
            }
            finally
            {
                UnityEngine.Profiling.Profiler.EndSample();
            }

            return output;
        }
    }
}
