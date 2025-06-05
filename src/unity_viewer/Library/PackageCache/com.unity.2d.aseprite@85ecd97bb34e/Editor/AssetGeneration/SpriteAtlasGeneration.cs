using System.Collections.Generic;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.U2D;
using UnityEditor.U2D.Aseprite.Common;

namespace UnityEditor.U2D.Aseprite
{
    internal static class SpriteAtlasGeneration
    {
        public static void Generate(
            AssetImportContext ctx,
            AsepriteImporter importer,
            string assetName)
        {
            var atlas = CreateDefaultSpriteAtlas();
            var textures = GetTextures(ctx);

            var texturesToPack = new Object[textures.Count];
            for (var i = 0; i < textures.Count; ++i)
                texturesToPack[i] = textures[i];
            atlas.Add(texturesToPack);
            atlas.name = $"{assetName}_Atlas";
            
            InternalEditorBridge.RegisterAndPackSpriteAtlas(atlas, ctx, importer, null);
            ctx.AddObjectToAsset(atlas.name, atlas);
        }
        
        static SpriteAtlas CreateDefaultSpriteAtlas()
        {
            var spriteAtlas = new SpriteAtlas();
            InternalEditorBridge.SetSpriteAtlasToV2(spriteAtlas);
            spriteAtlas.SetTextureSettings(new SpriteAtlasTextureSettings()
            {
                filterMode = FilterMode.Point,
                anisoLevel = 0,
                generateMipMaps = false,
                sRGB = true,
            });

            var platformSettings = new TextureImporterPlatformSettings();
            PlatformSettingsUtilities.SetupPlatformSettingsWithDefaultVales(platformSettings);
            spriteAtlas.SetPlatformSettings(platformSettings);
            return spriteAtlas;
        }  
        
        static List<Texture2D> GetTextures(AssetImportContext ctx)
        {
            var assetObjects = new List<Object>();
            ctx.GetObjects(assetObjects);

            var textures = new List<Texture2D>();
            foreach (var obj in assetObjects)
            {
                if (obj is Texture2D texture)
                    textures.Add(texture);
            }
            return textures;            
        }        
    }
}