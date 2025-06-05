using System.Collections.Generic;
using UnityEditor.U2D.Aseprite.Common;

namespace UnityEditor.U2D.Aseprite
{
    internal static class PlatformSettingsUtilities
    {
        public static TextureImporterPlatformSettings GetPlatformTextureSettings(BuildTarget buildTarget, in List<TextureImporterPlatformSettings> platformSettings)
        {
            var buildTargetName = TexturePlatformSettingsHelper.GetBuildTargetGroupName(buildTarget);
            TextureImporterPlatformSettings settings = null;
            foreach (var platformSetting in platformSettings)
            {
                if (platformSetting.name == buildTargetName && platformSetting.overridden)
                    settings = platformSetting;
                else if (platformSetting.name == TexturePlatformSettingsHelper.defaultPlatformName)
                    settings = platformSetting;
            }

            if (settings != null)
                return settings;
            return CreateDefaultSettings(buildTargetName);
        }

        public static TextureImporterPlatformSettings CreateDefaultSettings(string buildTargetName)
        {
            var settings = new TextureImporterPlatformSettings();
            settings.name = buildTargetName;
            settings.overridden = false;

            SetupPlatformSettingsWithDefaultVales(settings);
            
            return settings;
        }

        public static void SetupPlatformSettingsWithDefaultVales(TextureImporterPlatformSettings settings)
        {
            // Default settings
            settings.textureCompression = TextureImporterCompression.Uncompressed;
            settings.maxTextureSize = 16384;
        }
    }
}
