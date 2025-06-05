using UnityEngine;
using System;

namespace UnityEditor.U2D.Aseprite.Common
{
    internal interface ITexturePlatformSettingsDataProvider
    {
        bool textureTypeHasMultipleDifferentValues { get; }
        TextureImporterType textureType { get; }
        SpriteImportMode spriteImportMode { get; }

        int GetTargetCount();
        TextureImporterPlatformSettings GetPlatformTextureSettings(int i, string name);
        bool ShowPresetSettings();
        bool DoesSourceTextureHaveAlpha(int v);
        bool IsSourceTextureHDR(int v);
        void SetPlatformTextureSettings(int i, TextureImporterPlatformSettings platformSettings);
        void GetImporterSettings(int i, TextureImporterSettings settings);
        string GetBuildTargetName(SerializedProperty sp);
        SerializedProperty platformSettingsArray { get; }
    }

    [Serializable]
    internal class TexturePlatformSettings : BaseTextureImportPlatformSettings
    {
        [SerializeField]
        TextureImportPlatformSettingsData m_Data = new();
        ITexturePlatformSettingsDataProvider m_DataProvider;
        Func<BaseTextureImportPlatformSettings> DefaultImportSettings;

        public override TextureImportPlatformSettingsData model => m_Data;

        public TexturePlatformSettings(string name, BuildTarget target, ITexturePlatformSettingsDataProvider inspector, Func<BaseTextureImportPlatformSettings> defaultPlatform)
            : base(name, target)
        {
            m_DataProvider = inspector;
            DefaultImportSettings = defaultPlatform;
            Init();
            CacheSerializedProperties(inspector.platformSettingsArray);
        }

        public void CacheSerializedProperties(SerializedProperty platformSettingsArray)
        {
            if (model.platformTextureSettingsProp != null && model.platformTextureSettingsProp.isValid && m_DataProvider.GetBuildTargetName(model.platformTextureSettingsProp) == model.platformTextureSettings.name)
                return;
            
            if (platformSettingsArray.arraySize == 0)
            {
                model.platformTextureSettingsProp = null;
                throw new UnityException("Cannot find any Platform Settings, including the Default Platform. This is incorrect, did initialization fail?");
            }

            for (var index = 0; index < platformSettingsArray.arraySize; ++index)
            {
                var arrayElementAtIndex = platformSettingsArray.GetArrayElementAtIndex(index);
                if (m_DataProvider.GetBuildTargetName(arrayElementAtIndex) == model.platformTextureSettings.name)
                {
                    model.platformTextureSettingsProp = arrayElementAtIndex;
                    break;
                }
            }

            model.alphaSplitProperty = model.platformTextureSettingsProp != null ? model.platformTextureSettingsProp.FindPropertyRelative("m_AllowsAlphaSplitting") : throw new UnityException(string.Format("Could not find the requested Platform Texture Settings {0}. This is incorrect, did initialization fail?", model.platformTextureSettings.name));
            model.androidETC2FallbackOverrideProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_AndroidETC2FallbackOverride");
            model.compressionQualityProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_CompressionQuality");
            model.crunchedCompressionProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_CrunchedCompression");
            model.maxTextureSizeProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_MaxTextureSize");
            model.overriddenProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_Overridden");
            model.resizeAlgorithmProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_ResizeAlgorithm");
            model.textureCompressionProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_TextureCompression");
            model.textureFormatProperty = model.platformTextureSettingsProp.FindPropertyRelative("m_TextureFormat");
        }

        public override bool textureTypeHasMultipleDifferentValues => m_DataProvider.textureTypeHasMultipleDifferentValues;
        public override TextureImporterType textureType => m_DataProvider.textureType;
        public override SpriteImportMode spriteImportMode => m_DataProvider.spriteImportMode;

        public override int GetTargetCount()
        {
            return m_DataProvider.GetTargetCount();
        }

        public override bool ShowPresetSettings()
        {
            return m_DataProvider.ShowPresetSettings();
        }

        public override TextureImporterSettings GetImporterSettings(int i)
        {
            var textureImporterSettings = new TextureImporterSettings();
            m_DataProvider.GetImporterSettings(i, textureImporterSettings);
            return textureImporterSettings;
        }

        public override bool IsSourceTextureHDR(int i)
        {
            return m_DataProvider.IsSourceTextureHDR(i);
        }

        public override bool DoesSourceTextureHaveAlpha(int i)
        {
            return m_DataProvider.DoesSourceTextureHaveAlpha(i);
        }

        public override TextureImporterPlatformSettings GetPlatformTextureSettings(int i, string name)
        {
            var temp = new TextureImporterPlatformSettings();
            m_DataProvider.GetPlatformTextureSettings(i, name).CopyTo(temp);
            return temp;
        }

        public override BaseTextureImportPlatformSettings GetDefaultImportSettings()
        {
            return DefaultImportSettings();
        }

        public override void SetPlatformTextureSettings(int i, TextureImporterPlatformSettings platformSettings)
        {
            platformSettings.name = GetFixedPlatformName(platformSettings.name);
            if (!model.overriddenIsDifferent)
                model.overriddenProperty.boolValue = model.platformTextureSettings.overridden;
            if (!model.textureFormatIsDifferent)
                model.textureFormatProperty.intValue = (int)model.platformTextureSettings.format;
            if (!model.maxTextureSizeIsDifferent)
                model.maxTextureSizeProperty.intValue = model.platformTextureSettings.maxTextureSize;
            if (!model.resizeAlgorithmIsDifferent)
                model.resizeAlgorithmProperty.intValue = (int)model.platformTextureSettings.resizeAlgorithm;
            if (!model.textureCompressionIsDifferent)
                model.textureCompressionProperty.intValue = (int)model.platformTextureSettings.textureCompression;
            if (!model.compressionQualityIsDifferent)
                model.compressionQualityProperty.intValue = model.platformTextureSettings.compressionQuality;
            if (!model.crunchedCompressionIsDifferent)
                model.crunchedCompressionProperty.boolValue = model.platformTextureSettings.crunchedCompression;
            if (!model.allowsAlphaSplitIsDifferent)
                model.alphaSplitProperty.boolValue = model.platformTextureSettings.allowsAlphaSplitting;
            if (!model.androidETC2FallbackOverrideIsDifferent)
                model.androidETC2FallbackOverrideProperty.intValue = (int)model.platformTextureSettings.androidETC2FallbackOverride;
            m_DataProvider.SetPlatformTextureSettings(i, platformSettings);
        }

        string GetFixedPlatformName(string platform)
        {
            var targetGroup = BuildPipeline.GetBuildTargetGroupByName(platform);
            if (targetGroup != BuildTargetGroup.Unknown)
                return BuildPipeline.GetBuildTargetGroupName(targetGroup);
            return platform;
        }
    }
}
