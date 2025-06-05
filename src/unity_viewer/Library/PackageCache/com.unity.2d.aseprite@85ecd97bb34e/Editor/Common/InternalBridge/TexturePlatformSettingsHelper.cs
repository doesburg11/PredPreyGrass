using System.Collections.Generic;
using UnityEngine;
using UnityEditor.Build;
using System; 

namespace UnityEditor.U2D.Aseprite.Common
{
    [Serializable]
    internal class TexturePlatformSettingsHelper
    {
        public delegate TextureImporterPlatformSettings CreateDefaultTextureImporterPlatformSettings(string platformName);
        
        [SerializeField] List<TexturePlatformSettings> m_PlatformSettings;
        ITexturePlatformSettingsDataProvider m_DataProvider;

        public static string defaultPlatformName => TextureImporterInspector.s_DefaultPlatformName;

        BaseTextureImportPlatformSettings GetDefaultTextureImportPlatformSettings() => m_PlatformSettings[0];

        internal static List<TextureImporterPlatformSettings> PlatformSettingsNeeded(CreateDefaultTextureImporterPlatformSettings createDefaultSettingsFunc)
        {
            var validPlatforms = BaseTextureImportPlatformSettings.GetBuildPlayerValidPlatforms();

            var platformSettings = new List<TextureImporterPlatformSettings>
            {
                createDefaultSettingsFunc(TextureImporterInspector.s_DefaultPlatformName)
            };

            foreach (var bp in validPlatforms)
            {
                var settings = createDefaultSettingsFunc.Invoke(bp.name);
                platformSettings.Add(settings);
            }

            return platformSettings;
        }

        public TexturePlatformSettingsHelper(ITexturePlatformSettingsDataProvider dataProvider)
        {
            m_DataProvider = dataProvider;
            var validPlatforms = BaseTextureImportPlatformSettings.GetBuildPlayerValidPlatforms();

            m_PlatformSettings = new List<TexturePlatformSettings>
            {
                CreateDefaultSettings(dataProvider, GetDefaultTextureImportPlatformSettings)
            };

            foreach (var bp in validPlatforms)
            {
                m_PlatformSettings.Add(new TexturePlatformSettings(bp.name, bp.defaultTarget, dataProvider,
                    GetDefaultTextureImportPlatformSettings));
            }
        }

        static TexturePlatformSettings CreateDefaultSettings(ITexturePlatformSettingsDataProvider dataProvider,
            Func<BaseTextureImportPlatformSettings> defaultPlatformFunc)
        {
            return new TexturePlatformSettings(TextureImporterInspector.s_DefaultPlatformName,
                BuildTarget.StandaloneWindows, dataProvider, defaultPlatformFunc);
        }

        public void ShowPlatformSpecificSettings()
        {
            var texturePlatformSettings = m_PlatformSettings.ConvertAll<BaseTextureImportPlatformSettings>(x => x);

            BaseTextureImportPlatformSettings.InitPlatformSettings(texturePlatformSettings);
            m_PlatformSettings.ForEach(settings =>
                settings.CacheSerializedProperties(m_DataProvider.platformSettingsArray));

            //Show platform grouping
            var selectedPage = EditorGUILayout.BeginPlatformGrouping(
                BaseTextureImportPlatformSettings.GetBuildPlayerValidPlatforms(),
                EditorGUIUtility.TrTextContent("Default"), EditorStyles.frameBox, idx =>
                {
                    var ps = m_PlatformSettings[idx + 1];
                    var model = ps.model;
                    if (model.isDefault)
                        return false;
                    return model.overriddenIsDifferent || model.allAreOverridden;
                });

            //Show platform settings
            using var changed = new EditorGUI.ChangeCheckScope();

            BaseTextureImportPlatformSettings.ShowPlatformSpecificSettings(texturePlatformSettings, selectedPage);
            // Doing it this way is slow, but it ensures Presets get updated correctly whenever the UI is being changed.
            if (changed.changed)
            {
                BaseTextureImportPlatformSettings.ApplyPlatformSettings(texturePlatformSettings);
            }
        }

        public bool HasModified()
        {
            foreach (var ps in m_PlatformSettings)
            {
                if (ps.model.HasChanged())
                    return true;
            }

            return false;
        }

        void SyncPlatformSettings()
        {
            foreach (var ps in m_PlatformSettings)
                ps.Sync();
        }

        public void Apply()
        {
            foreach (var ps in m_PlatformSettings)
                ps.Apply();
        }

        public static string GetBuildTargetGroupName(BuildTarget target)
        {
            var targetGroup = BuildPipeline.GetBuildTargetGroup(target);
            foreach (var bp in BuildPlatforms.instance.buildPlatforms)
            {
                if (bp.targetGroup == targetGroup)
                    return bp.name;
            }

            return TextureImporter.defaultPlatformName;
        }
    }
}