namespace UnityEditor.U2D.Aseprite
{
    internal static class TexturePlatformSettingsModal
    {
        public struct BuildPlatformData
        {
            public string buildTargetName;
            public TextureImporterFormat defaultTextureFormat;
            public BuildTarget[] buildTarget;
        }

        // Add new platforms here
        public static readonly BuildPlatformData[] validBuildPlatform = new BuildPlatformData[]
        {
            new BuildPlatformData()
            {
                buildTargetName = "Default",
                defaultTextureFormat = TextureImporterFormat.Automatic,
                buildTarget = new[]
                {
                    BuildTarget.NoTarget
                }
            },

            new BuildPlatformData()
            {
                buildTargetName = "PC, Mac & Linux Standalone",
                defaultTextureFormat = TextureImporterFormat.RGBA32,
                buildTarget = new[]
                {
                    BuildTarget.StandaloneWindows,
                    BuildTarget.StandaloneWindows64,
                    BuildTarget.StandaloneLinux64,
                    BuildTarget.StandaloneOSX,
                }
            },
            new BuildPlatformData()
            {
                buildTargetName = "iOS",
                defaultTextureFormat = TextureImporterFormat.RGBA32,
                buildTarget = new[] { BuildTarget.iOS }
            },
            new BuildPlatformData()
            {
                buildTargetName = "tvOS",
                defaultTextureFormat = TextureImporterFormat.RGBA32,
                buildTarget = new[] { BuildTarget.tvOS }
            },
            new BuildPlatformData()
            {
                buildTargetName = "Android",
                defaultTextureFormat = TextureImporterFormat.RGBA32,
                buildTarget = new[] { BuildTarget.Android }
            },
            new BuildPlatformData()
            {
                buildTargetName = "Universal Windows Platform",
                defaultTextureFormat = TextureImporterFormat.RGBA32,
                buildTarget = new[] { BuildTarget.WSAPlayer }
            },
        };

        static TexturePlatformSettingsModal() { }
    }
}
