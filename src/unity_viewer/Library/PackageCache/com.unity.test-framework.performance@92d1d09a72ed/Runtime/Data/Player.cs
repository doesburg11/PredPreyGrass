using System;

namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Represents player settings of a performance test, sampled at the beginning of a test run.
    /// </summary>
    [Serializable]
    public class Player
    {
        /// <summary>
        /// True if the player is a Development build. False if the player is a Release build.
        /// </summary>
        [RequiredMember] public bool Development;

        /// <summary>
        /// Screen resolution width.
        /// </summary>
        [RequiredMember] public int ScreenWidth;

        /// <summary>
        /// Screen resolution height.
        /// </summary>
        [RequiredMember] public int ScreenHeight;

        /// <summary>
        /// Screen refresh rate.
        /// </summary>
        [RequiredMember] public int ScreenRefreshRate;

        /// <summary>
        /// Whether the player is in fullscreen or windowed modes.
        /// </summary>
        [RequiredMember] public bool Fullscreen;

        /// <summary>
        /// The number of VSyncs that should pass between each frame. Use 'Don't Sync' (0) to not wait for VSync. Value must be 0, 1, 2, 3, or 4.
        /// </summary>
        [RequiredMember] public int Vsync;

        /// <summary>
        /// Anti aliasing.
        /// </summary>
        [RequiredMember] public int AntiAliasing;

        /// <summary>
        /// Whether player is in batchmode or not.
        /// </summary>
        [RequiredMember] public bool Batchmode;

        /// <summary>
        /// Render threading mode.
        /// </summary>
        [RequiredMember] public string RenderThreadingMode;

        /// <summary>
        /// Whether GPU skinning is enabled or not.
        /// </summary>
        [RequiredMember] public bool GpuSkinning;

        /// <summary>
        /// String representation of players RuntimePlatform.
        /// </summary>
        [RequiredMember] public string Platform;

        /// <summary>
        /// Color space.
        /// </summary>
        [RequiredMember] public string ColorSpace;

        /// <summary>
        /// Anisotropic filtering.
        /// </summary>
        [RequiredMember] public string AnisotropicFiltering;

        /// <summary>
        /// Blend weights.
        /// </summary>
        [RequiredMember] public string BlendWeights;

        /// <summary>
        /// Graphics API.
        /// </summary>
        [RequiredMember] public string GraphicsApi;

        // strings because their enums are editor only.
        /// <summary>
        /// Scripting backend.
        /// </summary>
        [RequiredMember] public string ScriptingBackend;
        /// <summary>
        /// Android target SDK version.
        /// </summary>
        [RequiredMember] public string AndroidTargetSdkVersion;
        /// <summary>
        /// Android build system.
        /// </summary>
        [RequiredMember] public string AndroidBuildSystem;
        /// <summary>
        /// Build target.
        /// </summary>
        [RequiredMember] public string BuildTarget;
        /// <summary>
        /// Stereo rendering path.
        /// </summary>
        [RequiredMember] public string StereoRenderingPath;
    }
}
