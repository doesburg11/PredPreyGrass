using UnityEngine;
using UnityEditor;
using System.IO;

#if UNITY_EDITOR
// Ensure class initializer is called whenever scripts recompile
[InitializeOnLoad]
internal static class CreateAssetBundle
{
    // Register an event handler when the class is initialized
    static CreateAssetBundle()
    {
        EditorApplication.playModeStateChanged += PlayModeStateChange;
    }

    static void PlayModeStateChange(PlayModeStateChange state)
    {
        if (state == UnityEditor.PlayModeStateChange.ExitingEditMode)
            CreateAssetBundles();
    }

    static void CreateAssetBundles()
    {
#if ASSETBUNDLE_ENABLED
        string assetBundleDirectory = "Assets/StreamingAssets";
        if (!Directory.Exists(Application.streamingAssetsPath))
        {
            Directory.CreateDirectory(assetBundleDirectory);
        }
        BuildPipeline.BuildAssetBundles(assetBundleDirectory, BuildAssetBundleOptions.None, EditorUserBuildSettings.activeBuildTarget);
#endif
    }
}
#endif
