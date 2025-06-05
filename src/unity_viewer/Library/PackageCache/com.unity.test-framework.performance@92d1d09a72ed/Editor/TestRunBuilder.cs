using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Unity.PerformanceTesting.Data;
using Unity.PerformanceTesting.Editor;
using Unity.PerformanceTesting.Runtime;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

[assembly: PrebuildSetup(typeof(TestRunBuilder))]
[assembly: PostBuildCleanup(typeof(TestRunBuilder))]

namespace Unity.PerformanceTesting.Editor
{
    internal class TestRunBuilder : IPrebuildSetup, IPostBuildCleanup, IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        private const string cleanResources = "PT_ResourcesCleanup";

        public int callbackOrder
        {
            get { return 0; }
        }

        public void OnPreprocessBuild(BuildReport report)
        {
            CreateResourcesFolder();

            var run = CreateBuildInfo();
            SaveToStorage(run, Utils.TestRunPath);

            var settings = new RunSettings(Environment.GetCommandLineArgs());
            SaveToStorage(settings, Utils.RunSettingsPath);
        }
        
        public void OnPostprocessBuild(BuildReport report)
        {
            Cleanup();
        }

        public void Setup()
        {
            EditorPrefs.SetBool(cleanResources, false);

            var run = CreateRunInfo();
            SaveToPrefs(run, Utils.PlayerPrefKeyRunJSON);

            var settings = new RunSettings(Environment.GetCommandLineArgs());
            SaveToPrefs(settings, Utils.PlayerPrefKeySettingsJSON);
        }

#if !UNITY_2021_1_OR_NEWER
        private static List<string> cachedDependencies;
#endif
        static List<string> GetPackageDependencies()
        {
#if !UNITY_2021_1_OR_NEWER
            if (cachedDependencies != null)
                return cachedDependencies;
#endif

            IEnumerable<UnityEditor.PackageManager.PackageInfo> packages;
#if !UNITY_2021_1_OR_NEWER
            var listRequest = UnityEditor.PackageManager.Client.List(true);
            while (!listRequest.IsCompleted)
                System.Threading.Thread.Sleep(10);
            if (listRequest.Status == UnityEditor.PackageManager.StatusCode.Failure)
                Debug.LogError("Failed to list local packages");
            packages = new List<UnityEditor.PackageManager.PackageInfo>(listRequest.Result);
#else
            packages = UnityEditor.PackageManager.PackageInfo.GetAllRegisteredPackages();
#endif
            var reformated = packages.Select(p => $"{p.name}@{p.version}").ToList();
#if !UNITY_2021_1_OR_NEWER
            cachedDependencies = reformated;
#endif
            return reformated;
        }

        public void Cleanup()
        {
            DeleteFileAndMeta(Utils.TestRunPath);
            DeleteFileAndMeta(Utils.RunSettingsPath);

            if (EditorPrefs.GetBool(cleanResources) && Directory.Exists("Assets/Resources"))
            {
                Directory.Delete("Assets/Resources/", true);
                if(File.Exists("Assets/Resources.meta")) {File.Delete("Assets/Resources.meta");}
            }

            AssetDatabase.Refresh();
        }

        private void DeleteFileAndMeta(string path)
        {
            if (File.Exists(path)) { File.Delete(path); }
            var metaPath = path + ".meta";
            if (File.Exists(metaPath)) { File.Delete(metaPath); }
        } 

        private static Data.Editor GetEditorInfo()
        {
            var fullVersion = UnityEditorInternal.InternalEditorUtility.GetFullUnityVersion();
            const string pattern = @"(.+\.+.+\.\w+)|((?<=\().+(?=\)))";
            var matches = Regex.Matches(fullVersion, pattern);

            return new Data.Editor
            {
                Branch = GetEditorBranch(),
                Version = matches[0].Value,
                Changeset = matches[1].Value,
                Date = UnityEditorInternal.InternalEditorUtility.GetUnityVersionDate(),
            };
        }

        private static string GetEditorBranch()
        {
            foreach (var method in typeof(UnityEditorInternal.InternalEditorUtility).GetMethods())
            {
                if (method.Name.Contains("GetUnityBuildBranch"))
                {
                    return (string) method.Invoke(null, null);
                }
            }

            return "null";
        }

        private static void SetBuildSettings(Run run)
        {
            if (run.Player == null) run.Player = new Player();

            run.Player.GpuSkinning = PlayerSettings.gpuSkinning;
            #if UNITY_2021_2_OR_NEWER
            run.Player.ScriptingBackend = PlayerSettings
                .GetScriptingBackend(NamedBuildTarget.FromBuildTargetGroup(EditorUserBuildSettings.selectedBuildTargetGroup)).ToString();
            #else 
            run.Player.ScriptingBackend = PlayerSettings
                .GetScriptingBackend(EditorUserBuildSettings.selectedBuildTargetGroup).ToString();
            #endif
            run.Player.RenderThreadingMode = PlayerSettings.graphicsJobs ? "GraphicsJobs" :
                PlayerSettings.MTRendering ? "MultiThreaded" : "SingleThreaded";
            run.Player.AndroidTargetSdkVersion = PlayerSettings.Android.targetSdkVersion.ToString();
            run.Player.AndroidBuildSystem = EditorUserBuildSettings.androidBuildSystem.ToString();
            run.Player.BuildTarget = EditorUserBuildSettings.activeBuildTarget.ToString();
            run.Player.StereoRenderingPath = PlayerSettings.stereoRenderingPath.ToString();
        }

        public Run CreateRunInfo()
        {
            var run = new Run();
            run.Editor = GetEditorInfo();
            run.Dependencies = GetPackageDependencies();
            SetBuildSettings(run);
            run.Date = Utils.ConvertToUnixTimestamp(DateTime.Now);

            return run;
        }
        public Run CreateBuildInfo()
        {
            var run = new Run();
            run.Editor = GetEditorInfo();
            run.Dependencies = GetPackageDependencies();
            SetBuildSettings(run);

            return run;
        }

        public Run GetPerformanceTestRun()
        {
            var run = CreateRunInfo();
            Metadata.SetRuntimeSettings(run);

            return run;
        }


        private void CreateResourcesFolder()
        {
            if (Directory.Exists(Utils.ResourcesPath))
            {
                EditorPrefs.SetBool(cleanResources, false);
                return;
            }

            EditorPrefs.SetBool(cleanResources, true);
            AssetDatabase.CreateFolder("Assets", "Resources");
        }

        private string SaveToStorage(object obj, string path)
        {
            var json = JsonUtility.ToJson(obj);
            File.WriteAllText(path, json);
            return json;
        }

        private string SaveToPrefs(object obj, string key)
        {
            var json = JsonUtility.ToJson(obj, true);
            PlayerPrefs.SetString(key, json);
            return json;
        }
    }
}
