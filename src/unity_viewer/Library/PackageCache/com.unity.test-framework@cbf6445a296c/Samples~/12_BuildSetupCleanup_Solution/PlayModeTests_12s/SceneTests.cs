using System.Collections;
using System.Linq;
using System.IO;
using NUnit.Framework;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace PlayModeTests_12s
{
    public class SceneTests : IPrebuildSetup, IPostBuildCleanup
    {
        private string originalScene;
        private const string k_SceneName = "Assets/MyGameScene.unity";

        public void Setup()
        {
#if UNITY_EDITOR
            if (EditorBuildSettings.scenes.Any(scene => scene.path == k_SceneName))
            {
                return;
            }
            
            var includedScenes = EditorBuildSettings.scenes.ToList();
            includedScenes.Add(new EditorBuildSettingsScene(k_SceneName, true));
            EditorBuildSettings.scenes = includedScenes.ToArray();
#endif
        }

        [UnitySetUp]
        public IEnumerator SetupBeforeTest()
        {
            originalScene = SceneManager.GetActiveScene().path;
            if (!File.Exists(k_SceneName))
            {
                Assert.Inconclusive("The path to the Scene is not correct. Set the correct path for the k_SceneName variable.");
            }
            SceneManager.LoadScene(k_SceneName);
            yield return null; // Skip a frame, allowing the scene to load.
        }

        [Test]
        public void VerifyScene()
        {
            var gameObject = GameObject.Find("GameObjectToTestFor");

            Assert.That(gameObject, Is.Not.Null, $"GameObjectToTestFor not found in {SceneManager.GetActiveScene().path}.");
        }

        [TearDown]
        public void TeardownAfterTest()
        {
            SceneManager.LoadScene(originalScene);
        }

        public void Cleanup()
        {
#if UNITY_EDITOR
            EditorBuildSettings.scenes = EditorBuildSettings.scenes.Where(scene => scene.path != k_SceneName).ToArray();
#endif
        }
    }
}
