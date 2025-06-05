using System.Collections;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace PlayModeTests_12
{
    public class SceneTests
    {
        private string originalScene;
        private const string k_SceneName = "Assets/MyGameScene.unity";
        
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
    }
}
