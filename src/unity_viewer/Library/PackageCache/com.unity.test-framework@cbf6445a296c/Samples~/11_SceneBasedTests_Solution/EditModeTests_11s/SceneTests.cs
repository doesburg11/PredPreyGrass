using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.TestTools;

namespace EditModeTests_11s
{
    public class SceneTests
    {
        private const string pathToScene = "Assets/MyGameScene.unity";
        
        [SetUp]
        public void Setup()
        {
            if (!File.Exists(pathToScene))
            {
                Assert.Inconclusive("The path to the Scene is not correct. Set the correct path for the pathToScene variable.");
            }
            EditorSceneManager.OpenScene(pathToScene);
        }
        
        [Test]
        public void VerifyScene()
        {
            var gameObject = GameObject.Find("GameObjectToTestFor");
            
            Assert.That(gameObject, Is.Not.Null);
        }

        [TearDown]
        public void Teardown()
        {
            EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        }
    }
}
