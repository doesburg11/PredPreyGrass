using System;
using System.Collections;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Tests_14
{
    internal class ScriptAddingTests
    {
        private const string k_filePath = @"Assets\\Tests";
        
        private string fileName;
        
        [UnitySetUp]
        public IEnumerator Setup()
        {
            fileName = Guid.NewGuid() + ".cs";
            CreateScript(fileName);
            yield return new RecompileScripts();
        }

        [Test]
        public void CreatedScriptIsVerified()
        {
            var verification = VerifyScript();
            
            Assert.That(verification, Is.EqualTo("OK"));
        }

        [UnityTearDown]
        public IEnumerator Teardown()
        {
            if (string.IsNullOrEmpty(fileName))
            {
                Assert.Fail("Could not cleanup file. File name is not available.");
            }
            
            var path = Path.Combine(k_filePath, fileName);
            if (!File.Exists(path))
            {
                yield break;
            }
            
            File.Delete(path);
            yield return new RecompileScripts();
        }
        
        private void CreateScript(string fileName)
        {
            var path = Path.Combine(k_filePath, fileName);
            try
            {
                File.WriteAllText(path, @"
                public class MyTempScript {
                    public string Verify()
                    {
                        return ""OK"";
                    }    
                }");
            }
            catch(DirectoryNotFoundException)
            {
                Assert.Inconclusive("The path to file is incorrect. Please make sure that the path to TempScript is valid.");
            }
        }

        private string VerifyScript()
        {
            Type type = Type.GetType("MyTempScript", true);
            
            object instance = Activator.CreateInstance(type);

            var verifyMethod = type.GetMethod("Verify", BindingFlags.Instance | BindingFlags.Public);

            var verifyResult = verifyMethod.Invoke(instance, new object[0]);
            return verifyResult as string;
        }
    }
}