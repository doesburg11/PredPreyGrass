using System;
using System.IO;
using System.Reflection;
using NUnit.Framework;

namespace Tests_13
{
    internal class ScriptAddingTests
    {
        private const string pathToFile = "Assets/Tests_13/TempScript.cs"; 
        
        [Test]
        public void YourTestGoesHere()
        {
            
        }
        
        private void CreateScript()
        {
            try
            {
                File.WriteAllText(pathToFile, @"
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