using System.Text.RegularExpressions;
using MyExercise_5s;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Tests_5s
{
    internal class MyLoggingClassTests
    {
        [Test]
        public void DoSomethingLogsMessage()
        {
            var loggingClassUnderTest = new MyLoggingClass();
            
            loggingClassUnderTest.DoSomething();
            
            LogAssert.Expect(LogType.Log, "Doing something");
        }

        [Test]
        public void DoSomethingElseLogsError()
        {
            var loggingClassUnderTest = new MyLoggingClass();
            
            loggingClassUnderTest.DoSomethingElse();
            
            LogAssert.Expect(LogType.Error, new Regex("An error happened. Code: \\d"));
        }
    }
}