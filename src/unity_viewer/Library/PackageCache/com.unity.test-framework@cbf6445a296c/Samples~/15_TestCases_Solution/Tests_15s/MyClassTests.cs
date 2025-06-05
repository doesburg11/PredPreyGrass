using System.Collections;
using NUnit.Framework;
using UnityEngine.TestTools;

namespace Tests_15s
{
    public class MyClassTests
    {
        [Test]
        [TestCase(24, 80, 104)]
        [TestCase(10, -15, -5)]
        [TestCase(int.MaxValue, 10, int.MinValue + 9)]
        public void AddCalculatesCorrectValue(int valueA, int valueB, int expectedResult)
        {
            var myClass = new MyClass();

            var result = myClass.Add(valueA, valueB);
            
            Assert.That(result, Is.EqualTo(expectedResult));
        }
        
        [UnityTest]
        public IEnumerator AddAsyncCalculatesCorrectValue([ValueSource(nameof(AdditionCases))] AddCase addCase)
        {
            var myClass = new MyClass();

            var enumerator = myClass.AddAsync(addCase.valueA, addCase.valueB);
            while (enumerator.MoveNext())
            {
                yield return null;
            }
            var result = enumerator.Current;
            
            Assert.That(result, Is.EqualTo(addCase.expectedResult));
        }

        private static IEnumerable AdditionCases()
        {
            yield return new AddCase {valueA = 24, valueB = 80, expectedResult = 104};
            yield return new AddCase {valueA = 10, valueB = -15, expectedResult = -5};
            yield return new AddCase {valueA = int.MaxValue, valueB = 10, expectedResult = int.MinValue + 9};
        }

        public struct AddCase
        {
            public int valueA;
            public int valueB;
            public int expectedResult;

            public override string ToString()
            {
                return $"{valueA} + {valueB} = {expectedResult}";
            }
        }
    }
}
