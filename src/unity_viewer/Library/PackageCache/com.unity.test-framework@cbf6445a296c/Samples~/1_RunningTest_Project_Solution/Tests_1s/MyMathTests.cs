using MyExercise_1s;
using NUnit.Framework;

namespace Tests_1s
{
    public class MyMathTests
    {
        [Test]
        public void AddsTwoPositiveIntegers()
        {
            Assert.AreEqual(3, MyMath.Add(1, 2));
        }
        
        [Test]
        public void AddAPositiveAndNegativeInteger()
        {
            Assert.AreEqual(1, MyMath.Add(3, -2));
        }
        
        [Test]
        public void SubtractAPositiveInteger()
        {
            Assert.AreEqual(3, MyMath.Subtract(5, 2));
        }
        
        [Test]
        public void SubtractANegativeInteger()
        {
            Assert.AreEqual(7, MyMath.Subtract(5, -2));
        }
    }
}
