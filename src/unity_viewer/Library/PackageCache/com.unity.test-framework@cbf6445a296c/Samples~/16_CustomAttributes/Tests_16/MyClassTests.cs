using NUnit.Framework;

namespace Tests_16
{
    public class MyClassTests
    {
        [Test]
        public void SomethingReturnsTrue()
        {
            var classUnderTest = new MyClass();

            var result = classUnderTest.DoSomething();
            
            Assert.That(result, Is.True);
        }
        
        [Test]
        public void SomethingElseReturnsTrue()
        {
            var classUnderTest = new MyClass();

            var result = classUnderTest.DoSomethingElse();
            
            Assert.That(result, Is.True);
        }
    }
}