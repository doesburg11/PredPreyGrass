using NUnit.Framework;

namespace Tests_16s
{
    public class MyClassTests
    {
        [Test]
        [FasterThan500ms]
        public void SomethingReturnsTrue()
        {
            var classUnderTest = new MyClass();

            var result = classUnderTest.DoSomething();
            
            Assert.That(result, Is.True);
        }
        
        [Test]
        [FasterThan500ms]
        public void SomethingElseReturnsTrue()
        {
            var classUnderTest = new MyClass();

            var result = classUnderTest.DoSomethingElse();
            
            Assert.That(result, Is.True);
        }
    }
}