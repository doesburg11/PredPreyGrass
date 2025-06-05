using MyExercise_3s;
using NUnit.Framework;

namespace Tests_3s
{
    internal class ValueOutputterTests
    {
        [Test]
        public void GivesExpectedInt()
        {
            var outputterUnderTest = new ValueOutputter();

            var number = outputterUnderTest.GetInt();
            
            Assert.That(number, Is.EqualTo(11));
        }
        
        [Test]
        public void GivesExpectedString()
        {
            var outputterUnderTest = new ValueOutputter();

            var str = outputterUnderTest.GetString();
            
            Assert.That(str, Does.Contain("string").And.Contain("asserted"));
        }
        
        [Test]
        public void GivesExpectedFloat()
        {
            var outputterUnderTest = new ValueOutputter();

            var number = outputterUnderTest.GetFloat();
            
            Assert.That(number, Is.GreaterThan(19.33f).And.LessThan(19.34f));
        }
    }
}