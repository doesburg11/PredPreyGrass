using MyExercise_4s;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools.Utils;

namespace Tests_4s
{
    internal class ValueOutputterTests
    {
        [Test]
        public void Vector3ReturnsCorrectValue()
        {
            var valueOutputterUnderTest = new ValueOutputter();

            var vector3 = valueOutputterUnderTest.GetVector3();

            var expected = new Vector3(10.333f, 3f, 9.666f);
            Assert.That(vector3, Is.EqualTo(expected).Using(new Vector3EqualityComparer(0.001f)));
        }
        
        [Test]
        public void FloatReturnsCorrectValue()
        {
            var valueOutputterUnderTest = new ValueOutputter();

            var actualFloat = valueOutputterUnderTest.GetFloat();

            Assert.That(actualFloat, Is.EqualTo(19.333f).Using(new FloatEqualityComparer(0.001f)));
        }
        
        [Test]
        public void QuaternionReturnsCorrectValue()
        {
            var valueOutputterUnderTest = new ValueOutputter();

            var actualValue = valueOutputterUnderTest.GetQuaternion();

            var expectedValue = new Quaternion(10f, 0f, 7.33333f, 0f);
            Assert.That(actualValue, Is.EqualTo(expectedValue).Using(new QuaternionEqualityComparer(0.001f)));
        }
    }
}