# 4\. Custom comparison

## Learning objectives

This exercise will cover the custom equality comparers included in Unity Test Framework, such as `Vector3EqualityComparer`. These are used to assert on e.g. Vectors.

## Intro and motivation

We have extended the assertion capabilities of NUnit with some custom comparisons for Unity-specific objects. A good example of this is the ability to compare two `Vector3` objects.  
  
An example of its use is:  
  
```
actual = new Vector3(0.01f, 0.01f, 0f);
expected = new Vector3(0.01f, 0.01f, 0f);

Assert.That(actual, Is.EqualTo(expected).Using(Vector3EqualityComparer.Instance));
```
  
This allows us to verify that the two vectors are identical within a given tolerence. By default the tolerance is 0.0001f. The tolerance can be changed by providing a new `Vector3EqualityComparer`, instead of using the default in .instance. For example you can up the tolerance to 0.01f with the following:  
  
```
Assert.That(actual, Is.EqualTo(expected).Using(new Vector3EqualityComparer(0.01f));
```
  
For a list of all available custom comparers, see [Custom equality comparers](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-custom-equality-comparers.html).

## Exercise

Similar to the project for exercise 3, the [sample](./welcome.md#import-samples) `4_CustomComparison` contains a `ValueOutputter` class.  
  
Verify that the `ValueOutputter` returns the correct values from its methods: 

*   `GetVector3()` should return a `Vector3` that is roughly equal to (10.333, 3, 9.666).
  
*   `GetFloat()` should return a `float` that is roughly 19.333. This is the same as previous exercise, but you can try to solve this with a `FloatEqualityComparer`.
  
*   `GetQuaternion` should return a [Quaternion](https://docs.unity3d.com/ScriptReference/Quaternion.html) object that should be roughly equal to (10f, 0f, 7.33333f, 0f).

## Hints

*   For some of the exercises, you might need to provide a custom error tolerance to the comparer.
*   If the comparison fails, the comparers give a message about the actual and expected value, just like a normal assertion. However, because `ToString` on `Vector3` rounds the value off before displaying it, the two values in the string message might be equal, even when their `Vector3` values are not.

## Solution

The full solution is available in the sample `4_CustomComparison_Solution`.  
  
```
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
```

## Further reading and resources

[Custom equality comparers](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-custom-equality-comparers.html)
