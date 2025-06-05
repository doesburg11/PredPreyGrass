using UnityEngine;
using UnityEditor;

namespace UnityEditor.U2D.Common.Path
{
    internal interface ISnapping<T>
    {
        T Snap(T value);
    }
}
